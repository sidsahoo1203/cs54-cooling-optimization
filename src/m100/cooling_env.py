"""
CS54 PS3 — the cooling control environment.

A Gymnasium environment wrapping the learned twin from 06_build_twin.py.

WHAT THE AGENT CONTROLS
    the CRAC supply-air temperature setpoint, rate-limited to a realistic
    +/- 0.5 degC per 5-minute control step. Real CRAC units cannot jump.

WHAT IT CANNOT CONTROL
    the workload. IT power and outdoor temperature are replayed from the real
    trace. This is not a simplification, it is the physical truth: cooling
    control does not decide which jobs run. It is also what makes the oracle
    forecast arm legitimate — the future load is genuinely exogenous, so
    reading it ahead from the trace is a valid upper bound, not cheating.

THE THREE OBSERVATION MODES — this is the experiment
    none      the agent sees only the present
    oracle    the agent additionally sees the TRUE future IT power
    forecast  the agent sees a real prediction of it

    The reward, the dynamics and the agent architecture are identical in all
    three. Only the observation differs. So any difference in performance is
    attributable to the forecast and to nothing else, which is precisely the
    decomposition the literature has never reported.

REWARD
    -(cooling power / scale)  -  VIOLATION_WEIGHT * max(0, T_hot - 27)^2

    Energy is the objective; thermal safety is a penalty rather than a hard
    constraint so the agent can learn the trade-off rather than being clipped
    out of exploring it. The weight is reported with every result and swept in
    08_compare.py, because a conclusion that only holds at one weight is not a
    conclusion.
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # older installs
    import gym
    from gym import spaces


class CoolingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df, thermal, power, meta, cfg,
                 obs_mode="none", forecast_h=12, forecast_fn=None,
                 forecast_noise=0.0, episode_steps=None, rng_seed=0,
                 violation_weight=None):
        super().__init__()
        self.cfg = cfg
        self.state_col = meta["state"]
        self.action_col = meta["action"]
        self.tfeat = meta["thermal_features"]
        self.pfeat = meta.get("power_features")
        self.tmodel = thermal
        self.pmodel = power
        self.obs_mode = obs_mode
        self.forecast_h = int(forecast_h)
        self.forecast_fn = forecast_fn
        self.forecast_noise = float(forecast_noise)
        self.sp_lo = float(meta["setpoint_min"])
        self.sp_hi = float(meta["setpoint_max"])
        self.rate = float(cfg.SETPOINT_RATE_LIMIT)
        self.hi_limit = float(cfg.VIOLATION_LIMIT)
        self.vw = float(cfg.VIOLATION_WEIGHT if violation_weight is None
                        else violation_weight)
        self.ep_len = int(episode_steps or cfg.EPISODE_STEPS)
        self.rng = np.random.default_rng(rng_seed)

        # ---- the replay trace -------------------------------------------
        need = list(dict.fromkeys(self.tfeat + (self.pfeat or []) +
                                  [self.state_col, "it_power"]))
        self.df = df.dropna(subset=[c for c in need if c in df.columns]).reset_index(drop=True)
        if len(self.df) < self.ep_len + 50:
            raise ValueError(f"only {len(self.df)} usable rows; need "
                             f"{self.ep_len + 50}. Lower EPISODE_STEPS.")
        self.X = self.df[self.tfeat].to_numpy("float64")
        self.XP = (self.df[self.pfeat].to_numpy("float64")
                   if self.pfeat else None)
        self.state_series = self.df[self.state_col].to_numpy("float64")
        self.load = self.df["it_power"].to_numpy("float64")
        self.load_mu = float(self.load.mean())
        self.load_sd = float(self.load.std() + 1e-9)

        # Hotspot offset: the models work on the rack-MEAN inlet, but ASHRAE is
        # judged on the hottest rack. A fixed offset calibrated from the data is
        # crude but honest, and it is stated as a limitation in the paper.
        if "rack_inlet_p95" in self.df.columns:
            self.hot_offset = float(
                (self.df["rack_inlet_p95"] - self.df[self.state_col]).median())
            self.hot_basis = "rack_inlet_p95"
        elif "rack_inlet_max" in self.df.columns:
            self.hot_offset = float(
                (self.df["rack_inlet_max"] - self.df[self.state_col]).median())
            self.hot_basis = "rack_inlet_max"
        else:
            self.hot_offset, self.hot_basis = 0.0, self.state_col

        # Calibration check on the thermal limit. If the RECORDED trace already
        # violates at almost every step, the penalty is a near-constant the agent
        # cannot influence, so it carries no gradient and the controller learns
        # nothing from it. That happens when the hotspot offset is taken from the
        # single hottest node in the whole room rather than a representative one.
        rec_hot = self.df[self.state_col].to_numpy("float64") + self.hot_offset
        self.recorded_violation_rate = float((rec_hot > self.hi_limit).mean())
        if self.recorded_violation_rate > 0.5:
            # Fall back to the ASHRAE ALLOWABLE bound. A penalty the recorded
            # record incurs at every single step is a constant: it carries no
            # gradient, so no policy can differ from any other on it.
            alt = float(cfg.ASHRAE_ALLOWABLE_A1[1])
            alt_rate = float(((self.df[self.state_col].to_numpy("float64")
                               + self.hot_offset) > alt).mean())
            if alt_rate < self.recorded_violation_rate:
                self.hi_limit = alt
                self.recorded_violation_rate = alt_rate
        if self.recorded_violation_rate > 0.5:
            import warnings as _w
            _w.warn(
                f"\n  !! The recorded trace itself violates the {self.hi_limit} degC "
                f"limit in {self.recorded_violation_rate * 100:.1f}% of steps "
                f"(hotspot offset {self.hot_offset:+.2f} degC).\n"
                f"     The penalty is then near-constant and the agent cannot move "
                f"it, so every policy will look identical.\n"
                f"     Either widen the limit to the ASHRAE ALLOWABLE bound "
                f"({32.0} degC) or use a representative rack rather than the "
                f"room-wide maximum.\n", RuntimeWarning, stacklevel=2)

        self.cool_scale = (float(self.df["cooling_power"].mean())
                           if "cooling_power" in self.df.columns
                           and self.df["cooling_power"].notna().any() else 1.0)

        # ---- feature bookkeeping for the autoregressive state ------------
        self.i_state = self.tfeat.index(self.state_col)
        self.i_action = self.tfeat.index(self.action_col)
        self.lag_idx = [(l, self.tfeat.index(f"{self.state_col}_lag{l}"))
                        for l in cfg.TWIN_LAGS
                        if f"{self.state_col}_lag{l}" in self.tfeat]
        self.d_idx = [(k, self.tfeat.index(f"{self.state_col}_d{k}"))
                      for k in (1, 2, 3)
                      if f"{self.state_col}_d{k}" in self.tfeat]
        self.buflen = max([l for l, _ in self.lag_idx] + [3]) + 1
        self.ip_action = (self.pfeat.index(self.action_col)
                          if self.pfeat and self.action_col in self.pfeat else None)
        self.ip_state = (self.pfeat.index(self.state_col)
                         if self.pfeat and self.state_col in self.pfeat else None)

        # ---- spaces ------------------------------------------------------
        # Action is the CHANGE in setpoint, scaled to [-1, 1]. Asking the agent
        # for a delta rather than an absolute value makes the rate limit part of
        # the action space instead of a post-hoc clip it has to learn around.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        n_base = 8
        n_fc = self.forecast_h if obs_mode in ("oracle", "forecast") else 0
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(n_base + n_fc,), dtype=np.float32)
        self._t = 0
        self._start = 0

    # ---------------------------------------------------------------- obs
    def _forecast(self):
        """Future IT power, normalised. Empty in 'none' mode."""
        if self.obs_mode == "none":
            return np.zeros(0, dtype=np.float32)
        j = self._start + self._t
        end = min(j + self.forecast_h, len(self.load))
        fut = self.load[j:end]
        if len(fut) < self.forecast_h:
            fut = np.pad(fut, (0, self.forecast_h - len(fut)), mode="edge")
        if self.obs_mode == "forecast":
            if self.forecast_fn is not None:
                fut = self.forecast_fn(self.df, j, self.forecast_h)
            if self.forecast_noise > 0:
                # Degrading the oracle with calibrated noise is how the accuracy
                # sweep works: it traces the whole curve between a perfect
                # forecast and no forecast, without needing a different model at
                # every accuracy level.
                fut = fut + self.rng.normal(0, self.forecast_noise * self.load_sd,
                                            size=len(fut))
        return ((np.asarray(fut) - self.load_mu) / self.load_sd).astype(np.float32)

    def _obs(self):
        j = self._start + self._t
        row = self.X[j]
        hot = self.state + self.hot_offset
        base = np.array([
            self.state,
            hot - self.hi_limit,                   # headroom to the ASHRAE limit
            (self.load[j] - self.load_mu) / self.load_sd,
            self.setpoint,
            self.setpoint - self.sp_lo,
            self.sp_hi - self.setpoint,
            row[self.tfeat.index("tod_sin")] if "tod_sin" in self.tfeat else 0.0,
            row[self.tfeat.index("tod_cos")] if "tod_cos" in self.tfeat else 0.0,
        ], dtype=np.float32)
        return np.concatenate([base, self._forecast()]).astype(np.float32)

    # ---------------------------------------------------------------- api
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        hi = len(self.df) - self.ep_len - 2
        self._start = int(self.rng.integers(self.buflen, max(self.buflen + 1, hi)))
        self._t = 0
        j = self._start
        self.buf = list(self.state_series[j - self.buflen + 1: j + 1])
        self.state = float(self.buf[-1])
        self.setpoint = float(np.clip(self.X[j, self.i_action], self.sp_lo, self.sp_hi))
        self.ep_cool = 0.0
        self.ep_viol = 0.0
        self.ep_over = 0
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).ravel()[0], -1, 1))
        self.setpoint = float(np.clip(self.setpoint + a * self.rate,
                                      self.sp_lo, self.sp_hi))
        j = self._start + self._t

        row = self.X[j].copy()
        row[self.i_state] = self.buf[-1]
        row[self.i_action] = self.setpoint
        for l, i_ in self.lag_idx:
            row[i_] = self.buf[-1 - l]
        for k, i_ in self.d_idx:
            row[i_] = self.buf[-1] - self.buf[-1 - k]
        d = float(self.tmodel.predict(row.reshape(1, -1).astype("float32"))[0])
        self.state = self.buf[-1] + d
        self.buf.append(self.state); self.buf.pop(0)

        if self.pmodel is not None and self.XP is not None:
            prow = self.XP[j].copy()
            if self.ip_action is not None:
                prow[self.ip_action] = self.setpoint
            if self.ip_state is not None:
                prow[self.ip_state] = self.state
            cool = float(self.pmodel.predict(prow.reshape(1, -1).astype("float32"))[0])
        else:
            cool = self.cool_scale

        hot = self.state + self.hot_offset
        over = max(0.0, hot - self.hi_limit)
        reward = -(cool / self.cool_scale) - self.vw * over ** 2

        self.ep_cool += cool * (self.cfg.STEP_MINUTES / 60.0)   # kWh
        self.ep_viol += over
        self.ep_over += int(over > 0)
        self._t += 1
        done = self._t >= self.ep_len
        info = {}
        if done:
            info = {"cooling_kWh": self.ep_cool,
                    "violation_degC_steps": self.ep_viol,
                    "violation_steps": self.ep_over,
                    "violation_rate": self.ep_over / self.ep_len,
                    "final_setpoint": self.setpoint}
        return self._obs(), float(reward), done, False, info


class FixedSetpointPolicy:
    """
    The conventional strategy the case study asks you to beat: a fixed setpoint,
    held regardless of conditions. It is what most data centres actually do, and
    it is the honest comparator — not a deliberately bad controller.
    """

    def __init__(self, target):
        self.target = float(target)

    def __call__(self, env):
        err = self.target - env.setpoint
        return np.array([np.clip(err / env.rate, -1, 1)], dtype=np.float32)


class ReactivePolicy:
    """
    A slightly smarter rule: lower the setpoint when the hotspot approaches the
    ASHRAE limit, raise it when there is comfortable headroom. This is roughly
    what a well-tuned conventional control loop does, and beating it is a much
    more meaningful claim than beating a fixed setpoint.
    """

    def __init__(self, limit, margin=2.0, step=1.0):
        self.limit, self.margin, self.step = float(limit), float(margin), float(step)

    def __call__(self, env):
        hot = env.state + env.hot_offset
        head = self.limit - hot
        if head < self.margin:
            return np.array([-self.step], dtype=np.float32)
        if head > 2 * self.margin:
            return np.array([+self.step], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)


def evaluate(env, policy, episodes=10, seed=0):
    """Run a callable policy (or an SB3 model) and return per-episode metrics."""
    rows = []
    for e in range(episodes):
        obs, _ = env.reset(seed=seed + e)
        done = False
        total = 0.0
        while not done:
            if hasattr(policy, "predict"):          # stable-baselines3 model
                a, _ = policy.predict(obs, deterministic=True)
            else:                                   # rule-based callable
                a = policy(env)
            obs, r, done, _, info = env.step(a)
            total += r
        info["reward"] = total
        rows.append(info)
    return rows
