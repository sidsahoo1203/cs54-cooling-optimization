#!/usr/bin/env python3
"""
CS54 PS3 Step 3 — train the controller and run the three-way comparison.

THIS IS THE EXPERIMENT THAT ANSWERS THE RESEARCH GAP.

From the 32-journal review, four things are already established and cannot be
claimed as new: ML predicts data-centre temperature accurately; RL beats
rule-based cooling by 5-13%; thermal safety during RL training is solved; and
coupling a forecast into an RL controller is already published (papers 20, 30,
31). What nobody has done is separate how much of a coupled system's benefit
comes from the FORECAST and how much from the CONTROLLER.

So the same PPO agent, the same reward, the same dynamics, three observations:

    none      sees only the present                  -> reactive ceiling
    oracle    sees the TRUE future IT load           -> theoretical ceiling
    forecast  sees a real prediction of it           -> what is achievable

    none -> oracle    is what anticipation is worth AT ALL
    oracle -> forecast is what imperfect prediction COSTS you

Those two numbers are the contribution. The oracle arm is free: the complete
historical trace is on disk, so the true future can simply be read ahead.

Two rule-based baselines are run first, because "RL beats cooling control" is
only interesting against a comparator anyone would actually deploy.

    python 07_train_rl.py --quick        # tiny run, proves the chain works
    python 07_train_rl.py                # full: 3 modes x 5 seeds
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "m100"))
import config_ps3 as P
from cooling_env import CoolingEnv, FixedSetpointPolicy, ReactivePolicy, evaluate
from common import banner


def make_env(df, th, pw, meta, mode, horizon, seed, noise=0.0, vw=None, ep=None):
    return CoolingEnv(df, th, pw, meta, P, obs_mode=mode, forecast_h=horizon,
                      forecast_noise=noise, rng_seed=seed, violation_weight=vw,
                      episode_steps=ep)


def summarise(rows, label):
    d = pd.DataFrame(rows)
    return {
        "policy": label,
        "cooling_kWh_mean": d["cooling_kWh"].mean(),
        "cooling_kWh_std": d["cooling_kWh"].std(),
        "violation_rate": d["violation_rate"].mean(),
        "violation_degC_steps": d["violation_degC_steps"].mean(),
        "reward": d["reward"].mean(),
        "episodes": len(d),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--timesteps", type=int, default=P.PPO_TIMESTEPS)
    ap.add_argument("--seeds", type=int, nargs="+", default=P.RL_SEEDS)
    ap.add_argument("--horizon", type=int, default=P.DEFAULT_FORECAST_H)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--episode-steps", type=int, default=None)
    ap.add_argument("--forecast-noise", type=float, default=0.15,
                    help="stand-in predictor error, as a multiple of the load's "
                         "own std, used ONLY until the real PS1 predictor is wired in")
    args = ap.parse_args()
    if args.quick:
        args.timesteps = min(args.timesteps, 4000)
        args.seeds = args.seeds[:1]
        args.episodes = 3
        args.episode_steps = args.episode_steps or 48

    if not (P.MODELS_DIR / "twin_meta.json").exists():
        print("ERROR: twin not built. Run 06_build_twin.py first.")
        return 1

    banner("CS54 PS3 — controller training and the three-way comparison")
    import joblib
    meta = json.loads((P.MODELS_DIR / "twin_meta.json").read_text())
    th = joblib.load(P.MODELS_DIR / "twin_thermal.joblib")["model"]
    pw = joblib.load(P.MODELS_DIR / "twin_power.joblib")["model"] \
        if (P.MODELS_DIR / "twin_power.joblib").exists() else None

    src = P.PROCESSED_DIR / f"twin_features_{P.MONTH}.parquet"
    if not src.exists():
        print(f"ERROR: {src} not found. 06_build_twin.py should have written it.")
        return 1
    df = pd.read_parquet(src)
    print(f"  trace {len(df):,} rows · state '{meta['state']}' · "
          f"action '{meta['action']}' in "
          f"[{meta['setpoint_min']:.2f}, {meta['setpoint_max']:.2f}] degC")
    vh = int(meta.get("validity_horizon_min", 0) or 0)
    if vh:
        cap = max(1, vh // P.STEP_MINUTES)
        if args.horizon > cap:
            print(f"  !! forecast horizon {args.horizon * P.STEP_MINUTES} min exceeds the")
            print(f"     twin's validity horizon of {vh} min. Beyond that the twin is")
            print(f"     beaten by holding temperature constant, so the agent would be")
            print(f"     planning inside a model that is worse than no model.")
            print(f"     Capping the horizon to {cap} steps ({vh} min).")
            args.horizon = cap
    print(f"  forecast horizon {args.horizon} steps "
          f"({args.horizon * P.STEP_MINUTES} min) · "
          f"violation weight {P.VIOLATION_WEIGHT}"
          + (f" · twin valid to {vh} min" if vh else ""))

    results = []

    # ------------------------------------------------ rule-based baselines
    banner("1. Conventional baselines")
    probe = make_env(df, th, pw, meta, "none", args.horizon, 0, ep=args.episode_steps)
    med_sp = float(np.median(df[meta["action"]].dropna()))
    unit = P.UNITS.get(meta["action"], ("", ""))[0] or "units"
    for name, pol in (
        (f"fixed_{meta['action']}_{med_sp:.1f}{unit}", FixedSetpointPolicy(med_sp)),
        ("reactive_rule", ReactivePolicy(P.VIOLATION_LIMIT)),
    ):
        r = summarise(evaluate(probe, pol, args.episodes, seed=7), name)
        results.append(r)
        print(f"  {name:<26} cooling {r['cooling_kWh_mean']:>9,.1f} kWh   "
              f"violations {r['violation_rate'] * 100:5.2f}%")

    # ------------------------------------------------ PPO, three observations
    banner("2. PPO under three observation conditions")
    print("  Identical agent, reward and dynamics in all three. Only what the")
    print("  agent can SEE changes.\n")
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("  stable-baselines3 not installed:")
        print("      TMPDIR=~/tmp pip install stable-baselines3 gymnasium")
        return 1

    if args.forecast_noise > 0:
        print(f"  !! The 'forecast' arm is currently a DEGRADED ORACLE: the true")
        print(f"     future load plus Gaussian noise at {args.forecast_noise:.2f} x its")
        print(f"     standard deviation. That is a stand-in, NOT your PS1 predictor.")
        print(f"     Until the real LSTM is wired in via forecast_fn, report this")
        print(f"     arm as a sensitivity analysis, never as 'the achievable case'.")
        print(f"     With --forecast-noise 0 it becomes identical to the oracle and")
        print(f"     the third arm tells you nothing.\n")

    for mode in P.OBS_MODES:
        per_seed = []
        noise = args.forecast_noise if mode == "forecast" else 0.0
        for sd in args.seeds:
            t0 = _time.time()
            env = make_env(df, th, pw, meta, mode, args.horizon, sd,
                           noise=noise, ep=args.episode_steps)
            model = PPO("MlpPolicy", env, seed=sd, verbose=0,
                        n_steps=512, batch_size=128, learning_rate=3e-4)
            model.learn(total_timesteps=args.timesteps)
            ev = make_env(df, th, pw, meta, mode, args.horizon, 10_000 + sd,
                          noise=noise, ep=args.episode_steps)
            rows = evaluate(ev, model, args.episodes, seed=10_000 + sd)
            s = summarise(rows, f"ppo_{mode}_seed{sd}")
            s["mode"], s["seed"] = mode, sd
            per_seed.append(s)
            print(f"  {mode:<9} seed {sd:<5} cooling "
                  f"{s['cooling_kWh_mean']:>9,.1f} kWh   "
                  f"violations {s['violation_rate'] * 100:5.2f}%   "
                  f"({_time.time() - t0:,.0f}s)")
        d = pd.DataFrame(per_seed)
        agg = {"policy": f"ppo_{mode}", "mode": mode,
               "cooling_kWh_mean": d["cooling_kWh_mean"].mean(),
               "cooling_kWh_std": d["cooling_kWh_mean"].std(),
               "violation_rate": d["violation_rate"].mean(),
               "violation_degC_steps": d["violation_degC_steps"].mean(),
               "reward": d["reward"].mean(), "episodes": int(d["episodes"].sum()),
               "n_seeds": len(d)}
        results.append(agg)
        print(f"  {'-> ' + mode:<26} {agg['cooling_kWh_mean']:>9,.1f} "
              f"+/- {agg['cooling_kWh_std']:,.1f} kWh over {len(d)} seeds\n")

    res = pd.DataFrame(results)
    res.to_csv(P.RESULTS_DIR / "ps3_controller_results.csv", index=False)

    # ------------------------------------------------ the decomposition
    underpowered = args.timesteps < 50_000 or len(args.seeds) < 3
    banner("3. THE RESULT — decomposing the benefit")
    if underpowered:
        print(f"  NOT REPORTABLE. This run used {args.timesteps:,} timesteps on "
              f"{len(args.seeds)} seed(s).")
        print("  PPO needs on the order of 100k+ timesteps and several seeds before")
        print("  its policy means anything. The figures below are from an agent that")
        print("  is still close to random, and the decomposition is the headline")
        print("  result of the whole paper — it must never come from a run like this.")
        print()
        print("  A reliable sign of an untrained agent: more information making")
        print("  performance WORSE. If the oracle arm loses to the no-forecast arm,")
        print("  that is impossible for a converged policy and tells you the numbers")
        print("  are noise.")
        print()
        print("  For a reportable run:")
        print("      nohup python 07_train_rl.py > ps3_full.log 2>&1 &")
        print(f"      (3 arms x {len(P.RL_SEEDS)} seeds x {P.PPO_TIMESTEPS:,} "
              f"timesteps — several hours, leave it overnight)")
        print()
    def get(p):
        r = res[res["policy"] == p]
        return float(r["cooling_kWh_mean"].iloc[0]) if len(r) else None

    base = get(f"fixed_{meta['action']}_{med_sp:.1f}{unit}")
    react = get("reactive_rule")
    none_, oracle, fc = get("ppo_none"), get("ppo_oracle"), get("ppo_forecast")
    pct = lambda a, b: (1 - b / a) * 100 if a and b else float("nan")
    def tag(a, b):
        v = pct(a, b)
        return f"{abs(v):5.2f}% {'SAVING' if v >= 0 else 'WORSE '}"

    print(f"  fixed setpoint          {base:>10,.1f} kWh   (reference)")
    print(f"  reactive rule           {react:>10,.1f} kWh   {tag(base, react)} vs fixed")
    print(f"  PPO, no forecast        {none_:>10,.1f} kWh   {tag(base, none_)} vs fixed")
    print(f"  PPO, real forecast      {fc:>10,.1f} kWh   {tag(base, fc)} vs fixed")
    print(f"  PPO, oracle forecast    {oracle:>10,.1f} kWh   {tag(base, oracle)} vs fixed")
    if none_ and base and none_ > base:
        print()
        print("  !! PPO is using MORE energy than a fixed setpoint. With a short")
        print("     --timesteps budget that simply means the agent is undertrained")
        print("     and is still close to random. Do not read anything into the")
        print("     decomposition below until a full run converges.")
    print()
    if none_ and oracle and oracle > none_:
        print("  !! The ORACLE arm used MORE energy than the NO-FORECAST arm.")
        print("     A perfect forecast cannot make a converged agent worse — it can")
        print("     always ignore the extra inputs. This ordering means the policies")
        print("     have not converged and the gaps below are sampling noise.")
        print()
    print(f"  VALUE OF ANTICIPATION   none -> oracle   {pct(none_, oracle):+.2f}%")
    print(f"  COST OF IMPERFECTION    oracle -> real   {pct(oracle, fc):+.2f}%")
    antic = pct(none_, oracle)
    if antic is not None and abs(antic) >= 0.5:
        print(f"  REALISED FRACTION       real / oracle    "
              f"{pct(none_, fc) / antic * 100:.1f}%  of the available benefit")
    else:
        print(f"  REALISED FRACTION       not reportable — the anticipation gap "
              f"({antic:+.2f}%) is within noise,")
        print(f"                          so a ratio against it is meaningless.")
    print()
    print("  These two numbers are the paper. Nobody has reported them, because")
    print("  published work measures the coupled system end to end and never")
    print("  runs the no-forecast and oracle arms that separate the two effects.")
    print()
    print("  Interpret honestly:")
    print("    anticipation gap LARGE  -> forecasting genuinely matters here, and")
    print("        the realised fraction says how much of it today's predictors")
    print("        actually capture")
    print("    anticipation gap SMALL  -> an equally publishable negative result:")
    print("        the field has been adding forecast machinery for little gain,")
    print("        and you have the controlled experiment that shows it")

    with open(P.RESULTS_DIR / "ps3_results.md", "w") as fh:
        fh.write("# CS54 Sub-problem (c) — controller results\n\n")
        fh.write(f"Month `{P.MONTH}` · {P.STEP_MINUTES}-min control interval · "
                 f"forecast horizon {args.horizon * P.STEP_MINUTES} min · "
                 f"violation weight {P.VIOLATION_WEIGHT} · "
                 f"{len(args.seeds)} seeds · {args.timesteps:,} timesteps\n\n")
        fh.write(res.to_markdown(index=False) + "\n\n")
        fh.write("## Decomposition\n\n")
        fh.write(f"- value of anticipation (none -> oracle): "
                 f"**{pct(none_, oracle):+.2f}%**\n")
        fh.write(f"- cost of imperfect prediction (oracle -> real): "
                 f"**{pct(oracle, fc):+.2f}%**\n")

    banner("Saved")
    print(f"  {P.RESULTS_DIR / 'ps3_controller_results.csv'}")
    print(f"  {P.RESULTS_DIR / 'ps3_results.md'}")
    print("\n  NEXT:  python 08_compare.py   (horizon and accuracy sweeps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
