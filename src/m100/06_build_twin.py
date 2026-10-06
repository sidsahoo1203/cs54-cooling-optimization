#!/usr/bin/env python3
"""
CS54 PS3 Step 2 — learn the controllable thermal + power twin.

Two models, both from real data:

  THERMAL   (state, lags, IT power, SETPOINT, outdoor temp, time) -> change in
            rack inlet temperature over one 5-minute step
  POWER     (SETPOINT, IT power, outdoor temp, rack inlet)        -> cooling kW

Together they are the environment the RL agent practises in. Three design
decisions carry the weight:

1. PREDICT THE CHANGE, NOT THE LEVEL — same reason as PS2. A model predicting
   the level scores R2 > 0.99 by echoing its input and has learned no physics.

2. PHYSICS IMPOSED AS MONOTONICITY CONSTRAINTS. Observational data is
   confounded: operators move the setpoint in response to conditions, so a
   purely statistical fit can learn a physically backwards response — and an RL
   agent will find and exploit exactly that error, producing a controller that
   "saves energy" by doing something impossible. XGBoost's monotone_constraints
   forbid it:
       raising the supply setpoint can never LOWER the rack inlet temperature
       raising the supply setpoint can never RAISE cooling power
   This is the "physics-guided" idea from papers 2, 5 and 32, in its cheapest
   usable form.

3. THE CAUSAL SANITY CHECK IS A FIRST-CLASS OUTPUT. The script sweeps the
   setpoint through both models and reports the response in degC per degC and
   kW per degC. If those gains are ~0, the agent has no lever and the result
   must say so — that is the lesson PS2's fan-speed analysis already taught.

    python 06_build_twin.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "m100"))
import config_ps3 as P
from common import banner, chronological_split, regression_metrics

STATE = P.THERMAL_STATE
ACTION = P.ACTION_VAR


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").copy()
    for col in [c for c in (STATE, "it_power", "supply_temp", "cooling_power")
                if c in df.columns]:
        for lag in P.TWIN_LAGS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        for w in P.TWIN_ROLL:
            r = df[col].rolling(w, min_periods=max(2, w // 2))
            df[f"{col}_rmean{w}"] = r.mean()
            df[f"{col}_rstd{w}"] = r.std()
    for k in (1, 2, 3):
        df[f"{STATE}_d{k}"] = df[STATE] - df[STATE].shift(k)
    return df


def capacity_for(n_train: int, n_feat: int) -> dict:
    """
    Match model capacity to the data actually available.

    The room-level facility table has ~5k training rows, 30x fewer than the
    rack-level PS2 table. Running 700 trees at depth 6 on that memorises noise
    and the test period then looks like a different process — which is what an
    R2 of -3.35 means. It is overfitting, not absence of signal: no signal gives
    R2 near zero, not large and negative.
    """
    if n_train < 12_000:
        return dict(n_estimators=300, max_depth=3, learning_rate=0.03,
                    min_child_weight=20, subsample=0.7, colsample_bytree=0.6,
                    reg_lambda=10.0, reg_alpha=1.0)
    return dict(n_estimators=700, max_depth=6, learning_rate=0.05,
                min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, reg_alpha=0.0)


def fit(df, feats, label, monotone_on, monotone_sign, name):
    """XGBoost with a monotonicity constraint on one named feature."""
    from xgboost import XGBRegressor
    cons = [monotone_sign if f == monotone_on else 0 for f in feats]
    stamps = np.sort(df["time"].unique())
    tr_s, va_s, te_s = chronological_split(len(stamps), P.TEST_FRACTION, P.VAL_FRACTION)
    t_tr, t_va, t_te = stamps[tr_s], stamps[va_s], stamps[te_s]
    need = feats + [label]
    tr = df[df.time.isin(t_tr)].dropna(subset=need)
    va = df[df.time.isin(t_va)].dropna(subset=need)
    te = df[df.time.isin(t_te)].dropna(subset=need)
    if len(tr) < 300 or len(te) < 50:
        print(f"  {name}: too few complete rows (train {len(tr)}, test {len(te)})")
        return None, None, None
    cap = capacity_for(len(tr), len(feats))
    m = XGBRegressor(**cap, random_state=P.SEED, n_jobs=-1, tree_method="hist",
                     monotone_constraints="(" + ",".join(map(str, cons)) + ")",
                     early_stopping_rounds=40, eval_metric="rmse")
    ev = [(va[feats].to_numpy("float32"), va[label].to_numpy("float32"))] \
        if len(va) > 50 else [(te[feats].to_numpy("float32"),
                               te[label].to_numpy("float32"))]
    m.fit(tr[feats].to_numpy("float32"), tr[label].to_numpy("float32"),
          eval_set=ev, verbose=False)
    met = regression_metrics(te[label], m.predict(te[feats].to_numpy("float32")),
                             include_mape=False)
    print(f"  {name:<22} train {len(tr):>7,}  test {len(te):>6,}  "
          f"RMSE {met['RMSE']:.4f}  R2 {met['R2']:.4f}  "
          f"[depth {cap['max_depth']}, {cap['n_estimators']} trees, "
          f"{len(tr) / max(len(feats), 1):,.0f} rows/feature]")
    return m, (t_tr, t_va, t_te), te


def sweep(model, feats, te, action_col, lo, hi, n=13):
    """
    Average model response to moving the action across its observed range.

    Returns NaNs when that range has zero width: fitting a slope through
    identical x-values produces a number with no meaning and an arbitrary sign.
    """
    if not np.isfinite([lo, hi]).all() or (hi - lo) < 1e-6:
        return np.full(n, lo), np.full(n, np.nan)
    base = te[feats].to_numpy("float64")
    idx = feats.index(action_col)
    grid = np.linspace(lo, hi, n)
    out = []
    for v in grid:
        X = base.copy()
        X[:, idx] = v
        out.append(float(model.predict(X.astype("float32")).mean()))
    return grid, np.asarray(out)


def main() -> int:
    src = P.PROCESSED_DIR / f"facility_{P.RESAMPLE}_{P.MONTH}.parquet"
    if not src.exists():
        print(f"ERROR: {src} not found. Run 05_facility_extract.py first.")
        return 1

    banner(f"CS54 PS3 — building the controllable twin · {P.MONTH}")
    df = pd.read_parquet(src)

    step = int(getattr(P, "TWIN_STEP", 1))
    smooth = int(getattr(P, "TWIN_SMOOTH", 1))
    df["time"] = pd.to_datetime(df["time"])
    if smooth > 1:
        num = [c for c in df.columns
               if c != "time" and pd.api.types.is_numeric_dtype(df[c])]
        df[num] = df[num].rolling(smooth, min_periods=max(2, smooth // 2)).mean()
        df = df.dropna(subset=[STATE])
        print(f"  trailing {smooth * P.STEP_MINUTES}-min smoothing "
              f"(noise down ~{smooth ** 0.5:.1f}x), stride still "
              f"{P.STEP_MINUTES} min — {len(df):,} rows kept")
        print(f"  STATE THIS IN THE PAPER: the twin models the "
              f"{smooth * P.STEP_MINUTES}-minute smoothed thermal state, not the")
        print(f"  raw 5-minute signal. Smoothing raises R2 partly BY CONSTRUCTION,")
        print(f"  because it removes the high-frequency component that is hardest")
        print(f"  to predict. The justification is physical, not statistical: a")
        print(f"  valve or CRAC change takes 20-40 min to appear in the room, so")
        print(f"  the controller acts on the smoothed state anyway. The persistence")
        print(f"  baseline below is computed on the SAME smoothed target, so the")
        print(f"  comparison is fair; quote the margin over persistence, never the")
        print(f"  raw R2 on its own.")
    if step > 1:
        print(f"  prediction target: the change over the next "
              f"{step * P.STEP_MINUTES} min")

    need = [STATE, ACTION, "it_power"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"ERROR: required columns absent: {missing}")
        print("The twin cannot be built without a thermal state, an action and a load.")
        return 1
    for c in ("outdoor_temp", "cooling_power"):
        if c not in df.columns:
            print(f"  NOTE: '{c}' absent — continuing without it")

    df = build_features(df)
    lo, hi = float(np.nanpercentile(df[ACTION], 1)), float(np.nanpercentile(df[ACTION], 99))
    _au = P.UNITS.get(ACTION, ("", ""))[0] or "units"
    print(f"  action '{ACTION}' operating range (1st-99th pct): "
          f"{lo:.2f} .. {hi:.2f} {_au}")

    # ------------------------------------------------ thermal model
    banner("1. Thermal transition model")
    df["d_state"] = df[STATE].shift(-step) - df[STATE]
    t_feats = [c for c in
               [STATE, ACTION, "it_power", "outdoor_temp", "supply_temp",
                "tod_sin", "tod_cos"]
               + [c for c in df.columns if c.startswith(f"{STATE}_")]
               + [c for c in df.columns if c.startswith("it_power_")]
               if c in df.columns]
    t_feats = list(dict.fromkeys(t_feats))
    med = set(P.THERMAL_MEDIATORS)
    dropped = sorted({c for c in t_feats
                      if c in med or any(c.startswith(f"{m}_") for m in med)})
    t_feats = [c for c in t_feats if c not in dropped]
    print(f"  {len(t_feats)} features · monotone {P.MONOTONE_THERMAL:+d} on '{ACTION}'")
    if dropped:
        print(f"  excluded as MEDIATORS of '{ACTION}': {', '.join(dropped)}")
        print("  (conditioning on a mediator blocks the action's path to the state)")
    tmodel, splits, t_te = fit(df, t_feats, "d_state", ACTION,
                               P.MONOTONE_THERMAL, "thermal (delta state)")
    if tmodel is None:
        return 1
    _tm = regression_metrics(t_te["d_state"],
                             tmodel.predict(t_te[t_feats].to_numpy("float32")),
                             include_mape=False)

    free_model = None
    if getattr(P, "COMPARE_CONSTRAINED", False) and P.MONOTONE_THERMAL != 0:
        free_model, _, f_te = fit(df, t_feats, "d_state", ACTION, 0,
                                  "thermal (unconstrained)")
        if free_model is not None:
            _fm = regression_metrics(
                f_te["d_state"],
                free_model.predict(f_te[t_feats].to_numpy("float32")),
                include_mape=False)
            gfree, rfree = sweep(free_model, t_feats, f_te, ACTION, *
                                 np.nanpercentile(df[ACTION].dropna(), [1, 99]))
            slope_free = (float(np.polyfit(gfree, rfree, 1)[0])
                          if np.isfinite(rfree).all() else float("nan"))
            want = "negative" if P.MONOTONE_THERMAL < 0 else "positive"
            got = "negative" if slope_free < 0 else "positive"
            print(f"  unconstrained fit: R2 {_fm['R2']:.4f} vs constrained "
                  f"{_tm['R2']:.4f}; its '{ACTION}' response is {got} "
                  f"({slope_free:+.4f}), physics wants {want}")
            if got != want:
                print("  -> The record is CONFOUNDED: without the constraint the")
                print("     model learns a physically impossible response, because")
                print("     operators move this actuator IN RESPONSE to conditions.")
                print("     Report this comparison; it is evidence, not a nuisance.")

    dz = t_te["d_state"].to_numpy()
    base = regression_metrics(dz, np.zeros_like(dz), include_mape=False)
    print(f"  {'persistence (delta=0)':<22} {'':>7}  {'':>6}  "
          f"RMSE {base['RMSE']:.4f}  R2 {base['R2']:.4f}")
    if base["RMSE"] <= _tm["RMSE"]:
        print("  !! The thermal model is WORSE than predicting no change at all.")
        print("     A negative R2 means it is adding error, not signal. Usually")
        print("     this means the action carries no information — check the")
        print("     action scan before going further.")

    # ------------------------------------------------ power model
    banner("2. Cooling power model")
    pmodel = p_te = None
    if "cooling_power" in df.columns and df["cooling_power"].notna().sum() > 400:
        p_feats = [c for c in [ACTION, "it_power", "outdoor_temp", STATE,
                               "supply_temp", "tod_sin", "tod_cos",
                               "it_power_lag1", "it_power_rmean6"]
                   if c in df.columns]
        print(f"  {len(p_feats)} features · monotone {P.MONOTONE_POWER:+d} on '{ACTION}'")
        pmodel, _, p_te = fit(df, p_feats, "cooling_power", ACTION,
                              P.MONOTONE_POWER, "cooling power")
        if pmodel is not None:
            mean_cp = float(df["cooling_power"].mean())
            print(f"  mean measured cooling power {mean_cp:,.1f} kW")
    else:
        print("  cooling_power unavailable — the reward cannot use measured power.")
        print("  Fall back to a COP-curve model; see PS3_README.md 'If power is absent'.")

    # ------------------------------------------------ causal sanity check
    banner("3. Causal sanity check — does the agent actually have a lever?")
    report = {}
    g, resp = sweep(tmodel, t_feats, t_te, ACTION, lo, hi)
    ok_t = np.isfinite(resp).all() and (hi - lo) >= 1e-6
    gain_t = float(np.polyfit(g, resp, 1)[0]) if ok_t else float("nan")
    spread_t = float(resp.max() - resp.min()) if ok_t else 0.0
    au = P.UNITS.get(ACTION, ("", ""))[0] or "units"
    print(f"  thermal gain   {gain_t:+.4f} degC of inlet change per {au} of '{ACTION}'")
    print(f"  response spread over the operating range: {spread_t:.4f} degC/step")
    if "it_power" in t_feats:
        imp = dict(zip(t_feats, tmodel.feature_importances_))
        rank = sorted(imp.values(), reverse=True).index(imp[ACTION]) + 1
        print(f"  '{ACTION}' gain importance {imp[ACTION]:.5f} "
              f"(rank {rank} of {len(imp)})")
        report["thermal_action_rank"] = rank
    report.update(thermal_gain=gain_t, thermal_spread=spread_t)

    if pmodel is not None:
        gp, respp = sweep(pmodel, p_feats, p_te, ACTION, lo, hi)
        ok_p = np.isfinite(respp).all() and (hi - lo) >= 1e-6
        gain_p = float(np.polyfit(gp, respp, 1)[0]) if ok_p else float("nan")
        print(f"  power gain     {gain_p:+.3f} kW per {au} of '{ACTION}'")
        print(f"  over the {hi - lo:.1f} {au} operating range that is "
              f"{abs(gain_p) * (hi - lo):,.1f} kW of headroom")
        report.update(power_gain=gain_p,
                      power_headroom_kW=abs(gain_p) * (hi - lo))

    print()
    # A zero-width operating range makes every "gain" below meaningless: polyfit
    # is handed identical x-values and returns noise with an arbitrary sign.
    # 22-01 hit exactly this — the setpoint never left 16.00 degC, so the power
    # gain came back +5.657 kW/degC, the wrong sign for a monotone -1 model, and
    # the old check read that as evidence the action worked. It must fail first.
    degenerate = (hi - lo) < 1e-6
    no_signal = report.get("thermal_action_rank", 0) == len(t_feats) and spread_t < 1e-6
    if degenerate or no_signal:
        print("  VERDICT: NO USABLE ACTION.")
        if degenerate:
            print(f"  '{ACTION}' never moves: its 1st-99th percentile range is")
            print(f"  {lo:.4f} to {hi:.4f}, a width of {hi - lo:.2e}.")
            print("  Every gain printed above is a degenerate fit through identical")
            print("  x-values and carries no information — ignore the numbers and")
            print("  the signs, including any that look physically correct.")
        if no_signal:
            print(f"  The model gives '{ACTION}' the lowest importance of all")
            print("  features and a flat response.")
        print()
        print("  Do NOT train an agent on this twin. Run 05_facility_extract.py's")
        print("  action scan, pick a lever the facility actually used, and rebuild:")
        print("      CS54_ACTION=<that variable> python 06_build_twin.py")
        return 1
    if spread_t < 1e-3 and (pmodel is None or abs(report.get("power_gain", 0)) < 1e-3):
        print("  VERDICT: the action has essentially NO effect in the learned twin.")
        print("  An RL agent here will produce a meaningless policy. Do not train")
        print("  one and report its savings. Switch to the hybrid physics twin")
        print("  described in PS3_README.md, and say in the paper that the control")
        print("  response is imposed rather than identified.")
    else:
        print("  VERDICT: the action moves both temperature and power in the")
        print("  physically correct direction, with usable magnitude. The twin is")
        print("  a legitimate RL environment.")
        print("  Note for the paper: the SIGN is guaranteed by the monotonicity")
        print("  constraint; the MAGNITUDE is what the data supports. Report both")
        print("  facts rather than implying the response was discovered freely.")

    # ------------------------------------------------ rollout validation
    banner("4. Rollout validation of the thermal model")
    te = t_te.sort_values("time")
    X = te[t_feats].to_numpy("float64")
    truth = te[STATE].to_numpy("float64")
    si = t_feats.index(STATE)
    lag_i = [(l, t_feats.index(f"{STATE}_lag{l}")) for l in P.TWIN_LAGS
             if f"{STATE}_lag{l}" in t_feats]
    d_i = [(k, t_feats.index(f"{STATE}_d{k}")) for k in (1, 2, 3)
           if f"{STATE}_d{k}" in t_feats]
    buf_len = max([l for l, _ in lag_i] + [3]) + 1

    roll_rows = []
    for H in (1, 3, 6, 12):
        preds, pers, acts = [], [], []
        for s in range(buf_len, len(te) - H, 6):
            buf = list(truth[s - buf_len + 1: s + 1])
            ok = True
            for k in range(H):
                row = X[s + k].copy()
                if not np.isfinite(row).all():
                    ok = False
                    break
                row[si] = buf[-1]
                for l, i_ in lag_i:
                    row[i_] = buf[-1 - l]
                for kk, i_ in d_i:
                    row[i_] = buf[-1] - buf[-1 - kk]
                buf.append(buf[-1] + float(tmodel.predict(row.reshape(1, -1)
                                                          .astype("float32"))[0]))
                buf.pop(0)
            if ok and np.isfinite(truth[s + H]):
                preds.append(buf[-1]); pers.append(truth[s]); acts.append(truth[s + H])
        if not preds:
            continue
        ml = regression_metrics(acts, preds, include_mape=False)
        mp = regression_metrics(acts, pers, include_mape=False)
        roll_rows.append({"horizon_min": H * P.STEP_MINUTES * step,
                          "twin_RMSE": ml["RMSE"], "persist_RMSE": mp["RMSE"],
                          "twin_R2": ml["R2"], "n": ml["n"]})
        tag = "TWIN WINS" if ml["RMSE"] < mp["RMSE"] else "persistence wins"
        print(f"  {H * P.STEP_MINUTES * step:>3} min: twin {ml['RMSE']:.4f} vs "
              f"persistence {mp['RMSE']:.4f} degC   {tag}")
    roll = pd.DataFrame(roll_rows)
    validity = 0
    if not roll.empty:
        for _, r in roll.sort_values("horizon_min").iterrows():
            if r["twin_RMSE"] < r["persist_RMSE"]:
                validity = int(r["horizon_min"])
            else:
                break   # contiguous from the shortest: once it fails, it is done
        print()
        if validity:
            print(f"  VALIDITY HORIZON: {validity} min.")
            print(f"  Beyond this the twin is beaten by simply holding the")
            print(f"  temperature constant, so its predictions there are worse")
            print(f"  than no model at all. The controller must not plan further")
            print(f"  ahead than this — 07 will cap its forecast horizon to it.")
        else:
            print("  VALIDITY HORIZON: 0 — the twin loses even at one step.")
    if not roll.empty and (roll.twin_RMSE >= roll.persist_RMSE).all():
        print("\n  The twin never beats persistence. It is not yet a usable")
        print("  environment: an agent trained inside it would be optimising noise.")

    # ------------------------------------------------ save
    import joblib
    joblib.dump({"model": tmodel, "features": t_feats}, P.MODELS_DIR / "twin_thermal.joblib")
    if pmodel is not None:
        joblib.dump({"model": pmodel, "features": p_feats},
                    P.MODELS_DIR / "twin_power.joblib")
    meta = {
        "month": P.MONTH, "state": STATE, "action": ACTION,
        "setpoint_min": lo, "setpoint_max": hi,
        "step_minutes": P.STEP_MINUTES, "twin_step": step,
        "twin_smooth": smooth, "validity_horizon_min": validity, "seed": P.SEED,
        "thermal_features": t_feats,
        "power_features": p_feats if pmodel is not None else None,
        "monotone_thermal": P.MONOTONE_THERMAL,
        "monotone_power": P.MONOTONE_POWER,
        "sanity": report,
    }
    (P.MODELS_DIR / "twin_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    if not roll.empty:
        roll.to_csv(P.RESULTS_DIR / "ps3_twin_rollout.csv", index=False)

    # The environment replays this exact engineered frame, so the feature columns
    # the models were trained on are guaranteed to exist with the same meaning.
    # Rebuilding them separately in the env would be a silent source of drift.
    feat_path = P.PROCESSED_DIR / f"twin_features_{P.MONTH}.parquet"
    keep = list(dict.fromkeys(
        ["time"] + t_feats + (p_feats if pmodel is not None else [])
        + [c for c in (STATE, "rack_inlet_max", "cooling_power", "it_power",
                       "pue", "total_power") if c in df.columns]))
    df[keep].to_parquet(feat_path, index=False)
    print(f"  wrote {feat_path}  ({feat_path.stat().st_size / 1e6:,.1f} MB)")

    # ------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(g, resp, "o-", color="#b5442e")
    ax[0].set_xlabel(f"{ACTION} (degC)"); ax[0].set_ylabel("predicted inlet change (degC)")
    ax[0].set_title("Thermal response to the action")
    if pmodel is not None:
        ax[1].plot(gp, respp, "o-", color="#3b6ea5")
        ax[1].set_xlabel(f"{ACTION} (degC)"); ax[1].set_ylabel("cooling power (kW)")
        ax[1].set_title("Power response to the action")
    for a in ax:
        a.grid(alpha=.3, lw=.5)
    fig.tight_layout()
    fig.savefig(P.FIGURES_DIR / "ps3_twin_response.png", dpi=150)
    plt.close(fig)

    banner("Saved")
    print(f"  {P.MODELS_DIR / 'twin_thermal.joblib'}")
    if pmodel is not None:
        print(f"  {P.MODELS_DIR / 'twin_power.joblib'}")
    print(f"  {P.MODELS_DIR / 'twin_meta.json'}")
    print(f"  {P.FIGURES_DIR / 'ps3_twin_response.png'}")

    # v7: gate on the VALIDITY HORIZON. The previous test was .any(), so a win
    # at 5 minutes passed a twin that diverged badly at 30 and 60 — the range the
    # controller actually plans over.
    usable = validity > 0
    banner("NEXT")
    if usable:
        h = max(1, validity // P.STEP_MINUTES)
        print(f"      python 07_train_rl.py --horizon {h}   "
              f"# {validity} min, the twin's validity horizon")
        print()
        print("      --quick is for checking the chain runs. Its numbers are NOT")
        print("      results: 4,000 timesteps on one seed is an untrained agent.")
        print("      For anything you would report:")
        print("          nohup python 07_train_rl.py --horizon "
              f"{h} > ps3_full.log 2>&1 &")
    else:
        # Previously this printed the same NEXT line whatever happened, which
        # sent a failed twin straight into controller training.
        print("      DO NOT run 07_train_rl.py.")
        print("      The twin loses to persistence at every horizon, so an agent")
        print("      trained inside it would be optimising noise and any saving")
        print("      it reported would be an artefact of the model's own error.")
        print()
        print("      Report this as a negative result and say what it rules out.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
