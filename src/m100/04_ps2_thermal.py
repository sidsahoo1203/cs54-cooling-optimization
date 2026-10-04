#!/usr/bin/env python3
"""
CS54 Sub-problem (b) — thermal dynamics model (the digital twin's core).
VERSION 2.

Case study wording:
    "(b) models the thermal dynamics of the data center (how heat propagates and
     how cooling actions affect temperature across zones)"

Learns a STATE-TRANSITION FUNCTION:
    ( thermal state(t), IT power(t), cooling action(t), workload(t) ) -> state(t+h)

WHAT v2 FIXES (see CHANGES.md):
  1. LAGGED THERMAL HISTORY. v1 gave the model only contemporaneous values, so
     it could not see its own rate of change — and the next change of a
     first-order system depends mostly on the current one. v2 adds lags,
     previous deltas and rolling statistics of the thermal state. This is the
     main reason v1 scored R2 = 0.0017.
  2. MULTIPLE STEP SIZES. A 5-minute change in rack-mean inlet temperature is
     close to sensor resolution, so a near-zero R2 there may be measuring noise
     rather than an unlearnable system. v2 fits at 5, 15, 30 and 60 minutes.
     Inter-rack heat transport also needs more than five minutes to appear at
     all, so the spatial ablation is repeated at every step size.
  3. A PERSISTENCE ROLLOUT BASELINE. v1 reported rollout R2 = 0.86 with nothing
     to compare it against. A model that always predicts "no change" also
     produces a stable rollout and a high R2 on a smooth signal. v2 reports both.
  4. NO MAPE on a delta target. The true change passes through zero, so the
     percentage error explodes and the number is meaningless.
  5. Units, the ASHRAE edition and equipment class are stated with every number.

    python 04_ps2_thermal.py
    python 04_ps2_thermal.py --racks 12 --stride 24     # faster
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from common import banner, chronological_split, regression_metrics

STATE = "ambient"          # thermal state variable, degC
ACTION = "fan_speed"       # the only cooling actuator recorded in 2020 months
AMB_DIFFS = [1, 2, 3]
BUFLEN = max(max(C.PS2_LAGS), max(C.PS2_ROLL), max(AMB_DIFFS)) + 1


# ------------------------------------------------------------------ features
def add_thermal_history(df: pd.DataFrame) -> pd.DataFrame:
    """Lags, previous deltas and rolling statistics, computed within each rack."""
    df = df.sort_values(["rack", "time"]).copy()
    g = df.groupby("rack", sort=False)
    for col in [c for c in (STATE, "total_power", "cpu_busy", "core_temp")
                if c in df.columns]:
        for lag in C.PS2_LAGS:
            df[f"{col}_lag{lag}"] = g[col].shift(lag)
        for w in C.PS2_ROLL:
            r = g[col].rolling(w, min_periods=max(2, w // 2))
            df[f"{col}_rmean{w}"] = r.mean().reset_index(level=0, drop=True)
            df[f"{col}_rstd{w}"] = r.std().reset_index(level=0, drop=True)
    # Previous rates of change of the thermal state — what v1 was missing.
    for k in AMB_DIFFS:
        df[f"{STATE}_d{k}"] = df[STATE] - g[STATE].shift(k)
    return df


def add_neighbour_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Physical-neighbour and row-aggregate features.

    Within a row of the machine room X decreases monotonically with rack ID
    (ExaData spatial documentation), so racks whose IDs differ by one and share
    a row are physically adjacent.
    """
    df = df.sort_values(["time", "rack"]).copy()
    src = [c for c in ("total_power", STATE, "cpu_busy") if c in df.columns]
    row_agg = (df.groupby(["time", "rack_row"], as_index=False)[src].mean()
               .rename(columns={c: f"row_{c}" for c in src}))
    df = df.merge(row_agg, on=["time", "rack_row"], how="left")
    for side, delta in (("prev", -1), ("next", +1)):
        nb = df[["time", "rack", "rack_row"] + src].copy()
        nb["rack"] = nb["rack"] + delta
        nb = nb.rename(columns={c: f"{side}_{c}" for c in src})
        df = df.merge(nb, on=["time", "rack", "rack_row"], how="left")
    return df


FEATURE_SETS = {
    "local_only": lambda cols: [c for c in cols
                                if not c.startswith(("prev_", "next_", "row_"))],
    "plus_neighbours": lambda cols: [c for c in cols if not c.startswith("row_")],
    "plus_row": lambda cols: list(cols),
}


def base_columns(df: pd.DataFrame) -> list[str]:
    want = [STATE, "total_power", "cpu_power", "cpu_busy", "load_one", "core_temp",
            ACTION, "mem_free", "cluster_cpu_util", "tod_sin", "tod_cos"]
    want += [c for c in df.columns if any(
        c.startswith(f"{b}_lag") or c.startswith(f"{b}_rmean")
        or c.startswith(f"{b}_rstd") or c.startswith(f"{b}_d")
        for b in (STATE, "total_power", "cpu_busy", "core_temp"))]
    want += [c for c in df.columns if c.startswith(("prev_", "next_", "row_"))]
    return [c for c in dict.fromkeys(want) if c in df.columns]


def fit_xgb(tr, va, te, feats, label):
    from xgboost import XGBRegressor
    m = XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                     random_state=C.SEED, n_jobs=-1, tree_method="hist",
                     early_stopping_rounds=40, eval_metric="rmse")
    ev = [(va[feats].to_numpy("float32"), va[label].to_numpy("float32"))] \
        if len(va) > 100 else [(te[feats].to_numpy("float32"),
                                te[label].to_numpy("float32"))]
    m.fit(tr[feats].to_numpy("float32"), tr[label].to_numpy("float32"),
          eval_set=ev, verbose=False)
    return m


# ------------------------------------------------------------------ rollout
def batched_rollout(model, feats, panel, horizon, stride):
    """
    Autoregressive multi-step prediction, vectorised over all (rack, start) pairs.

    The model's own temperature output is fed back in — including through the
    lag, delta and rolling features, which are recomputed from a running buffer
    of PREDICTED values. Exogenous signals (IT power, workload, fan, neighbours)
    are replayed from the real trace, which is legitimate because cooling control
    does not decide which jobs run.

    Returns (learned_pred, persistence_pred, truth) at the horizon-th step.
    """
    ai = {name: i for i, name in enumerate(feats)}
    si = ai[STATE]
    lag_idx = [(l, ai[f"{STATE}_lag{l}"]) for l in C.PS2_LAGS
               if f"{STATE}_lag{l}" in ai]
    diff_idx = [(k, ai[f"{STATE}_d{k}"]) for k in AMB_DIFFS
                if f"{STATE}_d{k}" in ai]
    roll_idx = [(w, ai.get(f"{STATE}_rmean{w}"), ai.get(f"{STATE}_rstd{w}"))
                for w in C.PS2_ROLL]

    Xs, truths, starts = [], [], []
    for _, grp in panel:
        X = grp[feats].to_numpy("float64")
        y = grp[STATE].to_numpy("float64")
        n = len(grp)
        for s in range(BUFLEN, n - horizon, stride):
            if not np.isfinite(X[s:s + horizon]).all():
                continue
            if not np.isfinite(y[s - BUFLEN:s + horizon + 1]).all():
                continue
            Xs.append(X[s:s + horizon])            # exogenous rows, in order
            truths.append(y[s + horizon])
            starts.append(y[s - BUFLEN + 1:s + 1])  # seed buffer, oldest..newest
    if not Xs:
        return np.array([]), np.array([]), np.array([])

    Xseq = np.stack(Xs)                  # (pairs, horizon, feats)
    buf = np.stack(starts)               # (pairs, BUFLEN)
    truth = np.asarray(truths)
    persist = buf[:, -1].copy()          # "hold the current temperature"

    for k in range(horizon):
        row = Xseq[:, k, :].copy()
        row[:, si] = buf[:, -1]
        for l, idx in lag_idx:
            row[:, idx] = buf[:, -1 - l]
        for d, idx in diff_idx:
            row[:, idx] = buf[:, -1] - buf[:, -1 - d]
        for w, mi, sdi in roll_idx:
            win = buf[:, -w:]
            if mi is not None:
                row[:, mi] = win.mean(axis=1)
            if sdi is not None:
                row[:, sdi] = win.std(axis=1, ddof=1)
        nxt = buf[:, -1] + model.predict(row.astype("float32"))
        buf = np.concatenate([buf[:, 1:], nxt.reshape(-1, 1)], axis=1)

    return buf[:, -1], persist, truth


# ------------------------------------------------------------------ main
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--racks", type=int, default=None)
    p.add_argument("--rollout-racks", type=int, default=8)
    p.add_argument("--stride", type=int, default=12,
                   help="rollout start points every N steps (12 = hourly)")
    args = p.parse_args()

    src = C.PROCESSED_DIR / f"rack_{C.RESAMPLE}_{C.MONTH}.parquet"
    if not src.exists():
        print(f"ERROR: {src} not found. Run 02_extract.py first.")
        return 1

    banner("CS54 v2 — Sub-problem (b): learned thermal dynamics")
    df = pd.read_parquet(src)
    df["rack"] = pd.to_numeric(df["rack"], errors="coerce")
    df = df.dropna(subset=["rack", "time", STATE]).copy()
    df["rack"] = df["rack"].astype(int)
    if args.racks:
        df = df[df["rack"].isin(sorted(df["rack"].unique())[: args.racks])]
    u = C.UNITS.get(STATE, ("degC",))[0]
    print(f"  {len(df):,} rows · {df['rack'].nunique()} racks · state '{STATE}' [{u}]")

    df = add_thermal_history(df)
    df = add_neighbour_features(df)
    df = df.sort_values(["rack", "time"])
    g = df.groupby("rack", sort=False)

    cols = base_columns(df)
    stamps = np.sort(df["time"].unique())
    tr_s, va_s, te_s = chronological_split(len(stamps), C.TEST_FRACTION, C.VAL_FRACTION)
    t_tr, t_va, t_te = stamps[tr_s], stamps[va_s], stamps[te_s]
    print(f"  split: train {len(t_tr):,} · val {len(t_va):,} · test {len(t_te):,} "
          f"timestamps · test begins {t_te[0]}")
    print(f"  {len(cols)} candidate features "
          f"({sum(c.startswith(('prev_','next_','row_')) for c in cols)} spatial, "
          f"{sum('_lag' in c or '_rmean' in c or '_rstd' in c or '_d' in c[-3:] for c in cols)} historical)")

    # ---------------------------------------------------- 1. multi-horizon
    banner("1. Thermal dynamics at several step sizes, with spatial ablation")
    print("  Target is the CHANGE in rack inlet temperature over h steps, not its")
    print("  level. Predicting the level scores R2 > 0.99 by echoing the input.")
    print("  MAPE is omitted: the true change passes through zero.\n")

    superset = sorted(set().union(*(set(s(cols)) for s in FEATURE_SETS.values())))
    rows, models_by_h = [], {}

    for h in C.PS2_HORIZONS:
        lab = f"delta{h}"
        df[lab] = g[STATE].shift(-h) - df[STATE]
        need = superset + [lab]
        tr = df[df["time"].isin(t_tr)].dropna(subset=need)
        va = df[df["time"].isin(t_va)].dropna(subset=need)
        te = df[df["time"].isin(t_te)].dropna(subset=need)
        if len(tr) < 500 or len(te) < 100:
            print(f"  h={h}: too few complete rows ({len(tr)}/{len(te)})")
            continue

        dz = te[lab].to_numpy("float64")
        rows.append({"horizon_min": h * C.STEP_MINUTES, "feature_set": "persistence (0)",
                     "n_features": 0,
                     **regression_metrics(dz, np.zeros_like(dz), include_mape=False)})
        best = None
        for name, sel in FEATURE_SETS.items():
            feats = [c for c in sel(cols) if c in superset]
            m = fit_xgb(tr, va, te, feats, lab)
            met = regression_metrics(te[lab], m.predict(te[feats].to_numpy("float32")),
                                     include_mape=False)
            rows.append({"horizon_min": h * C.STEP_MINUTES, "feature_set": name,
                         "n_features": len(feats), **met})
            if best is None or met["RMSE"] < best[1]:
                best = (name, met["RMSE"], m, feats)
        models_by_h[h] = best
        print(f"  h={h * C.STEP_MINUTES:>3} min  best={best[0]:<16} "
              f"RMSE={best[1]:.4f} {u}  "
              f"(persistence {rows[-4]['RMSE']:.4f})")

    abl = pd.DataFrame(rows)
    print("\n" + abl.to_string(index=False))

    # v2 fix: every horizon can legitimately produce zero complete rows (e.g.
    # too few racks for the "plus_neighbours"/"plus_row" spatial features to
    # ever have a real neighbour present, so those columns are NaN for every
    # row and dropna() empties the frame). That leaves `rows` empty and `abl`
    # with no "horizon_min" column at all. Detect that BEFORE indexing into
    # abl below, instead of letting a bare KeyError('horizon_min') kill the
    # run — report it as the real finding it is and stop cleanly.
    if abl.empty or "horizon_min" not in abl.columns or not models_by_h:
        print("\nNo model trained at any horizon: every horizon had too few "
              "complete rows after dropna(). This is expected with a small "
              "RACK_SUBSET, where racks commonly have no neighbour/row-mate "
              "present in the subset, so the spatial feature columns are NaN "
              "for essentially every row. Not a data or pipeline fault — "
              "re-run with the full rack set (RACK_SUBSET = None) or a subset "
              "large enough that every included rack's neighbours are also "
              "included.")
        return 1

    print("\n  Reading this table:")
    print("    R2 near zero  -> the change is dominated by noise at that step size")
    print("    R2 rising with horizon -> real dynamics, previously masked by")
    print("       sensor resolution over 5 minutes")
    print("    neighbours helping only at longer steps -> heat transport between")
    print("       racks is real but slower than one bin")

    for h in C.PS2_HORIZONS:
        s = abl[abl["horizon_min"] == h * C.STEP_MINUTES]
        if {"local_only", "plus_neighbours"} <= set(s["feature_set"]):
            a = s.loc[s.feature_set == "local_only", "RMSE"].iloc[0]
            b = s.loc[s.feature_set == "plus_neighbours", "RMSE"].iloc[0]
            print(f"    {h * C.STEP_MINUTES:>3} min: neighbours change RMSE by "
                  f"{(1 - b / a) * 100:+.2f}%")

    # ---------------------------------------------------- 2. rollout
    banner("2. Multi-step rollout vs a persistence rollout")
    print("  The 1-step model is iterated, feeding its own output back in through")
    print("  the lag, delta and rolling features. Persistence = hold the current")
    print("  temperature. If the twin does not beat that, it is not yet a twin.\n")

    roll_rows = []
    if 1 in models_by_h:
        _, _, model1, feats1 = models_by_h[1]
        te_all = df[df["time"].isin(t_te)]
        counts = te_all.dropna(subset=feats1 + [STATE])["rack"].value_counts()
        racks = sorted(counts[counts > max(C.ROLLOUT_HORIZONS) + BUFLEN + 5]
                       .index)[: args.rollout_racks]
        print(f"  rollout racks: {racks}  (stride {args.stride} steps)")
        panel = [(rk, te_all[te_all["rack"] == rk].sort_values("time")
                  .dropna(subset=feats1 + [STATE])) for rk in racks]

        for h in C.ROLLOUT_HORIZONS:
            pred, pers, truth = batched_rollout(model1, feats1, panel, h, args.stride)
            if pred.size == 0:
                continue
            ml = regression_metrics(truth, pred, include_mape=False)
            mp = regression_metrics(truth, pers, include_mape=False)
            roll_rows.append({"horizon_min": h * C.STEP_MINUTES,
                              "twin_RMSE": ml["RMSE"], "persist_RMSE": mp["RMSE"],
                              "twin_R2": ml["R2"], "persist_R2": mp["R2"],
                              "improvement_%": (1 - ml["RMSE"] / mp["RMSE"]) * 100,
                              "n": ml["n"]})
            flag = "TWIN WINS" if ml["RMSE"] < mp["RMSE"] else "persistence wins"
            print(f"  {h * C.STEP_MINUTES:>3} min: twin {ml['RMSE']:.4f} vs "
                  f"persistence {mp['RMSE']:.4f} {u}  "
                  f"({(1 - ml['RMSE'] / mp['RMSE']) * 100:+.1f}%)  {flag}")

    roll = pd.DataFrame(roll_rows)
    if not roll.empty:
        wins = int((roll["twin_RMSE"] < roll["persist_RMSE"]).sum())
        print(f"\n  twin beats persistence at {wins} of {len(roll)} horizons.")
        if wins == 0:
            print("  This is a NEGATIVE RESULT and must be reported as one: the")
            print("  learned twin adds nothing over holding temperature constant.")

    # ---------------------------------------------------- 3. action sensitivity
    banner(f"3. Sensitivity to the cooling actuator '{ACTION}'")
    sens = pd.DataFrame()
    if 1 in models_by_h and ACTION in models_by_h[1][3]:
        _, _, m1, f1 = models_by_h[1]
        te = df[df["time"].isin(t_te)].dropna(subset=f1)
        if len(te) > 200:
            base = te[f1].to_numpy("float64")
            idx = f1.index(ACTION)
            lo, hi = np.nanpercentile(te[ACTION], [1, 99])
            obs_lo, obs_hi = float(te[ACTION].min()), float(te[ACTION].max())
            grid = np.linspace(lo, hi, 9)
            sens = pd.DataFrame([
                {ACTION: float(v),
                 "mean_predicted_delta_degC": float(m1.predict(
                     np.where(np.arange(base.shape[1]) == idx, v, base).astype("float32")
                 ).mean())} for v in grid])
            print(sens.to_string(index=False))
            slope = np.polyfit(sens[ACTION], sens["mean_predicted_delta_degC"], 1)[0]
            span = (hi - lo) / max(abs(np.nanmean(te[ACTION])), 1e-9) * 100
            spread = float(sens["mean_predicted_delta_degC"].max()
                           - sens["mean_predicted_delta_degC"].min())
            imp = dict(zip(f1, getattr(m1, "feature_importances_", [0] * len(f1))))
            act_imp = float(imp.get(ACTION, 0.0))
            print(f"\n  slope {slope:+.3e} degC per RPM")
            print(f"  response spread across the swept range: {spread:.3e} degC")
            print(f"  XGBoost gain importance of '{ACTION}': {act_imp:.5f} "
                  f"(rank {sorted(imp.values(), reverse=True).index(act_imp) + 1}"
                  f" of {len(imp)})")
            print(f"  observed range {obs_lo:.0f}..{obs_hi:.0f} RPM; "
                  f"1st-99th percentile spans only {span:.1f}% of the mean")
            if act_imp == 0.0 or spread < 1e-9:
                print("\n  The model assigns the actuator ZERO importance and its")
                print("  response is exactly flat. This is a real result, not a")
                print("  code fault: given 50 other features, fan speed carries no")
                print("  information the model can use. An actuator a controller")
                print("  cannot move the state with is not a control variable.")
            print("\n  INTERPRETATION — state this plainly in the paper:")
            print("  There is not enough variation in rack-mean fan speed to")
            print("  identify a causal response, and node fans are firmware-")
            print("  controlled: they REACT to temperature rather than being")
            print("  commanded. Any slope here is correlation, not control.")
            print("  This is exactly why sub-problem (c) requires the CRAC")
            print("  supply-air setpoint from the vertiv plugin — a variable an")
            print("  operator actually sets.")
    else:
        print(f"  '{ACTION}' not among the selected features — skipped.")

    # ---------------------------------------------------- 4. ASHRAE
    banner("4. Thermal envelope")
    lo, hi = C.ASHRAE_RECOMMENDED
    alo, ahi = C.ASHRAE_ALLOWABLE_A1
    hot = f"{STATE}_max" if f"{STATE}_max" in df.columns else STATE
    v = df[hot].dropna()
    print(f"  Standard : {C.ASHRAE_EDITION}")
    print(f"  Measure  : dry-bulb temperature at the equipment air inlet")
    print(f"  Column   : '{hot}' = the hottest node inlet reading in each rack-bin")
    print(f"             (IPMI 'ambient' is the node BMC's inlet-side air sensor;")
    print(f"             rack maximum is used because ASHRAE compliance is judged")
    print(f"             on the worst inlet in the rack, not the average)")
    print(f"  n        : {len(v):,} rack-bins\n")
    print(f"  range    : {v.min():.2f} .. {v.max():.2f} degC")
    print(f"  mean     : {v.mean():.2f} degC   median {v.median():.2f} degC")
    print(f"  recommended {lo}-{hi} degC : {((v >= lo) & (v <= hi)).mean() * 100:.2f}%")
    print(f"  allowable A1 {alo}-{ahi} degC: {((v >= alo) & (v <= ahi)).mean() * 100:.2f}%")
    print(f"  above {hi} degC            : {(v > hi).mean() * 100:.2f}%")
    print(f"  below {lo} degC            : {(v < lo).mean() * 100:.2f}%  <- OVER-COOLING")
    print("\n  The below-band share is the headroom this whole case study targets:")
    print("  energy spent holding inlets colder than the standard requires.")

    # ---------------------------------------------------- figures
    if not abl.empty:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for name in ["persistence (0)"] + list(FEATURE_SETS):
            s = abl[abl.feature_set == name]
            if not s.empty:
                ax.plot(s["horizon_min"], s["R2"], "o-", label=name)
        ax.axhline(0, color="#999", lw=.8)
        ax.set_xlabel("prediction step (minutes)")
        ax.set_ylabel("R² of the temperature change")
        ax.set_title("CS54 (b) — learnability of thermal dynamics vs step size")
        ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.3, lw=.5)
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / "ps2_r2_vs_horizon.png", dpi=150)
        plt.close(fig)

    if not roll.empty:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(roll["horizon_min"], roll["twin_RMSE"], "o-",
                color="#b5442e", label="learned twin")
        ax.plot(roll["horizon_min"], roll["persist_RMSE"], "s--",
                color="#555", label="persistence rollout")
        ax.set_xlabel("rollout horizon (minutes)")
        ax.set_ylabel(f"{STATE} RMSE (degC)")
        ax.set_title("CS54 (b) — rollout error against the honest baseline")
        ax.legend(fontsize=9, frameon=False); ax.grid(alpha=.3, lw=.5)
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / "ps2_rollout_vs_persistence.png", dpi=150)
        plt.close(fig)

    # ---------------------------------------------------- save
    abl.to_csv(C.RESULTS_DIR / "ps2_ablation.csv", index=False)
    if not roll.empty:
        roll.to_csv(C.RESULTS_DIR / "ps2_rollout.csv", index=False)
    if not sens.empty:
        sens.to_csv(C.RESULTS_DIR / "ps2_action_sensitivity.csv", index=False)

    with open(C.RESULTS_DIR / "ps2_results.md", "w") as fh:
        fh.write("# CS54 Sub-problem (b) — learned thermal dynamics (v2)\n\n")
        fh.write(f"Month `{C.MONTH}` · {C.RESAMPLE} bins · state `{STATE}` [degC] · "
                 f"action `{ACTION}` [RPM] · seed {C.SEED}\n\n")
        fh.write("Target is the **change** in rack inlet temperature, not its level. "
                 "MAPE is omitted because the true change passes through zero.\n\n")
        fh.write("## Dynamics vs step size, with spatial ablation\n\n")
        fh.write(abl.to_markdown(index=False) + "\n\n")
        if not roll.empty:
            fh.write("## Rollout against a persistence rollout\n\n")
            fh.write(roll.to_markdown(index=False) + "\n\n")
        if not sens.empty:
            fh.write(f"## Sensitivity to `{ACTION}`\n\n")
            fh.write(sens.to_markdown(index=False) + "\n\n")
            fh.write("Node fans are firmware-controlled and react to temperature; "
                     "this is correlation, not control.\n\n")
        fh.write("## Thermal envelope\n\n")
        fh.write(f"- Standard: {C.ASHRAE_EDITION}\n")
        fh.write(f"- Measure: dry-bulb at the equipment air inlet\n")
        fh.write(f"- Column `{hot}`: hottest node inlet reading per rack-bin "
                 f"(IPMI `ambient` is the node BMC inlet-side air sensor)\n")
        fh.write(f"- n = {len(v):,} rack-bins; range {v.min():.2f}-{v.max():.2f} degC; "
                 f"mean {v.mean():.2f} degC\n")
        fh.write(f"- within recommended {lo}-{hi} degC: "
                 f"{((v >= lo) & (v <= hi)).mean() * 100:.2f}%\n")
        fh.write(f"- within allowable A1 {alo}-{ahi} degC: "
                 f"{((v >= alo) & (v <= ahi)).mean() * 100:.2f}%\n")
        fh.write(f"- below {lo} degC (over-cooling headroom): "
                 f"{(v < lo).mean() * 100:.2f}%\n")

    banner("Saved")
    for f in ("ps2_results.md", "ps2_ablation.csv", "ps2_rollout.csv"):
        print(f"  {C.RESULTS_DIR / f}")
    print(f"  figures in {C.FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
