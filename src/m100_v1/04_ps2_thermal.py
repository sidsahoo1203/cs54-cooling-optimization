#!/usr/bin/env python3
"""
CS54 Sub-problem (b) — thermal dynamics model (the digital twin's core).

Case study wording:
    "(b) models the thermal dynamics of the data center (how heat propagates and
     how cooling actions affect temperature across zones)"

What this script learns is a STATE-TRANSITION FUNCTION:

    ( thermal state(t), IT power(t), cooling action(t), workload(t) )  ->  state(t+1)

That is exactly the object an RL controller later needs as its environment, and
it is what papers #18 and #32 of the literature review build. Three design
choices deserve explanation, because they are the substance of the method:

1. IT PREDICT THE CHANGE, NOT THE LEVEL.
   Room temperature is highly autocorrelated: ambient(t+1) is very close to
   ambient(t). A model predicting the LEVEL scores R^2 > 0.99 by echoing its
   input and has learned no physics at all. Predicting delta = ambient(t+1) -
   ambient(t) removes that free ride, so the score reflects real dynamics.
   The persistence baseline (delta = 0) makes the comparison explicit.

2. MULTI-STEP ROLLOUT IS THE REAL TEST.
   One-step accuracy is easy. A simulator is only useful if errors do not
   compound when its own output is fed back in. So the model is rolled out
   autoregressively: predicted temperature becomes the next step's input, while
   exogenous signals (IT power, workload) are replayed from the real trace.
   This is legitimate precisely because cooling control does not decide which
   jobs run — workload is set by users and the scheduler.

3. SPATIAL ABLATION IS THE PROPAGATION EVIDENCE.
   "Heat propagates between zones" is a claim, not an assumption. Three nested
   feature sets — own rack only, plus physical neighbours, plus the whole row —
   are compared. If neighbours reduce error, propagation is demonstrated from
   data rather than asserted.

    python 04_ps2_thermal.py
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

STATE = "ambient"          # the thermal state variable we model
ACTION = "fan_speed"       # the cooling actuator recorded in this month


# ------------------------------------------------------------------ features
def add_neighbour_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Physical-neighbour and row-aggregate features.

    Within a row of the machine room, X decreases monotonically with rack ID
    (confirmed in the ExaData spatial documentation), so racks whose IDs differ
    by one and share a row are physically adjacent.
    """
    df = df.sort_values(["time", "rack"]).copy()
    src = [c for c in ("total_power", "ambient", "cpu_busy") if c in df.columns]

    # Row aggregates at each instant.
    row_agg = (
        df.groupby(["time", "rack_row"], as_index=False)[src]
        .mean()
        .rename(columns={c: f"row_{c}" for c in src})
    )
    df = df.merge(row_agg, on=["time", "rack_row"], how="left")

    # Immediate neighbours: shift rack index within the same timestamp and row.
    for side, delta in (("prev", -1), ("next", +1)):
        nb = df[["time", "rack", "rack_row"] + src].copy()
        nb["rack"] = nb["rack"] + delta          # so it lines up with rack r
        nb = nb.rename(columns={c: f"{side}_{c}" for c in src})
        df = df.merge(nb, on=["time", "rack", "rack_row"], how="left")

    return df


FEATURE_SETS = {
    "local_only": lambda cols: [
        c for c in cols
        if not c.startswith(("prev_", "next_", "row_"))
    ],
    "plus_neighbours": lambda cols: [
        c for c in cols if not c.startswith("row_")
    ],
    "plus_row": lambda cols: list(cols),
}


def base_columns(df: pd.DataFrame) -> list[str]:
    want = [
        STATE, "total_power", "cpu_power", "cpu_busy", "load_one", "core_temp",
        ACTION, "mem_free", "cluster_cpu_util", "tod_sin", "tod_cos",
        "prev_total_power", "prev_ambient", "prev_cpu_busy",
        "next_total_power", "next_ambient", "next_cpu_busy",
        "row_total_power", "row_ambient", "row_cpu_busy",
    ]
    return [c for c in want if c in df.columns]


# ------------------------------------------------------------------ rollout
def rollout(model, grp: pd.DataFrame, feats: list[str], horizon: int,
            state_col: str = STATE) -> tuple[np.ndarray, np.ndarray]:
    """
    Autoregressive multi-step prediction over one rack's contiguous test slice.

    At each start point the model predicts `horizon` steps forward, feeding its
    own temperature output back in while exogenous columns come from the real
    trace. Returns (predicted, actual) at the horizon-th step.
    """
    X = grp[feats].to_numpy("float64").copy()
    truth = grp[state_col].to_numpy("float64")
    si = feats.index(state_col)
    n = len(grp)
    preds, actuals = [], []

    for start in range(0, n - horizon):
        row = X[start].copy()
        if not np.isfinite(row).all():
            continue
        state = truth[start]
        ok = True
        for k in range(horizon):
            exo = X[start + k].copy()
            if not np.isfinite(exo).all():
                ok = False
                break
            exo[si] = state                       # our own state, not the truth
            d = float(model.predict(exo.reshape(1, -1))[0])
            state = state + d
        if ok and np.isfinite(truth[start + horizon]):
            preds.append(state)
            actuals.append(truth[start + horizon])
    return np.asarray(preds), np.asarray(actuals)


# ------------------------------------------------------------------ main
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--racks", type=int, default=None)
    p.add_argument("--rollout-racks", type=int, default=6,
                   help="racks used for the (slow) rollout evaluation")
    args = p.parse_args()

    src = C.PROCESSED_DIR / f"rack_{C.RESAMPLE}_{C.MONTH}.parquet"
    if not src.exists():
        print(f"ERROR: {src} not found. Run 02_extract.py first.")
        return 1

    banner("CS54 — Sub-problem (b): learned thermal dynamics model")
    df = pd.read_parquet(src)
    df["rack"] = pd.to_numeric(df["rack"], errors="coerce")
    df = df.dropna(subset=["rack", "time", STATE]).copy()
    df["rack"] = df["rack"].astype(int)
    if args.racks:
        keep = sorted(df["rack"].unique())[: args.racks]
        df = df[df["rack"].isin(keep)]
    print(f"  loaded {len(df):,} rows · {df['rack'].nunique()} racks")
    if ACTION not in df.columns:
        print(f"  NOTE: '{ACTION}' absent — the model will have no cooling "
              f"actuator input. Dynamics are still learnable; action "
              f"sensitivity is not.")

    df = add_neighbour_features(df)

    # Target: the one-step CHANGE in thermal state.
    df = df.sort_values(["rack", "time"])
    df["delta"] = df.groupby("rack", sort=False)[STATE].shift(-1) - df[STATE]
    df = df.dropna(subset=["delta"])

    cols = base_columns(df)
    stamps = np.sort(df["time"].unique())
    tr_s, va_s, te_s = chronological_split(len(stamps), C.TEST_FRACTION, C.VAL_FRACTION)
    t_tr, t_va, t_te = stamps[tr_s], stamps[va_s], stamps[te_s]
    print(f"  chronological split: train {len(t_tr):,} · val {len(t_va):,} "
          f"· test {len(t_te):,} timestamps")
    print(f"  test period begins {t_te[0]}")

    train = df[df["time"].isin(t_tr)]
    val = df[df["time"].isin(t_va)]
    test = df[df["time"].isin(t_te)]

    # ------------------------------------------------ 1. spatial ablation
    banner("1. Spatial ablation — is there measurable heat propagation?")
    from xgboost import XGBRegressor

    ablation, models = [], {}

    # Every feature set MUST be scored on identical rows, otherwise the
    # comparison measures sample selection rather than feature value. Edge racks
    # have no left/right neighbour, so the widest feature set drops them; we
    # therefore fix the evaluation rows once, using the union of all features,
    # and use those same rows for every variant including persistence.
    superset = sorted(set().union(*(set(sel(cols)) for sel in FEATURE_SETS.values())))
    need = superset + ["delta"]
    train_c = train.dropna(subset=need)
    val_c = val.dropna(subset=need)
    test_c = test.dropna(subset=need)
    print(f"  common complete rows — train {len(train_c):,} · val {len(val_c):,} "
          f"· test {len(test_c):,}  (identical for every feature set)")
    if len(train_c) < 500 or len(test_c) < 100:
        print("  !! too few rows after requiring neighbour completeness.")
        print("     With very few racks, drop 'plus_neighbours'/'plus_row' or")
        print("     extract more racks. Continuing with local_only only.")
        superset = FEATURE_SETS["local_only"](cols)
        need = superset + ["delta"]
        train_c, val_c, test_c = (d.dropna(subset=need) for d in (train, val, test))

    dz = test_c["delta"].to_numpy("float64")
    ablation.append({"feature_set": "persistence (delta=0)",
                     "n_features": 0,
                     **regression_metrics(dz, np.zeros_like(dz))})

    for name, selector in FEATURE_SETS.items():
        feats = [c for c in selector(cols) if c in superset]
        if not feats:
            continue
        tr, va, te = train_c, val_c, test_c
        if len(tr) < 500 or len(te) < 100:
            print(f"  {name}: too few complete rows ({len(tr)}/{len(te)})")
            continue
        m = XGBRegressor(
            n_estimators=700, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=C.SEED, n_jobs=-1, tree_method="hist",
            early_stopping_rounds=40, eval_metric="rmse",
        )
        ev = [(va[feats].to_numpy("float32"), va["delta"].to_numpy("float32"))] \
            if len(va) > 100 else [(te[feats].to_numpy("float32"),
                                    te["delta"].to_numpy("float32"))]
        m.fit(tr[feats].to_numpy("float32"), tr["delta"].to_numpy("float32"),
              eval_set=ev, verbose=False)
        yp = m.predict(te[feats].to_numpy("float32"))
        ablation.append({"feature_set": name, "n_features": len(feats),
                         **regression_metrics(te["delta"], yp)})
        models[name] = (m, feats)

    abl = pd.DataFrame(ablation)
    # MAPE is meaningless for a delta target: true deltas pass through zero, so
    # the percentage error explodes. Drop it rather than print a misleading number.
    abl_show = abl.drop(columns=[c for c in ("MAPE_%",) if c in abl.columns])
    print("\n" + abl_show.to_string(index=False))

    if {"local_only", "plus_neighbours"} <= set(models):
        a = abl.loc[abl["feature_set"] == "local_only", "RMSE"].iloc[0]
        b = abl.loc[abl["feature_set"] == "plus_neighbours", "RMSE"].iloc[0]
        print(f"\n  adding physical neighbours changes delta-RMSE by "
              f"{(1 - b / a) * 100:+.2f}%  (same {int(abl['n'].iloc[0]):,} test rows)")
        print("  A clear improvement is direct evidence of inter-rack thermal")
        print("  coupling; a negligible change is itself a finding worth stating.")

    if not models:
        print("\nNo model trained — cannot continue.")
        return 1

    best_name = abl[abl["feature_set"].isin(models)].sort_values("RMSE").iloc[0][
        "feature_set"]
    model, feats = models[best_name]
    print(f"\n  best feature set: {best_name}")

    # ------------------------------------------------ 2. rollout stability
    banner("2. Multi-step rollout — does the twin stay stable?")
    # Draw rollout racks from the COMPLETE-ROW test set: edge racks have no
    # left/right neighbour, so under the wider feature sets they contain no
    # usable rows at all and would silently contribute nothing.
    # dict.fromkeys de-duplicates while preserving order: STATE is normally also
    # a feature, and a repeated label would widen grp[feats] and break predict().
    usable = test_c[list(dict.fromkeys(feats + [STATE, "rack", "time"]))].dropna()
    roll_racks = [
        rk for rk, cnt in usable["rack"].value_counts().sort_index().items()
        if cnt > max(C.ROLLOUT_HORIZONS) + 30
    ][: args.rollout_racks]
    print(f"  rollout racks: {roll_racks}")
    roll_rows = []
    for h in C.ROLLOUT_HORIZONS:
        P, A = [], []
        for rk in roll_racks:
            grp = usable[usable["rack"] == rk].sort_values("time")
            if len(grp) <= h + 2:
                continue
            pp, aa = rollout(model, grp, feats, h)
            P.append(pp)
            A.append(aa)
        if not P:
            continue
        P, A = np.concatenate(P), np.concatenate(A)
        met = regression_metrics(A, P)
        roll_rows.append({"horizon_steps": h, "horizon_min": h * 5, **met})
        print(f"  {h:>3} steps ({h * 5:>3} min): RMSE {met['RMSE']:.4f} °C  "
              f"MAE {met['MAE']:.4f}  R2 {met['R2']:.4f}  n={met['n']:,}")

    roll = pd.DataFrame(roll_rows)
    if len(roll) > 1 and np.isfinite(roll["RMSE"]).all():
        growth = roll["RMSE"].iloc[-1] / max(roll["RMSE"].iloc[0], 1e-9)
        span = roll["horizon_steps"].iloc[-1] / roll["horizon_steps"].iloc[0]
        verdict = "SUB-linear" if growth < span else "SUPER-linear"
        print(f"\n  error grows {growth:.2f}x while the horizon grows {span:.0f}x "
              f"→ {verdict}.")
        if growth < span:
            print("  Sub-linear growth means errors are not compounding badly:")
            print("  the twin is stable enough to train an RL agent inside.")
        else:
            print("  Super-linear growth means errors compound. Before using this")
            print("  as an RL environment, either shorten the control interval or")
            print("  add more thermal history to the state.")

    # ------------------------------------------------ 3. action sensitivity
    banner("3. Cooling-action sensitivity — does the model respond to the actuator?")
    sens = pd.DataFrame()
    if ACTION in feats:
        te = test.dropna(subset=feats).copy()
        if len(te) > 200:
            base = te[feats].to_numpy("float64")
            ai = feats.index(ACTION)
            lo, hi = np.nanpercentile(te[ACTION], [5, 95])
            grid = np.linspace(lo, hi, 9)
            rows = []
            for v in grid:
                X = base.copy()
                X[:, ai] = v
                rows.append({ACTION: float(v),
                             "mean_predicted_delta_C": float(model.predict(X).mean())})
            sens = pd.DataFrame(rows)
            print("\n" + sens.to_string(index=False))
            slope = np.polyfit(sens[ACTION], sens["mean_predicted_delta_C"], 1)[0]
            print(f"\n  slope = {slope:.3e} °C per unit {ACTION}")
            print("  A NEGATIVE slope is physically correct: more airflow removes")
            print("  more heat. A positive or flat slope means the model has")
            print("  learned correlation, not causation — say so in the paper")
            print("  rather than hiding it. Node fans are firmware-controlled and")
            print("  RESPOND to temperature, so confounding is expected here; this")
            print("  is precisely why the CRAC setpoint from the vertiv plugin is")
            print("  needed for genuine control in sub-problem (c).")
    else:
        print(f"  '{ACTION}' not among model features — skipped.")

    # ------------------------------------------------ 4. ASHRAE envelope
    banner("4. ASHRAE TC 9.9 thermal envelope in the observed data")
    lo, hi = C.ASHRAE_RECOMMENDED
    alo, ahi = C.ASHRAE_ALLOWABLE_A1
    hot = f"{STATE}_max" if f"{STATE}_max" in df.columns else STATE
    v = df[hot].dropna()
    print(f"  using '{hot}' as the rack hotspot inlet temperature, n={len(v):,}")
    print(f"  observed range      : {v.min():.2f} .. {v.max():.2f} °C")
    print(f"  mean                : {v.mean():.2f} °C")
    print(f"  within recommended  ({lo}-{hi} °C): "
          f"{((v >= lo) & (v <= hi)).mean() * 100:.2f}%")
    print(f"  within allowable A1 ({alo}-{ahi} °C): "
          f"{((v >= alo) & (v <= ahi)).mean() * 100:.2f}%")
    print(f"  above {hi} °C          : {(v > hi).mean() * 100:.2f}%")
    print(f"  below {lo} °C          : {(v < lo).mean() * 100:.2f}%")
    print("\n  A large 'below recommended' share is the over-cooling headroom the")
    print("  whole case study is about: energy spent holding inlets colder than")
    print("  ASHRAE requires.")

    # ------------------------------------------------ figures
    if not roll.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(roll["horizon_min"], roll["RMSE"], "o-", color="#b5442e")
        ax.set_xlabel("rollout horizon (minutes)")
        ax.set_ylabel(f"{STATE} RMSE (°C)")
        ax.set_title("CS54 (b) — thermal twin error vs rollout horizon")
        ax.grid(alpha=0.3, lw=0.5)
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / "ps2_rollout_error.png", dpi=150)
        plt.close(fig)

    trained = abl[abl["feature_set"].isin(models)]
    if not trained.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.bar(trained["feature_set"], trained["RMSE"], color="#3b6ea5")
        ax.set_ylabel("one-step delta RMSE (°C)")
        ax.set_title("CS54 (b) — spatial feature ablation")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / "ps2_spatial_ablation.png", dpi=150)
        plt.close(fig)

    # an example rollout trace on one rack
    if roll_racks:
        rk = roll_racks[0]
        grp = usable[usable["rack"] == rk].sort_values("time")
        h = C.ROLLOUT_HORIZONS[-1]
        if len(grp) > h + 30:
            pp, aa = rollout(model, grp, feats, h)
            tt = grp["time"].to_numpy()[h: h + len(pp)]
            fig, ax = plt.subplots(figsize=(13, 4.2))
            ax.plot(tt, aa, color="#111", lw=1.6, label="actual")
            ax.plot(tt, pp, color="#b5442e", lw=1.2, label=f"twin, {h * 5} min rollout")
            ax.axhspan(lo, hi, color="#2e7d32", alpha=0.08,
                       label=f"ASHRAE recommended {lo}-{hi} °C")
            ax.set_title(f"CS54 (b) — learned thermal twin · rack {rk} · test period")
            ax.set_ylabel(f"{STATE} (°C)")
            ax.legend(fontsize=8, ncol=3, frameon=False)
            ax.grid(alpha=0.25, lw=0.5)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(C.FIGURES_DIR / f"ps2_rollout_rack{rk}.png", dpi=150)
            plt.close(fig)

    # ------------------------------------------------ save
    abl.to_csv(C.RESULTS_DIR / "ps2_spatial_ablation.csv", index=False)
    if not roll.empty:
        roll.to_csv(C.RESULTS_DIR / "ps2_rollout.csv", index=False)
    if not sens.empty:
        sens.to_csv(C.RESULTS_DIR / "ps2_action_sensitivity.csv", index=False)

    with open(C.RESULTS_DIR / "ps2_results.md", "w") as fh:
        fh.write("# CS54 Sub-problem (b) — learned thermal dynamics\n\n")
        fh.write(f"Month `{C.MONTH}` · {C.RESAMPLE} bins · state `{STATE}` · "
                 f"action `{ACTION}` · seed {C.SEED}\n\n")
        fh.write("Target is the one-step CHANGE in rack inlet temperature, not "
                 "its level, so autocorrelation cannot inflate the score.\n\n")
        fh.write("## Spatial ablation\n\n")
        fh.write(abl.to_markdown(index=False) + "\n\n")
        if not roll.empty:
            fh.write("## Multi-step rollout\n\n")
            fh.write(roll.to_markdown(index=False) + "\n\n")
        if not sens.empty:
            fh.write(f"## Sensitivity to `{ACTION}`\n\n")
            fh.write(sens.to_markdown(index=False) + "\n\n")
        fh.write("## ASHRAE TC 9.9\n\n")
        fh.write(f"- hotspot column: `{hot}`, n={len(v):,}\n")
        fh.write(f"- observed {v.min():.2f}–{v.max():.2f} °C, mean {v.mean():.2f} °C\n")
        fh.write(f"- within recommended {lo}–{hi} °C: "
                 f"{((v >= lo) & (v <= hi)).mean() * 100:.2f}%\n")
        fh.write(f"- below {lo} °C (over-cooling headroom): "
                 f"{(v < lo).mean() * 100:.2f}%\n")

    banner("Saved")
    for f in ("ps2_results.md", "ps2_spatial_ablation.csv", "ps2_rollout.csv"):
        print(f"  {C.RESULTS_DIR / f}")
    print(f"  figures in {C.FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
