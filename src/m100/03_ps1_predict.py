#!/usr/bin/env python3
"""
CS54 Sub-problem (a) — near-term rack-level heat and temperature prediction.
VERSION 2.

Case study wording:
    "(a) predicts near-term server/rack-level heat generation based on workload
     scheduling data and historical thermal patterns"

Targets, both forecast at several horizons:
    total_power  — mean node power per rack, WATTS. This IS the heat generation.
    ambient      — mean node inlet air temperature per rack, DEGREES CELSIUS.

Models:
    persistence      value now, carried forward. Zero parameters. The floor.
    seasonal_naive   value at the same clock time yesterday. Captures the daily
                     cycle for free, and is a much harder baseline than
                     persistence on anything diurnal.
    ridge            L2-regularised linear regression
    xgboost          gradient-boosted trees — strongest tabular method in the
                     literature (papers 3, 13, 27 of the review)
    lstm / gru       the recurrent models the case study specifies

WHAT v2 FIXES (see CHANGES.md):
  1. EVERY model is now scored on EXACTLY the same rows. In v1 the recurrent
     models were evaluated on 26,529 rows while the flat models used 25,001,
     which made the comparison meaningless. v2 intersects the flat and sequence
     key sets and evaluates all five models on that intersection.
  2. The sequence window was off by one against the flat path, so the two were
     not forecasting the same thing. Both now use: features up to and including
     t, target at t+h.
  3. Four horizons instead of one: 5, 15, 30 and 60 minutes.
  4. A seasonal-naive baseline is added.
  5. Units are printed and saved with every number.

    python 03_ps1_predict.py                  # full run, all horizons
    python 03_ps1_predict.py --quick          # 12 racks, 1 horizon, 6 epochs
    python 03_ps1_predict.py --no-deep        # skip LSTM/GRU
    python 03_ps1_predict.py --horizons 1 12  # only 5-min and 60-min
"""
from __future__ import annotations

import argparse
import os
import sys
import time as _time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from common import banner, chronological_split, regression_metrics


# ------------------------------------------------------------------ features
def build_supervised(df: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """
    Supervised table. For every (rack, t): features describe up to and including
    t; y is the target at t + horizon. Lags and rolling windows are computed
    WITHIN each rack, so no rack's history bleeds into another's.
    """
    df = df.sort_values(["rack", "time"]).copy()
    g = df.groupby("rack", sort=False)

    drivers = [c for c in (target, "cpu_busy", "load_one", "cpu_power",
                           "ambient", "core_temp", "fan_speed", "total_power")
               if c in df.columns]
    drivers = list(dict.fromkeys(drivers))

    for col in drivers:
        for lag in C.LAGS:
            df[f"{col}_lag{lag}"] = g[col].shift(lag)
        for w in C.ROLL_WINDOWS:
            r = g[col].rolling(w, min_periods=max(2, w // 2))
            df[f"{col}_rmean{w}"] = r.mean().reset_index(level=0, drop=True)
            df[f"{col}_rstd{w}"] = r.std().reset_index(level=0, drop=True)
        df[f"{col}_diff1"] = g[col].diff(1)

    # The seasonal-naive reference: same clock time, 24 h earlier.
    df["_seasonal"] = g[target].shift(C.SEASONAL_LAG)
    # The label. A NEGATIVE shift looks FORWARD, which is the point.
    df["y"] = g[target].shift(-horizon)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"time", "y", "rack", "_seasonal"}
    # *_max columns are hotspot diagnostics we report on, not model inputs:
    # they carry same-timestamp information about the target.
    return [c for c in df.columns
            if c not in drop and not c.endswith("_max")
            and pd.api.types.is_numeric_dtype(df[c])]


# ------------------------------------------------------------------ sequences
def build_sequences(df: pd.DataFrame, target: str, horizon: int, lookback: int,
                    feats: list[str]):
    """
    3-D tensor (samples, lookback, features) for the recurrent models, built per
    rack so windows never straddle two racks.

    ALIGNMENT, matching the flat path exactly:
        window = rows [i-lookback+1 .. i]   (inclusive of i)
        target = value at i + horizon
        key    = (rack, time at i)
    """
    Xs, ys, ts, racks = [], [], [], []
    for rack, grp in df.groupby("rack", sort=True):
        grp = grp.sort_values("time")
        arr = grp[feats].to_numpy(dtype="float32")
        tgt = grp[target].to_numpy(dtype="float32")
        tim = grp["time"].to_numpy()
        n = len(grp)
        for i in range(lookback - 1, n - horizon):
            win = arr[i - lookback + 1: i + 1]
            y = tgt[i + horizon]
            if not np.isfinite(win).all() or not np.isfinite(y):
                continue
            Xs.append(win); ys.append(y); ts.append(tim[i]); racks.append(rack)
    if not Xs:
        return (np.empty((0, lookback, len(feats)), "float32"),
                np.empty(0, "float32"), np.array([]), np.array([]))
    return (np.stack(Xs), np.asarray(ys, "float32"),
            np.asarray(ts), np.asarray(racks))


def make_rnn(kind: str, lookback: int, n_feat: int, units: int = 48):
    """Keras 2 and Keras 3 compatible: explicit Input layer, no legacy kwargs."""
    from tensorflow import keras
    cell = keras.layers.LSTM if kind == "lstm" else keras.layers.GRU
    m = keras.Sequential([
        keras.Input(shape=(lookback, n_feat)),
        cell(units),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ], name=kind)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


# ------------------------------------------------------------------ one run
def run(df: pd.DataFrame, target: str, horizon: int, args) -> pd.DataFrame:
    unit = C.UNITS.get(target, ("", ""))[0]
    banner(f"(a) target={target} [{unit}] · horizon={horizon} steps "
           f"({horizon * C.STEP_MINUTES} min)")

    sup = build_supervised(df, target, horizon)
    feats = feature_columns(sup)
    sup = sup.dropna(subset=["y"])

    stamps = np.sort(sup["time"].unique())
    tr_s, va_s, te_s = chronological_split(len(stamps), C.TEST_FRACTION, C.VAL_FRACTION)
    t_tr, t_va, t_te = stamps[tr_s], stamps[va_s], stamps[te_s]
    print(f"  split on {len(stamps):,} timestamps — "
          f"train {len(t_tr):,} · val {len(t_va):,} · test {len(t_te):,}")
    print(f"  test period {t_te[0]} .. {t_te[-1]}")

    train = sup[sup["time"].isin(t_tr)].dropna(subset=feats)
    val = sup[sup["time"].isin(t_va)].dropna(subset=feats)
    test = sup[sup["time"].isin(t_te)].dropna(subset=feats)
    if len(train) < 500 or len(test) < 100:
        print("  !! too few complete rows to model")
        return pd.DataFrame()

    # ---- sequences, and THE COMMON KEY SET -------------------------------
    seq = None
    if not args.no_deep:
        seq_feats = [c for c in C.SEQ_FEATURES if c in df.columns]
        Xs, ys, ts, rk = build_sequences(df, target, horizon, C.LOOKBACK_STEPS, seq_feats)
        if Xs.shape[0] > 1000:
            seq = (Xs, ys, ts, rk, seq_feats)
            print(f"  sequence tensor {Xs.shape} from {len(seq_feats)} signals "
                  f"({Xs.nbytes / 1e6:,.0f} MB)")
        else:
            print("  not enough complete sequences — skipping recurrent models")

    flat_key = pd.MultiIndex.from_arrays([test["rack"].to_numpy(),
                                          test["time"].to_numpy()])
    if seq is not None:
        Xs, ys, ts, rk, seq_feats = seq
        m_te = np.isin(ts, t_te)
        seq_key = pd.MultiIndex.from_arrays([rk[m_te], ts[m_te]])
        common = flat_key.intersection(seq_key)
    else:
        common = flat_key

    if len(common) < 100:
        print(f"  !! only {len(common)} rows common to all models — cannot compare")
        return pd.DataFrame()
    print(f"  EVALUATION SET: {len(common):,} rows common to every model "
          f"({len(common) / len(flat_key) * 100:.1f}% of flat-usable test rows)")

    te = test.set_index(flat_key).loc[common]
    Xte = te[feats].to_numpy("float32")
    yte = te["y"].to_numpy("float64")
    Xtr, ytr = train[feats].to_numpy("float32"), train["y"].to_numpy("float32")

    results, preds = [], {}

    def record(name, yp):
        results.append({"model": name, **regression_metrics(yte, yp)})
        preds[name] = np.asarray(yp, "float64")

    # -- persistence: the honest floor
    record("persistence", te[target].to_numpy("float64"))

    # -- seasonal naive: same clock time yesterday
    sn = te["_seasonal"].to_numpy("float64")
    if np.isfinite(sn).mean() > 0.5:
        record("seasonal_naive", np.where(np.isfinite(sn), sn,
                                          te[target].to_numpy("float64")))

    # -- ridge. Scaler fit on TRAIN ONLY — fitting on everything leaks test
    #    statistics into training and is a classic silent error.
    sc = StandardScaler().fit(Xtr)
    ridge = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
    record("ridge", ridge.predict(sc.transform(Xte)))

    # -- xgboost
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                           random_state=C.SEED, n_jobs=-1, tree_method="hist",
                           early_stopping_rounds=40, eval_metric="rmse")
        ev = [(val[feats].to_numpy("float32"), val["y"].to_numpy("float32"))] \
            if len(val) > 100 else [(Xte, yte.astype("float32"))]
        xgb.fit(Xtr, ytr, eval_set=ev, verbose=False)
        record("xgboost", xgb.predict(Xte))
        imp = xgb.feature_importances_
        order = np.argsort(imp)[::-1][:20]
        fig, ax = plt.subplots(figsize=(8, 6.5))
        ax.barh([feats[i] for i in order][::-1], imp[order][::-1], color="#3b6ea5")
        ax.set_title(f"XGBoost gain — {target}, {horizon * C.STEP_MINUTES} min")
        ax.grid(axis="x", alpha=.25, lw=.5)
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / f"ps1_{target}_h{horizon}_importance.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"  xgboost skipped: {exc}")

    # -- recurrent models, on the SAME rows
    if seq is not None:
        Xs, ys, ts, rk, seq_feats = seq
        m_tr, m_va = np.isin(ts, t_tr), np.isin(ts, t_va)
        m_te = np.isin(ts, t_te)
        mu = Xs[m_tr].reshape(-1, Xs.shape[2]).mean(0)
        sd = Xs[m_tr].reshape(-1, Xs.shape[2]).std(0) + 1e-8
        ymu, ysd = ys[m_tr].mean(), ys[m_tr].std() + 1e-8

        # position of every test sequence, then reorder to `common`
        key_te = pd.MultiIndex.from_arrays([rk[m_te], ts[m_te]])
        pos = pd.Series(np.arange(m_te.sum()), index=key_te)
        take = pos.loc[common].to_numpy()
        Xte_seq = (Xs[m_te][take] - mu) / sd

        from tensorflow import keras
        for kind in ("lstm", "gru"):
            keras.utils.set_random_seed(C.SEED)
            t0 = _time.time()
            model = make_rnn(kind, Xs.shape[1], Xs.shape[2])
            model.fit((Xs[m_tr] - mu) / sd, (ys[m_tr] - ymu) / ysd,
                      validation_data=(((Xs[m_va] - mu) / sd,
                                        (ys[m_va] - ymu) / ysd)
                                       if m_va.sum() > 100 else None),
                      epochs=args.epochs, batch_size=C.BATCH_SIZE,
                      callbacks=[keras.callbacks.EarlyStopping(
                          monitor="val_loss" if m_va.sum() > 100 else "loss",
                          patience=args.patience, restore_best_weights=True)],
                      verbose=args.verbose)
            record(kind, model.predict(Xte_seq, verbose=0).ravel() * ysd + ymu)
            print(f"  {kind} trained in {_time.time() - t0:,.0f}s")

    # ---- report ----------------------------------------------------------
    res = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    res.insert(0, "target", target)
    res.insert(1, "unit", unit)
    res.insert(2, "horizon_min", horizon * C.STEP_MINUTES)
    print("\n" + res.to_string(index=False))

    base = res.loc[res["model"] == "persistence"].iloc[0]
    print(f"\n  vs persistence (RMSE {base['RMSE']:.4g} {unit}, "
          f"MAE {base['MAE']:.4g} {unit}):")
    for _, r in res.iterrows():
        if r["model"] == "persistence":
            continue
        dr = (1 - r["RMSE"] / base["RMSE"]) * 100
        dm = (1 - r["MAE"] / base["MAE"]) * 100
        verdict = "BEATS on both" if dr > 0 and dm > 0 else \
                  "RMSE only" if dr > 0 else "WORSE on both" if dm <= 0 else "MAE only"
        print(f"    {r['model']:<16} RMSE {dr:+6.2f}%   MAE {dm:+6.2f}%   {verdict}")
    print("\n  A model is only genuinely better if it wins on BOTH. Winning on")
    print("  RMSE alone means it avoids a few large errors while being worse on")
    print("  the typical prediction.")

    # ---- figure ----------------------------------------------------------
    if preds:
        rack = int(te["rack"].mode().iloc[0])
        sel = (te["rack"] == rack).to_numpy()
        if sel.sum() > 20:
            fig, ax = plt.subplots(figsize=(13, 4.5))
            ax.plot(te.loc[sel, "time"].to_numpy(), yte[sel], lw=1.6,
                    color="#111", label="actual", zorder=5)
            for name, yp in preds.items():
                ax.plot(te.loc[sel, "time"].to_numpy(), yp[sel], lw=1.0,
                        alpha=.85, label=name)
            ax.set_title(f"CS54 (a) — {target} forecast "
                         f"{horizon * C.STEP_MINUTES} min ahead · rack {rack}")
            ax.set_ylabel(f"{target} ({unit})")
            ax.legend(ncol=6, fontsize=8, frameon=False)
            ax.grid(alpha=.25, lw=.5)
            fig.autofmt_xdate(); fig.tight_layout()
            fig.savefig(C.FIGURES_DIR / f"ps1_{target}_h{horizon}_rack{rack}.png", dpi=150)
            plt.close(fig)
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-deep", action="store_true")
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--patience", type=int, default=C.PATIENCE)
    p.add_argument("--racks", type=int, default=None)
    p.add_argument("--horizons", type=int, nargs="+", default=None)
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()
    horizons = args.horizons or C.HORIZONS
    if args.quick:
        args.racks = args.racks or 12
        args.epochs = min(args.epochs, 6)
        horizons = horizons[-1:]

    src = C.PROCESSED_DIR / f"rack_{C.RESAMPLE}_{C.MONTH}.parquet"
    if not src.exists():
        print(f"ERROR: {src} not found. Run 02_extract.py first.")
        return 1

    banner("CS54 v2 — Sub-problem (a): heat and temperature prediction")
    df = pd.read_parquet(src)
    df["rack"] = pd.to_numeric(df["rack"], errors="coerce")
    df = df.dropna(subset=["rack", "time"])
    df["rack"] = df["rack"].astype(int)
    if args.racks:
        keep = sorted(df["rack"].unique())[: args.racks]
        df = df[df["rack"].isin(keep)]
        print(f"  restricted to {len(keep)} racks")
    print(f"  {df.shape[0]:,} rows x {df.shape[1]} cols · seed {C.SEED} · "
          f"horizons {[h * C.STEP_MINUTES for h in horizons]} min")

    allr = []
    for target in [t for t in (C.TARGET, C.SECOND_TARGET) if t in df.columns]:
        for h in horizons:
            r = run(df, target, h, args)
            if not r.empty:
                allr.append(r)
    if not allr:
        print("\nNo results produced.")
        return 1

    final = pd.concat(allr, ignore_index=True)
    final.to_csv(C.RESULTS_DIR / "ps1_results.csv", index=False)
    with open(C.RESULTS_DIR / "ps1_results.md", "w") as fh:
        fh.write("# CS54 Sub-problem (a) — prediction results (v2)\n\n")
        fh.write(f"Month `{C.MONTH}` · {C.RESAMPLE} bins · rack level · seed "
                 f"{C.SEED} · chronological split "
                 f"{int((1 - C.TEST_FRACTION) * 100)}/{int(C.TEST_FRACTION * 100)}\n\n")
        fh.write("Every model at a given target and horizon is scored on "
                 "**identical rows**.\n\n")
        for target in final["target"].unique():
            sub = final[final["target"] == target]
            u = sub["unit"].iloc[0]
            fh.write(f"## `{target}` ({u})\n\n")
            fh.write(sub.drop(columns=["target", "unit"]).to_markdown(index=False))
            fh.write("\n\n")

    banner("Saved")
    print(f"  {C.RESULTS_DIR / 'ps1_results.csv'}")
    print(f"  {C.RESULTS_DIR / 'ps1_results.md'}")
    print(f"  figures in {C.FIGURES_DIR}")
    print("\n  NEXT:  python 04_ps2_thermal.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
