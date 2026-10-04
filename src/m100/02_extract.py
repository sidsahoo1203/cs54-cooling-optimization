#!/usr/bin/env python3
"""
CS54 Step 2 — Extract a rack-level working dataset from the raw M100 Parquet.
VERSION 2.

Turns ~5 GB of per-node, 20-second telemetry into one tidy table of 49 racks x
5-minute bins. Output is a few tens of MB, so every experiment afterwards runs
in seconds rather than minutes.

Memory strategy: each metric is streamed in row batches and reduced on the fly
to (time_bin, rack) sums and counts. Peak memory tracks the batch size, not the
file size — this is what makes it work on 8 GB of RAM.

Aggregation convention: values are averaged ACROSS THE NODES IN A RACK, so every
column is "mean per node in that rack". Temperatures also get a _max column — the
rack hotspot, which is what ASHRAE compliance is judged on.

WHAT v2 FIXES (see CHANGES.md):
  1. Missing workload is no longer turned into zero workload. v1 computed
     cpu_busy with .fillna(0), which told the models the servers were idle in
     the ~46% of bins where Ganglia simply had not reported. v2 forward-fills
     within a bounded window and leaves longer gaps as NaN, with an audit flag.
  2. cpu_busy is now 100 - cpu_idle, a directly measured complement, rather
     than a sum of three sparser columns.
  3. Racks are restricted to 0..48. Ganglia reports node IDs past 979, which
     produced a phantom rack 49.
  4. Physically implausible sensor readings are removed (20-06 contains a 44 degC
     "rack inlet", which is a fault, not a measurement).

    python 02_extract.py
"""
from __future__ import annotations

import sys
import time as _time
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from common import (banner, bounded_ffill, clip_implausible, metric_dir,
                    rack_row, stream_metric_to_zones)

TEMP_LIKE = ("ambient", "temp")


def main() -> int:
    banner(f"CS54 v2 — extracting month {C.MONTH} at {C.RESAMPLE}, level={C.AGG_LEVEL}")
    t0 = _time.time()

    frames: list[pd.DataFrame] = []
    print("\nPer-node metrics (aggregated to racks 0..%d):" % (C.N_RACKS - 1))
    for metric, plugin in C.METRICS.items():
        path = metric_dir(C.RAW_ROOT, C.MONTH, plugin, metric)
        if path is None:
            print(f"    {metric:16s} SKIPPED (not on disk)")
            continue
        try:
            df = stream_metric_to_zones(
                path, metric,
                resample=C.RESAMPLE, nodes_per_rack=C.NODES_PER_RACK,
                agg_level=C.AGG_LEVEL, rack_subset=C.RACK_SUBSET,
                n_racks=C.N_RACKS,
                want_max=any(k in metric for k in TEMP_LIKE))
        except Exception as exc:
            print(f"    {metric:16s} FAILED: {exc}")
            continue
        if not df.empty:
            frames.append(df)

    if not frames:
        print("\nERROR: nothing extracted. Run 01_inspect.py and fix config.METRICS.")
        return 1

    print("\nJoining metrics on (time, zone) ...")
    wide = reduce(lambda a, b: pd.merge(a, b, on=["time", "zone"], how="outer"), frames)
    wide = wide.sort_values(["zone", "time"]).reset_index(drop=True)
    wide = wide.rename(columns={"zone": "rack"})

    print("\nCluster-wide metrics (broadcast to every rack):")
    for metric, plugin in C.CLUSTER_METRICS.items():
        path = metric_dir(C.RAW_ROOT, C.MONTH, plugin, metric)
        if path is None:
            print(f"    {metric:16s} SKIPPED (not on disk)")
            continue
        try:
            df = stream_metric_to_zones(path, metric, resample=C.RESAMPLE,
                                        agg_level="cluster")
        except Exception as exc:
            print(f"    {metric:16s} FAILED: {exc}")
            continue
        if df.empty:
            continue
        wide = pd.merge(wide, df.groupby("time", as_index=False)[metric].mean(),
                        on="time", how="left")

    # ---------------------------------------------------- v2: outlier removal
    banner("Sensor plausibility filter")
    wide, removed = clip_implausible(wide, C.PLAUSIBLE)
    if removed:
        for col, n in sorted(removed.items(), key=lambda kv: -kv[1]):
            lo, hi = C.PLAUSIBLE[col.replace("_max", "")]
            print(f"  {col:22s} {n:>8,} readings outside [{lo}, {hi}] -> NaN")
        print("\n  These are sensor faults, not measurements. A 44 degC rack inlet")
        print("  is not a real inlet temperature and would distort both the")
        print("  ASHRAE statistics and the thermal model.")
    else:
        print("  no implausible readings found")

    # ---------------------------------------------------- v2: bounded ffill
    banner("Workload gap handling (the v1 bug)")
    gang = [m for m, p in C.METRICS.items() if p == "ganglia_pub" and m in wide.columns]
    before = {c: float(wide[c].isna().mean() * 100) for c in gang}
    wide, imputed = bounded_ffill(wide, gang, group="rack", limit=C.IMPUTE_LIMIT_BINS)
    wide["workload_imputed"] = imputed.astype(int)
    after = {c: float(wide[c].isna().mean() * 100) for c in gang}

    print(f"  forward-fill limit: {C.IMPUTE_LIMIT_BINS} bins "
          f"({C.IMPUTE_LIMIT_BINS * C.STEP_MINUTES} min)\n")
    print(f"  {'column':<18}{'missing before':>16}{'missing after':>16}")
    for c in gang:
        print(f"  {c:<18}{before[c]:>15.2f}%{after[c]:>15.2f}%")
    print(f"\n  rows with at least one imputed workload value: "
          f"{imputed.mean() * 100:.2f}%")
    print("  Rows still NaN after this are genuine gaps and will be dropped by")
    print("  the models — honestly, rather than filled with a fabricated zero.")

    # ---------------------------------------------------- derived features
    banner("Derived features")
    wide["rack_row"] = wide["rack"].apply(lambda r: rack_row(int(r), C.RACK_ROWS))

    # v2: cpu_busy from the DIRECTLY MEASURED idle fraction. No fillna anywhere.
    if "cpu_idle" in wide.columns:
        wide["cpu_busy"] = 100.0 - wide["cpu_idle"]
    if {"cpu_user", "cpu_system", "cpu_wio"}.issubset(wide.columns):
        wide["cpu_busy_alt"] = wide[["cpu_user", "cpu_system", "cpu_wio"]].sum(
            axis=1, min_count=3)

    if {"p0_power", "p1_power"}.issubset(wide.columns):
        # min_count=2: if either socket is missing the sum is NaN, not a half-sum.
        wide["cpu_power"] = wide[["p0_power", "p1_power"]].sum(axis=1, min_count=2)
    if {"fan0_0", "fan0_1"}.issubset(wide.columns):
        wide["fan_speed"] = wide[["fan0_0", "fan0_1"]].mean(axis=1)
    elif "fan0_0" in wide.columns:
        wide["fan_speed"] = wide["fan0_0"]
    if {"p0_core0_temp", "p1_core0_temp"}.issubset(wide.columns):
        wide["core_temp"] = wide[["p0_core0_temp", "p1_core0_temp"]].mean(axis=1)

    # Cyclic time. Sine AND cosine together so 23:55 and 00:00 are adjacent in
    # feature space rather than maximally far apart.
    t = pd.to_datetime(wide["time"])
    mod = t.dt.hour * 60 + t.dt.minute
    wide["tod_sin"] = np.sin(2 * np.pi * mod / 1440.0)
    wide["tod_cos"] = np.cos(2 * np.pi * mod / 1440.0)
    wide["dow_sin"] = np.sin(2 * np.pi * t.dt.dayofweek / 7.0)
    wide["dow_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 7.0)
    wide["is_weekend"] = (t.dt.dayofweek >= 5).astype(int)
    for c in ("cpu_busy", "cpu_busy_alt", "cpu_power", "fan_speed", "core_temp"):
        if c in wide.columns:
            print(f"  {c:<16} missing {wide[c].isna().mean() * 100:6.2f}%")

    # ---------------------------------------------------- report + save
    banner("Extracted dataset")
    print(f"  shape      : {wide.shape[0]:,} rows x {wide.shape[1]} columns")
    print(f"  time span  : {wide['time'].min()}  ..  {wide['time'].max()}")
    print(f"  racks      : {wide['rack'].nunique()}  (expected {C.N_RACKS})")
    print(f"  memory     : {wide.memory_usage(deep=True).sum() / 1e6:,.1f} MB")

    print("\n  missing-data rate by column:")
    for col, pct in (wide.isna().mean() * 100).sort_values(ascending=False).items():
        u = C.UNITS.get(col.replace("_max", ""), ("", ""))[0]
        flag = "  <-- mostly empty" if pct > 50 else ""
        print(f"      {col:<22}{pct:6.2f} %  {u:<6}{flag}")

    out = C.PROCESSED_DIR / f"rack_{C.RESAMPLE}_{C.MONTH}.parquet"
    wide.to_parquet(out, index=False)
    print(f"\n  wrote {out}  ({out.stat().st_size / 1e6:,.1f} MB)")

    wide.describe().T.to_csv(C.RESULTS_DIR / f"extract_summary_{C.MONTH}.csv")

    # v2: a units table, so no number is ever reported unitless again.
    with open(C.RESULTS_DIR / f"units_{C.MONTH}.md", "w") as fh:
        fh.write(f"# CS54 — units and meanings ({C.MONTH})\n\n")
        fh.write("| column | unit | meaning |\n|---|---|---|\n")
        for col, (unit, mean) in C.UNITS.items():
            if col in wide.columns:
                fh.write(f"| `{col}` | {unit} | {mean} |\n")
        fh.write("\nAll per-node metrics are averaged across the nodes in a rack, "
                 "so every value is *mean per node in that rack*. Temperature "
                 "columns additionally carry `_max`, the rack hotspot.\n")
        fh.write(f"\nThermal standard: {C.ASHRAE_EDITION}. "
                 f"Recommended {C.ASHRAE_RECOMMENDED[0]}-{C.ASHRAE_RECOMMENDED[1]} degC, "
                 f"allowable {C.ASHRAE_ALLOWABLE_A1[0]}-{C.ASHRAE_ALLOWABLE_A1[1]} degC "
                 f"dry-bulb at the equipment inlet.\n")
    print(f"  wrote {C.RESULTS_DIR / f'units_{C.MONTH}.md'}")

    print(f"\n  elapsed {(_time.time() - t0) / 60:.1f} min")
    banner("NEXT")
    print("      python 03_ps1_predict.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
