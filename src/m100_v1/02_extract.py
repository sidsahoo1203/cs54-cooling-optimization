#!/usr/bin/env python3
"""
CS54 Step 2 — Extract a rack-level working dataset from the raw M100 Parquet.

Turns ~5 GB of per-node, 20-second telemetry into one tidy table of roughly
49 racks x 5-minute bins. Output is a few tens of MB, so every experiment
afterwards runs in seconds instead of minutes.

Memory strategy: each metric is streamed in row batches and reduced to
(time_bin, rack) sums and counts on the fly. Peak memory stays near the batch
size, not the file size — this is what makes it work on 8 GB of RAM.

Aggregation convention: values are averaged ACROSS THE NODES IN A RACK, so
every column is "mean per node in that rack". Temperatures also get a _max
column, which is the rack hotspot and what ASHRAE compliance is judged on.
Per-node means avoid the confound of racks having different numbers of live
nodes at different times.

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
from common import banner, metric_dir, rack_row, stream_metric_to_zones

TEMP_LIKE = ("ambient", "temp")


def main() -> int:
    banner(f"CS54 — extracting month {C.MONTH} at {C.RESAMPLE}, level={C.AGG_LEVEL}")
    t0 = _time.time()

    frames: list[pd.DataFrame] = []

    # ------------------------------------------------ per-node metrics
    print("\nPer-node metrics (aggregated to racks):")
    for metric, plugin in C.METRICS.items():
        path = metric_dir(C.RAW_ROOT, C.MONTH, plugin, metric)
        if path is None:
            print(f"    {metric:16s} SKIPPED (not on disk)")
            continue
        want_max = any(k in metric for k in TEMP_LIKE)
        try:
            df = stream_metric_to_zones(
                path,
                metric,
                resample=C.RESAMPLE,
                nodes_per_rack=C.NODES_PER_RACK,
                agg_level=C.AGG_LEVEL,
                rack_subset=C.RACK_SUBSET,
                want_max=want_max,
            )
        except Exception as exc:  # keep going; one bad metric must not kill the run
            print(f"    {metric:16s} FAILED: {exc}")
            continue
        if not df.empty:
            frames.append(df)

    if not frames:
        print("\nERROR: nothing extracted. Run 01_inspect.py and fix config.METRICS.")
        return 1

    # ------------------------------------------------ join into a wide table
    print("\nJoining metrics on (time, zone) ...")
    wide = reduce(
        lambda a, b: pd.merge(a, b, on=["time", "zone"], how="outer"), frames
    )
    wide = wide.sort_values(["zone", "time"]).reset_index(drop=True)
    wide = wide.rename(columns={"zone": "rack"})

    # ------------------------------------------------ cluster-wide metrics
    print("\nCluster-wide metrics (broadcast to every rack):")
    for metric, plugin in C.CLUSTER_METRICS.items():
        path = metric_dir(C.RAW_ROOT, C.MONTH, plugin, metric)
        if path is None:
            print(f"    {metric:16s} SKIPPED (not on disk)")
            continue
        try:
            df = stream_metric_to_zones(
                path, metric, resample=C.RESAMPLE, agg_level="cluster"
            )
        except Exception as exc:
            print(f"    {metric:16s} FAILED: {exc}")
            continue
        if df.empty:
            continue
        # Collapse any entity dimension: one value per time bin.
        df = df.groupby("time", as_index=False)[metric].mean()
        wide = pd.merge(wide, df, on="time", how="left")

    # ------------------------------------------------ derived features
    print("\nDeriving features ...")
    wide["rack_row"] = wide["rack"].apply(lambda r: rack_row(int(r), C.RACK_ROWS))

    if {"cpu_user", "cpu_system", "cpu_wio"}.issubset(wide.columns):
        wide["cpu_busy"] = (
            wide["cpu_user"].fillna(0)
            + wide["cpu_system"].fillna(0)
            + wide["cpu_wio"].fillna(0)
        )
    elif "cpu_idle" in wide.columns:
        wide["cpu_busy"] = 100.0 - wide["cpu_idle"]

    if {"p0_power", "p1_power"}.issubset(wide.columns):
        wide["cpu_power"] = wide["p0_power"].fillna(0) + wide["p1_power"].fillna(0)

    if {"fan0_0", "fan0_1"}.issubset(wide.columns):
        wide["fan_speed"] = wide[["fan0_0", "fan0_1"]].mean(axis=1)
    elif "fan0_0" in wide.columns:
        wide["fan_speed"] = wide["fan0_0"]

    if {"p0_core0_temp", "p1_core0_temp"}.issubset(wide.columns):
        wide["core_temp"] = wide[["p0_core0_temp", "p1_core0_temp"]].mean(axis=1)

    # Time-of-day / day-of-week as cyclic features. Sine and cosine together so
    # 23:55 and 00:00 are adjacent in feature space rather than maximally apart.
    t = pd.to_datetime(wide["time"])
    minute_of_day = t.dt.hour * 60 + t.dt.minute
    wide["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    wide["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    wide["dow_sin"] = np.sin(2 * np.pi * t.dt.dayofweek / 7.0)
    wide["dow_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 7.0)
    wide["is_weekend"] = (t.dt.dayofweek >= 5).astype(int)

    # ------------------------------------------------ report + save
    banner("Extracted dataset")
    print(f"  shape      : {wide.shape[0]:,} rows x {wide.shape[1]} columns")
    print(f"  time span  : {wide['time'].min()}  ..  {wide['time'].max()}")
    print(f"  racks      : {wide['rack'].nunique()}  "
          f"(rows/rack ~ {len(wide) // max(wide['rack'].nunique(), 1):,})")
    print(f"  memory     : {wide.memory_usage(deep=True).sum() / 1e6:,.1f} MB")

    print("\n  missing-data rate by column:")
    miss = (wide.isna().mean() * 100).sort_values(ascending=False)
    for col, pct in miss.items():
        flag = "  <-- mostly empty" if pct > 50 else ""
        print(f"      {col:22s} {pct:6.2f} %{flag}")

    out = C.PROCESSED_DIR / f"rack_{C.RESAMPLE}_{C.MONTH}.parquet"
    wide.to_parquet(out, index=False)
    print(f"\n  wrote {out}  ({out.stat().st_size / 1e6:,.1f} MB)")

    summary = wide.describe().T
    summary_path = C.RESULTS_DIR / f"extract_summary_{C.MONTH}.csv"
    summary.to_csv(summary_path)
    print(f"  wrote {summary_path}")

    print(f"\n  elapsed {(_time.time() - t0) / 60:.1f} min")

    banner("NEXT")
    print("  Check the missing-data table above. Any column over ~50% empty")
    print("  should be removed from config.METRICS before modelling.")
    print("      python 03_ps1_predict.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
