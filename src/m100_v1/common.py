"""
Shared helpers for the CS54 M100 pipeline.

The M100 ExaData Parquet files are "long" tables: one row per
(timestamp, entity, value). Exact column names are not documented
identically everywhere, so everything here detects the schema rather than
assuming it. That is deliberate — it is the difference between a script that
runs on your machine first try and one that throws KeyError.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

warnings.filterwarnings("ignore", category=FutureWarning)

# Candidate column names, in priority order.
TIME_CANDIDATES = ["timestamp", "time", "ts", "_timestamp", "datetime", "date"]
VALUE_CANDIDATES = ["value", "val", "v", "reading"]
ENTITY_CANDIDATES = ["node", "device", "panel", "host", "hostname", "nodeid"]


# --------------------------------------------------------------- paths
def metric_dir(raw_root: Path, month: str, plugin: str, metric: str) -> Path | None:
    """Locate a metric directory, tolerating both Hive and bare layouts."""
    candidates = [
        raw_root / f"year_month={month}" / f"plugin={plugin}" / f"metric={metric}",
        raw_root / month / f"plugin={plugin}" / f"metric={metric}",
        raw_root / f"year_month={month}" / plugin / metric,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    # Last resort: glob for it anywhere under the root.
    hits = list(raw_root.glob(f"**/plugin={plugin}/metric={metric}"))
    return hits[0] if hits else None


def list_available_metrics(raw_root: Path, month: str) -> dict[str, list[str]]:
    """Map plugin -> sorted metric names actually present on disk."""
    out: dict[str, list[str]] = {}
    for plugin_dir in sorted(raw_root.glob(f"**/plugin=*")):
        if not plugin_dir.is_dir():
            continue
        if month and month not in str(plugin_dir):
            continue
        plugin = plugin_dir.name.split("=", 1)[1]
        metrics = sorted(
            d.name.split("=", 1)[1]
            for d in plugin_dir.iterdir()
            if d.is_dir() and d.name.startswith("metric=")
        )
        out[plugin] = metrics
    return out


# --------------------------------------------------------------- schema
def detect_columns(names: list[str]) -> tuple[str | None, str | None, str | None]:
    """Return (time_col, value_col, entity_col) guessed from column names."""
    lower = {n.lower(): n for n in names}

    def pick(cands):
        for c in cands:
            if c in lower:
                return lower[c]
        return None

    time_col = pick(TIME_CANDIDATES)
    value_col = pick(VALUE_CANDIDATES)
    entity_col = pick(ENTITY_CANDIDATES)

    # Fallbacks: anything with "time" in it; the lone leftover column as value.
    if time_col is None:
        for n in names:
            if "time" in n.lower() or "date" in n.lower():
                time_col = n
                break
    if value_col is None:
        leftovers = [n for n in names if n not in (time_col, entity_col)]
        if len(leftovers) == 1:
            value_col = leftovers[0]
    return time_col, value_col, entity_col


def to_datetime_series(s: pd.Series) -> pd.Series:
    """Convert a time column to datetime64, inferring epoch units if integer."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        med = float(np.nanmedian(v.to_numpy(dtype="float64", na_value=np.nan)))
        # Epoch magnitude -> unit. 2020 is ~1.58e9 seconds.
        if med > 1e17:
            unit = "ns"
        elif med > 1e14:
            unit = "us"
        elif med > 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(v, unit=unit, errors="coerce")
    return pd.to_datetime(s, errors="coerce", format="mixed")


# --------------------------------------------------------------- streaming
def stream_metric_to_zones(
    path: Path,
    metric: str,
    resample: str = "5min",
    nodes_per_rack: int = 20,
    agg_level: str = "rack",
    rack_subset=None,
    batch_rows: int = 500_000,
    compact_every: int = 24,
    want_max: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Read one metric's Parquet directory in bounded memory and return a tidy
    frame with columns [time, zone, <metric>] (and <metric>_max if want_max).

    Aggregation is exact: partial sums and counts are accumulated, then divided
    once at the end. (Averaging pre-averaged batches would be wrong whenever
    batches straddle a time bin unevenly.)
    """
    ds = pads.dataset(str(path), format="parquet")
    names = list(ds.schema.names)
    time_col, value_col, entity_col = detect_columns(names)

    if time_col is None or value_col is None:
        raise ValueError(
            f"{metric}: could not identify time/value columns in {names}. "
            f"Edit TIME_CANDIDATES / VALUE_CANDIDATES in common.py."
        )

    cols = [c for c in (time_col, value_col, entity_col) if c is not None]
    partials: list[pd.DataFrame] = []
    accum: pd.DataFrame | None = None
    n_rows = 0

    def compact(frames, acc):
        if not frames:
            return acc
        chunk = pd.concat(frames, ignore_index=True)
        if acc is not None:
            chunk = pd.concat([acc, chunk], ignore_index=True)
        return chunk.groupby(["time", "zone"], as_index=False).agg(
            _sum=("_sum", "sum"), _cnt=("_cnt", "sum"), _max=("_max", "max")
        )

    for batch in ds.to_batches(columns=cols, batch_size=batch_rows):
        if batch.num_rows == 0:
            continue
        df = batch.to_pandas()
        n_rows += len(df)

        df["time"] = to_datetime_series(df[time_col]).dt.floor(resample)
        df["_v"] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["time", "_v"])
        if df.empty:
            continue

        if entity_col is not None and agg_level == "rack":
            ent = pd.to_numeric(df[entity_col], errors="coerce")
            if ent.notna().any():
                df["zone"] = (ent // nodes_per_rack).astype("Int64")
            else:
                # Non-numeric entity (device/panel names): keep as its own zone.
                df["zone"] = df[entity_col].astype("string")
        elif entity_col is not None:
            df["zone"] = pd.to_numeric(df[entity_col], errors="coerce").astype("Int64")
        else:
            df["zone"] = -1  # cluster-wide metric

        if rack_subset is not None and pd.api.types.is_integer_dtype(
            pd.Series(df["zone"]).dtype
        ):
            df = df[df["zone"].isin(list(rack_subset))]
            if df.empty:
                continue

        g = (
            df.groupby(["time", "zone"], dropna=True)["_v"]
            .agg(_sum="sum", _cnt="count", _max="max")
            .reset_index()
        )
        partials.append(g)

        if len(partials) >= compact_every:
            accum = compact(partials, accum)
            partials = []

    accum = compact(partials, accum)
    if accum is None or accum.empty:
        if verbose:
            print(f"    !! {metric}: no usable rows", file=sys.stderr)
        return pd.DataFrame(columns=["time", "zone", metric])

    accum[metric] = accum["_sum"] / accum["_cnt"].replace(0, np.nan)
    keep = ["time", "zone", metric]
    if want_max:
        accum[f"{metric}_max"] = accum["_max"]
        keep.append(f"{metric}_max")
    out = accum[keep].sort_values(["time", "zone"])
    if verbose:
        print(
            f"    {metric:16s} rows_read={n_rows:>12,}  bins={len(out):>9,}  "
            f"zones={out['zone'].nunique():>4}  "
            f"[{out['time'].min()} .. {out['time'].max()}]"
        )
    return out.reset_index(drop=True)


# --------------------------------------------------------------- metrics/eval
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype="float64").ravel()
    y_pred = np.asarray(y_pred, dtype="float64").ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if y_true.size == 0:
        return {k: float("nan") for k in ("RMSE", "MAE", "MAPE_%", "R2", "n")}
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE_%": float(np.nanmean(np.abs(err / denom)) * 100.0),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(y_true.size),
    }


def chronological_split(n: int, test_fraction: float, val_fraction: float):
    """
    Index slices for train/val/test, split by TIME ORDER only.

    This is the single most-scrutinised methodological point in time-series
    papers. Random splitting leaks the future into the training set and
    inflates every metric. Never shuffle.
    """
    n_test = int(round(n * test_fraction))
    n_trainval = n - n_test
    n_val = int(round(n_trainval * val_fraction))
    n_train = n_trainval - n_val
    return (
        slice(0, n_train),
        slice(n_train, n_train + n_val),
        slice(n_train + n_val, n),
    )


def rack_row(rack_id: int, rack_rows: dict) -> int:
    for row, rng in rack_rows.items():
        if rack_id in rng:
            return row
    return -1


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
