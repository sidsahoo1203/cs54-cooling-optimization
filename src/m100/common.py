"""
Shared helpers for the CS54 M100 pipeline. VERSION 2.

The M100 ExaData Parquet files are "long" tables: one row per
(timestamp, entity, value). Column names are detected rather than assumed —
that is the difference between a script that runs first try and one that throws
KeyError on someone else's machine.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

warnings.filterwarnings("ignore", category=FutureWarning)

TIME_CANDIDATES = ["timestamp", "time", "ts", "_timestamp", "datetime", "date"]
VALUE_CANDIDATES = ["value", "val", "v", "reading"]
ENTITY_CANDIDATES = ["node", "device", "panel", "host", "hostname", "nodeid"]


# --------------------------------------------------------------- paths
def metric_dir(raw_root: Path, month: str, plugin: str, metric: str) -> Path | None:
    """
    Locate a metric directory, tolerating the naming variation that actually
    occurs in M100 ExaData.

    Three real differences this has to absorb:
      * plugins are named `vertiv_pub`, `logics_pub`, `schneider_pub` on disk,
        while the paper and the docs call them `vertiv`, `logics`, `schneider`
      * Schneider metrics carry a PLC prefix: the documented `Temp_mandata` is
        stored as `PLC_PLC_Q101.Temp_mandata`
      * layouts differ between Hive (`year_month=22-01`) and bare (`22-01`)

    The match is restricted to the requested month: without that, a glob happily
    returns another month's directory, which is how slurm data from 20-06 once
    surfaced in a 22-01 inspection.
    """
    plugin_names = list(dict.fromkeys([
        plugin, f"{plugin}_pub",
        plugin[:-4] if plugin.endswith("_pub") else plugin]))

    for pl in plugin_names:
        for base in (raw_root / f"year_month={month}", raw_root / month):
            for d in (base / f"plugin={pl}" / f"metric={metric}",
                      base / pl / metric):
                if d.is_dir():
                    return d

    def in_month(p: Path) -> bool:
        return any(part in (month, f"year_month={month}") for part in p.parts)

    # Exact metric, any plugin spelling, inside the month.
    for pl in plugin_names:
        hits = [h for h in raw_root.glob(f"**/plugin={pl}/metric={metric}")
                if in_month(h)]
        if hits:
            return hits[0]

    # Suffix match, for prefixed names like PLC_PLC_Q101.Temp_mandata.
    for pl in plugin_names:
        pdirs = [d for d in raw_root.glob(f"**/plugin={pl}") if in_month(d)]
        for pd in pdirs:
            cands = [d for d in pd.iterdir()
                     if d.is_dir() and d.name.startswith("metric=")
                     and d.name.split("=", 1)[1].split(".")[-1] == metric]
            if cands:
                return sorted(cands)[0]
    return None


def list_available_metrics(raw_root: Path, month: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for plugin_dir in sorted(raw_root.glob("**/plugin=*")):
        if not plugin_dir.is_dir():
            continue
        if month and month not in str(plugin_dir):
            continue
        plugin = plugin_dir.name.split("=", 1)[1]
        out[plugin] = sorted(
            d.name.split("=", 1)[1]
            for d in plugin_dir.iterdir()
            if d.is_dir() and d.name.startswith("metric=")
        )
    return out


# --------------------------------------------------------------- schema
def detect_columns(names: list[str]):
    lower = {n.lower(): n for n in names}

    def pick(cands):
        for c in cands:
            if c in lower:
                return lower[c]
        return None

    time_col = pick(TIME_CANDIDATES)
    value_col = pick(VALUE_CANDIDATES)
    entity_col = pick(ENTITY_CANDIDATES)
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
    """Datetime64, inferring the epoch unit if the column is integer."""
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s
    elif pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        med = float(np.nanmedian(v.to_numpy(dtype="float64", na_value=np.nan)))
        unit = "ns" if med > 1e17 else "us" if med > 1e14 else "ms" if med > 1e11 else "s"
        out = pd.to_datetime(v, unit=unit, errors="coerce", utc=True)
    else:
        out = pd.to_datetime(s, errors="coerce", utc=True)
    # Normalise to tz-naive UTC so merges never fail on mismatched tz-awareness.
    if getattr(out.dtype, "tz", None) is not None:
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out


# --------------------------------------------------------------- streaming
def stream_metric_to_zones(
    path: Path,
    metric: str,
    resample: str = "5min",
    nodes_per_rack: int = 20,
    agg_level: str = "rack",
    rack_subset=None,
    n_racks: int | None = None,
    batch_rows: int = 500_000,
    compact_every: int = 24,
    want_max: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Read one metric's Parquet directory in bounded memory; return a tidy frame
    [time, zone, <metric>] (plus <metric>_max when want_max).

    Aggregation is exact: partial sums and counts accumulate, then divide once.
    Averaging pre-averaged batches would be wrong wherever batches straddle a
    time bin unevenly.
    """
    ds = pads.dataset(str(path), format="parquet")
    names = list(ds.schema.names)
    time_col, value_col, entity_col = detect_columns(names)
    if time_col is None or value_col is None:
        raise ValueError(f"{metric}: no time/value column in {names}")

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
            _sum=("_sum", "sum"), _cnt=("_cnt", "sum"), _max=("_max", "max"))

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

        if agg_level == "cluster":
            # Collapse every entity into one room-level signal. Facility plugins
            # carry a STRING entity ("CDZ1", "Q101"); coercing that to a number
            # yields NaN and the groupby would then silently drop every row.
            df["zone"] = -1
        elif entity_col is not None and agg_level == "rack":
            ent = pd.to_numeric(df[entity_col], errors="coerce")
            if ent.notna().any():
                df["zone"] = (ent // nodes_per_rack).astype("Int64")
            else:
                df["zone"] = df[entity_col].astype("string")
        elif entity_col is not None:
            df["zone"] = pd.to_numeric(df[entity_col], errors="coerce").astype("Int64")
        else:
            df["zone"] = -1

        if pd.api.types.is_integer_dtype(pd.Series(df["zone"]).dtype):
            if n_racks is not None:
                # v2: Ganglia reports node IDs past 979 (login/service nodes),
                # which produced a phantom rack 49. M100 has racks 0..48.
                df = df[(df["zone"] >= 0) & (df["zone"] < n_racks)]
            if rack_subset is not None:
                df = df[df["zone"].isin(list(rack_subset))]
        if df.empty:
            continue

        partials.append(
            df.groupby(["time", "zone"], dropna=True)["_v"]
            .agg(_sum="sum", _cnt="count", _max="max").reset_index())
        if len(partials) >= compact_every:
            accum, partials = compact(partials, accum), []

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
        print(f"    {metric:16s} rows_read={n_rows:>12,}  bins={len(out):>9,}  "
              f"zones={out['zone'].nunique():>4}  "
              f"[{out['time'].min()} .. {out['time'].max()}]")
    return out.reset_index(drop=True)


# --------------------------------------------------------------- v2 helpers
def bounded_ffill(df: pd.DataFrame, cols: list[str], group: str,
                  limit: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Forward-fill `cols` within each `group`, at most `limit` consecutive rows.

    A sensor reading stays valid for a short while; it does not stay valid
    forever, and it is never a substitute for a measurement that was simply not
    taken. Gaps longer than `limit` remain NaN so dropna removes them honestly.

    Returns the frame and a boolean Series marking rows where any column was
    imputed, so the effect can be audited and reported.
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df, pd.Series(False, index=df.index)
    df = df.sort_values([group, "time"]).copy()
    before = df[cols].isna()
    df[cols] = df.groupby(group, sort=False)[cols].ffill(limit=limit)
    after = df[cols].isna()
    imputed = (before & ~after).any(axis=1)
    return df, imputed


def clip_implausible(df: pd.DataFrame, bounds: dict) -> tuple[pd.DataFrame, dict]:
    """Set physically impossible sensor readings to NaN. Returns counts removed."""
    removed = {}
    for col, (lo, hi) in bounds.items():
        for c in (col, f"{col}_max"):
            if c not in df.columns:
                continue
            bad = (df[c] < lo) | (df[c] > hi)
            n = int(bad.sum())
            if n:
                df.loc[bad, c] = np.nan
                removed[c] = n
    return df, removed


# --------------------------------------------------------------- metrics
def regression_metrics(y_true, y_pred, include_mape: bool = True) -> dict:
    """
    RMSE, MAE, R2 and optionally MAPE.

    include_mape=False for any target that passes through zero — a temperature
    CHANGE, for instance. MAPE divides by the true value, so near-zero truths
    make it explode and the number becomes meaningless rather than merely large.
    """
    y_true = np.asarray(y_true, dtype="float64").ravel()
    y_pred = np.asarray(y_pred, dtype="float64").ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    keys = ("RMSE", "MAE", "MAPE_%", "R2", "n") if include_mape else ("RMSE", "MAE", "R2", "n")
    if y_true.size == 0:
        return {k: (0 if k == "n" else float("nan")) for k in keys}
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    out = {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(y_true.size),
    }
    if include_mape:
        denom = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
        out["MAPE_%"] = float(np.nanmean(np.abs(err / denom)) * 100.0)
        out = {k: out[k] for k in ("RMSE", "MAE", "MAPE_%", "R2", "n")}
    return out


def chronological_split(n: int, test_fraction: float, val_fraction: float):
    """
    Index slices for train/val/test, split by TIME ORDER only.

    The most-scrutinised methodological point in any time-series paper. Random
    splitting leaks the future backwards and inflates every metric. Never shuffle.
    """
    n_test = int(round(n * test_fraction))
    n_trainval = n - n_test
    n_val = int(round(n_trainval * val_fraction))
    n_train = n_trainval - n_val
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def rack_row(rack_id: int, rack_rows: dict) -> int:
    for row, rng in rack_rows.items():
        if rack_id in rng:
            return row
    return -1


def unit_of(col: str, units: dict) -> str:
    base = col
    for suffix in ("_max", "_lag", "_rmean", "_rstd", "_diff"):
        if suffix in base:
            base = base.split(suffix)[0]
            break
    return units.get(base, ("", ""))[0]


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
