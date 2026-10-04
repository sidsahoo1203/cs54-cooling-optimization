#!/usr/bin/env python3
"""
CS54 Step 1 — Inspect the extracted M100 data.

Reads nothing into memory beyond a few thousand rows. Produces:
  * the full plugin -> metric inventory actually present on disk
  * the Parquet schema of every metric we intend to use
  * a sample of rows, the detected time/value/entity columns, and the time span
  * a written data dictionary at results/data_dictionary_<MONTH>.md

Run this FIRST. Everything downstream depends on what it reports.

    python 01_inspect.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.dataset as pads

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from common import (
    banner,
    detect_columns,
    list_available_metrics,
    metric_dir,
    to_datetime_series,
)


def describe_metric(plugin: str, metric: str) -> dict | None:
    path = metric_dir(C.RAW_ROOT, C.MONTH, plugin, metric)
    if path is None:
        print(f"  MISSING  {plugin}/{metric}")
        return None

    ds = pads.dataset(str(path), format="parquet")
    names = list(ds.schema.names)
    tcol, vcol, ecol = detect_columns(names)
    files = list(path.glob("*.parquet"))
    size_mb = sum(f.stat().st_size for f in files) / 1e6

    # Read a small head only.
    head = next(ds.to_batches(batch_size=5000)).to_pandas()
    tspan = ("?", "?")
    if tcol:
        t = to_datetime_series(head[tcol])
        tspan = (str(t.min()), str(t.max()))

    n_entities = head[ecol].nunique() if ecol else 0
    vstats = {}
    if vcol:
        v = pd.to_numeric(head[vcol], errors="coerce")
        vstats = {
            "min": float(v.min()) if v.notna().any() else float("nan"),
            "max": float(v.max()) if v.notna().any() else float("nan"),
            "mean": float(v.mean()) if v.notna().any() else float("nan"),
            "null_%": float(v.isna().mean() * 100),
        }

    print(f"  {plugin}/{metric}")
    print(f"      files={len(files)}  size={size_mb:,.1f} MB  columns={names}")
    print(f"      detected: time={tcol!r} value={vcol!r} entity={ecol!r}")
    print(f"      head sample: {len(head):,} rows, entities={n_entities}, "
          f"time {tspan[0]} .. {tspan[1]}")
    if vstats:
        print(f"      value: min={vstats['min']:.4g} max={vstats['max']:.4g} "
              f"mean={vstats['mean']:.4g} null={vstats['null_%']:.2f}%")
    print(f"      dtypes: {dict(zip(names, [str(t) for t in ds.schema.types]))}")

    return {
        "plugin": plugin,
        "metric": metric,
        "n_files": len(files),
        "size_mb": round(size_mb, 1),
        "columns": ", ".join(names),
        "time_col": tcol,
        "value_col": vcol,
        "entity_col": ecol,
        "entities_in_sample": n_entities,
        "sample_time_min": tspan[0],
        "sample_time_max": tspan[1],
        **{f"val_{k}": round(v, 4) for k, v in vstats.items()},
    }


def main() -> int:
    banner(f"CS54 — M100 inspection · month {C.MONTH}")
    print(f"RAW_ROOT = {C.RAW_ROOT}")
    if not C.RAW_ROOT.exists():
        print(f"\nERROR: {C.RAW_ROOT} does not exist. Fix RAW_ROOT in config.py.")
        return 1

    banner("A. Plugin / metric inventory present on disk")
    inventory = list_available_metrics(C.RAW_ROOT, C.MONTH)
    if not inventory:
        print("No 'plugin=*' directories found. Is the month extracted?")
        return 1
    for plugin, metrics in inventory.items():
        print(f"\n  {plugin}  ({len(metrics)} metrics)")
        for i in range(0, len(metrics), 6):
            print("      " + "  ".join(f"{m:<22}" for m in metrics[i:i + 6]))

    banner("B. Requested metrics — schema and sanity")
    rows = []
    wanted = {**C.METRICS, **C.CLUSTER_METRICS}
    for metric, plugin in wanted.items():
        info = describe_metric(plugin, metric)
        if info:
            rows.append(info)
        print()

    banner("C. Missing metrics (rename or drop these in config.py)")
    missing = [
        f"{p}/{m}" for m, p in wanted.items()
        if metric_dir(C.RAW_ROOT, C.MONTH, p, m) is None
    ]
    print("  none" if not missing else "\n".join(f"  {x}" for x in missing))

    # ------------------------------------------------ write data dictionary
    out_md = C.RESULTS_DIR / f"data_dictionary_{C.MONTH}.md"
    with open(out_md, "w") as fh:
        fh.write(f"# CS54 — M100 data dictionary ({C.MONTH})\n\n")
        fh.write("Siddharth Sahoo · RA2512005010041\n\n")
        fh.write("## Inventory present on disk\n\n")
        for plugin, metrics in inventory.items():
            fh.write(f"### `{plugin}` — {len(metrics)} metrics\n\n")
            fh.write(", ".join(f"`{m}`" for m in metrics) + "\n\n")
        if rows:
            fh.write("## Metrics selected for this study\n\n")
            fh.write(pd.DataFrame(rows).to_markdown(index=False))
            fh.write("\n")
        if missing:
            fh.write("\n## Requested but absent\n\n")
            fh.write("\n".join(f"- `{x}`" for x in missing) + "\n")
    print(f"\nWrote {out_md}")

    if rows:
        csv = C.RESULTS_DIR / f"data_dictionary_{C.MONTH}.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"Wrote {csv}")

    banner("NEXT")
    print("  If section C lists missing metrics, correct their names in")
    print("  config.py using section A, then re-run. Otherwise:")
    print("      python 02_extract.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
