#!/usr/bin/env python3
"""
Synthetic M100-shaped Parquet tree for TESTING THE PIPELINE ONLY. v2.

Never appears in results. Its purpose is to prove the scripts run end to end
before they touch the real 5 GB extract. It reproduces the real layout

    year_month=YY-MM/plugin=<p>/metric=<m>/a_0.parquet
    columns: timestamp (epoch ms, UTC), value, node (string)

and deliberately reproduces the three awkward features of the real 20-06 data
that v2 had to handle:

  * Ganglia sampled far more sparsely than IPMI, with gaps  -> exercises the
    bounded forward-fill instead of the v1 fillna(0) bug
  * a handful of physically impossible inlet temperatures   -> exercises the
    plausibility filter
  * Ganglia node IDs past 979                               -> exercises the
    rack 0..48 restriction

It also embeds genuine first-order thermal dynamics with inter-rack coupling and
real thermal inertia, so the models have something true to find and the spatial
ablation is meaningful.

    python make_fixture.py /tmp/fake_m100
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RACKS, NODES_PER_RACK = 8, 20
DAYS, PERIOD_S = 8, 120
START = pd.Timestamp("2020-06-01 00:00:00")
RNG = np.random.default_rng(7)


def simulate():
    n_nodes = RACKS * NODES_PER_RACK
    n_t = DAYS * 24 * 3600 // PERIOD_S
    t = START + pd.to_timedelta(np.arange(n_t) * PERIOD_S, unit="s")
    hod = t.hour + t.minute / 60.0

    base = 45 + 28 * np.sin((hod - 9) / 24 * 2 * np.pi)
    base = base * np.where(t.dayofweek >= 5, 0.72, 1.0)
    bursts = np.zeros(n_t)
    for _ in range(60):
        s = RNG.integers(0, n_t - 60)
        bursts[s:s + RNG.integers(10, 80)] += RNG.uniform(8, 28)

    cpu_busy = np.empty((n_t, n_nodes), "float32")
    for j in range(n_nodes):
        r = j // NODES_PER_RACK
        v = base + bursts * RNG.uniform(.5, 1.4) + 6 * np.sin(r + hod / 3)
        cpu_busy[:, j] = np.clip(v + RNG.normal(0, 4, n_t), 1, 99)

    total_power = (110 + 3.1 * cpu_busy + .006 * cpu_busy ** 2
                   + RNG.normal(0, 7, (n_t, n_nodes))).astype("float32")
    p0 = (total_power * .34 + RNG.normal(0, 3, total_power.shape)).astype("float32")
    p1 = (total_power * .32 + RNG.normal(0, 3, total_power.shape)).astype("float32")

    ambient = np.empty_like(total_power)
    fan = np.empty_like(total_power)
    core = np.empty_like(total_power)
    amb = np.full(n_nodes, 22., "float32")
    fsp = np.full(n_nodes, 6000., "float32")
    rack_of = np.arange(n_nodes) // NODES_PER_RACK
    rack_power = np.stack([total_power[:, rack_of == r].mean(1)
                           for r in range(RACKS)], axis=1)

    alpha = .05                      # thermal inertia
    for i in range(n_t):
        nb = np.zeros(n_nodes, "float32")
        for r in range(RACKS):
            left = rack_power[i, r - 1] if r > 0 else rack_power[i, r]
            right = rack_power[i, r + 1] if r < RACKS - 1 else rack_power[i, r]
            nb[rack_of == r] = .5 * (left + right)
        equil = 17. + .018 * total_power[i] + .010 * nb - .00035 * fsp
        amb = amb + alpha * (equil - amb) + RNG.normal(0, .04, n_nodes).astype("float32")
        ambient[i] = amb
        c = amb + .16 * total_power[i] + RNG.normal(0, .6, n_nodes).astype("float32")
        core[i] = c
        fsp = np.clip(3000 + 62 * (c - 45), 2800, 13000).astype("float32")
        fan[i] = fsp

    # a few sensor faults, as in the real 20-06
    for _ in range(40):
        ambient[RNG.integers(0, n_t), RNG.integers(0, n_nodes)] = RNG.uniform(43, 46)

    return t, {
        "total_power": total_power, "ambient": ambient,
        "p0_power": p0, "p1_power": p1,
        "fan0_0": fan, "fan0_1": fan * RNG.uniform(.97, 1.03, fan.shape),
        "p0_core0_temp": core, "p1_core0_temp": core + 1.2,
        "cpu_user": cpu_busy * .72, "cpu_system": cpu_busy * .2,
        "cpu_idle": 100 - cpu_busy, "cpu_wio": cpu_busy * .05,
        "load_one": cpu_busy * .4, "load_five": cpu_busy * .38,
        "mem_free": 2.9e8 - 6e5 * cpu_busy,
        "bytes_in": 1e6 * cpu_busy, "bytes_out": 9e5 * cpu_busy,
        "proc_run": np.maximum(1, cpu_busy / 9),
    }


IPMI = {"total_power", "ambient", "p0_power", "p1_power", "fan0_0", "fan0_1",
        "p0_core0_temp", "p1_core0_temp"}


def main(root: Path) -> int:
    if root.exists():
        shutil.rmtree(root)
    month = f"{START.year % 100:02d}-{START.month:02d}"
    t, series = simulate()
    n_t, n_nodes = series["ambient"].shape
    epoch_ms = ((t - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")).to_numpy()
    nodes = np.arange(n_nodes)

    for metric, arr in series.items():
        plugin = "ipmi_pub" if metric in IPMI else "ganglia_pub"
        ts = np.repeat(epoch_ms, n_nodes)
        nd = np.tile(nodes, n_t)
        val = arr.reshape(-1).astype("float32")

        if plugin == "ganglia_pub":
            # sparse and gappy, like the real thing (~54% bin coverage)
            keep = RNG.random(ts.size) < 0.22
            gap_start = n_t // 3
            in_gap = np.repeat(
                (np.arange(n_t) >= gap_start) & (np.arange(n_t) < gap_start + 600),
                n_nodes)
            keep &= ~in_gap
            ts, nd, val = ts[keep], nd[keep], val[keep]
            # a few readings from service nodes beyond the compute range
            extra = max(1, ts.size // 200)
            ts = np.concatenate([ts, ts[:extra]])
            nd = np.concatenate([nd, RNG.integers(980, 1000, extra)])
            val = np.concatenate([val, val[:extra]])

        d = root / f"year_month={month}" / f"plugin={plugin}" / f"metric={metric}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": pd.to_datetime(ts, unit="ms", utc=True),
                      "value": val,
                      "node": nd.astype(str)}).to_parquet(d / "a_0.parquet", index=False)

    for name, vals in (("cluster_cpu_util", series["cpu_user"].mean(1)),
                       ("cluster_memory_util", series["cpu_idle"].mean(1))):
        d = root / f"year_month={month}" / "plugin=slurm_pub" / f"metric={name}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": pd.to_datetime(epoch_ms, unit="ms", utc=True),
                      "value": vals.astype("float32")}).to_parquet(
            d / "a_0.parquet", index=False)

    size = sum(f.stat().st_size for f in root.rglob("*.parquet")) / 1e6
    print(f"fixture at {root}")
    print(f"  month={month} racks={RACKS} nodes={n_nodes} steps={n_t:,} "
          f"size={size:,.1f} MB")
    print("  ganglia deliberately sparse + one long gap + out-of-range node IDs")
    print("  ambient contains 40 injected sensor faults (43-46 degC)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake_m100")))
