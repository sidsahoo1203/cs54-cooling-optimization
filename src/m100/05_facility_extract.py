#!/usr/bin/env python3
"""
CS54 PS3 Step 1 — extract the 2022 facility data and decide whether PS3 is possible.

This script does two jobs, and the second one matters more than the first.

JOB 1: build the facility table — CRAC setpoint and supply/return temperatures,
cooling and IT power, measured PUE, outdoor weather — on the same 5-minute grid
as the node data, so the thermal twin can be learned from it.

JOB 2: THE GO / NO-GO TEST. PS3 trains an agent to set the CRAC supply-air
temperature. That is only learnable if the recorded setpoint actually MOVED
during the month. If CINECA's operators held it fixed, no amount of modelling
recovers what changing it would have done, and the agent ends up with an action
that does nothing — exactly what happened to node fan speed in PS2, where the
model gave the "actuator" a gain importance of rank 55 out of 56.

So this script measures the setpoint's variation first and prints a verdict.
Run it before writing a single line of RL code.

    python 05_facility_extract.py
"""
from __future__ import annotations

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

try:
    from common import banner, metric_dir, stream_metric_to_zones, to_datetime_series
except ImportError:
    print("ERROR: common.py not found. Put the PS3 scripts next to the v2 pipeline,")
    print("or copy common.py into this folder.")
    raise


def pull(plugin: str, metrics: list[str], scale: dict | None = None) -> pd.DataFrame:
    """Read a facility plugin's metrics onto the 5-minute grid."""
    out = None
    for m in metrics:
        path = metric_dir(P.FACILITY_RAW, P.MONTH, plugin, m)
        if path is None:
            print(f"    {m:38s} ABSENT")
            continue
        try:
            df = stream_metric_to_zones(path, m, resample=P.RESAMPLE,
                                        agg_level="cluster", verbose=False)
        except Exception as exc:
            print(f"    {m:38s} FAILED: {exc}")
            continue
        if df.empty:
            print(f"    {m:38s} empty")
            continue
        # Facility devices (CRAC units, panels) are averaged into one room-level
        # signal. Per-unit control is a natural extension, not needed for v1 of (c).
        s = df.groupby("time", as_index=False)[m].mean()
        if scale and m in scale:
            s[m] = s[m] / scale[m]
        resolved = f"{path.parent.name}/{path.name}"
        note = "" if resolved == f"plugin={plugin}/metric={m}" else f"  <- {resolved}"
        print(f"    {m:38s} {len(s):>7,} bins  "
              f"[{s[m].min():.2f} .. {s[m].max():.2f}]  "
              f"mean {s[m].mean():.2f}  std {s[m].std():.3f}{note}")
        out = s if out is None else out.merge(s, on="time", how="outer")
    return out if out is not None else pd.DataFrame(columns=["time"])


def scan_actions(df: pd.DataFrame) -> str | None:
    """
    Rank every candidate control variable by whether it is actually EXERCISED.

    PS3 trains an agent to move one variable. That is only learnable if the
    variable moved in the recorded data. 22-01 showed the CRAC supply-air
    setpoint pinned at 16.00 degC for the whole month — so no model can recover
    what changing it would have done, and the agent's action would be inert.
    Rather than fail on that one variable, this scans all of them and names the
    best lever the facility actually used.
    """
    banner("ACTION SCAN — which control variable is actually exercised?")
    rows = []
    for col in P.ACTION_CANDIDATES:
        if col not in df.columns or df[col].notna().sum() < 100:
            rows.append({"action": col, "status": "absent"})
            continue
        v = df[col].dropna()
        p5, p95 = np.percentile(v, [5, 95])
        span, sd = float(p95 - p5), float(v.std())
        changes = int((v.diff().abs() > 1e-6).sum())
        mid = float(np.median(v))
        cv = sd / abs(mid) * 100 if abs(mid) > 1e-9 else 0.0
        rows.append({"action": col, "status": "ok", "min": float(v.min()),
                     "max": float(v.max()), "std": sd, "span_5_95": span,
                     "cv_%": cv, "changes": changes,
                     "usable": span > 1e-6 and cv >= 1.0 and changes >= P.MIN_SETPOINT_CHANGES})

    scan = pd.DataFrame(rows)
    show = scan[scan.status == "ok"].copy()
    if not show.empty:
        show = show.sort_values("cv_%", ascending=False)
        print(f"  {'action':<22}{'range':>20}{'std':>9}{'5-95 span':>11}"
              f"{'cv %':>8}{'changes':>9}  usable")
        for _, r in show.iterrows():
            rng = f"{r['min']:.2f}-{r['max']:.2f}"
            print(f"  {r['action']:<22}{rng:>20}{r['std']:>9.3f}"
                  f"{r['span_5_95']:>11.3f}{r['cv_%']:>8.2f}{r['changes']:>9,}"
                  f"  {'YES' if r['usable'] else 'no'}")
    for _, r in scan[scan.status == "absent"].iterrows():
        print(f"  {r['action']:<22}{'absent from this month':>20}")

    scan.to_csv(P.RESULTS_DIR / f"ps3_action_scan_{P.MONTH}.csv", index=False)
    good = show[show.usable] if not show.empty else show

    print()
    chosen = P.ACTION_VAR
    if good.empty:
        print("  NO USABLE ACTION in this month. Every candidate is either held")
        print("  constant or absent. PS3 cannot be validated from this record as a")
        print("  data-driven control problem — see PS3_README.md for the hybrid")
        print("  physics twin, which imposes the response instead of learning it.")
        return None

    best = good.iloc[0]["action"]
    print(f"  BEST LEVER: '{best}'  "
          f"(cv {good.iloc[0]['cv_%']:.1f}%, {int(good.iloc[0]['changes']):,} changes)")
    if chosen != best:
        print(f"  config_ps3.ACTION_VAR is currently '{chosen}'.")
        print(f"  To use the scan's choice:  CS54_ACTION={best} python 06_build_twin.py")
    if chosen in set(good["action"]):
        print(f"  '{chosen}' is usable, so 06_build_twin.py can proceed as configured.")
    else:
        print(f"  '{chosen}' is NOT usable. Change it before building the twin, or")
        print(f"  the agent will have an action with no effect.")
    print()
    print("  Say this in the paper: the control variable was selected on measured")
    print("  variation in the operational record, not assumed. At this facility in")
    print("  winter the supply-air setpoint is a fixed policy value and free")
    print("  cooling does the regulating, so the valve is the real lever.")
    return best


def main() -> int:
    banner(f"CS54 PS3 — facility extraction · month {P.MONTH}")
    print(f"  FACILITY_RAW = {P.FACILITY_RAW}")
    if not P.FACILITY_RAW.exists():
        print("\n  ERROR: that directory does not exist. Download first — see PS3_README.md")
        return 1

    print("\nvertiv (CRAC units):")
    vert = pull("vertiv", P.VERTIV_METRICS)
    print("\nschneider (chilled-water plant, values scaled /10):")
    schn = pull("schneider", P.SCHNEIDER_METRICS, P.SCHNEIDER_SCALE)
    print("\nlogics (facility power metering):")
    logi = pull("logics", P.LOGICS_METRICS)
    print("\nweather:")
    weat = pull("weather", P.WEATHER_METRICS)

    frames = [f for f in (vert, schn, logi, weat) if not f.empty]
    if not frames:
        print("\n  ERROR: no facility data found. Check the download and the month.")
        return 1

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on="time", how="outer")
    df = df.sort_values("time").reset_index(drop=True)

    # -------------------------------------------------- canonical names
    ren = {
        "Supply_Air_Temperature_Set_Point": "setpoint",
        "Supply_Air_Temperature": "supply_temp",
        "Return_Air_Temperature": "return_temp",
        "Fan_Speed": "crac_fan",
        "Compressor_Utilization": "compressor",
        "Tot_ict": "it_power",
        "Tot": "total_power",
        "Pue": "pue",
        "temp": "outdoor_temp",
        "humidity": "outdoor_humidity",
        "Free_Cooling_Valve_Open_Position": "free_cooling_valve",
        "Free_Cooling_Status": "free_cooling_status",
        "Free_Cooling_Fluid_Temperature": "free_cooling_fluid_temp",
        "Portata_attiva": "water_flow",
        "Temp_mandata": "water_supply_temp",
        "Temp_ritorno": "water_return_temp",
    }
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})

    cool_parts = [c for c in ("Tot_cdz", "Tot_chiller", "Tot_qpompe") if c in df.columns]
    if cool_parts:
        df["cooling_power"] = df[cool_parts].sum(axis=1, min_count=1)
        print(f"\n  cooling_power = {' + '.join(cool_parts)}")

    t = pd.to_datetime(df["time"])
    mod = t.dt.hour * 60 + t.dt.minute
    df["tod_sin"] = np.sin(2 * np.pi * mod / 1440.0)
    df["tod_cos"] = np.cos(2 * np.pi * mod / 1440.0)

    # -------------------------------------------------- node-side thermal state
    banner("Node-side rack inlet temperatures for the same month")
    npath = metric_dir(P.NODE_RAW, P.MONTH, "ipmi_pub", "ambient")
    if npath is None:
        print(f"  ipmi_pub/ambient not found for {P.MONTH} under {P.NODE_RAW}.")
        print("  The twin needs the rack inlet temperature it is controlling.")
        print("  Re-run the download WITHOUT the --wildcards filter, or add")
        print("  '*plugin=ipmi_pub*' to it, so ambient comes down too.")
        print("  Continuing with facility-only columns; 06_build_twin.py will")
        print("  fall back to the CRAC return-air temperature as the state.")
    else:
        amb = stream_metric_to_zones(npath, "ambient", resample=P.RESAMPLE,
                                     nodes_per_rack=20, agg_level="rack",
                                     n_racks=49, want_max=True, verbose=True)
        if not amb.empty:
            room = amb.groupby("time").agg(
                rack_inlet=("ambient", "mean"),
                rack_inlet_max=("ambient_max", "max")).reset_index()
            # 95th percentile across racks: "95% of racks are below this". The
            # room-wide maximum is one node and sits far above the mean, so
            # using it as the compliance point makes every step a violation.
            p95 = (amb.groupby("time")["ambient_max"]
                   .quantile(0.95).reset_index(name="rack_inlet_p95"))
            room = room.merge(p95, on="time", how="left")
            df = df.merge(room, on="time", how="left")
            print(f"  rack_inlet      mean {df['rack_inlet'].mean():.2f} degC")
            if "rack_inlet_p95" in df.columns:
                print(f"  rack_inlet_p95  mean {df['rack_inlet_p95'].mean():.2f} degC "
                      f"(offset {(df['rack_inlet_p95'] - df['rack_inlet']).median():+.2f})")
            print(f"  rack_inlet_max  max  {df['rack_inlet_max'].max():.2f} degC "
                  f"(offset {(df['rack_inlet_max'] - df['rack_inlet']).median():+.2f} "
                  f"— one node, not a compliance point)")

    if "rack_inlet" not in df.columns and "return_temp" in df.columns:
        df["rack_inlet"] = df["return_temp"]
        df["rack_inlet_max"] = df["return_temp"]
        print("  FALLBACK: using CRAC return-air temperature as the thermal state.")

    # -------------------------------------------------- plausibility + gaps
    banner("Plausibility filter and gap handling")
    removed = {}
    for col, (lo, hi) in P.PLAUSIBLE.items():
        if col not in df.columns:
            continue
        bad = (df[col] < lo) | (df[col] > hi)
        n = int(bad.sum())
        if n:
            df.loc[bad, col] = np.nan
            removed[col] = n
            print(f"  {col:<18}{n:>7,} readings outside [{lo}, {hi}] -> NaN")
    if not removed:
        print("  no implausible readings")
    else:
        print("\n  These are metering dropouts, not measurements. Tot_ict reached")
        print("  0.00 kW on a running machine, which is what produced PUE = 167.")

    wcols = [c for c in ("outdoor_temp", "outdoor_humidity", "dew_point",
                         "pressure", "wind_speed") if c in df.columns]
    if wcols:
        before = float(df[wcols[0]].isna().mean() * 100)
        df[wcols] = df[wcols].ffill(limit=P.WEATHER_FFILL_BINS)
        after = float(df[wcols[0]].isna().mean() * 100)
        print(f"\n  weather sampled every 10 min against a 5-min grid: "
              f"{before:.1f}% empty -> {after:.1f}% after a "
              f"{P.WEATHER_FFILL_BINS * P.STEP_MINUTES}-min carry-forward")

    # -------------------------------------------------- report
    banner("Facility table")
    print(f"  shape     {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  span      {df['time'].min()}  ..  {df['time'].max()}")
    print("\n  column coverage:")
    for c, pct in (df.isna().mean() * 100).sort_values().items():
        u = P.UNITS.get(c, ("", ""))[0]
        print(f"      {c:<26}{100 - pct:6.2f} % present  {u}")

    if "pue" in df.columns and df["pue"].notna().any():
        p = df["pue"].dropna()
        print(f"\n  MEASURED PUE: mean {p.mean():.4f}  "
              f"min {p.min():.4f}  max {p.max():.4f}")
        print("  This is the facility's own number, not an estimate. It is the")
        print("  headline metric of the whole case study.")
    elif {"total_power", "it_power"} <= set(df.columns):
        df["pue_computed"] = df["total_power"] / df["it_power"].replace(0, np.nan)
        p = df["pue_computed"].dropna()
        print(f"\n  COMPUTED PUE (Tot / Tot_ict): mean {p.mean():.4f}")

    out = P.PROCESSED_DIR / f"facility_{P.RESAMPLE}_{P.MONTH}.parquet"
    df.to_parquet(out, index=False)
    print(f"\n  wrote {out}  ({out.stat().st_size / 1e6:,.1f} MB)")

    # -------------------------------------------------- the decision
    best = scan_actions(df)
    ok = best is not None

    # -------------------------------------------------- figures
    if "setpoint" in df.columns and df["setpoint"].notna().any():
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
        axes[0].plot(df["time"], df["setpoint"], lw=1, color="#b5442e",
                     label="setpoint (action)")
        if "supply_temp" in df:
            axes[0].plot(df["time"], df["supply_temp"], lw=.9, color="#3b6ea5",
                         alpha=.8, label="supply temp (realised)")
        axes[0].set_ylabel("degC"); axes[0].legend(fontsize=8, frameon=False)
        axes[0].set_title("CS54 (c) — is the control variable exercised?")
        if "rack_inlet_max" in df:
            axes[1].plot(df["time"], df["rack_inlet_max"], lw=.9, color="#111")
            axes[1].axhspan(*P.ASHRAE_RECOMMENDED, color="#2e7d32", alpha=.08)
            axes[1].set_ylabel("rack inlet max\n(degC)")
        if "cooling_power" in df:
            axes[2].plot(df["time"], df["cooling_power"], lw=.9, color="#6a3d9a")
            axes[2].set_ylabel("cooling power\n(kW)")
        for a in axes:
            a.grid(alpha=.25, lw=.5)
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(P.FIGURES_DIR / "ps3_facility_overview.png", dpi=150)
        plt.close(fig)
        print(f"  wrote {P.FIGURES_DIR / 'ps3_facility_overview.png'}")

    banner("NEXT")
    if ok:
        print(f"      CS54_ACTION={best} python 06_build_twin.py")
    else:
        print("      No usable action in this month. Do NOT train an agent on it.")
        print("      See PS3_README.md for the hybrid physics twin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
