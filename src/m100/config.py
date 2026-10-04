"""
CS54 — AI-Based Cooling Optimization in Data Centers
Configuration for the M100 ExaData pipeline.  VERSION 2.

Siddharth Sahoo · RA2512005010041 · Guide: Dr. P. Akilandeswari

Everything tunable lives here. Edit this file, not the scripts.
See CHANGES.md for what v2 fixes and why.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
# Override without editing:  CS54_RAW_ROOT=/some/path python 02_extract.py
RAW_ROOT = Path(os.environ.get(
    "CS54_RAW_ROOT", Path.home() / "research/cs54-cooling/data/m100_raw"))
PROJECT_ROOT = Path(os.environ.get(
    "CS54_PROJECT_ROOT", Path.home() / "research/cs54-cooling"))

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

MONTH = os.environ.get("CS54_MONTH", "20-06")

# ---------------------------------------------------------------- metrics
METRICS = {
    # ---- ipmi_pub: node hardware sensors, ~20 s per node
    "total_power":   "ipmi_pub",   # node total power draw  <- HEAT GENERATION
    "ambient":       "ipmi_pub",   # node inlet / ambient air temperature
    "p0_power":      "ipmi_pub",   # CPU socket 0 power
    "p1_power":      "ipmi_pub",   # CPU socket 1 power
    "fan0_0":        "ipmi_pub",   # node fan speed  <- node-level actuator
    "fan0_1":        "ipmi_pub",
    "p0_core0_temp": "ipmi_pub",   # representative core temperature
    "p1_core0_temp": "ipmi_pub",
    # ---- ganglia_pub: OS-level workload (SPARSER — see IMPUTE_LIMIT below)
    "cpu_user":      "ganglia_pub",
    "cpu_system":    "ganglia_pub",
    "cpu_idle":      "ganglia_pub",
    "cpu_wio":       "ganglia_pub",
    "load_one":      "ganglia_pub",
    "load_five":     "ganglia_pub",
    "mem_free":      "ganglia_pub",
    "bytes_in":      "ganglia_pub",
    "bytes_out":     "ganglia_pub",
    "proc_run":      "ganglia_pub",
}

CLUSTER_METRICS = {
    "cluster_cpu_util":    "slurm_pub",
    "cluster_memory_util": "slurm_pub",
}

# ---------------------------------------------------------------- UNITS
# Verified against the ExaData plugin documentation and the observed ranges in
# 20-06. Quoted in every results file so no number is ever reported unitless.
# NOTE mem_free: Ganglia reports kilobytes, not bytes. Observed ~2.9e8 kB
# = ~287 GB per node, which is correct for an M100 node. Read as bytes it
# would be an impossible 287 MB.
UNITS = {
    "total_power":   ("W",    "node total power draw (= heat dissipated)"),
    "ambient":       ("degC", "node ambient/inlet air temperature (BMC sensor)"),
    "p0_power":      ("W",    "CPU socket 0 power"),
    "p1_power":      ("W",    "CPU socket 1 power"),
    "cpu_power":     ("W",    "p0_power + p1_power"),
    "fan0_0":        ("RPM",  "node fan 0 tachometer"),
    "fan0_1":        ("RPM",  "node fan 1 tachometer"),
    "fan_speed":     ("RPM",  "mean of fan0_0 and fan0_1"),
    "p0_core0_temp": ("degC", "CPU0 core 0 die temperature"),
    "p1_core0_temp": ("degC", "CPU1 core 0 die temperature"),
    "core_temp":     ("degC", "mean of p0_core0_temp and p1_core0_temp"),
    "cpu_user":      ("%",    "CPU time in user mode"),
    "cpu_system":    ("%",    "CPU time in kernel mode"),
    "cpu_idle":      ("%",    "CPU time idle"),
    "cpu_wio":       ("%",    "CPU time waiting on I/O"),
    "cpu_busy":      ("%",    "100 - cpu_idle (directly measured complement)"),
    "cpu_busy_alt":  ("%",    "cpu_user + cpu_system + cpu_wio (cross-check)"),
    "load_one":      ("procs", "1-minute load average"),
    "load_five":     ("procs", "5-minute load average"),
    "mem_free":      ("kB",   "free memory, KILOBYTES (Ganglia convention)"),
    "bytes_in":      ("B/s",  "network bytes received per second"),
    "bytes_out":     ("B/s",  "network bytes sent per second"),
    "proc_run":      ("count", "runnable processes"),
    "cluster_cpu_util":    ("%", "cluster-wide CPU utilisation (SLURM)"),
    "cluster_memory_util": ("%", "cluster-wide memory utilisation (SLURM)"),
}

# ---------------------------------------------------------------- resampling
RESAMPLE = "5min"
STEP_MINUTES = 5
AGG_LEVEL = "rack"
NODES_PER_RACK = 20
N_RACKS = 49                # valid rack IDs are 0..48. Anything else is dropped.

RACK_ROWS = {0: range(0, 18), 1: range(18, 33), 2: range(33, 49)}
RACK_SUBSET = None          # e.g. list(range(12)) while developing

# ---------------------------------------------------------------- v2: gaps
# Ganglia samples far more sparsely than IPMI: in 20-06 only ~54% of 5-minute
# bins carry a workload reading. v1 filled those with ZERO, which told every
# model the servers were idle when they simply had not been measured. v2
# forward-fills within a bounded window and leaves longer gaps as NaN, and
# flags every imputed row so the effect is auditable.
IMPUTE_LIMIT_BINS = 6       # 6 x 5min = 30 min. Beyond that, stays NaN.

# ---------------------------------------------------------------- v2: outliers
# Physically implausible readings are sensor faults, not data. 20-06 contains a
# rack inlet reading of 44.0 degC, which is not a real inlet temperature.
PLAUSIBLE = {
    "ambient":       (5.0, 40.0),     # degC
    "p0_core0_temp": (10.0, 110.0),   # degC
    "p1_core0_temp": (10.0, 110.0),
    "total_power":   (50.0, 3000.0),  # W
    "fan0_0":        (500.0, 20000.0),
    "fan0_1":        (500.0, 20000.0),
}

# ---------------------------------------------------------------- PS1
TARGET = "total_power"
SECOND_TARGET = "ambient"

# v2: multiple horizons, not just 60 min. Steps of 5 minutes.
HORIZONS = [1, 3, 6, 12]    # 5, 15, 30, 60 minutes
LOOKBACK_STEPS = 24         # 2 hours of history for the sequence models
LAGS = [1, 2, 3, 6, 12, 24]
ROLL_WINDOWS = [6, 12, 24]
SEASONAL_LAG = 288          # 24 h at 5-min steps -> the seasonal-naive baseline

TEST_FRACTION = 0.20
VAL_FRACTION = 0.15

SEED = 42
EPOCHS = 25
BATCH_SIZE = 512
PATIENCE = 5

SEQ_FEATURES = [
    "total_power", "ambient", "cpu_busy", "load_one",
    "cpu_power", "fan_speed", "core_temp", "mem_free",
    "tod_sin", "tod_cos",
]

# ---------------------------------------------------------------- PS2
# ASHRAE TC 9.9, "Thermal Guidelines for Data Processing Environments",
# 4th edition (2015) / 5th edition (2021). Class A1 equipment.
ASHRAE_EDITION = "ASHRAE TC 9.9 Thermal Guidelines, 4th ed. (2015); Class A1"
ASHRAE_RECOMMENDED = (18.0, 27.0)     # degC, dry-bulb at the equipment inlet
ASHRAE_ALLOWABLE_A1 = (15.0, 32.0)    # degC

# v2: the thermal model is fitted at several step sizes. A 5-minute change in
# rack-mean inlet temperature is close to sensor resolution, so a near-zero R2
# there may be noise rather than an unlearnable system. Longer steps have a
# far better signal-to-noise ratio, and inter-rack heat transport needs more
# than five minutes to show up at all.
PS2_HORIZONS = [1, 3, 6, 12]
ROLLOUT_HORIZONS = [1, 3, 6, 12]

# Lagged thermal history for the dynamics model. v1 used only contemporaneous
# values, which leaves a first-order system blind to its own rate of change.
PS2_LAGS = [1, 2, 3, 6, 12]
PS2_ROLL = [6, 12]

for _d in (PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
