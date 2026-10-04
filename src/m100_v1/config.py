"""
CS54 — AI-Based Cooling Optimization in Data Centers
Configuration for the M100 ExaData pipeline.

Siddharth Sahoo · RA2512005010041 · Guide: Dr. P. Akilandeswari

Everything tunable lives here. Edit this file, not the scripts.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
# Root of the extracted M100 data. Must contain a "year_month=YY-MM" folder.
# Override without editing this file:  CS54_RAW_ROOT=/some/path python 02_extract.py
RAW_ROOT = Path(os.environ.get(
    "CS54_RAW_ROOT", Path.home() / "research/cs54-cooling/data/m100_raw"))

# Where processed outputs go.
PROJECT_ROOT = Path(os.environ.get(
    "CS54_PROJECT_ROOT", Path.home() / "research/cs54-cooling"))
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Which month to process.
MONTH = os.environ.get("CS54_MONTH", "20-06")

# ---------------------------------------------------------------- metrics
# Metric name -> plugin. Only these are read. Keep the list tight:
# every extra metric costs a full pass over that metric's Parquet files.
METRICS = {
    # ---- ipmi_pub: node hardware sensors, ~20 s per node
    "total_power":   "ipmi_pub",   # node total power draw (W)  <- HEAT GENERATION
    "ambient":       "ipmi_pub",   # node inlet / ambient air temperature (C)
    "p0_power":      "ipmi_pub",   # CPU socket 0 power (W)
    "p1_power":      "ipmi_pub",   # CPU socket 1 power (W)
    "fan0_0":        "ipmi_pub",   # node fan speed (RPM) <- node-level cooling actuator
    "fan0_1":        "ipmi_pub",
    "p0_core0_temp": "ipmi_pub",   # a representative core temperature
    "p1_core0_temp": "ipmi_pub",
    # ---- ganglia_pub: OS-level workload
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

# Metrics that are cluster-wide (no node column) rather than per-node.
# Handled separately: broadcast to every rack.
CLUSTER_METRICS = {
    "cluster_cpu_util":    "slurm_pub",
    "cluster_memory_util": "slurm_pub",
}

# ---------------------------------------------------------------- resampling
RESAMPLE = "5min"          # target time grid
AGG_LEVEL = "rack"         # "rack" (49 zones) or "node" (980 — heavy)
NODES_PER_RACK = 20        # confirmed: rack = node // 20

# Rack rows in the machine room (confirmed from ExaData documentation).
# Within a row, X decreases monotonically with rack ID, so racks whose IDs
# differ by 1 within the same row are physically adjacent.
RACK_ROWS = {0: range(0, 18), 1: range(18, 33), 2: range(33, 49)}

# Optional: limit to a subset of racks while developing. None = all 49.
RACK_SUBSET = None         # e.g. list(range(0, 10))

# ---------------------------------------------------------------- PS1
TARGET = "total_power"     # what we forecast: rack heat generation (W)
SECOND_TARGET = "ambient"  # also forecast rack inlet temperature (C)

HORIZON_STEPS = 12         # 12 x 5min = 1 hour ahead
LOOKBACK_STEPS = 24        # 24 x 5min = 2 hours of history
LAGS = [1, 2, 3, 6, 12, 24]
ROLL_WINDOWS = [6, 12, 24]

TEST_FRACTION = 0.20       # last 20% of time -> test. CHRONOLOGICAL, never shuffled.
VAL_FRACTION = 0.15        # of the training portion, the last 15% -> validation

# Compact signal set fed to the sequence models (LSTM/GRU). The recurrent layer
# derives its own history, so no lag columns are needed here — and keeping this
# list short is what stops the 3-D tensor from exhausting 8 GB of RAM.
SEQ_FEATURES = [
    "total_power", "ambient", "cpu_busy", "load_one",
    "cpu_power", "fan_speed", "core_temp", "mem_free",
    "tod_sin", "tod_cos",
]

SEED = 42
EPOCHS = 40
BATCH_SIZE = 256
PATIENCE = 6

# ---------------------------------------------------------------- PS2
# Thermal safety envelope (ASHRAE TC 9.9)
ASHRAE_RECOMMENDED = (18.0, 27.0)
ASHRAE_ALLOWABLE_A1 = (15.0, 32.0)

ROLLOUT_HORIZONS = [1, 3, 6, 12]   # steps ahead to test the thermal model

for _d in (PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
