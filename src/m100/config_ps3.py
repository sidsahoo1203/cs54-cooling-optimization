"""
CS54 Sub-problem (c) — configuration.

Siddharth Sahoo · RA2512005010041

PS3 uses a DIFFERENT month from PS1/PS2: the 2020 records contain no facility
data at all. Everything here points at the 2022 extract.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(os.environ.get(
    "CS54_PROJECT_ROOT", Path.home() / "research/cs54-cooling"))
FACILITY_RAW = Path(os.environ.get(
    "CS54_FACILITY_RAW", PROJECT_ROOT / "data/m100_2022"))
NODE_RAW = Path(os.environ.get(
    "CS54_RAW_ROOT", PROJECT_ROOT / "data/m100_raw"))

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

MONTH = os.environ.get("CS54_PS3_MONTH", "22-01")

# ---------------------------------------------------------------- plugins
# Verified against gitlab.com/ecs-lab/exadata documentation/plugins/*.md
VERTIV_METRICS = [
    "Supply_Air_Temperature_Set_Point",   # <-- THE CONTROL ACTION
    "Supply_Air_Temperature",             # what the setpoint produced
    "Return_Air_Temperature",             # room thermal state
    "Actual_Return_Air_Temperature_Set_Point",
    "Fan_Speed",                          # CRAC fan, %
    "Compressor_Utilization",             # mechanical cooling load, %
    "Free_Cooling_Status",
    "Free_Cooling_Valve_Open_Position",
    "Free_Cooling_Fluid_Temperature",
    "Return_Humidity",
    "Ext_Air_Sensor_A_Temperature",
]
SCHNEIDER_METRICS = [
    "Temp_mandata",      # chilled-water supply temperature  (x10)
    "Temp_ritorno",      # chilled-water return temperature  (x10)
    "Set_temperatura",   # water temperature setpoint        (x10)
    "Portata_attiva",    # active flow rate                  (x10)
]
# Schneider stores integers scaled by 10 (x50 for Portata_1/2). Documented in
# the plugin reference. Applying the wrong factor silently gives temperatures
# ten times too large, which is the kind of error that survives into a paper.
SCHNEIDER_SCALE = {m: 10.0 for m in SCHNEIDER_METRICS}
SCHNEIDER_PANEL_KEEP = None   # None = keep all; set "Q101" to drop the redundant panel

LOGICS_METRICS = [
    "Tot_cdz",      # CRAC power            kW
    "Tot_chiller",  # chiller power         kW
    "Tot_qpompe",   # pump power            kW
    "Tot_ict",      # IT power              kW
    "Tot",          # total facility power  kW
    "Pue",          # facility-computed PUE
]
WEATHER_METRICS = ["temp", "humidity", "dew_point", "pressure", "wind_speed"]

UNITS = {
    "setpoint": ("degC", "CRAC supply-air temperature setpoint — THE ACTION"),
    "supply_temp": ("degC", "CRAC supply-air temperature (realised)"),
    "return_temp": ("degC", "CRAC return-air temperature"),
    "crac_fan": ("%", "CRAC fan speed"),
    "compressor": ("%", "compressor utilisation"),
    "cooling_power": ("kW", "Tot_cdz + Tot_chiller + Tot_qpompe"),
    "it_power": ("kW", "Tot_ict — IT equipment power"),
    "total_power": ("kW", "Tot — total facility power"),
    "pue": ("-", "Power Usage Effectiveness, facility-computed"),
    "outdoor_temp": ("degC", "outdoor dry-bulb temperature"),
    "free_cooling_valve": ("%", "free-cooling valve open position — THE ACTION"),
    "free_cooling_status": ("-", "free-cooling engagement"),
    "water_flow": ("m3/h", "active chilled-water flow rate"),
    "water_supply_temp": ("degC", "chilled-water supply temperature"),
    "water_return_temp": ("degC", "chilled-water return temperature"),
    "crac_fan": ("%", "CRAC fan speed"),
    "rack_inlet": ("degC", "mean node inlet temperature across racks"),
    "rack_inlet_max": ("degC", "hottest node inlet in the room — one node"),
    "rack_inlet_p95": ("degC", "95th-percentile rack inlet — the compliance point"),
}

# ---------------------------------------------------------------- plausibility
# Metering dropouts, not measurements. Tot_ict reached 0.00 kW on a machine that
# was plainly running, and dividing by it produced PUE = 167. Quoting a mean PUE
# without filtering these would put a wrong headline number in the paper.
PLAUSIBLE = {
    "it_power":      (50.0, 2000.0),    # kW
    "total_power":   (50.0, 3000.0),    # kW
    "cooling_power": (1.0, 2000.0),     # kW
    "pue":           (1.0, 3.0),        # dimensionless; < 1 is impossible
    "rack_inlet":    (5.0, 40.0),       # degC
    "rack_inlet_max": (5.0, 40.0),
    "rack_inlet_p95": (5.0, 40.0),
    "supply_temp":   (5.0, 35.0),
    "return_temp":   (5.0, 45.0),
}

# Weather is sampled every 10 min against a 5-min grid, so ~50% of bins are
# empty by construction. Outdoor conditions move slowly; carrying a reading for
# 30 min is physically reasonable and is flagged in the output.
WEATHER_FFILL_BINS = 6

# ---------------------------------------------------------------- grid
RESAMPLE = "5min"
STEP_MINUTES = 5

# Control interval, as a multiple of the 5-minute extraction grid.
#
# A valve or CRAC change takes 20-40 minutes to show in the room, so a 5-minute
# control step is finer than the physics it controls. Worse, the 5-minute change
# in rack-mean inlet temperature has std ~0.19 degC, close to sensor resolution:
# the model spends its capacity fitting noise. Aggregating to a coarser step
# averages away that noise (by about sqrt(TWIN_STEP)) AND targets a change large
# enough to sit above it. PS2 measured exactly this effect — R2 of the
# temperature change rose from 0.045 at 15 min to 0.085 at 60 min.
#
# Resampling to a coarser grid would ALSO divide the row count by the same
# factor, and too few rows was already the problem. So instead:
#
#   TWIN_SMOOTH  trailing rolling-mean window applied to the noisy signals.
#                Cuts measurement noise by about sqrt(window) while keeping the
#                5-minute stride, so no rows are lost. Trailing, never centred —
#                a centred window would leak the future into the features.
#   TWIN_STEP    how many 5-minute bins ahead the delta target looks.
#
# Default: smooth over 30 min, predict the change over the next 30 min, at a
# 5-minute stride. Row count is preserved.
TWIN_SMOOTH = int(os.environ.get("CS54_TWIN_SMOOTH", "6"))
TWIN_STEP = int(os.environ.get("CS54_TWIN_STEP", "1"))  # keep at 1: the rollout
# iterates one stride at a time, so a multi-stride target would be over-applied.

# ---------------------------------------------------------------- go/no-go
# PS3 only works if the recorded setpoint actually MOVES. If operators held it
# fixed, no model can learn what changing it does, and the agent will have an
# action that does nothing — exactly what happened with node fan speed in PS2.
# These are the thresholds 05_facility_extract.py tests against.
MIN_SETPOINT_STD = 0.15        # degC
MIN_SETPOINT_RANGE = 1.0       # degC between 5th and 95th percentile
MIN_SETPOINT_CHANGES = 50      # distinct change events in the month

# ---------------------------------------------------------------- twin
THERMAL_STATE = "rack_inlet"

# The control variable. 22-01 proved the CRAC supply-air SETPOINT is held at a
# constant 16.00 degC all month (std 0.002, 16 change events), so it carries no
# information and an agent set to move it would be pressing a disconnected
# button. 05_facility_extract.py now scans every facility variable, ranks them
# by usable variation, and names the best candidate; set this to what it says.
#   free_cooling_valve  valve position, 2.4-42.1% open, std 3.82 -> commanded AND varying
#   supply_temp         realised supply air, 13.3-23.3 degC, std 1.00 -> varies, not commanded
#   setpoint            DEAD in 22-01
ACTION_VAR = os.environ.get("CS54_ACTION", "free_cooling_valve")

# Candidates the scan ranks, with the direction physics requires for each:
#   thermal: +1 means raising it cannot LOWER the rack inlet temperature
#   power:   -1 means raising it cannot RAISE cooling power
# Opening the free-cooling valve admits more outside cooling, so it lowers both
# temperature and compressor power: thermal -1, power -1.
ACTION_CANDIDATES = {
    "setpoint":           (+1, -1),
    "supply_temp":        (+1, -1),
    "free_cooling_valve": (-1, -1),
    "crac_fan":           (-1, +1),   # more airflow cools, but costs fan power
    "water_flow":         (-1, +1),
}
EXOGENOUS = ["it_power", "outdoor_temp", "tod_sin", "tod_cos"]

TWIN_LAGS = [1, 2, 3, 6, 12]
TWIN_ROLL = [6, 12]

# Physics the learned model is NOT allowed to violate, imposed as XGBoost
# monotone constraints:
#   +1  raising the CRAC supply setpoint cannot LOWER the rack inlet temperature
#   -1  raising the setpoint cannot RAISE cooling power (less work to do)
# Observational data is confounded — operators move the setpoint in response to
# conditions — so without this the model can learn a physically backwards
# response and the controller will exploit it. This is the same "physics-guided"
# idea as papers 2, 5 and 32 of the literature review.
MONOTONE_THERMAL, MONOTONE_POWER = ACTION_CANDIDATES.get(ACTION_VAR, (+1, -1))

# Variables that lie BETWEEN the action and the thermal state on the causal
# chain. They must be excluded from the thermal model, because conditioning on a
# mediator blocks the path you are trying to measure: given the supply air
# temperature, the free-cooling valve explains nothing further about the rack
# inlet, and a monotonicity constraint on the valve then fights the data instead
# of encoding physics. In 22-01 that combination produced R2 = -9.07.
#
#   valve opens -> supply air colder -> rack inlet drops
#   (action)       (MEDIATOR)           (state)
MEDIATORS = {
    "free_cooling_valve": ["supply_temp", "water_supply_temp", "compressor",
                           "free_cooling_fluid_temp", "free_cooling_status"],
    "water_flow":         ["supply_temp", "water_supply_temp", "compressor"],
    "crac_fan":           ["supply_temp", "compressor"],
    "supply_temp":        [],   # the direct cause; nothing mediates it
    "setpoint":           ["supply_temp"],
}
THERMAL_MEDIATORS = MEDIATORS.get(ACTION_VAR, [])

# Fit the thermal model both with and without the monotonicity constraint and
# report both. The comparison IS a result: if the unconstrained fit is far
# better but gives a physically impossible action response, that is the
# confounding in the operational record, measured rather than asserted.
COMPARE_CONSTRAINED = True

TEST_FRACTION = 0.20
VAL_FRACTION = 0.15
SEED = 42

# ---------------------------------------------------------------- environment
EPISODE_STEPS = 288            # 24 h at 5-minute control intervals
SETPOINT_MIN = 16.0            # degC — overwritten from the data's observed range
SETPOINT_MAX = 24.0
SETPOINT_RATE_LIMIT = 0.5      # degC per control step; real CRACs cannot jump

ASHRAE_EDITION = "ASHRAE TC 9.9 Thermal Guidelines, 4th ed. (2015); Class A1"
ASHRAE_RECOMMENDED = (18.0, 27.0)
ASHRAE_ALLOWABLE_A1 = (15.0, 32.0)

# Reward = -(cooling power) - VIOLATION_WEIGHT * (overheat degC)^2
# The weight sets how many kW a degree of thermal risk is worth. Report it, and
# show the result is not an artefact of one choice: 08_compare.py sweeps it.
VIOLATION_WEIGHT = 50.0
VIOLATION_LIMIT = ASHRAE_RECOMMENDED[1]   # penalise above 27 degC

# ---------------------------------------------------------------- experiment
# The three observation conditions that answer the research gap.
OBS_MODES = ["none", "oracle", "forecast"]
FORECAST_HORIZONS = [3, 6, 12, 24]        # 15, 30, 60, 120 minutes of lookahead
DEFAULT_FORECAST_H = 12
RL_SEEDS = [42, 123, 456, 789, 1011]
PPO_TIMESTEPS = 200_000

# Degradation levels for the forecast-accuracy sweep: Gaussian noise added to
# the oracle, as a multiple of the target's own standard deviation.
ACCURACY_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]

# Asymmetric loss: under-predicting heat risks a thermal violation, over-
# predicting wastes a little energy. ALPHA is how many times more costly an
# under-prediction is. ALPHA = 1 recovers plain MSE.
ASYMMETRY_ALPHAS = [1.0, 2.0, 4.0, 8.0]

for _d in (PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
