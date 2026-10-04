# CS54 — M100 pipeline v2 (sub-problems a and b)

**Siddharth Sahoo · RA2512005010041 · Guide: Dr. P. Akilandeswari**

Runs sub-problem (a) heat and temperature prediction and sub-problem (b) thermal
dynamics modelling on the M100 ExaData month already on your disk.
**No new download needed.** `CHANGES.md` says what v2 fixes and why.

---

## Install

Already done last time, but if you are on a fresh machine:

```bash
conda activate cs54
TMPDIR=~/tmp pip install pyarrow xgboost scikit-learn matplotlib tabulate
```

`TMPDIR=~/tmp` is required on your laptop — `/tmp` is a 1.5 GB RAM disk and large
wheels fail to unpack there.

## Put the files in place

Replace the old `src/m100` contents with these. Keep the old ones if you want to
compare:

```bash
cd ~/research/cs54-cooling/src
mv m100 m100_v1          # keep the old run for comparison
mkdir -p m100
# copy the v2 files into m100/
cd m100
```

## Run

```bash
bash run_all.sh quick     # ~10 min  — prove it works first
bash run_all.sh           # full run — leave it going
```

Or step by step:

```bash
python 01_inspect.py        # ~1 min    schema + data dictionary
python 02_extract.py        # ~16 min   the slow one, run once
python 03_ps1_predict.py    # ~1-2 h    4 horizons x 2 targets x 5 models
python 04_ps2_thermal.py    # ~20 min   dynamics + rollout
```

**Do the `quick` run first.** It uses 12 racks, one horizon and six epochs, and
finishes in minutes. If that is clean, start the full run and leave it.

Useful flags:

```bash
python 03_ps1_predict.py --no-deep          # skip LSTM/GRU entirely
python 03_ps1_predict.py --horizons 1 12    # only 5-min and 60-min
python 04_ps2_thermal.py --stride 24        # faster rollout, fewer start points
```

`run_all.sh` writes `run_<mode>_<timestamp>.log`. **Send me that file** — it has
everything I need.

---

## What each script does

| File | Purpose | Output |
|---|---|---|
| `config.py` | Every setting, including the units table. **Edit this, not the scripts.** | — |
| `common.py` | Schema-adaptive Parquet reading, bounded streaming, gap handling, metrics, chronological split | — |
| `01_inspect.py` | Inventory plugins and metrics on disk; detect each file's schema; sanity-check values | `results/data_dictionary_<MONTH>.md` |
| `02_extract.py` | Stream ~5 GB of 20-second per-node telemetry into a 49-rack × 5-minute table | `data/processed/rack_5min_<MONTH>.parquet`, `results/units_<MONTH>.md` |
| `03_ps1_predict.py` | **(a)** forecast rack heat and inlet temperature at 5/15/30/60 min | `results/ps1_results.md`, figures |
| `04_ps2_thermal.py` | **(b)** learn the thermal state transition; rollout vs persistence; spatial ablation; ASHRAE | `results/ps2_results.md`, figures |
| `make_fixture.py` | Synthetic test data. **Testing only — never appears in results.** | — |
| `run_all.sh` | Chains everything with a log | `run_*.log` |

---

## How to read the results

### PS1 — `results/ps1_results.md`

One table per target, one block of rows per horizon. For each model you get
RMSE, MAE, MAPE and R², **all on identical rows**.

Below each table the script prints a verdict per model:

```
ridge            RMSE +23.53%   MAE +22.61%   BEATS on both
gru              RMSE +18.55%   MAE +15.73%   BEATS on both
xgboost          RMSE  +3.58%   MAE  +5.18%   BEATS on both
```

**A model only genuinely wins if it beats persistence on both.** Winning on RMSE
alone means it avoids a few large errors while being worse on the typical
prediction — which is what happened in v1 and is worth knowing.

Expect persistence to be hard to beat at 5 minutes and easy to beat at 60. That
shape is informative: it tells you the timescale on which forecasting has value,
which is directly the question sub-problem (c) asks.

### PS2 — `results/ps2_results.md`

**Table 1, dynamics vs step size.** R² of the temperature *change*, at 5/15/30/60
minutes, for three feature sets. Read the shape:

- flat near zero everywhere → the change really is unpredictable
- rising with step size → real dynamics, masked at 5 min by sensor resolution
- neighbours helping only at longer steps → inter-rack heat transport is real
  but slower than one bin

**Table 2, rollout vs persistence.** The important one. The learned twin is
iterated forward, feeding its own output back in; persistence holds the current
temperature. Every row says `TWIN WINS` or `persistence wins`. If the twin never
wins, that is a negative result and the script says so in those words — report
it, do not bury it.

**Section 3, actuator sensitivity.** Reports the response spread and the
actuator's importance *and rank*. If the model assigns fan speed zero importance,
the output says so explicitly, and distinguishes that from a code fault.

**Section 4, thermal envelope.** With the ASHRAE edition, the equipment class,
the measure (dry-bulb at the equipment inlet) and what `ambient_max` actually is.

---

## The methodological commitments

Enforced in code, not left to discipline. These are what a panel attacks first.

**Chronological splitting.** The test period is strictly later than training.
Random shuffling leaks the future backwards and inflates every metric — the most
common fatal flaw in applied ML papers. The boundary is one instant shared by all
racks.

**Scalers fit on training data only.** Fitting on everything leaks test mean and
variance into training. Silent, and it inflates results.

**Persistence reported everywhere.** Zero parameters. On a smooth signal it is a
genuinely strong prediction, and without it an R² of 0.94 looks like a triumph
when persistence also scores 0.939.

**Identical evaluation rows.** Every model at a given target and horizon is
scored on the intersection of what all of them can serve. v1 compared 25,001 rows
against 26,529 and the table was meaningless.

**PS2 predicts the change, not the level.** Predicting the level scores R² > 0.99
by echoing the input and learns no physics.

**No MAPE on a delta target.** The true change passes through zero, so the
percentage error explodes and the number is meaningless.

**Missing data stays missing.** Short gaps are forward-filled within 30 minutes
and flagged; longer gaps remain NaN and the rows are dropped honestly. Nothing is
filled with a fabricated zero.

---

## Aggregation convention

Per-node values are **averaged across the nodes in a rack**, so every column is
"mean per node in that rack". This avoids the confound of racks having different
numbers of live nodes at different times. Temperatures also carry a `_max`
column — the rack hotspot, which is what ASHRAE compliance is judged on.

`rack = node // 20`, from the ExaData spatial documentation: 49 racks, 20 nodes
each, node IDs 0–979. Rows are racks 0–17, 18–32, 33–48. Node IDs past 979 are
login and service nodes and are dropped.

---

## Known limits of the 2020 months — say these, do not hide them

1. **No cooling control variable.** The only actuator present is node fan speed,
   which is firmware-controlled and reacts to temperature rather than being
   commanded. The script quantifies how little the model uses it.
2. **PUE cannot be computed.** It needs `logics.Tot_cdz`, `logics.Tot_chiller`
   and `logics.Tot_ict`, which are in the 2022 records.
3. **Sub-problem (c) is not executable here.** It needs
   `vertiv.Supply_Air_Temperature_Set_Point`.

None of this weakens (a) or (b). They run on real, peer-reviewed, publicly
released supercomputer telemetry — which is what was asked for.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Everything MISSING in `01_inspect.py` | `RAW_ROOT` wrong in `config.py`, or the month is not extracted |
| A metric shows MISSING in section C | Name differs. Find it in section A, correct `config.METRICS` |
| `02_extract.py` killed / machine freezes | Set `RACK_SUBSET = list(range(12))` in `config.py` |
| Timestamps come out as 1970 | Epoch-unit guess was wrong. Send me `01_inspect.py`'s dtype line |
| A column is >50% empty after extraction | Remove it from `config.METRICS` — it will otherwise discard rows via dropna |
| PS1 takes hours | `--no-deep`, or `--horizons 1 12`, or lower `EPOCHS` in `config.py` |
| PS2 rollout slow | `--stride 24` (start points every 2 h instead of every hour) |
| Persistence still wins everywhere | A real finding at short horizons. Check the 60-min rows before concluding anything is broken |

## Reproducibility

Seed fixed at `config.SEED = 42` and printed in every results file alongside the
month, bin size and split fractions. Record your versions in the research log:

```bash
conda list | grep -E "python |pandas|numpy|scikit|xgboost|tensorflow|pyarrow"
```
