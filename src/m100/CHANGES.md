# CS54 pipeline v2 — what changed and why

**Siddharth Sahoo · RA2512005010041 · 1 October 2026**

Every item in your review is addressed below, with the file and the reasoning.
Nothing was changed that you did not identify, except three bugs I found while
fixing yours.

---

## Your verdict → what I changed

### PS1

| Your finding | Status | What was done |
|---|---|---|
| "Models used different test sample counts, making comparison unfair" | **FIXED** | `03_ps1_predict.py` now intersects the flat-model key set with the sequence-model key set and evaluates **all five models on exactly those rows**. The run prints `EVALUATION SET: N rows common to every model`. |
| "Only the 60-minute horizon and one month were tested" | **FIXED (horizons)** | Four horizons: 5, 15, 30, 60 min (`config.HORIZONS`). Multiple months is a config change — see "Running more months" below. |
| "Units and meanings of total_power and ambient are missing" | **FIXED** | `config.UNITS` defines unit and meaning for every column; printed in every header, written to `results/units_<MONTH>.md`, and carried as a column in `ps1_results.csv`. |
| "Test all models on identical samples at 5, 15, 30 and 60-minute horizons" | **DONE** | Exactly that. |
| "Persistence performs better in MAE and MAPE" | **NOW EXPLICIT** | Each run prints a per-model verdict: `BEATS on both` / `RMSE only` / `WORSE on both`. A model winning on RMSE alone is called out, not hidden. |
| "GRU and LSTM perform worse than persistence" | **LIKELY CAUSED BY A BUG — see below** | The workload features they depend on were corrupted. |

### PS2

| Your finding | Status | What was done |
|---|---|---|
| "Local model R² = 0.00165, learned almost no temperature-change behaviour" | **FIXED** | **Lagged thermal history added** — the single biggest change. See below. |
| "Adding neighbouring and row features reduced performance" | **RE-TESTED PROPERLY** | Ablation now repeated at every step size. Heat transport between racks cannot show up inside one 5-minute bin. |
| "Spatial heat propagation is not demonstrated" | **NOW TESTABLE** | Per-horizon neighbour effect is printed as a percentage change in RMSE. |
| "Rollout R² is high, but no persistence rollout baseline is provided" | **FIXED** | `batched_rollout` returns the learned prediction **and** a persistence rollout. Every horizon prints `TWIN WINS` or `persistence wins`. |
| "Fan-speed effect is extremely small and non-monotonic" | **QUANTIFIED AND EXPLAINED** | Now reports the response spread, the actuator's XGBoost gain importance **and its rank**. If the model assigns it zero importance, the output says so explicitly and distinguishes that from a code fault. |
| "Current model is unsuitable as a controllable thermal digital twin" | **CORRECT, AND NOW PROVABLE EITHER WAY** | The twin-vs-persistence table is the evidence. If the twin loses, the script says so in those words. |
| "MAPE is inappropriate for near-zero temperature changes" | **FIXED** | `regression_metrics(..., include_mape=False)` for all delta targets. Removed from the table and the CSV, not just the display. |
| "Add workload, cooling setpoint, supply temperature, airflow and lagged thermal features" | **PARTIALLY DONE** | Lagged thermal features and workload: done. Setpoint, supply temperature and airflow **do not exist in the 2020 months** — they are in the `vertiv`/`schneider` plugins of the 2022 records. That is PS3's dataset step. |

### ASHRAE

| Your finding | Status | What was done |
|---|---|---|
| "Confirm that ambient_max represents server-inlet temperature" | **DOCUMENTED** | The output now states: IPMI `ambient` is the node BMC's inlet-side air sensor; `ambient_max` is the hottest node inlet in each rack-bin, used because ASHRAE compliance is judged on the worst inlet in a rack, not the average. |
| "Mention the ASHRAE edition and equipment class" | **FIXED** | `ASHRAE TC 9.9 Thermal Guidelines for Data Processing Environments, 4th ed. (2015); Class A1`, with the measure stated as dry-bulb at the equipment air inlet. Printed and saved. |

---

## Three bugs I found while fixing yours

### Bug 1 — missing workload was being turned into zero workload

**The most damaging one, and the likely cause of the weak PS1 results.**

Ganglia samples far more sparsely than IPMI: in your 20-06 run, 224,089 bins
against IPMI's 414,981. **46% of 5-minute bins had no workload reading.**

v1 computed:

```python
wide["cpu_busy"] = (wide["cpu_user"].fillna(0)
                  + wide["cpu_system"].fillna(0)
                  + wide["cpu_wio"].fillna(0))
```

`fillna(0)` converts *"not measured"* into *"100% idle"*. That is a fabricated
observation, for nearly half the dataset. Your own output shows it: every Ganglia
column reported ~47% missing while `cpu_busy` reported **0.00%**.

`cpu_busy` is the physical cause of the power you are predicting, and it is one
of the ten signals in `SEQ_FEATURES` — the compact set the LSTM and GRU use. So
the recurrent models were trained on a corrupted version of the most important
input. That is almost certainly why they finished last.

**v2:** bounded forward-fill (a reading stays valid 30 minutes, not forever),
longer gaps stay NaN, `cpu_busy = 100 - cpu_idle` from the directly measured
complement, a `workload_imputed` audit flag, and a before/after missingness
table printed at extraction.

### Bug 2 — the sequence window was off by one against the flat path

v1's flat model used features up to time *t* and predicted *t+h*. v1's sequence
builder used a window ending at *t−1* and predicted *t+h−1*, while labelling the
sample with time *t*. **The two paths were forecasting different things**, so
even with identical rows the comparison would have been wrong.

**v2:** both use window `[t−lookback+1 .. t]`, target at `t+h`, key `(rack, t)`.

### Bug 3 — a phantom rack 49

Ganglia reports node IDs past 979 (login and service nodes), and `rack = node // 20`
turned those into a rack 49 that does not physically exist. Your extract said
`racks: 50`. M100 has 49 racks, IDs 0–48.

**v2:** racks outside `0..N_RACKS-1` are dropped during streaming.

---

## The scientific change: why PS2 scored 0.0017

Two reasons, and both are now addressed.

**1. The model could not see its own rate of change.** v1 gave it only
contemporaneous values — temperature now, power now, workload now. But the next
change of a first-order thermal system depends mostly on the *current* change.
Without lagged state the model is structurally blind to the dynamics it is meant
to learn.

v2 adds, per rack: lags of the thermal state at 1, 2, 3, 6 and 12 steps;
previous deltas at 1, 2 and 3 steps; rolling mean and standard deviation over 6
and 12 steps; and the same treatment for power, workload and core temperature.
Roughly 39 historical features where v1 had none.

**2. A 5-minute change may be below the noise floor.** Rack-mean inlet
temperature over 20 nodes, averaged over 5 minutes, barely moves — and the IPMI
sensor has limited resolution. A near-zero R² there may be measuring sensor noise
rather than an unlearnable system.

v2 therefore fits the dynamics at **5, 15, 30 and 60 minutes** and reports R² at
each. The shape of that curve is itself a result:

- R² flat near zero at every step → the state genuinely is not predictable
- R² rising with step size → real dynamics, previously masked by resolution
- neighbours helping only at longer steps → **inter-rack heat transport is real
  but slower than one bin**, which is physically exactly what you would expect
  and is a better finding than v1's flat negative

---

## Running more months

The second half of your "only one month" point. Ten months of node-side data are
already on your disk. To add one:

```bash
# edit config.py:  MONTH = "20-07"
python 02_extract.py && python 03_ps1_predict.py && python 04_ps2_thermal.py
```

Or without editing anything:

```bash
CS54_MONTH=20-07 python 02_extract.py
CS54_MONTH=20-07 python 03_ps1_predict.py
```

Results land in month-independent filenames, so **copy `results/` aside between
months** or they will be overwritten. Reporting across three months instead of
one is the difference between "it worked" and "it generalises", and it is the
cheapest credibility you can buy before Review 2.

Avoid `20-04` — it is one of four anomalously small months in the collection and
looks like a monitoring outage.

---

## What is still a known limitation

State these; do not let a panelist find them first.

1. **No cooling control variable exists in 2020.** Node fan speed is
   firmware-controlled — it reacts to temperature rather than being commanded.
   v2 quantifies how little the model uses it. The real actuator,
   `vertiv.Supply_Air_Temperature_Set_Point`, is in the 2022 records.
2. **PUE cannot be computed from these months.** It needs `logics.Tot_cdz`,
   `logics.Tot_chiller` and `logics.Tot_ict`.
3. **One month, one machine.** Marconi-100 is air-cooled plus direct liquid
   cooling at CINECA; results are not automatically transferable to a
   conventional CRAC-only data centre.
4. **Rack-mean aggregation** hides intra-rack variation. The `_max` columns
   partially recover it for the ASHRAE analysis, but the models work on means.
