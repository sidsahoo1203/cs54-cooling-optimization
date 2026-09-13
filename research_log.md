# CS54 Research Log — Siddharth Sahoo
**Project:** AI-Based Cooling Optimization in Data Centers
**Guide:** Dr. P. Akilandeswari
**Reg. No:** RA2512005010041

---

## 2026-09-09 (Day 1) — Environment Setup

**Phase:** Phase 0 — Setup

**Goal:** Working Linux development environment for CS54 implementation.

**Done:**
- Found broken WSL Ubuntu registration left over from an old uninstall;
  unregistered and reinstalled clean.
- WSL2 + Ubuntu (resolute) installed, user account created.
- System updated: 152 packages upgraded.
- Installed build-essential (GCC 15.2.0), git 2.53.0, curl 8.18.0, wget.
- Installed Miniconda 26.7.1 at /home/siddharth/miniconda3.
- Created conda env `cs54` with Python 3.10.21.
- Created project structure and initialized Git repository.

**Environment decisions:**
- WSL2 rather than Colab: persistent filesystem, no session timeouts,
  standard Linux toolchain for research code.
- Conda pins Python 3.10 (SustainDC requirement) independently of
  Ubuntu's system Python 3.14.
- Accepted Anaconda default-channel ToS (free for academic use).

**Problems:**
- Broken pre-existing WSL distro: /bin/sh missing, getpwuid failures.
  Resolved via `wsl --unregister Ubuntu` then fresh install.

**Next:** Clone SustainDC, install requirements, run demo training.

## 2026-09-09 (Day 1, cont.) — SustainDC Installation

**Done:**
- Cloned SustainDC (HewlettPackard/dc-rl) into project folder.
- Installed requirements.txt: torch 2.0.0+cpu, Gymnasium 0.29.1,
  tensorflow 2.12.0, ray 2.24.0, PsychroLib 2.5.0, opyplus 1.4.2.
- Verified `import sustaindc_env` succeeds.

**Undocumented dependencies (missing from their requirements.txt):**
1. matplotlib
2. dash
3. dash-bootstrap-components

**Version conflicts introduced by dash:**
- Werkzeug 3.0.2 -> 3.1.8
- typing_extensions 4.11.0 -> 4.16.0
(Both were pinned by SustainDC; dash required newer.)

**Problem:** pip install failed with [Errno 28] No space left on device.
Root cause: /tmp is a 1.5 GB tmpfs (RAM-backed); the TensorFlow 2.12
wheel (586 MB) exceeded it while unpacking. Disk had 953 GB free.
Fix: TMPDIR=~/tmp to redirect pip's temp directory to real disk.

**Bundled data confirmed:**
- Workload: Alibaba_CPU_Data_Hourly_1.csv, _2.csv,
  GoogleClusteData_CPU_Data_Hourly_1.csv
  Format: unnamed index + cpu_load (fraction 0-1), 8905 rows
- Weather: 11 .epw files (US locations)
- CarbonIntensity: 17 regional CSVs

**Next:** Explore environment structure, run a first episode.


## 2026-09-12 — Phase 1 Complete

**Delivered:** PDF requirement (b) — thermal digital twin + Gym environment.

**Baseline result (5 seeds, 7-day episodes, Jan, NY weather, Alibaba trace):**
- PUE: 1.4123 +/- 0.0045
- Cooling energy: 71,903 +/- 784 kWh
- Max inlet temp: 23.3 C (setpoint 18 + 5.3 offset)
- ASHRAE violations: 0

**Key findings:**
1. rack_inlet_temp = CRAC_setpoint + fixed offset. No thermal inertia.
   Prediction target must be IT power / heat generation, not inlet temp.
2. Control DOES have inertia: agent applies +/-1 C deltas, range 15.0-21.6,
   with acceleration after 3 consecutive same-direction actions.
   This is the mechanical basis for anticipatory control.
3. All 20 racks receive identical utilization. Racks differ in hardware
   (110-170 W) and approach temp (5.0/5.3 C), not load.
4. Setpoint bounds mean inlet temp can never exceed ASHRAE 27 C.
   Violations are structurally impossible -> report headroom instead.
5. reset() starts at a random day/hour with unseeded global random.
   Must seed random + numpy externally.
6. Workload is quantized to 2 decimal places before reaching the DC.

**Questions for guide:**
- Accept zero-violation metric + headroom, or widen setpoint bounds?
- Is uniform per-rack load a limitation to address or to note?

**Next:** Phase 2 — feature engineering and selection.
