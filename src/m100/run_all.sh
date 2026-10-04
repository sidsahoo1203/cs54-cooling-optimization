#!/usr/bin/env bash
# CS54 — run the whole PS1 + PS2 pipeline in order, with a log.
#
#   bash run_all.sh            full run  (~1.5-3 h, mostly LSTM/GRU training)
#   bash run_all.sh quick      fast sanity check (~10 min, 12 racks, 1 horizon)
#   bash run_all.sh nodeep     everything except LSTM/GRU (~25 min)
#
# Stops at the first failure rather than carrying a broken file forward.

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-full}"
STAMP=$(date +%Y%m%d_%H%M)
LOG="run_${MODE}_${STAMP}.log"

case "$MODE" in
  quick)  PS1_ARGS="--quick";              PS2_ARGS="--racks 12 --stride 24" ;;
  nodeep) PS1_ARGS="--no-deep";            PS2_ARGS="" ;;
  full)   PS1_ARGS="";                     PS2_ARGS="" ;;
  *) echo "usage: bash run_all.sh [full|quick|nodeep]"; exit 2 ;;
esac

echo "CS54 pipeline — mode=$MODE — logging to $LOG"
echo "Started $(date)" | tee "$LOG"

step () {
  echo ""                                        | tee -a "$LOG"
  echo "######## $1" | tee -a "$LOG"
  shift
  # shellcheck disable=SC2068
  if ! python $@ 2>&1 | tee -a "$LOG"; then
    echo "FAILED. See $LOG" | tee -a "$LOG"
    exit 1
  fi
}

step "01_inspect  — schema and data dictionary"  01_inspect.py
step "02_extract  — build the rack-level table"  02_extract.py
step "03_ps1      — sub-problem (a) prediction"  03_ps1_predict.py $PS1_ARGS
step "04_ps2      — sub-problem (b) dynamics"    04_ps2_thermal.py $PS2_ARGS

echo ""                   | tee -a "$LOG"
echo "Finished $(date)"   | tee -a "$LOG"
echo ""                   | tee -a "$LOG"
echo "Results : ~/research/cs54-cooling/results/"  | tee -a "$LOG"
echo "Figures : ~/research/cs54-cooling/figures/"  | tee -a "$LOG"
echo "Full log: $LOG  <- send me this file"        | tee -a "$LOG"
