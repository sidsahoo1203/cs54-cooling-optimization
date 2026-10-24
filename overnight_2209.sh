#!/usr/bin/env bash
# =============================================================================
# CS54 overnight run - fetch and inventory the summer month (22-09)
#
#   Stage 0  preflight: disk, tools, paths
#   Stage 1  resolve + download the tar from Zenodo (resumable)
#   Stage 2  verify MD5, extract
#   Stage 3  standalone inventory - what plugins and metrics are ACTUALLY there
#   Stage 4  optional: your own pipeline, only if the configs point at 22-09
#
# Every stage is gated on the one before it. If something fails the script stops
# and says where, instead of running the rest against nothing.
# =============================================================================
set -u

PROJECT="$HOME/research/cs54-cooling"
MONTH="22-09"
RECORD="7590583"
NEED_GB=75

cd "$PROJECT" || { echo "cannot cd to $PROJECT"; exit 1; }
mkdir -p logs data/m100_raw "$HOME/tmp"
export TMPDIR="$HOME/tmp"

LOG="logs/overnight_${MONTH}_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date '+%H:%M:%S')] $*"; }
rule() { echo "-------------------------------------------------------------"; }

say "CS54 overnight run - month $MONTH - log: $LOG"
rule

# ---------------------------------------------------------------- stage 0
say "STAGE 0  preflight"
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
say "  disk free       : ${FREE_GB} GB   (need ~${NEED_GB} GB for tar + extract)"
if [ "${FREE_GB:-0}" -lt "$NEED_GB" ]; then
    say "  ABORT: not enough disk space."
    say "  Free space, or move data/ to a bigger drive, then re-run."
    exit 1
fi
for t in wget tar python3 md5sum; do
    command -v "$t" >/dev/null || { say "  ABORT: '$t' not found"; exit 1; }
done
say "  tools           : ok"
rule

# ---------------------------------------------------------------- stage 1
say "STAGE 1  resolve the file on Zenodo record $RECORD"
python3 -u - "$RECORD" <<'PY' > .zenodo_target || { say "  ABORT: could not read the Zenodo record"; exit 1; }
import json, sys, urllib.request
rec = sys.argv[1]
with urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}", timeout=60) as r:
    meta = json.load(r)
files = meta.get("files", [])
if not files:
    sys.exit("no files listed on the record")
# pick the largest file; these records hold one monthly tar
f = max(files, key=lambda x: x.get("size", 0))
link = f.get("links", {}).get("self") or f.get("links", {}).get("download")
md5 = (f.get("checksum") or "").replace("md5:", "")
print(f["key"]); print(link); print(md5); print(f.get("size", 0))
PY

TARNAME=$(sed -n 1p .zenodo_target)
TARURL=$(sed -n 2p .zenodo_target)
TARMD5=$(sed -n 3p .zenodo_target)
TARSIZE=$(sed -n 4p .zenodo_target)
rm -f .zenodo_target

say "  file            : $TARNAME"
say "  size            : $(( TARSIZE / 1000000000 )) GB"
say "  md5 (expected)  : ${TARMD5:-none published}"

TARPATH="data/m100_raw/$TARNAME"
say "  downloading (resumable; safe to re-run if it dies)"
wget -c --progress=dot:giga --tries=20 --waitretry=15 --read-timeout=60 \
     -O "$TARPATH" "$TARURL"
RC=$?
if [ $RC -ne 0 ]; then
    say "  ABORT: download failed (exit $RC). Re-running this script resumes it."
    exit 1
fi
say "  downloaded      : $(du -h "$TARPATH" | cut -f1)"
rule

# ---------------------------------------------------------------- stage 2
say "STAGE 2  verify and extract"
if [ -n "$TARMD5" ]; then
    say "  checking md5 (a few minutes on 34 GB)..."
    GOT=$(md5sum "$TARPATH" | cut -d' ' -f1)
    if [ "$GOT" != "$TARMD5" ]; then
        say "  ABORT: md5 mismatch. got $GOT, expected $TARMD5"
        say "  The file is corrupt. Delete it and re-run to download again."
        exit 1
    fi
    say "  md5 ok"
fi

DEST="data/m100_raw/$MONTH"
mkdir -p "$DEST"
say "  extracting into $DEST ..."
tar -xf "$TARPATH" -C "$DEST" || { say "  ABORT: extraction failed"; exit 1; }
say "  extracted       : $(du -sh "$DEST" | cut -f1)"
say "  NOTE: $TARPATH is kept. Delete it to reclaim space once you are happy."
rule

# ---------------------------------------------------------------- stage 3
say "STAGE 3  inventory - what is actually on disk"
say "  (standalone; does not import your config, so it cannot run on the wrong month)"
python3 -u - "$DEST" <<'PY'
import os, sys
from pathlib import Path

root = Path(sys.argv[1])
# the tar may unpack into a nested month folder
cands = [root] + [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
base = root
for c in cands:
    if any((c / n).is_dir() for n in ("ipmi_pub", "vertiv_pub", "logics_pub")):
        base = c
        break

FACILITY = ["vertiv_pub", "schneider_pub", "logics_pub", "weather_pub"]
NODE     = ["ipmi_pub", "ganglia_pub", "slurm_pub", "job_table", "nagios_pub"]
KEY_METRICS = {
    "vertiv_pub":    ["Supply_Air_Temperature_Set_Point", "Supply_Air_Temperature",
                      "Free_Cooling_Valve_Open_Position", "Compressor_Utilization"],
    "logics_pub":    ["Tot_cdz", "Tot_chiller", "Tot_qpompe", "Tot_ict", "Pue"],
    "schneider_pub": [],
    "weather_pub":   [],
}

def folder_report(p):
    metrics, files, size = [], 0, 0
    if not p.is_dir():
        return None
    for m in sorted(p.iterdir()):
        if not m.is_dir():
            continue
        fs = list(m.rglob("*.parquet"))
        metrics.append((m.name, len(fs), sum(f.stat().st_size for f in fs)))
        files += len(fs)
        size += sum(f.stat().st_size for f in fs)
    return metrics, files, size

print(f"\nbase directory: {base}\n")
print(f"{'plugin':<18}{'metrics':>9}{'files':>9}{'size':>12}   verdict")
print("-" * 70)

present = {}
for group, names in (("FACILITY", FACILITY), ("NODE", NODE)):
    print(f"-- {group} --")
    for name in names:
        rep = folder_report(base / name)
        if rep is None:
            print(f"{name:<18}{'-':>9}{'-':>9}{'-':>12}   ABSENT")
            present[name] = 0
            continue
        metrics, files, size = rep
        gb = size / 1e9
        verdict = "EMPTY" if files == 0 else "ok"
        print(f"{name:<18}{len(metrics):>9}{files:>9}{gb:>11.2f}G   {verdict}")
        present[name] = files

print("\n-- key metrics needed for sub-problem 3 --")
missing = []
for plugin, wanted in KEY_METRICS.items():
    for w in wanted:
        hit = (base / plugin / w).is_dir()
        # Schneider names carry a PLC prefix in 2022 months
        if not hit and (base / plugin).is_dir():
            hit = any(w in d.name for d in (base / plugin).iterdir() if d.is_dir())
        print(f"  {'FOUND  ' if hit else 'MISSING'}  {plugin}/{w}")
        if not hit:
            missing.append(f"{plugin}/{w}")

print("\n" + "=" * 70)
fac_ok = all(present.get(p, 0) > 0 for p in FACILITY)
node_ok = present.get("ipmi_pub", 0) > 0
if fac_ok and node_ok and not missing:
    print("VERDICT: GO.  Facility and node telemetry are both present.")
    print("         22-09 can carry all three sub-problems.")
elif fac_ok and node_ok:
    print("VERDICT: GO WITH CARE.  Plugins present, but some named metrics are")
    print("         missing (listed above). Check before building the twin.")
else:
    print("VERDICT: STOP.  Required plugins are absent or empty.")
    print("         Do not spend compute on this month - tell Claude what is missing.")
print("=" * 70)
PY
rule

# ---------------------------------------------------------------- stage 4
say "STAGE 4  your pipeline - only if the configs point at $MONTH"
CFG_OK=1
for f in src/m100/config.py src/m100/config_ps3.py; do
    if [ -f "$f" ]; then
        LINE=$(grep -nE '^[[:space:]]*MONTH[[:space:]]*=' "$f" | head -1)
        say "  $f : ${LINE:-no MONTH line found}"
        echo "$LINE" | grep -q "$MONTH" || CFG_OK=0
    else
        say "  $f : not found"
        CFG_OK=0
    fi
done

if [ "$CFG_OK" -ne 1 ]; then
    say "  SKIPPED - configs are not set to $MONTH."
    say "  That is deliberate: better to skip than to run on the wrong month."
    say "  Set MONTH = \"$MONTH\" in both files and re-run this script;"
    say "  stages 1 and 2 will no-op and it will pick up from here."
else
    say "  configs look right - running the extraction chain"
    cd src/m100 || exit 1
    for step in 01_inspect.py 05_facility_extract.py 02_extract.py; do
        say "  >>> $step"
        python3 -u "$step"
        RC=$?
        if [ $RC -ne 0 ]; then
            say "  $step FAILED (exit $RC) - stopping here, later steps would be garbage"
            break
        fi
        say "  <<< $step done"
    done
    cd "$PROJECT" || exit 1
fi
rule

say "DONE. Read the VERDICT block above first."
say "Full log: $LOG"
