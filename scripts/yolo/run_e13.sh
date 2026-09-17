#!/usr/bin/env bash
# E13: loss gains + optimiser/LR. Waits for the current re-runs to finish first.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=results/yolo/e13.log
LOCK=results/yolo/.e13.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "[e13] already running as pid $(cat "$LOCK")"; exit 1
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT INT TERM
echo "[e13] waiting for the EF/latency re-runs to finish ..." | tee "$LOG"
while [ -e results/yolo/.rerun.lock ] && kill -0 "$(cat results/yolo/.rerun.lock 2>/dev/null)" 2>/dev/null; do
    sleep 60
done
echo "[e13] starting $(date)" | tee -a "$LOG"
$PY -u scripts/yolo/run_experiments.py --groups E13_optim 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"
$PY -u scripts/yolo/aggregate_results.py 2>&1 | tee -a "$LOG"
echo "[e13] done $(date)" | tee -a "$LOG"
