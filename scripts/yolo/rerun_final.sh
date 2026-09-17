#!/usr/bin/env bash
# Three re-runs needed after late bug fixes, then regenerate the paper assets.
#
#   1. nine baselines' EF -- the earlier attempt ran with the pre-fix code and
#      was killed; the official Simpson implementation takes
#      find_contours(...)[0], so predictions with a stray blob get measured
#      instead of the ventricle (r=0.030 vs r=0.911 with the fix).
#   2. EF on the DEFAULT configuration. The queue reported E1_seam, chosen as
#      "best" by a Dice difference of p=0.87 -- an argmax over noise. The paper
#      must report the recommended configuration, not the lucky one.
#   3. Latency likewise on the default configuration.
#
# Single instance, same lock discipline as the sweep queue.
set -u

cd "$(dirname "$0")/../.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=results/yolo/rerun_final.log
LOCK=results/yolo/.rerun.lock
REF=E1_filled                       # the default / recommended configuration
W="yolo_runs/$REF/weights/best.pt"

if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "[rerun] already running as pid $(cat "$LOCK")"; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

say() { echo "" | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }
echo "[rerun] started $(date)" | tee "$LOG"

say "1/3 nine baselines, EF at native resolution (component filter applied)"
$PY -u scripts/yolo/eval_baseline_ef.py 2>&1 | tee -a "$LOG"

say "2/3 EF on the default configuration ($REF)"
if [ -f "$W" ]; then
    $PY -u scripts/yolo/eval_ef.py --weights "$W" --name "$REF" \
        --encoding filled --imgsz 640 2>&1 | tee -a "$LOG"
else
    echo "[warn] missing $W" | tee -a "$LOG"
fi

say "3/3 fair latency benchmark on the default configuration"
$PY -u scripts/yolo/benchmark_latency.py --weights "$W" 2>&1 | tee -a "$LOG"

say "regenerate tables and figures"
$PY -u scripts/yolo/aggregate_results.py 2>&1 | tee -a "$LOG"
$PY -u scripts/yolo/make_figures.py --weights "$W" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[rerun] done $(date)" | tee -a "$LOG"
