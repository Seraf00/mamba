#!/usr/bin/env bash
# Everything queued behind the main sweep, in dependency order.
#
#   1. E10  falsifiable test of the encoding ceiling at 32 vertices
#   2.      compose the per-group winners into E11 / E12 (writes experiments.yaml)
#   3. E11  best composed configuration, 3 seeds
#   4. E12  boundary-weighted loss sweep
#   5.      the five inference-only sweeps (conf / TTA / FP16)
#   6.      nine Paper 1 baselines' EF at native resolution
#   7.      best YOLO run's EF, same pipeline
#   8.      fair latency benchmark, one protocol for every model
#   9.      regenerate tables and figures
#
# Resumable: rerunning skips anything already evaluated.
#
# SINGLE INSTANCE ONLY. Stopping the parent shell does not kill the Python
# children, so a naive relaunch stacks duplicate trainings that fight over the
# GPU and corrupt each other's run directories. The lock below makes a second
# instance refuse to start.
set -u

cd "$(dirname "$0")/../.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=results/yolo/after_sweep.log
SWEEP=results/yolo/sweep.log
LOCK=results/yolo/.after_sweep.lock

if [ -e "$LOCK" ]; then
    other=$(cat "$LOCK" 2>/dev/null)
    if kill -0 "$other" 2>/dev/null; then
        echo "[queue] already running as pid $other -- refusing to start a second copy."
        exit 1
    fi
    echo "[queue] clearing stale lock from dead pid $other"
fi
echo $$ > "$LOCK"
cleanup() { rm -f "$LOCK"; }
trap cleanup EXIT INT TERM

say() { echo "" | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }

echo "[queue] started pid $$ at $(date)" | tee -a "$LOG"
until grep -q "Total wall time" "$SWEEP" 2>/dev/null; do
    sleep 120
done

say "E10: falsifiable test of the encoding ceiling (32 vertices)"
$PY -u scripts/yolo/run_experiments.py --groups E10_ceiling_test 2>&1 | tee -a "$LOG"

say "compose the per-group winners into E11 / E12"
$PY -u scripts/yolo/compose_best.py 2>&1 | tee -a "$LOG"

say "E11 (best composed config, 3 seeds) + E12 (boundary loss)"
$PY -u scripts/yolo/run_experiments.py --groups E11_best E12_boundary 2>&1 | tee -a "$LOG"

say "inference-only sweeps (conf / TTA / FP16)"
$PY -u scripts/yolo/run_experiments.py 2>&1 | tee -a "$LOG"

say "nine Paper 1 baselines, EF at native resolution"
$PY -u scripts/yolo/eval_baseline_ef.py 2>&1 | tee -a "$LOG"

# Best run, WITH the encoding and image size it was actually trained under.
# Selecting on score alone previously picked a seam-encoded model and would
# have scored it as if it were filled -- silently wrong numbers.
read -r BEST BEST_ENC BEST_SZ <<EOF
$($PY - <<'PYEOF'
import json, pathlib
best, score, enc, sz = "E1_filled", -1, "filled", 640
for d in pathlib.Path("results/yolo").glob("*"):
    if d.name.startswith("_"):
        continue
    f = d / "evaluation.json"
    w = pathlib.Path("yolo_runs") / d.name / "weights" / "best.pt"
    if not (f.exists() and w.exists()):
        continue
    try:
        j = json.loads(f.read_text())
    except json.JSONDecodeError:
        continue
    v = j.get("dice_mean", -1)
    if v > score:
        c = j.get("config") or {}
        best, score = d.name, v
        enc, sz = c.get("encoding", "filled"), c.get("imgsz", 640)
print(best, enc, sz)
PYEOF
)
EOF
echo "[queue] best run: $BEST (encoding=$BEST_ENC, imgsz=$BEST_SZ)" | tee -a "$LOG"

W="yolo_runs/$BEST/weights/best.pt"
say "YOLO EF ($BEST), native resolution"
if [ -f "$W" ]; then
    $PY -u scripts/yolo/eval_ef.py --weights "$W" --name "$BEST" \
        --encoding "$BEST_ENC" --imgsz "$BEST_SZ" 2>&1 | tee -a "$LOG"
else
    echo "[warn] no weights at $W -- skipping YOLO EF" | tee -a "$LOG"
fi

say "fair latency benchmark (one protocol for every model)"
$PY -u scripts/yolo/benchmark_latency.py ${W:+--weights "$W"} 2>&1 | tee -a "$LOG"

say "regenerate tables and figures"
$PY -u scripts/yolo/aggregate_results.py 2>&1 | tee -a "$LOG"
$PY -u scripts/yolo/make_figures.py 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[queue] all done at $(date)" | tee -a "$LOG"
