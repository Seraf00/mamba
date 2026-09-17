#!/usr/bin/env bash
# Relaunch everything after a stop. Safe to run at any time -- nothing is
# recomputed.
#
#   * runs already evaluated are skipped (evaluation.json present)
#   * a run interrupted part-way resumes from weights/last.pt, so at most the
#     current epoch is lost
#   * the post-sweep queue re-arms itself behind the sweep
#
# Usage:  bash scripts/yolo/resume.sh
set -u

cd "$(dirname "$0")/../.." || exit 1
PY=./.venv/Scripts/python.exe

echo "=== state ==="
$PY - <<'EOF'
import json, pathlib, yaml
plan = yaml.safe_load(open("scripts/yolo/experiments.yaml"))
done = {d.name for d in pathlib.Path("results/yolo").glob("*")
        if (d / "evaluation.json").exists()}
todo = []
for g in plan["groups"]:
    for r in g["runs"]:
        if r["name"] not in done:
            todo.append(r["name"])
for s in plan.get("inference_sweeps", []):
    if s["name"] not in done:
        todo.append(s["name"])
part = [p.parent.parent.name for p in pathlib.Path("yolo_runs").glob("*/weights/last.pt")
        if p.parent.parent.name not in done]
print(f"  completed : {len(done - {d for d in done if d.startswith('_')})}")
print(f"  remaining : {len(todo)}")
if part:
    print(f"  partial   : {', '.join(sorted(part))}  (will resume from last.pt)")
EOF

echo ""
echo "=== relaunching ==="
nohup $PY -u scripts/yolo/run_experiments.py >> results/yolo/sweep.log 2>&1 &
echo "  sweep       pid $!"
nohup bash scripts/yolo/run_after_sweep.sh >> results/yolo/after_sweep.log 2>&1 &
echo "  post-queue  pid $!"
echo ""
echo "Tail progress with:  tail -f results/yolo/sweep.log"
