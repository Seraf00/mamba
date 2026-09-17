#!/usr/bin/env python3
"""
Finisher for E13.

The bash wrapper that launched E13 was killed with its session, but the python
runner it spawned survived and is still working through the remaining arms.
What died with the wrapper were the follow-up steps; this performs them once
the surviving runner exits:

    1. unet_v2's EF (it failed earlier: that model returns a dict, now handled)
    2. merge it into the baseline EF results
    3. regenerate tables and figures

Written in Python rather than bash on purpose. The first attempt at this used a
shell wait-loop with nested PowerShell quoting, mis-detected the running
process, and started the follow-ups while E13 was still training -- competing
for the GPU and nearly regenerating the paper's tables from incomplete results.
The detection here is a single, testable function.

Usage:
    python scripts/yolo/finish_e13.py            # wait, then finish
    python scripts/yolo/finish_e13.py --check    # just report what is running
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = ROOT / "results" / "yolo" / "finish_e13.log"

PS = (
    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
    "Where-Object { $_.CommandLine -like '*run_experiments*' } | "
    "Measure-Object | Select-Object -ExpandProperty Count"
)


def runner_count() -> int:
    """How many E13 runner processes are alive. 0 means it has finished."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", PS],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        return int(out) if out.isdigit() else 0
    except Exception:
        return 0


def say(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def sh(cmd: list[str]) -> int:
    say("  $ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    tail = [l for l in (p.stdout or "").splitlines() if "it/s" not in l][-12:]
    for l in tail:
        say("    " + l)
    return p.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()

    n = runner_count()
    if args.check:
        print(f"E13 runner processes alive: {n}")
        sys.exit(0)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    say(f"[finish] waiting for the E13 runner ({n} alive) ...")
    # Require two consecutive zero readings, so a momentary hiccup in the
    # process query cannot trigger the follow-ups early.
    zeros = 0
    while zeros < 2:
        time.sleep(args.poll)
        zeros = zeros + 1 if runner_count() == 0 else 0
    say(f"[finish] E13 runner exited at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    say("\n=== unet_v2 EF (dict-output bug fixed) ===")
    sh([PY, "-u", "scripts/yolo/eval_baseline_ef.py", "--models", "unet_v2",
        "--out", "results/yolo/baseline_ef_unetv2.json"])

    a = ROOT / "results" / "yolo" / "baseline_ef_native.json"
    b = ROOT / "results" / "yolo" / "baseline_ef_unetv2.json"
    if a.exists() and b.exists():
        m = json.loads(a.read_text())
        m.update(json.loads(b.read_text()))
        a.write_text(json.dumps(m, indent=1))
        say(f"  merged unet_v2 -> {a.name} ({len(m)} models)")

    say("\n=== regenerate tables and figures ===")
    sh([PY, "-u", "scripts/yolo/aggregate_results.py"])
    sh([PY, "-u", "scripts/yolo/make_figures.py"])
    say(f"[finish] done {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
