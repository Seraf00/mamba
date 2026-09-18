#!/usr/bin/env python3
"""End-to-end: kill train_all_models.py mid-model, relaunch, check it resumes.

test_trainer_resume.py proves the Trainer resumes bit-identically. This proves
the COMMAND you actually run does: real CAMUS data, real model, real
dataloader workers, the process killed from outside the way a Colab disconnect
kills it -- then the identical command relaunched, twice.

  launch 1   killed once epoch 2 is logged and last.pth exists
  launch 2   must say "Resuming", finish, and leave a clean result
  launch 3   must skip the finished model and keep it in all_results.json

    python scripts/test_resume_e2e.py --model deeplab_v3 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS: list[str] = []


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILS.append(msg)


def kill_tree(proc):
    """Kill the trainer AND its dataloader workers, as a disconnect would."""
    if os.name == 'nt':
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)],
                       capture_output=True)
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait()


def csv_rows(p):
    if not p.exists():
        return []
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    ei = lines[0].split(',').index('epoch')
    return [int(float(l.split(',')[ei])) for l in lines[1:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='deeplab_v3')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--data-dir', default=str(ROOT / 'data' / 'CAMUS'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    out = Path(args.out or (Path(os.environ.get('TEMP', '/tmp')) / 'e2e_resume'))
    exp = 'e2e'
    mdir = out / exp / args.model
    if (out / exp).exists():
        import shutil
        shutil.rmtree(out / exp)

    cmd = [sys.executable, str(ROOT / 'scripts' / 'train_all_models.py'),
           '--data_dir', args.data_dir, '--models', args.model,
           '--epochs', str(args.epochs), '--batch_size', '8',
           '--early_stopping', '0', '--pin', 'full',
           '--resume', '--resume_every', '1', '--checkpoint_every', '0',
           '--mixed_precision', '--skip_benchmark', '--num_workers', '2',
           '--output_dir', str(out), '--exp_name', exp]
    popen_kw = {} if os.name == 'nt' else {'start_new_session': True}

    # ------------------------------------------------------------ launch 1
    print('launch 1: train until epoch 2 is logged, then kill')
    log1 = open(out / 'launch1.log', 'w') if out.exists() else None
    out.mkdir(parents=True, exist_ok=True)
    log1 = open(out / 'launch1.log', 'w')
    p = subprocess.Popen(cmd, stdout=log1, stderr=subprocess.STDOUT, **popen_kw)
    t0 = time.time()
    while p.poll() is None:
        if len(csv_rows(mdir / 'training_log.csv')) >= 2 and (mdir / 'last.pth').exists():
            break
        if time.time() - t0 > 1800:
            break
        time.sleep(2)
    if p.poll() is None:
        kill_tree(p)
        print(f'  killed after {time.time() - t0:.0f}s')
    log1.close()
    before = csv_rows(mdir / 'training_log.csv')
    check(len(before) >= 2, f'epochs logged before the kill: {before}')
    check((mdir / 'last.pth').exists(), 'last.pth survived the kill')
    check(not (mdir / 'results.json').exists(), 'model not marked finished')
    txt1 = (out / 'launch1.log').read_text(errors='ignore')
    check('Early stopping (effective): OFF' in txt1,
          'log reports the EFFECTIVE early-stopping state as OFF')

    # ------------------------------------------------------------ launch 2
    print('\nlaunch 2: same command, must resume and finish')
    r = subprocess.run(cmd, capture_output=True, text=True, errors='ignore')
    (out / 'launch2.log').write_text(r.stdout + r.stderr)
    txt = r.stdout + r.stderr
    check(r.returncode == 0, f'exit code {r.returncode}')
    check('Resuming at epoch' in txt, 'says it is resuming, not restarting')
    rows = csv_rows(mdir / 'training_log.csv')
    check(rows == list(range(args.epochs)),
          f'CSV has each epoch exactly once: {rows}')
    res = json.loads((mdir / 'results.json').read_text()) if (mdir / 'results.json').exists() else {}
    check(res.get('epochs_trained') == args.epochs,
          f"results.json epochs_trained = {res.get('epochs_trained')}")
    check(not (mdir / 'last.pth').exists(), 'last.pth cleaned up')
    check((mdir / 'best_model.pth').exists(), 'best_model.pth present')

    # ------------------------------------------------------------ launch 3
    print('\nlaunch 3: same command again, must skip and keep the result')
    r = subprocess.run(cmd, capture_output=True, text=True, errors='ignore')
    txt = r.stdout + r.stderr
    check(r.returncode == 0, f'exit code {r.returncode}')
    check('already finished' in txt, 'skips the finished model')
    allr = json.loads((out / exp / 'all_results.json').read_text())
    check(len(allr) == 1 and allr[0].get('display_name') == args.model
          and 'error' not in allr[0],
          f'all_results.json still holds the finished model ({len(allr)} entries)')

    print()
    if FAILS:
        print(f'{len(FAILS)} FAILED  (logs in {out})')
        for f in FAILS:
            print(f'  - {f}')
        return 1
    print(f'all checks passed  (logs in {out})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
