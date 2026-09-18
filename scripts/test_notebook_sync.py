#!/usr/bin/env python3
"""Run the notebook's own save_to_drive / restore_from_drive against a fake tree.

The code under test is extracted from notebooks/colab_revision.ipynb as built,
not copied, so this fails if the notebook drifts. Needs rsync (Colab, Linux,
WSL). Checks the disconnect path end to end:

  * the right files reach Drive (resume state and results in; final_model,
    periodic checkpoints and TensorBoard out)
  * a finished model's last.pth does not linger on Drive
  * after the local disk is wiped, restore brings back exactly what --resume
    needs, and reports only genuinely interrupted models as resumable
  * a save on an empty local disk (forgot to restore) deletes nothing on Drive

    python3 scripts/test_notebook_sync.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / 'notebooks' / 'colab_revision.ipynb'
FAILS = []


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILS.append(msg)


def extract():
    """Pull _SYNC_FILTER, save_to_drive and restore_from_drive out of the cell."""
    cells = json.load(open(NB, encoding='utf-8'))['cells']
    src = next('\n'.join(c['source']) for c in cells
               if c['cell_type'] == 'code' and 'def save_to_drive' in ''.join(c['source']))
    start = src.index('_SYNC_FILTER = [')
    return src[start:]


def files(d):
    return sorted(str(p.relative_to(d)).replace('\\', '/')
                  for p in Path(d).rglob('*') if p.is_file())


def main():
    if not shutil.which('rsync'):
        print('rsync not found -- run under Colab, Linux or WSL')
        return 2
    code = extract()
    tmp = Path(tempfile.mkdtemp())
    R, D = tmp / 'results', tmp / 'drive'
    ns = {'os': __import__('os'), 'subprocess': subprocess, 'Path': Path,
          'RESULTS_DIR': str(R), 'DRIVE_RESULTS': str(D), 'print': print}
    exec(code, ns)
    save, restore = ns['save_to_drive'], ns['restore_from_drive']

    try:
        g = R / 'r1_canonical'
        fin, run = g / 'unet_v1', g / 'nnunet'
        for d in (fin, run, fin / 'logs'):
            d.mkdir(parents=True, exist_ok=True)
        # finished model: results.json, best, final; its last.pth already gone
        for f in ('best_model.pth', 'final_model.pth', 'results.json',
                  'training_log.csv', 'checkpoint_epoch_10.pth'):
            (fin / f).write_text('x')
        (fin / 'logs' / 'events.tfevents').write_text('x')
        # interrupted model: best + last, no results.json
        for f in ('best_model.pth', 'last.pth', 'training_log.csv'):
            (run / f).write_text('x')
        (g / 'shard0.log').write_text('x')
        (g / 'experiment_config.json').write_text('{}')

        # Simulate the finished model having had a last.pth synced earlier.
        (D / 'r1_canonical' / 'unet_v1').mkdir(parents=True)
        (D / 'r1_canonical' / 'unet_v1' / 'last.pth').write_text('stale')

        print('save_to_drive')
        save(quiet=True)
        on = files(D)
        check('r1_canonical/nnunet/last.pth' in on, 'interrupted model: last.pth on Drive')
        check('r1_canonical/unet_v1/results.json' in on, 'finished model: results.json on Drive')
        check('r1_canonical/unet_v1/last.pth' not in on,
              "finished model's stale last.pth removed from Drive")
        check(not any('final_model' in f or 'checkpoint_epoch' in f or 'tfevents' in f
                      for f in on), 'final_model / periodic checkpoints / TensorBoard stay off Drive')

        print('\nnew session: local disk empty, save BEFORE restore (the mistake)')
        shutil.rmtree(R)
        R.mkdir()
        before = files(D)
        save(quiet=True)
        check(files(D) == before, 'saving an empty local disk deletes nothing on Drive')

        print('\nrestore_from_drive')
        restore()
        back = files(R)
        check('r1_canonical/nnunet/last.pth' in back, 'resume state restored')
        check('r1_canonical/unet_v1/results.json' in back, 'finished marker restored')
        check('r1_canonical/unet_v1/last.pth' not in back,
              'finished model does not come back looking resumable')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILS:
        print(f'{len(FAILS)} FAILED')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
