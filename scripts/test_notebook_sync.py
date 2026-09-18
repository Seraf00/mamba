#!/usr/bin/env python3
"""Run the notebook's own Drive helpers against a fake results tree.

The code under test is extracted from notebooks/colab_revision.ipynb as built,
not copied, so this fails if the notebook drifts. Needs rsync (Colab, Linux,
WSL).

Sync / restore
  * the right files reach Drive (resume state, results, evaluation outputs in;
    final_model, periodic checkpoints and TensorBoard out)
  * a finished model's last.pth does not linger on Drive
  * saving an empty local disk (forgot to restore) deletes nothing on Drive
  * restore brings back what --resume needs, and reports only real interruptions

Offload (moving a finished group to the PC to free Drive)
  * offload_check refuses a group still training or not yet evaluated
  * it writes MANIFEST.json and syncs it
  * verify_offload.py passes on a faithful download and fails on a corrupt one
  * offload() without the confirmation flag deletes nothing
  * with it: checkpoints leave Drive, small files stay, the group is marked
  * THE KEY PROPERTY: the next sync does not re-upload the offloaded weights

Failure visibility
  * a failed sync returns False and sets the status-line warning

    python3 scripts/test_notebook_sync.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / 'notebooks' / 'colab_revision.ipynb'
FAILS = []


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILS.append(msg)


def extract():
    cells = json.load(open(NB, encoding='utf-8'))['cells']
    src = next('\n'.join(c['source']) for c in cells
               if c['cell_type'] == 'code' and 'def save_to_drive' in ''.join(c['source']))
    return src[src.index('_BASE_FILTER = ['):]


def files(d):
    return sorted(str(p.relative_to(d)).replace('\\', '/')
                  for p in Path(d).rglob('*') if p.is_file())


def load(R, D):
    ns = {'os': os, 'subprocess': subprocess, 'Path': Path, 'json': json,
          'time': time, 'shutil': shutil, 'hashlib': hashlib, 'sys': sys,
          'RESULTS_DIR': str(R), 'DRIVE_RESULTS': str(D), 'print': print}
    exec(extract(), ns)
    return ns


def main():
    if not shutil.which('rsync'):
        print('rsync not found -- run under Colab, Linux or WSL')
        return 2
    tmp = Path(tempfile.mkdtemp())
    R, D, PC = tmp / 'results', tmp / 'drive', tmp / 'pc'
    ns = load(R, D)
    save, restore = ns['save_to_drive'], ns['restore_from_drive']

    try:
        # ------------------------------------------------------ sync / restore
        g = R / 'r1_canonical'
        fin, run = g / 'unet_v1', g / 'nnunet'
        for d in (fin, run, fin / 'logs', g / 'evaluation'):
            d.mkdir(parents=True, exist_ok=True)
        for f in ('best_model.pth', 'final_model.pth', 'results.json',
                  'training_log.csv', 'checkpoint_epoch_10.pth'):
            (fin / f).write_text('x' * 1000)
        (fin / 'logs' / 'events.tfevents').write_text('x')
        for f in ('best_model.pth', 'last.pth', 'training_log.csv'):
            (run / f).write_text('y' * 1000)
        (g / 'shard0.log').write_text('x')
        (g / 'evaluation' / 'results_table.tex').write_text('x')
        (g / 'evaluation' / 'bland_altman.png').write_text('x')
        (D / 'r1_canonical' / 'unet_v1').mkdir(parents=True)
        (D / 'r1_canonical' / 'unet_v1' / 'last.pth').write_text('stale')

        print('save_to_drive')
        check(save(quiet=True) is True, 'returns True on success')
        on = files(D)
        check('r1_canonical/nnunet/last.pth' in on, 'interrupted model: last.pth on Drive')
        check('r1_canonical/unet_v1/results.json' in on, 'finished model: results.json on Drive')
        check('r1_canonical/evaluation/results_table.tex' in on
              and 'r1_canonical/evaluation/bland_altman.png' in on,
              'evaluation outputs (.tex, .png) reach Drive')
        check('r1_canonical/unet_v1/last.pth' not in on,
              "finished model's stale last.pth removed from Drive")
        check(not any('final_model' in f or 'checkpoint_epoch' in f or 'tfevents' in f
                      for f in on), 'final_model / periodic checkpoints / TensorBoard stay off Drive')

        print('\nnew session: empty local disk, save BEFORE restore')
        shutil.rmtree(R); R.mkdir()
        before = files(D)
        save(quiet=True)
        check(files(D) == before, 'deletes nothing on Drive')

        print('\nrestore_from_drive')
        restore()
        back = files(R)
        check('r1_canonical/nnunet/last.pth' in back, 'resume state restored')
        check('r1_canonical/unet_v1/last.pth' not in back,
              'finished model does not come back looking resumable')

        # --------------------------------------------------------------- offload
        print('\noffload_check on a group that is not ready')
        check(ns['offload_check']('r1_canonical') is False,
              'refuses: nnunet still training, group not evaluated')

        # make it a finished, evaluated group
        (g / 'nnunet' / 'last.pth').unlink()
        (g / 'nnunet' / 'results.json').write_text('{}')
        (g / 'all_results.json').write_text('[]')
        (g / 'evaluation' / 'evaluation_results.json').write_text('{}')
        (g / 'baseline_ef_native.json').write_text('{}')
        save(quiet=True)

        print('\noffload_check on a finished, evaluated group')
        check(ns['offload_check']('r1_canonical') is True, 'accepts')
        check((D / 'r1_canonical' / 'MANIFEST.json').exists(), 'MANIFEST.json synced to Drive')

        print('\nverify_offload.py on the PC copy')
        shutil.copytree(D / 'r1_canonical', PC / 'r1_canonical')
        v = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'verify_offload.py'),
                            str(PC / 'r1_canonical')], capture_output=True, text=True)
        check(v.returncode == 0 and 'ALL VERIFIED' in v.stdout, 'faithful download: ALL VERIFIED')
        bad = PC / 'r1_canonical' / 'nnunet' / 'best_model.pth'
        bad.write_text('y' * 999 + 'z')              # same size, one byte changed
        v = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'verify_offload.py'),
                            str(PC / 'r1_canonical')], capture_output=True, text=True)
        check(v.returncode != 0 and 'CORRUPT' in v.stdout, 'corrupted byte: refused')
        (PC / 'r1_canonical' / 'unet_v1' / 'best_model.pth').unlink()
        v = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'verify_offload.py'),
                            str(PC / 'r1_canonical')], capture_output=True, text=True)
        check(v.returncode != 0 and 'MISSING' in v.stdout, 'missing file: refused')

        print('\noffload()')
        ns['offload']('r1_canonical')                 # no confirmation
        check(any(f.endswith('.pth') for f in files(D / 'r1_canonical')),
              'without the confirmation flag nothing is deleted')
        ns['offload']('r1_canonical', i_have_verified_the_download=True)
        dg = files(D / 'r1_canonical')
        check(not any(f.endswith('.pth') for f in dg), 'checkpoints removed from Drive')
        check('unet_v1/results.json' in dg and 'evaluation/evaluation_results.json' in dg,
              'small files stay on Drive (resume needs results.json)')
        check('OFFLOADED.json' in dg, 'group marked offloaded on Drive')
        check(ns['_offloaded']() == ['r1_canonical'], '_offloaded() reports it')

        print('\nTHE KEY PROPERTY: next sync')
        check(save(quiet=True) is True, 'sync still succeeds')
        check(not any(f.endswith('.pth') for f in files(D / 'r1_canonical')),
              'offloaded weights are NOT re-uploaded (still on local disk)')
        check((R / 'r1_canonical' / 'unet_v1' / 'best_model.pth').exists(),
              'local copy untouched until the runtime ends')

        print('\nrestore in a later session')
        shutil.rmtree(R); R.mkdir()
        restore()
        check((R / 'r1_canonical' / 'unet_v1' / 'results.json').exists()
              and (R / 'r1_canonical' / 'OFFLOADED.json').exists(),
              'results and the offload mark come back')

        # --------------------------------------------------------- sync failure
        print('\nfailed sync')
        ns2 = load(R, tmp / 'readonly' / 'drive')
        (tmp / 'readonly').mkdir()
        os.chmod(tmp / 'readonly', 0o555)
        try:
            ok = ns2['save_to_drive'](quiet=True)
        except PermissionError:
            ok = 'raised'
        check(ok is False, f'returns False rather than raising or passing ({ok})')
        check(ns2['_SYNC_ERR'] is not None, 'status-line warning is set')
        os.chmod(tmp / 'readonly', 0o755)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILS:
        print(f'{len(FAILS)} FAILED')
        for f in FAILS:
            print(f'  - {f}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
