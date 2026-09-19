#!/usr/bin/env python3
"""Execute the notebook's own settings + helper cells against the real trainer.

Syntax checks pass on code that is still wrong. This runs the cells verbatim,
extracted from notebooks/colab_revision.ipynb as built, with only the
paths and epoch count overridden:

  * every group's dry run produces the expected model count
    (R1 9, R2 4 per seed, R3 6, R4 29, R5 68)
  * a real one-epoch group trains, exits 0, writes results.json and
    all_results.json, syncs to "Drive", and the log shows the effective
    early-stopping state
  * relaunching the same group appends to the log (two session headers) and
    skips the finished model
  * _require_mamba_fast() passes where mamba-ssm is installed

Needs rsync and a CUDA torch (Colab, Linux, WSL with .venv-wsl).
Run from the repo root:

    python scripts/test_notebook_rungroup.py
"""
import json
import os
import shutil
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


def cell(marker):
    cells = json.load(open(NB, encoding='utf-8'))['cells']
    hits = ['\n'.join(c['source']) for c in cells
            if c['cell_type'] == 'code' and marker in '\n'.join(c['source'])]
    assert len(hits) == 1, f'{len(hits)} cells contain {marker!r}'
    return hits[0]


def main():
    os.chdir(ROOT)
    tmp = Path(tempfile.mkdtemp(prefix='nb_rungroup_'))
    try:
        settings = cell('# ---- canonical settings')
        helpers = cell('def run_group(')
        # Paths and a data dir the dataset cell would have set.
        settings = (settings
                    .replace("RESULTS_DIR = '/content/results'",
                             f"RESULTS_DIR = {str(tmp / 'results')!r}")
                    .replace("DRIVE_RESULTS = '/content/drive/MyDrive/Paper1/results_revision'",
                             f"DRIVE_RESULTS = {str(tmp / 'drive')!r}"))
        assert str(tmp) in settings, 'path override did not apply'
        ns = {'__name__': '__nb__', 'os': os,
              'DATA_DIR': str(ROOT / 'data' / 'CAMUS')}
        exec(settings, ns)
        ns['NUM_WORKERS'] = 2          # this machine, not a 48-vCPU G4

        # ------------------------------------------------------------ dry runs
        print('\ndry runs (model count per group)')
        exec(helpers, ns)
        run_group = ns['run_group']
        R = Path(ns['RESULTS_DIR'])
        R.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / 'results' / 'param_config_r3.json',
                    R / 'param_config_r3.json')

        import io
        import contextlib

        def dry(name, extra):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_group(name, extra, dry_run=True)
            out = buf.getvalue()
            for line in out.splitlines():
                if line.startswith('Models to train:'):
                    return int(line.split(':')[1])
            print(out[-1500:])
            return None

        expect = [
            ('r1_canonical', ['--base_only'], 9),
            ('r2_seed1', ['--models', 'transunet', 'nnunet', 'unet_v1',
                          'unet_resnet', '--seed', '1'], 4),
            ('r3_param_matched', ['--models', 'unet_v1', 'unet_v2', 'swin_unet',
                                  'transunet', '--param_matched', '--wide_only',
                                  '--param_config', str(R / 'param_config_r3.json')], 4),
            ('r3b_param_matched_bf16', ['--models', 'dense_context_unet', 'fpn',
                                        '--param_matched', '--wide_only',
                                        '--amp_dtype', 'bfloat16',
                                        '--param_config', str(R / 'param_config_r3.json')], 2),
            ('r4_ssm_batch8', ['--mamba_only', '--mamba_variants', 'mamba',
                               'mamba2', 'vmamba'], 29),
            ('r5_position', ['--position_ablation', '--position_only',
                             '--mamba_variants', 'mamba', 'vmamba'], 68),
        ]
        for name, extra, n in expect:
            got = dry(name, extra)
            check(got == n, f'{name}: {got} models (expected {n})')

        # ------------------------------------------------------ mamba guard
        print('\n_require_mamba_fast')
        try:
            ns['_require_mamba_fast']()
            check(True, 'passes with mamba-ssm installed')
        except RuntimeError as e:
            check(False, f'raised: {str(e)[:200]}')

        # ------------------------------------------------- one real group
        print('\nreal group, 1 epoch')
        ns['EPOCHS'] = 1
        exec(helpers, ns)                  # rebuild BASE_ARGS with EPOCHS=1
        ns['SYNC_MINUTES'] = 0             # sync on every poll
        run_group = ns['run_group']
        g = 'nb_test'
        run_group(g, ['--models', 'nnunet'])
        d = R / g
        check((d / 'nnunet' / 'results.json').exists(), 'results.json written')
        allr = json.loads((d / 'all_results.json').read_text())
        check(len(allr) == 1 and 'error' not in allr[0],
              f'all_results.json: {len(allr)} entry, no error')
        log = (d / 'shard0.log').read_text(errors='ignore')
        check('Early stopping (effective): OFF' in log,
              'log shows the effective early-stopping state')
        check(log.count('===== session start') == 1, 'one session header')
        drive = Path(ns['DRIVE_RESULTS']) / g
        check((drive / 'nnunet' / 'results.json').exists()
              and (drive / 'nnunet' / 'best_model.pth').exists(),
              'results and best_model.pth synced to Drive')
        check(not (drive / 'nnunet' / 'last.pth').exists(),
              'no resume file left on Drive for a finished model')

        print('\nrelaunch the same group')
        run_group(g, ['--models', 'nnunet'])
        log = (d / 'shard0.log').read_text(errors='ignore')
        check(log.count('===== session start') == 2,
              'log appended, earlier session kept')
        check('already finished' in log, 'finished model skipped')
        allr = json.loads((d / 'all_results.json').read_text())
        check(len(allr) == 1 and 'error' not in allr[0],
              'all_results.json still holds the model')
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
