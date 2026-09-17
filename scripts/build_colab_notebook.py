#!/usr/bin/env python3
"""Generate notebooks/colab_revision.ipynb.

The notebook is a build artefact, not something to hand-edit: a .ipynb is JSON
with escaped source strings, so editing one in place is how command flags drift
out of sync with the scripts they call. Everything that matters -- the canonical
training settings, which groups exist, what each one costs -- lives here as
Python and is rendered once.

The notebook it produces replaces notebooks/colab_training.ipynb, which is not
stale so much as *actively wrong*: it is the pipeline that produced the runs the
referees objected to, with --batch_size 128 for base models against 16 for the
SSM arms (the batch confound, NEW-2) and --early_stopping 20 throughout (the
truncation defect, NEW-1). Running it would reproduce both.

Usage:
    python scripts/build_colab_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks' / 'colab_revision.ipynb'


def md(text: str) -> dict:
    return {'cell_type': 'markdown', 'metadata': {},
            'source': text.strip('\n').split('\n')}


def code(text: str) -> dict:
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': text.strip('\n').split('\n')}


CELLS = []
A = CELLS.append

# --------------------------------------------------------------------------
A(md("""
# CAMUS revision — canonical training programme

Replaces `colab_training.ipynb`. Every training call here uses **batch 8, 100
epochs, early stopping off, `--pin full`**, because those four settings are the
revision:

| setting | why | audit item |
|---|---|---|
| `--batch_size 8` everywhere | the old notebook used 128 for base and 16 for SSM, so every base-vs-SSM comparison crossed batch sizes | NEW-2 |
| `--early_stopping 0` | patience 20 fired while cosine LR was near peak: DeepLabV3+ stopped at 37 epochs and scored 0.8602; retrained to 100 it scores 0.9140 | NEW-1 |
| `--pin full` | TF32 is on by default from Ampere on, and disabling it moves FPN-UNet's EF by 0.88 points — more than the gaps between adjacent architectures | EF-1 |
| `--seed` explicit | R2 measures the seed floor the papers currently assert without evidence | SEED-1, NEW-3 |

**Do not change the batch size to fill a bigger GPU.** At batch 8 and 256 px
these models use a small fraction of a large card, and the temptation is to
raise the batch. That would reintroduce exactly the confound this revision
removes. Use `NUM_SHARDS` below instead: it runs several models side by side on
one GPU, which buys wall-clock time and changes nothing scientifically.
"""))

A(md('## 1. Setup — run once per session'))

A(code("""
from google.colab import drive
drive.mount('/content/drive')
"""))

A(code("""
# Pinned, not nightly. The EF numbers are reproducible only against a fixed
# environment: skimage 0.25.2 and 0.26.0 disagree on the connected-component
# filter, which is enough to move a patient's EF. If you must move to a newer
# torch, re-run the EF evaluation and expect the table to shift.
!pip install -q torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128
!pip install -q scikit-image==0.25.2 timm einops nibabel SimpleITK medpy
"""))

A(code("""
# mamba-ssm + causal-conv1d. Needed for R4 and R5 only; R1-R3 run without them.
!pip install -q causal-conv1d --no-build-isolation
!pip install -q mamba-ssm --no-build-isolation
"""))

A(code("""
import os
if os.path.exists('/content/mamba'):
    !cd /content/mamba && git pull
else:
    !git clone https://github.com/Seraf00/mamba.git /content/mamba
%cd /content/mamba
!pip install -q -r requirements.txt
"""))

A(code("""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRITON_F32_DEFAULT'] = 'ieee'
# Set before any CUDA work in a child process; --pin full sets it too, but the
# notebook's own probes below should agree with the training runs.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import warnings; warnings.filterwarnings('ignore')
"""))

A(md('## 2. Dataset'))

A(code("""
SOURCE = '/content/drive/MyDrive/CAMUS_public/database_nifti/'
DATA_DIR = '/content/mamba/data/CAMUS'
os.makedirs(DATA_DIR, exist_ok=True)
if len(os.listdir(DATA_DIR)) < 100:
    !rsync -ah --info=progress2 "{SOURCE}" "{DATA_DIR}"
else:
    print(f'CAMUS already present ({len(os.listdir(DATA_DIR))} items)')
"""))

A(md("""
## 3. Canonical settings and the concurrency helper

`NUM_SHARDS` is the one knob to tune to your GPU. Each shard is an independent
training process taking every Nth model of the same plan, so they split the work
without any model being trained twice.

Sizing it: a typical model holds ~3 GB at batch 8 and DenseContextU-Net holds
18.9 GB, so VRAM is rarely the limit — **vCPU is**. Each shard spawns
`NUM_WORKERS` dataloader processes, so `NUM_SHARDS * (NUM_WORKERS + 1)` should
stay under the machine's core count. On a 12-vCPU Colab runtime that means 3
shards at 3 workers; on a 32+ vCPU instance, 8 shards at 3.
"""))

A(code("""
import subprocess, sys, time, json, glob
from pathlib import Path

# ---- canonical settings: do not vary these between groups ----
EPOCHS      = 100
BATCH_SIZE  = 8      # every group, no exceptions -- this is NEW-2's fix
EARLY_STOP  = 0      # disabled, not "large" -- this is NEW-1's fix
PIN         = 'full' # TF32 off; see utils.misc.pin_determinism
IMG_SIZE    = 256

# ---- machine-dependent ----
NUM_SHARDS  = 3      # concurrent training processes on the one GPU
NUM_WORKERS = 3      # dataloader workers PER SHARD
RESULTS_DIR = '/content/results'
DRIVE_RESULTS = '/content/drive/MyDrive/Paper1/results'

import torch, multiprocessing
print(f'GPU        : {torch.cuda.get_device_name(0)}')
print(f'VRAM       : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB')
print(f'vCPU       : {multiprocessing.cpu_count()}')
print(f'torch      : {torch.__version__}  cuda {torch.version.cuda}')
print()
print(f'shards x (workers+1) = {NUM_SHARDS * (NUM_WORKERS + 1)} processes '
      f'against {multiprocessing.cpu_count()} vCPU')
if NUM_SHARDS * (NUM_WORKERS + 1) > multiprocessing.cpu_count():
    print('  ^ oversubscribed: lower NUM_SHARDS or NUM_WORKERS')
"""))

A(code("""
BASE_ARGS = [
    '--data_dir', DATA_DIR,
    '--output_dir', RESULTS_DIR,
    '--epochs', str(EPOCHS),
    '--batch_size', str(BATCH_SIZE),
    '--early_stopping', str(EARLY_STOP),
    '--pin', PIN,
    '--num_workers', str(NUM_WORKERS),
    '--img_size', str(IMG_SIZE),
    '--mixed_precision',
    '--skip_benchmark',
]


def _models_done(exp_dir):
    \"\"\"Models finished so far, across shards.

    A shard rewrites its result file after every model, so a read can land
    mid-write and see truncated JSON. That is a progress display, not a
    correctness check -- swallow it and try again next tick.
    \"\"\"
    n = 0
    for f in exp_dir.glob('all_results_shard*.json'):
        try:
            n += len(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    return n


def run_group(exp_name, extra_args, shards=None, dry_run=False):
    \"\"\"Train one group across `shards` concurrent processes, then merge.

    Returns the experiment directory. Each shard streams to its own log file;
    a failure in one shard does not stop the others, and the merge reports
    which models errored rather than silently dropping them.
    \"\"\"
    shards = shards or NUM_SHARDS
    args = BASE_ARGS + ['--exp_name', exp_name] + [str(a) for a in extra_args]
    exp_dir = Path(RESULTS_DIR) / exp_name

    if dry_run:
        out = subprocess.run([sys.executable, 'scripts/train_all_models.py',
                              *args, '--dry_run'], capture_output=True, text=True)
        print(out.stdout[-3000:] or out.stderr[-3000:])
        return exp_dir

    exp_dir.mkdir(parents=True, exist_ok=True)
    procs, logs = [], []
    for s in range(shards):
        log = open(exp_dir / f'shard{s}.log', 'w')
        logs.append(log)
        procs.append(subprocess.Popen(
            [sys.executable, 'scripts/train_all_models.py', *args,
             '--num_shards', str(shards), '--shard', str(s)],
            stdout=log, stderr=subprocess.STDOUT))
    print(f'{exp_name}: launched {shards} shards -> {exp_dir}/shard*.log')

    t0 = time.time()
    while any(p.poll() is None for p in procs):
        alive = sum(p.poll() is None for p in procs)
        print(f'\\r  {(time.time()-t0)/60:6.1f} min | {alive} shards running | '
              f'{_models_done(exp_dir)} models done', end='')
        time.sleep(60)
    print()

    for log in logs:
        log.close()
    codes = [p.returncode for p in procs]
    print(f'  shard exit codes: {codes}')

    subprocess.run([sys.executable, 'scripts/train_all_models.py',
                    '--data_dir', DATA_DIR, '--output_dir', RESULTS_DIR,
                    '--exp_name', exp_name, '--merge_shards'], check=True)
    return exp_dir


def save_to_drive():
    os.makedirs(DRIVE_RESULTS, exist_ok=True)
    !rsync -ah --info=progress2 \\
        --include='*/' --include='best_model.pth' --include='*.json' \\
        --include='*.csv' --include='*.log' --exclude='*' \\
        {RESULTS_DIR}/ {DRIVE_RESULTS}/
"""))

A(md("""
Check the plan before spending hours on it. `dry_run=True` prints the model list
and settings without training.
"""))

A(code("""
run_group('r1_canonical', ['--base_only'], dry_run=True)
"""))

A(md("""
---
## R1 — canonical baselines

Nine base models, batch 8, 100 epochs, no early stopping. This is the session
every other group is compared against, and it is what makes one batch size
canonical. Unblocks R2, R3 and R4.
"""))

A(code("""
run_group('r1_canonical', ['--base_only'])
save_to_drive()
"""))

A(md("""
---
## R2 — seed triplets

Two further seeds for the four models the papers lean on. Seed 42 comes free
from R1, so this is seeds 1 and 2 only.

This replaces the current "replicate spread" claim, which rests on a
contaminated pair: `unet_resnet_wide` early-stopped at epoch 54 against
`unet_resnet`'s 95, so its 0.0117 spread measures truncation, not seed noise.
The two pairs that both ran near 100 epochs give 0.0012 and 0.0011.
"""))

A(code("""
SEED_MODELS = ['transunet', 'nnunet', 'unet_v1', 'unet_resnet']

for seed in (1, 2):
    run_group(f'r2_seed{seed}',
              ['--models', *SEED_MODELS, '--seed', seed],
              shards=min(NUM_SHARDS, len(SEED_MODELS)))
save_to_drive()
"""))

A(md("""
---
## R3 — rebuilt parameter-matched controls

The shipped `param_config.json` had seven entries, of which three were **no-ops**
— the "widened" override equalled the model's own default, so `unet_resnet_wide`
(resnet34), `deeplab_v3_wide` (resnet50) and `nnunet_wide` (bf=32) were the
baseline under a second name — and `fpn_wide` missed its target by 57%.

`param_config_r3.json` fixes both and drops the three, because for those models
the control answers nothing: nnU-Net's Mamba variant is **smaller** than its
baseline (−1.9%), DeepLabV3+'s is +4.1%, UNet-ResNet's +11.1%. There are no
extra parameters to attribute a gain to. That is a sentence in the paper, not a
training run.

It also adds the two controls that were missing where the gap is largest:

| control | widening | matched | vs target |
|---|---|---|---|
| `transunet_wide` | `vit_layers=26` | 201.3 M | −0.44% |
| `fpn_wide` | `resnet101, fpn_channels=1024` | 169.1 M | −2.29% |
| `swin_unet_wide` | `embed_dim=114` | 59.0 M | −4.67% |
| `unet_v1_wide` | `bf=96` | 69.8 M | +1.67% |
| `unet_v2_wide` | `bf=104` | 87.1 M | +4.91% |
| `dense_context_unet_wide` | `bf=200` | 5.3 M | −1.86% |

TransUNet widens by **depth, not width**. Width lands equally close on
parameters (`vit_dim=1128` is −0.56%) but makes every ViT-B/16 tensor the wrong
shape, so the control would be a randomly initialised transformer competing
against a pretrained one — the extra parameters would not be what the comparison
measured. Depth keeps the first 12 blocks pretrained and adds fresh ones, which
is how the Mamba variant adds its capacity too.
"""))

A(code("""
# Regenerate rather than trusting the file in git -- the parameter counts are
# derived from the models as they are now.
!python scripts/param_match.py --mamba_type mamba \\
    --output_json {RESULTS_DIR}/param_config_r3.json

run_group('r3_param_matched',
          ['--base_only', '--param_matched',
           '--param_config', f'{RESULTS_DIR}/param_config_r3.json'])
save_to_drive()
"""))

A(md("""
---
## R4 — SSM arms at batch 8

Mamba, Mamba-2 and VMamba variants, all at the canonical batch. Without this,
Paper D's base-vs-SSM comparison still crosses batch sizes.

Two things will fail here and both are results, not bugs:
`mamba_dense_context_unet` + VMamba is skipped as architecturally incompatible
(>12 GiB backward spike), and the Mamba-2 arms whose padded head dimension pushes
the Triton chunk-scan past the shared-memory ceiling will raise. The failure
table is generated from what this session records, so let them fail.
"""))

A(code("""
run_group('r4_ssm_batch8',
          ['--mamba_only', '--mamba_variants', 'mamba', 'mamba2', 'vmamba'])
save_to_drive()
"""))

A(md("""
---
## R5 — SSM position ablation

68 arms: one per single position plus an all-off arm, over the nine models that
expose position flags, for 1D Mamba and VMamba. `--position_only` trains just
the arms — the all-on models come from R4 and the baselines from R1.

Uses Paper 1's production flags (`POSITION_SETS` in `train_all_models.py`), so
the ablated models are the models the papers report.
**Not** `Paper2/models/ablation_models.py`, which is a separate
`ConfigurableMambaUNet` appearing in no leaderboard.

Decoder is the thin cell: only `mamba_swin_unet`, `mamba_deeplab` and
`mamba_fpn` have one, so a bottleneck-vs-decoder claim rests on three
architectures, not nine. Say that rather than averaging a ragged grid.
"""))

A(code("""
run_group('r5_position',
          ['--position_ablation', '--position_only',
           '--mamba_variants', 'mamba', 'vmamba'])
save_to_drive()
"""))

A(md("""
---
## Evaluation

Inference only. `eval_baseline_ef.py` computes EF at **native resolution** —
the defect it fixes was computing volumes on the 256 px grid, which is EF-1.
"""))

A(code("""
SESSIONS = ['r1_canonical', 'r3_param_matched', 'r4_ssm_batch8', 'r5_position']

for s in SESSIONS:
    d = Path(RESULTS_DIR) / s
    if not d.exists():
        print(f'skip {s} (not trained)'); continue
    !python scripts/yolo/eval_baseline_ef.py \\
        --checkpoint-dir {d} --data-dir {DATA_DIR} --pin full \\
        --out {d}/baseline_ef_native.json
"""))

A(code("""
# Per-patient EF arrays (Bland-Altman), quality-stratified Dice, overlay grids.
for s in SESSIONS:
    d = Path(RESULTS_DIR) / s
    if not d.exists():
        continue
    !python scripts/colab_session.py \\
        --checkpoint_dir {d} --data_dir {DATA_DIR} \\
        --out_dir {RESULTS_DIR}/session_out/{s}
"""))

A(code("""
# Per-model test metrics: Dice, IoU, HD95, ASSD.
for s in SESSIONS:
    d = Path(RESULTS_DIR) / s
    if not d.exists():
        continue
    !python scripts/evaluate_all_models.py \\
        --checkpoint_dir {d} --data_dir {DATA_DIR}

save_to_drive()
"""))

A(md("""
---
## Tables

Because the consistency contract holds — every number in all four manuscripts
comes from a generator reading a named artefact — this is minutes, and
`check_number_provenance.py --strict` fails the build if anything was
hand-typed back in.
"""))

A(code("""
!python scripts/fill_tables.py --results_root {RESULTS_DIR} \\
    --benchmark_csv {RESULTS_DIR}/benchmark_efficiency.csv
!python scripts/yolo/make_journal_tables.py
!python scripts/check_number_provenance.py --strict
"""))

A(md("""
---
## Recovery

Shards are independent processes, so a disconnect loses only the models that
were mid-training. Re-running a group skips nothing automatically — use
`--resume_from`, or re-run with `--models` naming what is missing.
"""))

A(code("""
# What actually completed, per group.
for s in SESSIONS:
    p = Path(RESULTS_DIR) / s / 'all_results.json'
    shards = sorted((Path(RESULTS_DIR) / s).glob('all_results_shard*.json'))
    if p.exists():
        r = json.load(open(p))
        ok = [x for x in r if 'error' not in x]
        print(f'{s:20s} merged: {len(ok)} ok, {len(r)-len(ok)} failed')
    elif shards:
        n = sum(len(json.load(open(f))) for f in shards)
        print(f'{s:20s} UNMERGED: {n} models across {len(shards)} shards')
    else:
        print(f'{s:20s} not started')
"""))

A(code("""
# Merge a group whose shards finished but whose driver cell was interrupted.
# GROUP = 'r4_ssm_batch8'
# !python scripts/train_all_models.py --data_dir {DATA_DIR} \\
#     --output_dir {RESULTS_DIR} --exp_name {GROUP} --merge_shards
"""))

# --------------------------------------------------------------------------
nb = {
    'cells': CELLS,
    'metadata': {
        'accelerator': 'GPU',
        'colab': {'provenance': [], 'toc_visible': True},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4,
    'nbformat_minor': 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

n_code = sum(c['cell_type'] == 'code' for c in CELLS)
print(f'Wrote {OUT} -- {len(CELLS)} cells ({n_code} code, '
      f'{len(CELLS) - n_code} markdown)')
