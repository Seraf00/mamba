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
| `--early_stopping 0` | with it on, every model gets a different training budget, so architecture is confounded with when the patience counter fired. Patience 20 against a 100-epoch cosine schedule hit 20 of 48 runs. Test Dice: DeepLabV3+ stopped at 37 epochs scores 0.8602, the same model to 100 epochs 0.9140 (+0.054); five models that ran full length in both sessions differ by -0.007 to +0.006 | NEW-1 |
| `--pin full` | TF32 is on by default from Ampere on, and disabling it moves FPN-UNet's EF by 0.88 points — more than the gaps between adjacent architectures | EF-1 |
| `--seed` explicit | R2 measures the seed floor the papers currently assert without evidence | SEED-1, NEW-3 |

**Do not change the batch size to fill a bigger GPU.** At batch 8 and 256 px
these models use a small fraction of a large card, and the temptation is to
raise the batch. That would reintroduce exactly the confound this revision
removes. Run one job at a time instead: measured on an RTX PRO 6000 Blackwell
(188 SMs, 95 GB), eight concurrent jobs gave 1.06x the throughput of one,
because separate CUDA processes time-slice the GPU rather than run side by side.
A single job there does 45 it/s, and R1-R5 comes to roughly 49 hours.
"""))

A(md('## 1. Setup — run once per session'))

A(code("""
from google.colab import drive
drive.mount('/content/drive')
"""))

A(md("""
### Environment — pin it, and pin the version the shootout actually validated

CUDA itself is not installable here: Colab's driver and toolkit are fixed. What
you choose is the PyTorch wheel's CUDA build, and **cu128 is the only one that
covers all three candidate GPUs** — its arch list is
`sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`, i.e. A100 (sm_80), H100 (sm_90)
and Blackwell (sm_120). It is also what every EF number on disk was measured
against (torch 2.9.1 / 2.10.0, cuda 12.8).

Set `TORCH_SPEC = None` to keep Colab's preinstalled torch — that is the
combination most likely to have a matching prebuilt `mamba-ssm` wheel, and a
source build costs 30–60 minutes and may fail. Run the shootout
(`scripts/colab_gpu_shootout_cell.py`) first, see which works, then **hard-pin
that version here and never change it mid-programme**. Colab updates its default
torch silently; a programme spanning weeks would otherwise straddle two builds,
which is the same class of confound as mixing GPUs.
"""))

A(code("""
# None = keep Colab's preinstalled torch (pin the value once the shootout says
# which stack gives a working mamba-ssm fast path).
# Validated on Colab G4 (RTX PRO 6000 Blackwell, sm_120) on 2026-09-18 with
# torch 2.11.0+cu128, Triton 3.6.0, mamba-ssm 2.3.2.post1: full kernel stack
# runs and the Mamba-2 {4 train, 6 fail} partition replicates T10. Pinned so a
# silent Colab update cannot put R1 and R4 on different builds. If Colab
# already has this version, pip does nothing.
TORCH_SPEC = 'torch==2.11.0'

if TORCH_SPEC:
    !pip install -q {TORCH_SPEC} torchvision --index-url https://download.pytorch.org/whl/cu128

# skimage is pinned regardless of torch: 0.25.2 and 0.26.0 disagree on the
# connected-component filter, which is enough to move a patient's EF.
!pip install -q scikit-image==0.25.2 timm einops nibabel SimpleITK medpy

import torch
print(f'torch {torch.__version__}  cuda {torch.version.cuda}')
print(f'built for: {" ".join(torch.cuda.get_arch_list())}')
print('RECORD THIS and use the same stack for every group.')
"""))

A(code("""
# mamba-ssm + causal-conv1d. Needed for R4/R5 only; R1-R3 run without them.
# If this falls back to a source build, it is slow and may fail -- that is the
# signal to reconsider the GPU rather than to wait it out.
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
## 3. Canonical settings

**`NUM_SHARDS = 1`, and leave it there** unless the shootout cell measures
otherwise on your card. Sharding runs several training processes on one GPU, and
it was expected to be the main speed lever on a large card. Measured, it is not:

| card | 1 job | 8 jobs | efficiency |
|---|---|---|---|
| RTX 4060 Laptop (24 SMs) | 3.64 it/s | — (2 jobs: 2.76) | 38% at 2 |
| RTX PRO 6000 Blackwell (188 SMs) | 45.03 it/s | 47.73 it/s | 13% |

Without NVIDIA's MPS daemon, separate CUDA processes time-slice the GPU instead
of overlapping, so extra shards add contention and almost no throughput. The
machinery stays because it is harmless at 1 and would matter if MPS were
available; the default is what was measured.
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
NUM_SHARDS  = 1      # measured: extra shards time-slice, ~1.06x at 8
NUM_WORKERS = 8      # one job, so it can have more of the machine's cores
RESULTS_DIR = '/content/results'
# Its own folder: the old sessions lived in Paper1/results, and restoring from
# there would pull the pre-revision runs back onto the training disk.
DRIVE_RESULTS = '/content/drive/MyDrive/Paper1/results_revision'

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
    # Disconnect safety. --resume skips models that already have results.json
    # and continues an interrupted one from last.pth, written every 5 epochs --
    # the most a disconnect can cost. Periodic checkpoint_epoch_N.pth files are
    # off: nothing reads them and each carries full optimizer state.
    '--resume',
    '--resume_every', '5',
    '--checkpoint_every', '0',
]

SYNC_MINUTES = 10   # copy progress to Drive this often while training


def _models_done(exp_dir):
    \"\"\"Models finished so far, across shards.

    A shard rewrites its result file after every model, so a read can land
    mid-write and see truncated JSON. That is a progress display, not a
    correctness check -- swallow it and try again next tick.
    \"\"\"
    n = 0
    files = list(exp_dir.glob('all_results_shard*.json'))
    if not files and (exp_dir / 'all_results.json').exists():
        files = [exp_dir / 'all_results.json']     # single-shard mode
    for f in files:
        try:
            n += len(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    return n


def _current(exp_dir):
    # Last "Training: X" and "Epoch N/M |" lines across the shard logs. Plain
    # string parsing, no regex: this code lives inside a string in the builder,
    # and every backslash there needs escaping twice.
    model, epoch = '', ''
    for log in sorted(exp_dir.glob('shard*.log')):
        try:
            lines = log.read_text(errors='ignore')[-20000:].splitlines()
        except OSError:
            continue
        for ln in lines:
            if ln.startswith('Training: '):
                model = ln[len('Training: '):].strip()
            elif ln.startswith('Epoch ') and ' | ' in ln:
                epoch = ln.split()[1]
    return model, epoch


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

    t0 = last_sync = time.time()
    while any(p.poll() is None for p in procs):
        alive = sum(p.poll() is None for p in procs)
        model, epoch = _current(exp_dir)
        print(f'\\r  {(time.time()-t0)/60:6.1f} min | {_models_done(exp_dir)} done '
              f'| now: {model} epoch {epoch}' + ' ' * 10, end='')
        if time.time() - last_sync > SYNC_MINUTES * 60:
            save_to_drive(quiet=True)
            last_sync = time.time()
        time.sleep(60)
    print()
    save_to_drive(quiet=True)

    for log in logs:
        log.close()
    codes = [p.returncode for p in procs]
    print(f'  shard exit codes: {codes}')

    # With one shard the trainer writes all_results.json itself; merging would
    # find no shard files and fail. Only a sharded run needs it.
    if shards > 1:
        subprocess.run([sys.executable, 'scripts/train_all_models.py',
                        '--data_dir', DATA_DIR, '--output_dir', RESULTS_DIR,
                        '--exp_name', exp_name, '--merge_shards'], check=True)
        save_to_drive(quiet=True)

    if any(c != 0 for c in codes):
        tail = (exp_dir / 'shard0.log').read_text(errors='ignore')[-3000:]
        print('  a shard exited with an error -- last lines of its log:')
        print(tail)
    return exp_dir


# What survives a disconnect. last.pth is the resume state and must be here;
# final_model.pth, checkpoint_epoch_*.pth and TensorBoard logs are read by
# nothing downstream and would only fill Drive.
_SYNC_FILTER = ['--include=*/', '--include=best_model.pth', '--include=last.pth',
                '--include=*.json', '--include=*.csv', '--include=*.log',
                '--include=*.jsonl', '--exclude=*']


def save_to_drive(quiet=False):
    os.makedirs(DRIVE_RESULTS, exist_ok=True)
    r = subprocess.run(['rsync', '-a', *_SYNC_FILTER,
                        f'{RESULTS_DIR}/', f'{DRIVE_RESULTS}/'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f'\\n  [sync] rsync failed: {r.stderr[-300:]}')
        return
    # rsync never deletes, so a finished model's last.pth -- removed locally
    # when training completes -- would stay on Drive for good: full optimizer
    # state, up to ~3 GB per model. Not rsync --delete: on a fresh session run
    # before restore_from_drive(), that would wipe Drive. Instead remove only
    # resume files whose model has a results.json, i.e. is certainly finished.
    for lp in Path(DRIVE_RESULTS).glob('*/*/last.pth'):
        if (lp.parent / 'results.json').exists():
            lp.unlink()
    if not quiet:
        print(f'synced {RESULTS_DIR} -> {DRIVE_RESULTS}')


def restore_from_drive():
    # Bring previous sessions' progress back onto local disk. Run at the start
    # of every session: harmless on the first (Drive is empty); after a
    # disconnect it restores results.json for finished models and last.pth for
    # the interrupted one, which is what --resume reads.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.isdir(DRIVE_RESULTS):
        print('nothing on Drive yet -- fresh start')
        return
    subprocess.run(['rsync', '-a', *_SYNC_FILTER,
                    f'{DRIVE_RESULTS}/', f'{RESULTS_DIR}/'], check=True)
    done = sorted(Path(RESULTS_DIR).glob('*/*/results.json'))
    partial = sorted(pth for pth in Path(RESULTS_DIR).glob('*/*/last.pth')
                     if not (pth.parent / 'results.json').exists())
    print(f'restored from Drive: {len(done)} finished models, '
          f'{len(partial)} interrupted mid-training')
    for pth in partial:
        print(f'  will resume: {pth.parent.parent.name}/{pth.parent.name}')
"""))

A(md("""
## 4. Start of every session

Run these three cells every time, including after a disconnect.

1. **Restore** pulls earlier progress back from Drive: `results.json` for every
   finished model and `last.pth` for the one that was interrupted.
2. **Preflight** checks the machine. It exits with an error on anything that
   would silently ruin a run: missing scikit-image (every EF becomes `nan`),
   mamba-ssm without its CUDA kernels (~100x slower), and so on.
3. **Dry run** prints the plan without training.

Once R1 starts, check the first model's log (`results/r1_canonical/shard0.log`)
for `Early stopping (effective): OFF`. That line comes from the config the
trainer actually receives. For the first half of this revision the flag said
"disabled" while the trainer's own counter still stopped runs at patience 20.
"""))

A(code("""
restore_from_drive()
"""))

A(code("""
!python scripts/preflight_colab.py --data-dir {DATA_DIR} --results-dir {RESULTS_DIR} \\
    --shards {NUM_SHARDS} --workers {NUM_WORKERS} --need-mamba
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
## If training stops

**Nothing to edit. Re-run the same cells.**

1. Reconnect (Runtime > Connect). Choose the **same GPU type**: every group must
   run on one card, or base-vs-SSM is confounded with hardware.
2. Run section 1 (setup), section 2 (dataset) and section 3 (settings) again.
3. Run section 4. `restore_from_drive()` reports what it found, e.g.
   `restored from Drive: 5 finished models, 1 interrupted mid-training`.
4. Re-run **the group cell that was running** (e.g. the R1 cell).

The trainer then:

- **skips** every model whose `results.json` exists
  (`[resume] unet_v1: already finished ... -- skipping`), keeping its result in
  `all_results.json`;
- **continues** the interrupted model from `last.pth`
  (`Resuming at epoch 36/100`), with optimizer, scheduler, AMP scale and RNG
  restored. `scripts/test_trainer_resume.py` checks that a resumed run is
  bit-identical to an uninterrupted one;
- **starts** the models not yet begun.

**What a disconnect costs.** At most 5 epochs of the model that was running
(`last.pth` is written every 5 epochs), plus up to 10 minutes of progress not
yet synced to Drive. Finished models are never retrained.

**Watch progress** in the group cell's status line, or in
`results/<group>/shard0.log`. `training_log.csv` in each model folder gets one
row per epoch.
"""))

A(code("""
# Where everything stands, from what is on disk (restore first after a disconnect).
for g in ['r1_canonical', 'r2_seed1', 'r2_seed2', 'r3_param_matched',
          'r4_ssm_batch8', 'r5_position']:
    d = Path(RESULTS_DIR) / g
    if not d.exists():
        print(f'{g:18s} not started'); continue
    done = sorted(p.parent.name for p in d.glob('*/results.json'))
    part = sorted(p.parent.name for p in d.glob('*/last.pth'))
    errs = []
    if (d / 'all_results.json').exists():
        try:
            errs = [r['display_name'] for r in json.load(open(d / 'all_results.json'))
                    if 'error' in r]
        except Exception:
            pass
    print(f'{g:18s} {len(done):3d} finished' +
          (f' | resuming: {", ".join(part)}' if part else '') +
          (f' | FAILED: {", ".join(errs)}' if errs else ''))
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
