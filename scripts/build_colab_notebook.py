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

**Time, measured on G4 with `--pin full`:** R1 took 7.4 h (DenseContextU-Net
164 min and FPN 147 min of it), R2 about 2.4 h per seed, and the widened
DenseContext and FPN controls about 6 h and 10 h each. Deterministic algorithms
and TF32-off cost the heavy convolutional models 1.5-2x over the unpinned
benchmark that suggested "~49 h for R1-R5"; budget R4 and R5 at roughly 30 h and
45 h rather than trusting that figure.
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

`TORCH_SPEC` is pinned to **torch 2.11.0**, the stack the shootout validated on
G4 (Triton 3.6.0, mamba-ssm 2.3.2.post1; full kernel stack runs, and the Mamba-2
{4 train, 6 fail} split replicates the published table). **Do not change it
mid-programme.** Colab updates its default torch without warning, and a
programme spanning several sessions would otherwise straddle two builds — the
same class of confound as mixing GPUs. Every number the revision reports,
EF included, is re-measured on this one stack.
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
# Everything in requirements.txt EXCEPT torch and the SSM kernels, which the
# cells above installed and pinned. The file's own ">=" pins would not
# downgrade them -- but if the mamba-ssm build above failed, pip would retry it
# here with build isolation, fail, and abort the whole install, leaving the
# ordinary dependencies (albumentations, opencv, ...) missing too.
!grep -viE '^(torch|torchvision|torchaudio|mamba-ssm|causal-conv1d)' requirements.txt > /tmp/req_colab.txt
!pip install -q -r /tmp/req_colab.txt
"""))

A(code("""
# Does the code just pulled support everything this notebook asks of it?
# This notebook and the scripts it calls come from the same repository but can
# be at different versions -- e.g. a notebook opened from a newer upload while
# GitHub still has older scripts. Then a group cell fails hours in with
# "unrecognized arguments". Check every flag the cells use, now.
import subprocess, sys
NEEDS = {
    'scripts/train_all_models.py': ['--resume', '--resume_every', '--checkpoint_every',
                                    '--pin', '--wide_only', '--position_only',
                                    '--param_matched', '--num_shards',
                                    '--amp_dtype'],
    'scripts/evaluate_all_models.py': ['--pin'],
    'scripts/yolo/eval_baseline_ef.py': ['--pin', '--checkpoint-dir'],
}
missing = []
for script, flags in NEEDS.items():
    h = subprocess.run([sys.executable, script, '--help'],
                       capture_output=True, text=True).stdout
    missing += [f'{script} {f}' for f in flags if f not in h]
!git -C /content/mamba log -1 --format='code version: %h  %ad  %s' --date=short
if missing:
    raise RuntimeError('The code on this machine is OLDER than this notebook. '
                       'Missing: ' + ', '.join(missing) + '. Push your latest '
                       'commits to GitHub, then re-run the clone cell above.')
print('code matches this notebook')
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
import subprocess, sys, time, json, glob, shutil, hashlib
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
    # The model being trained and its last COMPLETED epoch, from the logs.
    # Model: the last "Training: X" line anywhere in the file -- tqdm output
    # pushes it out of any fixed-size tail within an epoch. Epoch: the trainer's
    # own summary line ("Epoch 37/100 | Train Loss: ..."); tqdm's bar also
    # starts with "Epoch" and contains " | ", so match on "Train Loss".
    model, epoch = '', ''
    for log in sorted(exp_dir.glob('shard*.log')):
        try:
            txt = log.read_text(errors='ignore')
        except OSError:
            continue
        i = txt.rfind('\\nTraining: ')
        if i >= 0:
            model = txt[i + len('\\nTraining: '):].split('\\n', 1)[0].strip()
        # Whole file, by the summary's own marker: no window can be assumed
        # to still contain it after a long stretch of progress bars.
        j = txt.rfind(' | Train Loss')
        # Only if it belongs to the CURRENT model: an epoch line from before
        # the latest "Training:" is the previous model's last epoch.
        if j > i:
            ls = max(txt.rfind(ch, 0, j) for ch in ('\\n', '\\r')) + 1
            head = txt[ls:j].split()
            if len(head) >= 2 and head[0] == 'Epoch':
                epoch = head[1]
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

    # Unbuffered: a child writing to a file block-buffers stdout, so the
    # per-epoch lines would reach the log -- and the status line below -- in
    # 8 KB bursts, dozens of epochs late, and whatever was buffered at a
    # disconnect would be lost.
    # TQDM_MININTERVAL: the progress bar redraws ~10 times a second, and into a
    # log file each redraw is a new line -- megabytes per group, all synced to
    # Drive. Every 30 s is plenty; the per-epoch summary line is unaffected.
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'TQDM_MININTERVAL': '30'}

    if dry_run:
        out = subprocess.run([sys.executable, 'scripts/train_all_models.py',
                              *args, '--dry_run'], capture_output=True, text=True,
                             env=env)
        print(out.stdout[-3000:])
        if out.returncode != 0:
            print(out.stderr[-3000:])
        return exp_dir

    exp_dir.mkdir(parents=True, exist_ok=True)
    procs, logs = [], []
    for s in range(shards):
        # Append, never truncate: after a disconnect this log was restored
        # from Drive, and it is the only record of the earlier session.
        log = open(exp_dir / f'shard{s}.log', 'a')
        log.write(f'\\n===== session start {time.strftime("%Y-%m-%d %H:%M:%S")} =====\\n')
        log.flush()
        logs.append(log)
        procs.append(subprocess.Popen(
            [sys.executable, 'scripts/train_all_models.py', *args,
             '--num_shards', str(shards), '--shard', str(s)],
            stdout=log, stderr=subprocess.STDOUT, env=env))
    print(f'{exp_name}: launched {shards} shards -> {exp_dir}/shard*.log')

    t0 = last_sync = time.time()
    while any(p.poll() is None for p in procs):
        alive = sum(p.poll() is None for p in procs)
        model, epoch = _current(exp_dir)
        warn = ' | !! DRIVE SYNC FAILING' if _SYNC_ERR else ''
        print(f'\\r  {(time.time()-t0)/60:6.1f} min | {_models_done(exp_dir)} done '
              f'| now: {model} epoch {epoch}{warn}' + ' ' * 10, end='')
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
# nothing downstream and would only fill Drive. .tex/.png/.pdf are evaluation
# outputs: Drive is how results leave Colab, so anything not listed never
# reaches your machine.
_BASE_FILTER = ['--include=*/', '--include=best_model.pth', '--include=last.pth',
                '--include=*.json', '--include=*.csv', '--include=*.log',
                '--include=*.jsonl', '--include=*.tex', '--include=*.png',
                '--include=*.pdf', '--exclude=*']
_SYNC_ERR = None   # last sync failure, shown in the status line until it clears


def _offloaded():
    # Groups whose weights you downloaded, verified and removed from Drive.
    # Marked by OFFLOADED.json in the group folder (on Drive, so the mark
    # survives a disconnect and is restored with everything else).
    marks = set()
    for root in (RESULTS_DIR, DRIVE_RESULTS):
        if os.path.isdir(root):
            marks |= {m.parent.name for m in Path(root).glob('*/OFFLOADED.json')}
    return sorted(marks)


def _sync_filter():
    # rsync applies the FIRST matching rule, so these excludes must come before
    # the includes. Without them the next sync would re-upload the weights you
    # just deleted from Drive, because they are still on this machine's disk.
    return [f'--exclude={g}/*/*.pth' for g in _offloaded()] + _BASE_FILTER


def _require_mamba_fast():
    # Without the CUDA fast path the trainer waits 10 seconds and then trains
    # SSM models roughly 100x slower -- silently, for days. Refuse instead.
    r = subprocess.run([sys.executable, '-c',
                        'from models.modules import MambaBlock; '
                        'import sys; sys.exit(0 if MambaBlock(dim=64, d_state=16).use_fast_path else 1)'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('mamba-ssm CUDA fast path unavailable -- fix the '
                           'install (setup cell) before running R4/R5.\\n'
                           + r.stderr[-500:])


def drive_usage():
    # Free space on the mounted Drive. Deleted files keep counting against the
    # quota until the Drive TRASH is emptied.
    try:
        u = shutil.disk_usage('/content/drive/MyDrive')
        print(f'Drive: {u.used / 1e9:.1f} GB used of {u.total / 1e9:.1f} GB, '
              f'{u.free / 1e9:.1f} GB free')
        return u.free / 1e9
    except OSError as e:
        print(f'Drive usage unavailable: {e}')
        return None


def _sync_failed(msg):
    global _SYNC_ERR
    _SYNC_ERR = msg[:120]
    print()
    print('!' * 78)
    print('!! DRIVE SYNC FAILED -- training continues, but progress since the last')
    print('!! successful sync exists ONLY on this machine. A disconnect loses it.')
    print(f'!! {_SYNC_ERR}')
    print('!! If Drive is full: empty the Drive Trash, and see "If Drive fills up".')
    print('!' * 78)
    return False


def save_to_drive(quiet=False):
    # Must never raise. It runs inside the training cell's polling loop, and an
    # exception there ends the loop -- training carries on headless with NO
    # further syncs, the worst outcome on a full or flaky Drive. Every failure
    # becomes a False return and a visible warning instead.
    global _SYNC_ERR
    try:
        os.makedirs(DRIVE_RESULTS, exist_ok=True)
        r = subprocess.run(['rsync', '-a', *_sync_filter(),
                            f'{RESULTS_DIR}/', f'{DRIVE_RESULTS}/'],
                           capture_output=True, text=True)
    except OSError as e:
        return _sync_failed(f'{type(e).__name__}: {e}')
    if r.returncode != 0:
        return _sync_failed((r.stderr.strip().splitlines() or ['rsync failed'])[-1])
    _SYNC_ERR = None
    # rsync never deletes, so a finished model's last.pth -- removed locally
    # when training completes -- would stay on Drive for good: full optimizer
    # state, up to ~3 GB per model. Not rsync --delete: on a fresh session run
    # before restore_from_drive(), that would wipe Drive. Instead remove only
    # resume files whose model has a results.json, i.e. is certainly finished.
    try:
        for lp in Path(DRIVE_RESULTS).glob('*/*/last.pth'):
            if (lp.parent / 'results.json').exists():
                lp.unlink()
    except OSError as e:
        print(f'  [sync] could not remove a stale last.pth: {e}')
    if not quiet:
        print(f'synced {RESULTS_DIR} -> {DRIVE_RESULTS}')
    return True


def restore_from_drive():
    # Bring previous sessions' progress back onto local disk. Run at the start
    # of every session: harmless on the first (Drive is empty); after a
    # disconnect it restores results.json for finished models and last.pth for
    # the interrupted one, which is what --resume reads.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    drive_usage()
    if not os.path.isdir(DRIVE_RESULTS):
        print('nothing on Drive yet -- fresh start')
        return
    subprocess.run(['rsync', '-a', *_sync_filter(),
                    f'{DRIVE_RESULTS}/', f'{RESULTS_DIR}/'], check=True)
    done = sorted(Path(RESULTS_DIR).glob('*/*/results.json'))
    partial = sorted(pth for pth in Path(RESULTS_DIR).glob('*/*/last.pth')
                     if not (pth.parent / 'results.json').exists())
    print(f'restored from Drive: {len(done)} finished models, '
          f'{len(partial)} interrupted mid-training')
    for pth in partial:
        print(f'  will resume: {pth.parent.parent.name}/{pth.parent.name}')
    off = _offloaded()
    if off:
        print(f'  offloaded to your PC (weights not on Drive): {", ".join(off)}')


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def offload_check(group):
    # Step 1 of freeing Drive: is this group safe to move off Drive, and what
    # exactly must your download contain? Writes MANIFEST.json (size + SHA-256
    # of every checkpoint) and syncs it, so scripts/verify_offload.py on your PC
    # can prove the download is complete BEFORE anything is deleted.
    d = Path(RESULTS_DIR) / group
    problems = []
    if not d.is_dir():
        print(f'{group}: not on this machine'); return False
    if not (d / 'all_results.json').exists():
        problems.append('group has not finished (no all_results.json)')
    pending = [x.parent.name for x in d.glob('*/last.pth')
               if not (x.parent / 'results.json').exists()]
    if pending:
        problems.append(f'still training: {", ".join(pending)}')
    if not (d / 'evaluation' / 'evaluation_results.json').exists():
        problems.append('not evaluated yet -- run the Evaluation cells first; '
                        'they need the weights, which will not be here afterwards')
    if not (d / 'baseline_ef_native.json').exists():
        problems.append('EF not evaluated yet (baseline_ef_native.json missing)')
    ckpts = sorted(d.glob('*/best_model.pth'))
    if not ckpts:
        problems.append('no checkpoints on this machine to fingerprint')
    if problems:
        print(f'{group}: NOT ready to offload')
        for p_ in problems:
            print(f'  - {p_}')
        return False
    print(f'{group}: fingerprinting {len(ckpts)} checkpoints ...')
    manifest = {str(c.relative_to(d)): {'bytes': c.stat().st_size,
                                        'sha256': _sha256(c)} for c in ckpts}
    (d / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1))
    if not save_to_drive(quiet=True):
        print('  sync failed -- free some Drive space first'); return False
    gb = sum(v['bytes'] for v in manifest.values()) / 1e9
    print(f'  ready. {gb:.1f} GB of checkpoints; MANIFEST.json is on Drive.')
    print('  Next, on your PC:')
    print(f'    1. download MyDrive/Paper1/results_revision/{group} into '
          f'D:/Papers/Paper1/results_revision/{group}')
    print(f'    2. python scripts/verify_offload.py results_revision/{group}')
    print(f'    3. only if it prints ALL VERIFIED: offload("{group}", '
          f'i_have_verified_the_download=True)')
    return True


def offload(group, i_have_verified_the_download=False):
    # Step 2: remove this group's checkpoints from Drive and mark it offloaded,
    # so no later sync re-uploads them and no later evaluation runs on a group
    # whose weights are gone. Small files (results, evaluation, logs) stay on
    # Drive: resume needs results.json to know these models are finished.
    # Deleted files go to the Drive Trash and still count until it is emptied.
    if not i_have_verified_the_download:
        print('Not deleting anything. Run scripts/verify_offload.py on your PC '
              'first; call again with i_have_verified_the_download=True only '
              'when it prints ALL VERIFIED.')
        return
    if not (Path(RESULTS_DIR) / group / 'MANIFEST.json').exists():
        print(f'{group}: run offload_check("{group}") first.'); return
    dd = Path(DRIVE_RESULTS) / group
    removed = 0
    for c in dd.glob('*/*.pth'):
        removed += c.stat().st_size
        c.unlink()
    mark = {'offloaded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'removed_bytes': removed}
    for root in (RESULTS_DIR, DRIVE_RESULTS):
        (Path(root) / group / 'OFFLOADED.json').write_text(json.dumps(mark))
    print(f'{group}: removed {removed / 1e9:.1f} GB of checkpoints from Drive '
          f'(now in the Drive Trash -- empty it to get the space back).')
    print('  The weights are still on this machine until the runtime ends, and '
          'on your PC.')
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

| control | widening | matched | vs target | first R3 run (G4) |
|---|---|---|---|---|
| `unet_v1_wide` | `bf=96` | 69.8 M | +1.67% | done, val Dice 0.8999 |
| `unet_v2_wide` | `bf=104` | 87.1 M | +4.91% | done, val Dice 0.9041 |
| `transunet_wide` | `vit_layers=26` | 201.3 M | −0.44% | done, val Dice 0.9035 |
| `swin_unet_wide` | `depths=[2,2,12,2]` | 63.2 M | +2.12% | **failed** as `embed_dim=114` |
| `dense_context_unet_wide` | `bf=200` | 5.3 M | −1.86% | **NaN** from epoch 14 (fp16) |
| `fpn_wide` | `resnet101, fpn_channels=1024` | 169.1 M | −2.29% | **NaN** from epoch 4 (fp16) |

**TransUNet and Swin widen by depth, not width.** Width changes every tensor's
shape, so the pretrained weights cannot load and the control would be a randomly
initialised network against a pretrained one — the extra parameters would not be
what the comparison measured. `embed_dim=114` for Swin did exactly that: it
matched parameters but could not even be built with Swin-Tiny's 96-wide weights,
so the first R3 run failed at construction. Depth keeps the pretrained blocks
and adds fresh ones — how the Mamba variants add capacity, and how the Swin
family itself scales (Swin-S is `[2,2,18,2]`). `param_match.py` now builds every
chosen control with pretraining on before emitting it.

**DenseContextU-Net and FPN run in a separate group, R3b, in bfloat16.** Both
went NaN under float16 with training loss falling normally until the moment it
did — the signature of fp16 activation overflow, and the same thing happened to
both in the original submission's runs, where early stopping at epoch 25 hid it.
bfloat16 has float32's range, so it cannot overflow that way. It is a stated
deviation: the baselines trained in float16, and each `results.json` records
`amp_dtype`. If bf16 also diverges, the trainer's divergence guard stops the
model at the first non-finite epoch and records it as a failure — minutes, not
the ~6 and ~10 hours the NaN runs took.
"""))

A(code("""
# Regenerate rather than trusting the file in git -- the parameter counts are
# derived from the models as they are now, and each control is built with
# pretraining on before it is emitted.
!python scripts/param_match.py --mamba_type mamba \\
    --output_json {RESULTS_DIR}/param_config_r3.json

# The four controls that train in float16. unet_v1/unet_v2/transunet finished in
# the first run and are skipped by --resume; this trains swin_unet_wide.
run_group('r3_param_matched',
          ['--models', 'unet_v1', 'unet_v2', 'swin_unet', 'transunet',
           '--param_matched', '--wide_only',
           '--param_config', f'{RESULTS_DIR}/param_config_r3.json'])
"""))

A(md("""
### R3b — the two controls that overflow in float16

Before the first launch of this cell, delete from Drive
`results_revision/r3_param_matched/dense_context_unet_wide/` and
`.../fpn_wide/` (their float16 runs are NaN and belong to no result), then empty
the Drive Trash. Roughly 6 h (DenseContext) and 10 h (FPN) on G4 — the two
slowest runs in the programme.
"""))

A(code("""
run_group('r3b_param_matched_bf16',
          ['--models', 'dense_context_unet', 'fpn',
           '--param_matched', '--wide_only', '--amp_dtype', 'bfloat16',
           '--param_config', f'{RESULTS_DIR}/param_config_r3.json'])
"""))

A(md("""
---
## R4 — SSM arms at batch 8

Mamba, Mamba-2 and VMamba variants, all at the canonical batch. Without this,
Paper D's base-vs-SSM comparison still crosses batch sizes.

Some runs fail here, and each failure is a result to report, not a bug to fix:

- `mamba_dense_context_unet` + VMamba is skipped as architecturally
  incompatible (>12 GiB backward spike).
- Six Mamba-2 arms fail, as in the published table. On G4, five hit the Triton
  shared-memory ceiling (Swin, UNet-V1, TransUNet, Pure-Mamba, FPN).
  `mamba_unet_v2` fails **earlier and differently**: its 682 channels are not a
  multiple of 8, which causal-conv1d requires. The original run recorded a
  shared-memory failure for it, so the outcome is the same but the cause changed
  with the software stack. Report the cause as measured.

Failures are recorded in `all_results.json` with their error message and cost
seconds each. Let them fail.
"""))

A(code("""
_require_mamba_fast()
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
_require_mamba_fast()
run_group('r5_position',
          ['--position_ablation', '--position_only',
           '--mamba_variants', 'mamba', 'vmamba'])
save_to_drive()
"""))

A(md("""
---
## Evaluation

Inference only, pinned like training, and **in this order**:

1. `evaluate_all_models.py`: test Dice, IoU, HD95, ASSD. It writes
   `evaluation_results.json` from scratch.
2. `eval_baseline_ef.py`: EF at **native resolution**. The defect it fixes was
   computing volumes on the 256 px grid (EF-1).
3. `colab_session.py`: injects per-patient EF arrays into
   `evaluation_results.json` for the Bland–Altman figure, plus quality-strata
   Dice and overlays. Run it before step 1 and step 1 erases what it injected.

R2's seed runs are evaluated too: their test metrics are the seed floor.
"""))

A(code("""
# R2 is included: its test metrics ARE the seed floor, which is the reason R2
# exists.
SESSIONS = ['r1_canonical', 'r2_seed1', 'r2_seed2', 'r3_param_matched',
            'r3b_param_matched_bf16', 'r4_ssm_batch8', 'r5_position']

# 1) Test metrics: Dice, IoU, HD95, ASSD -> <session>/evaluation/.
#    MUST run first: it writes evaluation_results.json from scratch, so running
#    it after colab_session.py would erase the per-patient EF arrays that
#    colab_session injects into that same file (the Bland-Altman data).
for s in SESSIONS:
    d = Path(RESULTS_DIR) / s
    if not d.exists():
        print(f'skip {s} (not trained)'); continue
    if s in _offloaded():
        print(f'skip {s} (offloaded -- evaluated before its weights left)'); continue
    !python scripts/evaluate_all_models.py --checkpoint_dir {d} \\
        --data_dir {DATA_DIR} --pin full
save_to_drive()
"""))

A(code("""
# 2) EF at native resolution (the EF-1 fix), pinned.
for s in SESSIONS:
    d = Path(RESULTS_DIR) / s
    if not d.exists() or s in _offloaded():
        continue
    !python scripts/yolo/eval_baseline_ef.py \\
        --checkpoint-dir {d} --data-dir {DATA_DIR} --pin full \\
        --out {d}/baseline_ef_native.json
save_to_drive()
"""))

A(code("""
# 3) Per-patient EF arrays injected into evaluation_results.json (Bland-Altman),
#    quality-stratified Dice, overlay grids. After step 1, never before.
for s in SESSIONS:
    d = Path(RESULTS_DIR) / s
    if not d.exists() or s in _offloaded():
        continue
    !python scripts/colab_session.py \\
        --checkpoint_dir {d} --data_dir {DATA_DIR} \\
        --out_dir {RESULTS_DIR}/session_out/{s}
save_to_drive()
"""))

A(md("""
---
## Tables — build them on your machine, not here

The manuscripts are not on this runtime (Paper D is a separate repository), so
tables generated here would land in a throwaway clone. After the last group:

1. Copy `MyDrive/Paper1/results_revision/` from Drive into
   `D:/Papers/Paper1/results_revision/` on your machine.
2. Run the table generators there.

**Before that works, the generators need one change.** `fill_tables.py` still
reads the pre-revision session names (`base_models`, `param_matched`, ...) and
takes EF from fixed files under `results/yolo/`. Pointed at the new sessions as
they stand, it would pair the NEW Dice with the OLD EF for every model whose
name did not change, and it would not raise an error. Wiring it to
`results_revision/` is the next piece of work, and it does not block training:
start R1 now.
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
          'r3b_param_matched_bf16', 'r4_ssm_batch8', 'r5_position']:
    d = Path(RESULTS_DIR) / g
    if not d.exists():
        print(f'{g:18s} not started'); continue
    done = sorted(p.parent.name for p in d.glob('*/results.json'))
    part = sorted(p.parent.name for p in d.glob('*/last.pth')
                  if not (p.parent / 'results.json').exists())
    errs = []
    if (d / 'all_results.json').exists():
        try:
            errs = [r['display_name'] for r in json.load(open(d / 'all_results.json'))
                    if 'error' in r]
        except Exception:
            pass
    print(f'{g:18s} {len(done):3d} finished' +
          (f' | resuming: {", ".join(part)}' if part else '') +
          (f' | FAILED: {", ".join(errs)}' if errs else '') +
          (' | OFFLOADED to PC' if g in _offloaded() else ''))
drive_usage()
"""))

A(md("""
---
## If Drive fills up

The programme's checkpoints total roughly **20–32 GB** (R5's 68 arms are most of
it), measured from the equivalent earlier sessions. A free Google account has
15 GB, shared with Gmail and Photos, so on a free plan it will not all fit at
once. Two options:

**A. More storage.** Google One 100 GB costs about the same per month as a
coffee. Nothing below is then needed.

**B. Move each finished group to your PC.** Only when the group has **finished
and been evaluated**:

1. Run the three Evaluation cells. They need the weights, and the weights will
   not be on Drive afterwards.
2. `offload_check('r1_canonical')` checks the group is finished and evaluated,
   fingerprints every checkpoint into `MANIFEST.json`, and syncs it.
3. On your PC, download `MyDrive/Paper1/results_revision/r1_canonical` into
   `D:/Papers/Paper1/results_revision/r1_canonical`, then run
   `python scripts/verify_offload.py results_revision/r1_canonical`.
4. **Only if it prints `ALL VERIFIED`**:
   `offload('r1_canonical', i_have_verified_the_download=True)`.
5. **Empty the Drive Trash.** Deleted files keep counting until you do.

After offloading, the notebook **never re-uploads** that group's weights (they
are still on this machine's disk until the runtime ends). It also **skips** the
group in Evaluation, so a later run cannot overwrite good results with empty
ones. The small files stay on Drive, because resume needs `results.json` to know
those models are finished.

**Do not delete checkpoints from Drive by hand while training runs.** The next
sync re-uploads anything still on this machine, and a hand-deleted group has no
offload mark to protect its evaluation.

If a sync fails, the status line shows `!! DRIVE SYNC FAILING` and a banner
explains it. Training carries on, but anything since the last good sync exists
only on this machine.
"""))

A(code("""
drive_usage()
# offload_check('r1_canonical')
# offload('r1_canonical', i_have_verified_the_download=True)
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
