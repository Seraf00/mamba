#!/usr/bin/env python3
"""Verify a machine is ready to run the canonical training programme.

Run this BEFORE launching R1. Every check here corresponds to something that
has already gone wrong once, or to a number the revision depends on being true:

  GPU / VRAM        sizes NUM_SHARDS, and decides whether DenseContextU-Net
                    micro-batches (a BatchNorm deviation you want to know about)
  vCPU              the real cap on concurrency -- not VRAM
  TF32 default      on from Ampere; unpinned it moved FPN-UNet EF by 0.88 points
  skimage version   0.25.2 and 0.26.0 disagree on connected components, which is
                    enough to change a patient's EF
  mamba-ssm         without the CUDA fast path, SSM training is ~100x slower;
                    R4/R5 become impossible rather than slow
  dataset           a silently short split trains a different experiment
  determinism       two identical forward passes must agree bit-for-bit
  throughput        the only honest input to a time estimate

Exit code is 0 if nothing blocking was found, 1 otherwise. Warnings that do not
block (e.g. mamba-ssm absent when you only intend to run R1-R3) are reported
but do not fail the run.

Usage:
    python scripts/preflight_colab.py --data-dir /content/mamba/data/CAMUS
    python scripts/preflight_colab.py --shards 8 --workers 3 --bench
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Canonical settings -- must match notebooks/colab_revision.ipynb.
EPOCHS = 100
BATCH_SIZE = 8
IMG_SIZE = 256
REQUIRED_SKIMAGE = '0.25.2'
EXPECTED_PATIENTS = 500

OK, WARN, FAIL = 'ok', 'warn', 'FAIL'
results: list[tuple[str, str, str]] = []


def report(status: str, name: str, detail: str = '') -> None:
    results.append((status, name, detail))
    mark = {'ok': '  ok  ', 'warn': ' warn ', 'FAIL': ' FAIL '}[status]
    print(f'[{mark}] {name}' + (f'\n         {detail}' if detail else ''))


def check_gpu(shards: int, workers: int) -> None:
    import torch
    if not torch.cuda.is_available():
        report(FAIL, 'CUDA', 'No GPU visible. In Colab: Runtime > Change '
                             'runtime type > GPU.')
        return

    props = torch.cuda.get_device_properties(0)
    gib = props.total_memory / 1024 ** 3
    cc = f'{props.major}.{props.minor}'
    report(OK, 'GPU', f'{torch.cuda.get_device_name(0)}  |  {gib:.1f} GiB  |  '
                      f'compute capability {cc}')
    report(OK, 'torch', f'{torch.__version__}  cuda {torch.version.cuda}  '
                        f'cudnn {torch.backends.cudnn.version()}')

    # DenseContextU-Net needs 18.9 GiB at batch 8. Below that it micro-batches,
    # which makes BatchNorm see 2 samples instead of 8 -- allowed, but it is a
    # documented deviation and you should know it is happening.
    if gib * 0.85 < 18.9:
        report(WARN, 'DenseContextU-Net memory',
               f'{gib:.1f} GiB cannot hold batch 8 (needs 18.9 GiB). It will '
               f'micro-batch with gradient accumulation: effective batch stays '
               f'8, but BatchNorm sees 2 samples. State this in Methods.')
    else:
        report(OK, 'DenseContextU-Net memory',
               'fits batch 8 without micro-batching')

    # Concurrency is capped by cores, not memory.
    import multiprocessing
    cpus = multiprocessing.cpu_count()
    procs = shards * (workers + 1)
    if procs > cpus:
        report(WARN, 'concurrency',
               f'{shards} shards x ({workers} workers + 1) = {procs} processes '
               f'against {cpus} vCPU. Oversubscribed -- lower NUM_SHARDS or '
               f'NUM_WORKERS, or shards will fight for cores.')
    else:
        report(OK, 'concurrency',
               f'{shards} shards x ({workers} workers + 1) = {procs} processes, '
               f'{cpus} vCPU available')

    # Rough VRAM headroom for the requested concurrency. ~3 GiB per typical
    # model at batch 8, measured; DenseContextU-Net is the exception above.
    if shards * 3.0 > gib * 0.85:
        report(WARN, 'VRAM vs shards',
               f'{shards} concurrent jobs need roughly {shards * 3.0:.0f} GiB; '
               f'{gib:.1f} GiB present.')


def check_tf32() -> None:
    """TF32 is the single largest measurement effect in the study."""
    import torch
    if not torch.cuda.is_available():
        return
    default_on = (torch.backends.cuda.matmul.allow_tf32
                  or torch.backends.cudnn.allow_tf32)
    if default_on:
        report(OK, 'TF32 default',
               'ON, as expected on Ampere and later. --pin full turns it off; '
               'do not train without it or the numbers will not be comparable '
               'to the EF tables.')
    else:
        report(OK, 'TF32 default', 'already off')

    from utils import pin_determinism
    state = pin_determinism(42, 'full')
    bad = [k for k in ('tf32_matmul', 'tf32_cudnn') if state[k]]
    if bad:
        report(FAIL, 'pin_determinism', f'failed to disable {bad}')
    else:
        report(OK, 'pin_determinism',
               f"TF32 off, cudnn deterministic="
               f"{state['cudnn_deterministic']}, benchmark="
               f"{state['cudnn_benchmark']}, "
               f"CUBLAS={state['cublas_workspace_config']}")


def check_versions(need_mamba: bool) -> None:
    try:
        import skimage
        if skimage.__version__ == REQUIRED_SKIMAGE:
            report(OK, 'scikit-image', skimage.__version__)
        else:
            report(WARN, 'scikit-image',
                   f'{skimage.__version__}, expected {REQUIRED_SKIMAGE}. The '
                   f'connected-component filter differs between versions and '
                   f'that changes EF. Re-run the EF evaluation if you proceed.')
    except ImportError:
        report(FAIL, 'scikit-image',
               'not installed -- every EF computation will silently skip all '
               '50 patients and report nan. pip install scikit-image==0.25.2')

    for mod in ('timm', 'einops', 'nibabel'):
        try:
            __import__(mod)
            report(OK, mod, '')
        except ImportError:
            report(FAIL, mod, 'not installed')

    # mamba-ssm: needed only for R4/R5, but needed *badly* there.
    try:
        import mamba_ssm  # noqa: F401
        from models.modules import MambaBlock
        block = MambaBlock(dim=64, d_state=16)
        if getattr(block, 'use_fast_path', False):
            report(OK, 'mamba-ssm', 'installed, CUDA fast path active')
        else:
            report(FAIL if need_mamba else WARN, 'mamba-ssm',
                   'installed but falling back to the PyTorch implementation '
                   '(~100x slower). R4/R5 are not feasible like this.')
    except ImportError:
        report(FAIL if need_mamba else WARN, 'mamba-ssm',
               'not installed. Required for R4 and R5; R1-R3 run without it.')


def check_dataset(data_dir: Path) -> None:
    if not data_dir.exists():
        report(FAIL, 'dataset', f'{data_dir} does not exist')
        return
    patients = sorted(p for p in data_dir.iterdir()
                      if p.is_dir() and p.name.startswith('patient'))
    if len(patients) < EXPECTED_PATIENTS:
        report(FAIL, 'dataset',
               f'{len(patients)} patient directories, expected '
               f'{EXPECTED_PATIENTS}. A partial rsync trains a different '
               f'experiment without telling you.')
        return
    report(OK, 'dataset', f'{len(patients)} patient directories in {data_dir}')

    try:
        from data import CAMUSDataset, get_transforms
        counts = {}
        for split in ('train', 'val', 'test'):
            ds = CAMUSDataset(root_dir=str(data_dir), split=split,
                              transform=get_transforms(
                                  split='val', img_size=(IMG_SIZE, IMG_SIZE)))
            counts[split] = len(ds)
        report(OK, 'splits',
               '  '.join(f'{k}={v}' for k, v in counts.items()) +
               f'   ({counts["train"] // BATCH_SIZE} batches/epoch at batch '
               f'{BATCH_SIZE})')
        if counts['test'] != 200:
            report(WARN, 'test split',
                   f'{counts["test"]} frames, expected 200 '
                   f'(50 patients x 2 views x 2 phases)')
    except Exception as e:  # noqa: BLE001
        report(FAIL, 'splits', f'{type(e).__name__}: {e}')


def check_disk(results_dir: Path) -> None:
    target = results_dir if results_dir.exists() else results_dir.parent
    try:
        free = shutil.disk_usage(target).free / 1024 ** 3
    except OSError as e:  # noqa: BLE001
        report(WARN, 'disk', f'could not stat {target}: {e}')
        return
    # R5 alone is 68 checkpoints; the SSM models are the large ones.
    if free < 60:
        report(WARN, 'disk',
               f'{free:.0f} GiB free at {target}. R4+R5 write ~90 checkpoints; '
               f'delete checkpoint_epoch_*.pth between groups.')
    else:
        report(OK, 'disk', f'{free:.0f} GiB free at {target}')


def check_determinism() -> None:
    """Two identical forward passes must agree bit-for-bit once pinned."""
    import torch
    if not torch.cuda.is_available():
        return
    try:
        from utils import pin_determinism
        from models import get_model
        pin_determinism(42, 'full')
        m = get_model('unet_v1', in_channels=1, num_classes=4).cuda().eval()
        x = torch.randn(2, 1, IMG_SIZE, IMG_SIZE, device='cuda')
        with torch.no_grad():
            a = m(x)
            b = m(x)
        if torch.equal(a, b):
            report(OK, 'determinism', 'repeated forward passes bit-identical')
        else:
            report(WARN, 'determinism',
                   f'passes differ by up to {(a - b).abs().max().item():.3e}')
    except Exception as e:  # noqa: BLE001
        report(WARN, 'determinism', f'{type(e).__name__}: {e}')


def check_widened_controls() -> None:
    """The R3 controls must build before you spend hours discovering they don't."""
    from models import get_model
    cases = [
        ('transunet', {'vit_layers': 26}, IMG_SIZE),
        ('fpn', {'backbone': 'resnet101', 'fpn_channels': 1024}, IMG_SIZE),
        ('swin_unet', {'embed_dim': 114, 'img_size': 224}, 224),
        ('unet_v1', {'base_features': 96}, IMG_SIZE),
        ('unet_v2', {'base_features': 104}, IMG_SIZE),
        ('dense_context_unet', {'base_features': 200}, IMG_SIZE),
    ]
    bad = []
    for name, kw, _size in cases:
        try:
            m = get_model(name, in_channels=1, num_classes=4,
                          pretrained=False, **kw)
            del m
        except Exception as e:  # noqa: BLE001
            bad.append(f'{name}: {type(e).__name__}: {e}')
    if bad:
        report(FAIL, 'R3 widened controls', '\n         '.join(bad))
    else:
        report(OK, 'R3 widened controls',
               f'all {len(cases)} construct (transunet_wide keeps the first 12 '
               f'ViT blocks pretrained, the other 14 are fresh)')


def check_throughput(data_dir: Path, shards: int) -> None:
    """The only honest input to a time estimate."""
    import subprocess
    print('\n  Measuring throughput (about 2 minutes)...')
    out = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'bench_dataloader.py'),
         '--data-dir', str(data_dir), '--workers', '4', '--batches', '60'],
        capture_output=True, text=True)
    line = [l for l in out.stdout.split('\n') if l.strip().startswith('4 ')]
    if not line:
        report(WARN, 'throughput', 'benchmark produced no result:\n' +
               (out.stdout or out.stderr)[-500:])
        return
    parts = line[0].split()
    rate, s_per_epoch = float(parts[-2]), float(parts[-1])
    single = s_per_epoch * EPOCHS / 3600
    report(OK, 'throughput',
           f'{rate:.2f} it/s, {s_per_epoch:.0f} s/epoch -> {single:.2f} h per '
           f'100-epoch model (unet_v1). Reference sessions ran 8.3 s/epoch; '
           f'an RTX 4060 laptop runs 81.')
    # R1..R5 total, scaled from the reference session's per-model costs.
    ref_hours = {'R1': 4.5, 'R2': 2.0, 'R3': 7.5, 'R4': 19.4, 'R5': 28.5}
    scale = s_per_epoch / 8.3
    print()
    print(f'         {"group":6s} {"1 job":>9s} {f"{shards} shards":>11s}')
    tot_1 = tot_n = 0.0
    for g, h in ref_hours.items():
        one = h * scale
        many = one / (shards * 0.75)   # 0.75: measured-ish scaling loss
        tot_1 += one
        tot_n += many
        print(f'         {g:6s} {one:8.1f}h {many:10.1f}h')
    print(f'         {"TOTAL":6s} {tot_1:8.1f}h {tot_n:10.1f}h')
    print('         (shard scaling assumes 75% efficiency; verify on the '
          'first group)')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=str(ROOT / 'data' / 'CAMUS'))
    ap.add_argument('--results-dir', default=str(ROOT / 'results'))
    ap.add_argument('--shards', type=int, default=3,
                    help='NUM_SHARDS you intend to use')
    ap.add_argument('--workers', type=int, default=3,
                    help='NUM_WORKERS per shard')
    ap.add_argument('--need-mamba', action='store_true',
                    help='Fail rather than warn if mamba-ssm is unusable '
                         '(set this when you intend to run R4 or R5)')
    ap.add_argument('--bench', action='store_true',
                    help='Also measure throughput and project group timings')
    args = ap.parse_args()

    print('=' * 74)
    print('PRE-FLIGHT: canonical training programme')
    print(f'  batch {BATCH_SIZE} | {EPOCHS} epochs | early stopping OFF | '
          f'--pin full | {IMG_SIZE} px')
    print('=' * 74)

    check_gpu(args.shards, args.workers)
    check_tf32()
    check_versions(args.need_mamba)
    check_dataset(Path(args.data_dir))
    check_disk(Path(args.results_dir))
    check_determinism()
    check_widened_controls()
    if args.bench:
        check_throughput(Path(args.data_dir), args.shards)

    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]
    print()
    print('=' * 74)
    print(f'{len(fails)} blocking, {len(warns)} warnings, '
          f'{len(results) - len(fails) - len(warns)} ok')
    if fails:
        print('\nBlocking:')
        for _, name, detail in fails:
            print(f'  - {name}: {detail.splitlines()[0] if detail else ""}')
        print('\nDo not start training until these are resolved.')
    elif warns:
        print('\nNo blockers. Review the warnings above, then launch R1.')
    else:
        print('\nReady. Launch R1.')
    print('=' * 74)
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
