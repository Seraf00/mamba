#!/usr/bin/env python3
"""Find the dataloader configuration this machine actually wants.

R1 ran at 3.6 it/s against ~24 it/s on the hardware that produced the reference
sessions, with the GPU oscillating 0-87% and averaging ~40% -- a starved GPU
rather than a slow one. Steady-state epochs (84/81/81 s) ruled out worker
respawn as the dominant cost, which leaves worker count and prefetch depth.

This times real forward+backward steps on the real dataset, so the number it
reports is throughput you can multiply by 200 batches to get an epoch.

Usage:
    python scripts/bench_dataloader.py
    python scripts/bench_dataloader.py --workers 0 4 8 12 16 --batches 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import CAMUSDataset, get_transforms  # noqa: E402
from models import get_model  # noqa: E402
from training import CombinedLoss  # noqa: E402


def bench(n_workers, batches, batch_size, persistent, prefetch, model, crit, opt,
          scaler, device, ds):
    kw = dict(batch_size=batch_size, shuffle=True, num_workers=n_workers,
              pin_memory=True)
    if n_workers > 0:
        kw['persistent_workers'] = persistent
        kw['prefetch_factor'] = prefetch
    loader = DataLoader(ds, **kw)

    it = iter(loader)
    # warm up outside the timed region: first batch pays worker spawn and
    # cuDNN algorithm selection, neither of which recurs per step.
    for _ in range(3):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        _step(b, model, crit, opt, scaler, device)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    done = 0
    while done < batches:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); continue
        _step(b, model, crit, opt, scaler, device)
        done += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    del loader, it
    return done / dt


def _step(batch, model, crit, opt, scaler, device):
    # CAMUSDataset yields a dict when include_info=True and a plain
    # (image, mask) pair otherwise; the trainer accepts both, so this does too.
    if isinstance(batch, dict):
        x, y = batch['image'], batch['mask']
    else:
        x, y = batch[0], batch[1]
    x = x.to(device, non_blocking=True).float()
    y = y.to(device, non_blocking=True).long()
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        loss = crit(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='unet_v1')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--batches', type=int, default=60)
    ap.add_argument('--workers', type=int, nargs='*', default=[0, 4, 8, 12, 16])
    ap.add_argument('--data-dir', default=str(ROOT / 'data' / 'CAMUS'))
    ap.add_argument('--json', default=None,
                    help='Write the measured rates here. Used by '
                         'gpu_shootout.py, which runs several of these at once '
                         'and needs a parse-free result per process.')
    ap.add_argument('--quiet', action='store_true',
                    help='Suppress the table; only the summary is printed')
    args = ap.parse_args()

    device = torch.device('cuda')
    print(f'{torch.cuda.get_device_name(0)} | torch {torch.__version__} | '
          f'{args.model} @ batch {args.batch_size}')

    ds = CAMUSDataset(root_dir=args.data_dir, split='train',
                      transform=get_transforms(split='train', img_size=(256, 256)))
    model = get_model(args.model, in_channels=1, num_classes=4).to(device).train()
    crit = CombinedLoss(dice_weight=1.0, ce_weight=1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    if not args.quiet:
        print(f"\n{'workers':>8s} {'persist':>8s} {'prefetch':>9s} "
              f"{'it/s':>7s} {'s/epoch':>9s}  (200 batches)")
    best = (None, 0.0)
    measured = []
    for w in args.workers:
        for persistent, prefetch in ([(False, 2)] if w == 0
                                     else [(False, 2), (True, 4)]):
            rate = bench(w, args.batches, args.batch_size, persistent, prefetch,
                         model, crit, opt, scaler, device, ds)
            measured.append({'workers': w, 'persistent': persistent,
                             'prefetch': prefetch, 'it_per_s': rate,
                             's_per_epoch': 200 / rate})
            if not args.quiet:
                print(f'{w:8d} {str(persistent):>8s} {prefetch:9d} '
                      f'{rate:7.2f} {200 / rate:9.1f}', flush=True)
            if rate > best[1]:
                best = ((w, persistent, prefetch), rate)

    (w, p, pf), rate = best
    print(f'\nbest: num_workers={w} persistent_workers={p} prefetch_factor={pf}'
          f'  ->  {rate:.2f} it/s, {200/rate:.1f} s/epoch')
    print(f'reference sessions ran 8.3 s/epoch; an RTX 4060 laptop runs 81.')

    if args.json:
        import json
        with open(args.json, 'w') as f:
            json.dump({
                'gpu': torch.cuda.get_device_name(0),
                'torch': torch.__version__,
                'model': args.model,
                'batch_size': args.batch_size,
                'best': {'workers': w, 'persistent': p, 'prefetch': pf,
                         'it_per_s': rate, 's_per_epoch': 200 / rate},
                'measured': measured,
            }, f, indent=2)


if __name__ == '__main__':
    main()
