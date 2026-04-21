#!/usr/bin/env python3
"""
Memory profiling for all base + Mamba-variant model combinations.

Purpose: quantify the per-combination GPU memory requirement for one
forward+backward step, so the paper can empirically justify architectural
incompatibilities (notably VMamba + DenseContextUNet) rather than reporting
them as bare "skipped".

Produces:
  - {output_dir}/memory_profile.csv    row per (base, mamba_type)
  - {output_dir}/memory_profile.json   same data, JSON format
  - {output_dir}/memory_profile.pdf    publication-quality bar chart
  - {output_dir}/memory_profile.png    preview raster of the same figure

Usage (Colab / GPU):
  python scripts/profile_memory.py --batch_size 4 --output_dir ./results/memory

Typical runtime: ~10 minutes for 39 combos on a Colab L4.

Note: this profiles the *training* memory footprint (forward + backward),
which is what determines trainability. Inference memory is typically
3-4x smaller.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from models import get_model


BASE_MODELS = [
    'unet_v1', 'unet_v2', 'unet_resnet', 'deeplab_v3', 'nnunet',
    'dense_context_unet', 'swin_unet', 'transunet', 'fpn',
]
MAMBA_MODELS = [
    'mamba_unet_v1', 'mamba_unet_v2', 'mamba_unet_resnet', 'mamba_deeplab',
    'mamba_nnunet', 'mamba_dense_context_unet', 'mamba_swin_unet',
    'mamba_transunet', 'mamba_fpn', 'pure_mamba_unet',
]
MAMBA_VARIANTS = ['mamba', 'mamba2', 'vmamba']
SWIN_MODELS = ['swin_unet', 'mamba_swin_unet']


def get_img_size(model_name: str) -> int:
    return 224 if model_name in SWIN_MODELS else 256


def _cleanup():
    """Aggressive memory cleanup between models."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def profile_one(
    model_name: str,
    mamba_type: Optional[str],
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Profile forward+backward GPU memory for one (model, mamba_type) pair."""
    display = f'{model_name}_{mamba_type}' if mamba_type else model_name
    img_size = get_img_size(model_name)

    result: Dict[str, Any] = {
        'model_name': model_name,
        'mamba_type': mamba_type if mamba_type else '-',
        'display_name': display,
        'batch_size': batch_size,
        'img_size': img_size,
        'status': 'pending',
    }

    _cleanup()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Capture locals that may fail to be created so we can clean up in finally.
    model = None
    x = None
    target = None
    output = None
    loss = None

    try:
        # --- Build model ---
        kwargs = {'in_channels': 1, 'num_classes': 4}
        if mamba_type:
            kwargs['mamba_type'] = mamba_type
        model = get_model(model_name, **kwargs).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        result['params_M'] = round(n_params / 1e6, 3)

        mem_after_load = (torch.cuda.memory_allocated(device) / 1e9
                          if torch.cuda.is_available() else 0.0)

        # --- Forward + backward ---
        x = torch.randn(batch_size, 1, img_size, img_size, device=device)
        target = torch.randint(0, 4, (batch_size, img_size, img_size), device=device)
        criterion = nn.CrossEntropyLoss()

        model.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        output = model(x)
        # Some models return dict or deep-supervision list; use primary output.
        if isinstance(output, dict):
            output = output.get('out', next(iter(output.values())))
        if isinstance(output, (list, tuple)):
            output = output[0]

        loss = criterion(output, target)
        loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        peak_mem = (torch.cuda.max_memory_allocated(device) / 1e9
                    if torch.cuda.is_available() else 0.0)

        result['mem_after_load_GB'] = round(mem_after_load, 3)
        result['peak_mem_GB'] = round(peak_mem, 3)
        result['activation_mem_GB'] = round(max(peak_mem - mem_after_load, 0.0), 3)
        result['fwd_bwd_sec'] = round(t1 - t0, 3)
        result['status'] = 'ok'

    except torch.cuda.OutOfMemoryError as e:
        result['status'] = 'oom'
        # Record how much memory PyTorch was trying to allocate (first line of error)
        err_str = str(e)
        result['error'] = err_str.splitlines()[0][:300]
        if torch.cuda.is_available():
            try:
                result['peak_mem_GB_before_oom'] = round(
                    torch.cuda.max_memory_allocated(device) / 1e9, 3)
            except Exception:
                pass

    except Exception as e:
        result['status'] = 'error'
        result['error'] = f'{type(e).__name__}: {str(e)[:300]}'

    finally:
        # Explicit cleanup of tensors / model before next iteration
        del model, x, target, output, loss
        _cleanup()

    return result


def profile_all(
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    include_base: bool = True,
) -> List[Dict[str, Any]]:
    combos: List[tuple] = []
    if include_base:
        for m in BASE_MODELS:
            combos.append((m, None))
    for m in MAMBA_MODELS:
        for v in MAMBA_VARIANTS:
            combos.append((m, v))

    print(f'Profiling {len(combos)} combinations at batch_size={batch_size}')
    print('-' * 80)

    results: List[Dict[str, Any]] = []
    for i, (model_name, mamba_type) in enumerate(combos, 1):
        display = f'{model_name}_{mamba_type}' if mamba_type else model_name
        print(f'[{i:3d}/{len(combos)}] {display:42s}', end=' ', flush=True)

        r = profile_one(model_name, mamba_type, batch_size, device)

        if r['status'] == 'ok':
            print(
                f"OK   peak {r['peak_mem_GB']:6.2f} GB  "
                f"act {r['activation_mem_GB']:6.2f} GB  "
                f"{r['params_M']:6.2f}M params  {r['fwd_bwd_sec']:5.2f}s"
            )
        elif r['status'] == 'oom':
            print(f"OOM  {r.get('error', '')[:80]}")
        else:
            print(f"FAIL {r.get('error', '')[:80]}")

        results.append(r)

    # Save CSV + JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'memory_profile.csv'
    json_path = output_dir / 'memory_profile.json'

    _write_csv(results, csv_path)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f'Saved CSV : {csv_path}')
    print(f'Saved JSON: {json_path}')

    return results


def _write_csv(results: List[Dict[str, Any]], path: Path):
    import csv
    # Collect all keys across rows
    keys: List[str] = []
    seen = set()
    for r in results:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def make_figure(results: List[Dict[str, Any]], output_dir: Path, gpu_mem_GB: float):
    """
    Grouped vertical bar chart on log-scale y-axis.
      X-axis: base Mamba-enhanced model
      Colour: SSM variant (Mamba / Mamba-2 / VMamba)
      OOM bars go off-scale with a red 'OOM' label; horizontal dashed line
      marks the GPU capacity.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Keep only Mamba combinations; base models form a separate sanity panel
    mamba_rows = [r for r in results if r['mamba_type'] in MAMBA_VARIANTS]

    # Canonical base-model order (short label: no 'mamba_' prefix)
    base_order = [
        'dense_context_unet', 'nnunet', 'unet_resnet', 'unet_v1',
        'deeplab', 'unet_v2', 'fpn', 'swin_unet', 'transunet', 'pure_mamba',
    ]

    def _short(model_name: str) -> str:
        s = model_name.replace('mamba_', '', 1)
        if s == 'pure_mamba_unet':
            return 'pure_mamba'
        return s

    variants = ['mamba', 'mamba2', 'vmamba']
    colors = {'mamba': '#1f77b4', 'mamba2': '#ff7f0e', 'vmamba': '#2ca02c'}
    labels = {'mamba': 'Mamba (S6)', 'mamba2': 'Mamba-2 (SSD)', 'vmamba': 'VMamba (SS2D)'}

    # Build a lookup: (short_base, variant) -> row
    lookup = {}
    for r in mamba_rows:
        lookup[(_short(r['model_name']), r['mamba_type'])] = r

    bases = [b for b in base_order if any((b, v) in lookup for v in variants)]

    # Compute tallest successful bar to scale OOM markers
    ok_mems = [r['peak_mem_GB'] for r in mamba_rows if r['status'] == 'ok']
    max_ok = max(ok_mems) if ok_mems else 1.0
    oom_height = max(gpu_mem_GB * 1.5, max_ok * 10)  # deliberately off-scale

    fig, ax = plt.subplots(figsize=(11, 5.5))

    x = np.arange(len(bases))
    width = 0.27

    for j, variant in enumerate(variants):
        heights = []
        is_oom_flags = []
        for base in bases:
            r = lookup.get((base, variant))
            if r is None or r['status'] == 'error':
                heights.append(0)
                is_oom_flags.append(False)
            elif r['status'] == 'oom':
                heights.append(oom_height)
                is_oom_flags.append(True)
            else:
                heights.append(r['peak_mem_GB'])
                is_oom_flags.append(False)

        xpos = x + (j - 1) * width
        bars = ax.bar(
            xpos, heights, width,
            color=colors[variant], label=labels[variant],
            edgecolor='black', linewidth=0.4,
        )

        for bar, is_oom in zip(bars, is_oom_flags):
            if is_oom:
                bar.set_hatch('///')
                bar.set_edgecolor('red')
                bar.set_linewidth(1.5)
                ax.annotate(
                    'OOM',
                    xy=(bar.get_x() + bar.get_width() / 2, gpu_mem_GB * 1.05),
                    ha='center', va='bottom',
                    color='red', fontweight='bold', fontsize=10,
                )

    # GPU capacity horizontal line
    ax.axhline(
        gpu_mem_GB, color='red', linestyle='--', linewidth=1.2, alpha=0.75,
        zorder=1,
    )
    ax.text(
        len(bases) - 0.5, gpu_mem_GB, f' GPU capacity: {gpu_mem_GB:.0f} GB',
        ha='right', va='bottom', color='red', fontsize=9, alpha=0.8,
    )

    ax.set_yscale('log')
    ax.set_ylim(bottom=0.1, top=oom_height * 2)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [b.replace('_', '\n') for b in bases],
        fontsize=9, rotation=0,
    )
    ax.set_ylabel('Peak forward+backward GPU memory (GB, log scale)', fontsize=10)
    batch = mamba_rows[0]['batch_size'] if mamba_rows else '?'
    ax.set_title(
        f'Training-time memory footprint per (base × SSM variant) at batch size {batch}',
        fontsize=11,
    )
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.35, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    pdf_path = output_dir / 'memory_profile.pdf'
    png_path = output_dir / 'memory_profile.png'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved fig : {pdf_path}')
    print(f'Saved fig : {png_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Memory profiling for Mamba-enhanced cardiac models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for profiling (small so most configs fit)')
    parser.add_argument('--output_dir', type=str, default='./results/memory_profile',
                        help='Where to write CSV, JSON, and figure')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu (cpu only for dry-run)')
    parser.add_argument('--skip_base', action='store_true',
                        help='Skip base (non-Mamba) models')
    parser.add_argument('--no_figure', action='store_true',
                        help='Do not generate the matplotlib figure')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('ERROR: CUDA requested but not available. Use --device cpu for dry-run only.')
        sys.exit(1)

    device = torch.device(args.device)
    print(f'Device    : {device}')
    if torch.cuda.is_available() and device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        gpu_mem_GB = props.total_memory / 1e9
        print(f'GPU       : {props.name}')
        print(f'GPU memory: {gpu_mem_GB:.1f} GB')
    else:
        gpu_mem_GB = 0.0
    print(f'Batch size: {args.batch_size}')
    print()

    output_dir = Path(args.output_dir)
    results = profile_all(
        args.batch_size, device, output_dir,
        include_base=not args.skip_base,
    )

    if not args.no_figure and gpu_mem_GB > 0:
        make_figure(results, output_dir, gpu_mem_GB)

    # Summary: how many OOMed?
    statuses: Dict[str, int] = {}
    for r in results:
        statuses[r['status']] = statuses.get(r['status'], 0) + 1
    print()
    print('Summary:', {k: v for k, v in statuses.items()})


if __name__ == '__main__':
    main()
