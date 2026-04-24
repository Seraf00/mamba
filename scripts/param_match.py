#!/usr/bin/env python3
"""
Parameter-matched baseline computation.

For fair comparison between base and Mamba-enhanced models, this script:
1. Computes parameter counts for all model pairs
2. Finds the base_features value that makes base models parameter-matched
   with their Mamba-enhanced counterparts
3. Generates a training config for parameter-matched experiments

Usage:
    python scripts/param_match.py
    python scripts/param_match.py --output_json param_config.json

A Q1 reviewer will ask: "Is the improvement from Mamba or from more parameters?"
This script provides the answer by enabling fair comparison at equal capacity.
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models import get_model


# Base-to-Mamba model pairs
MODEL_PAIRS = [
    ('unet_v1', 'mamba_unet_v1'),
    ('unet_v2', 'mamba_unet_v2'),
    ('unet_resnet', 'mamba_unet_resnet'),
    ('deeplab_v3', 'mamba_deeplab'),
    ('fpn', 'mamba_fpn'),
    ('nnunet', 'mamba_nnunet'),
    ('swin_unet', 'mamba_swin_unet'),
    ('transunet', 'mamba_transunet'),
    ('dense_context_unet', 'mamba_dense_context_unet'),
]


def count_params(model):
    """Count total model parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


# Per-model widening search space. Each entry is a list of kwargs dicts to try.
# - Models with a ``base_features`` knob get a continuous sweep.
# - Pretrained-encoder models get a discrete backbone tier sweep.
# - Models tied to a fixed pretrained Transformer (Swin-Tiny / ViT-B/16) cannot
#   be honestly param-matched by widening — changing dims forfeits the
#   pretrained weights. We mark those ``None`` and report ``N/A`` in the table.

_BF_SWEEP = [{'base_features': bf} for bf in
             sorted(set(list(range(32, 257, 8)) +
                        [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]))]

# resnet50 ~ 25M params, resnet101 ~ 44M, resnet152 ~ 60M
_RESNET_BACKBONES = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

WIDEN_SEARCH_SPACE = {
    # Continuous-width models
    'unet_v1':            _BF_SWEEP,
    'unet_v2':            _BF_SWEEP,
    'nnunet':             _BF_SWEEP,
    'dense_context_unet': _BF_SWEEP,
    # Discrete pretrained-CNN backbones
    'unet_resnet':        [{'backbone': b} for b in _RESNET_BACKBONES],
    'fpn':                [{'backbone': b, 'fpn_channels': fpn}
                           for b in _RESNET_BACKBONES
                           for fpn in (128, 256, 384, 512)],
    'deeplab_v3':         [{'backbone': 'resnet50'}, {'backbone': 'resnet101'}],
    # Pretrained-Transformer models — widening forfeits pretrained weights
    'swin_unet':          None,
    'transunet':          None,
}


def find_matched_widening(base_name, target_params_m):
    """
    Search the per-model widening space for the kwargs that give the param
    count closest to ``target_params_m``.

    Returns (kwargs_dict_or_None, actual_params_M, status_str). ``status_str``
    is one of ``'OK'`` (within 10%), ``'APPROX'`` (best-effort), or ``'N/A'``
    (model is tied to fixed pretrained weights and cannot be widened cleanly).
    """
    space = WIDEN_SEARCH_SPACE.get(base_name)
    if space is None:
        # Pretrained-Transformer model — no honest widening
        return None, 0.0, 'N/A'

    best_kwargs = None
    best_diff = float('inf')
    best_params = 0.0

    for trial_kwargs in space:
        try:
            full_kwargs = {'in_channels': 1, 'num_classes': 4, **trial_kwargs}
            model = get_model(base_name, **full_kwargs)
            params = count_params(model)
            diff = abs(params - target_params_m)

            if diff < best_diff:
                best_diff = diff
                best_kwargs = trial_kwargs
                best_params = params

            del model
        except Exception:
            # Some kwarg combos may be invalid (e.g. encoder/fpn mismatch);
            # skip and keep searching the rest of the space.
            continue

    if best_kwargs is None:
        return None, 0.0, 'N/A'

    err_pct = (best_diff / target_params_m) * 100 if target_params_m > 0 else 0
    status = 'OK' if err_pct < 10 else 'APPROX'
    return best_kwargs, best_params, status


def _kwargs_label(kwargs):
    """Render a widening-kwargs dict as a short human/LaTeX-friendly tag."""
    if not kwargs:
        return 'N/A'
    parts = []
    for k, v in kwargs.items():
        if k == 'base_features':
            parts.append(f'bf={v}')
        elif k == 'backbone':
            parts.append(str(v))
        elif k == 'fpn_channels':
            parts.append(f'fpn={v}')
        else:
            parts.append(f'{k}={v}')
    return ','.join(parts)


def main():
    parser = argparse.ArgumentParser(description='Compute parameter-matched baselines')
    parser.add_argument('--mamba_type', type=str, default='mamba',
                        choices=['mamba', 'mamba2', 'vmamba'],
                        help='Mamba variant to match against')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Save config to JSON file')
    args = parser.parse_args()

    print("=" * 80)
    print("PARAMETER-MATCHED BASELINE ANALYSIS")
    print(f"Mamba variant: {args.mamba_type}")
    print("=" * 80)

    results = []

    print(f"\n{'Base Model':<20} {'Base':>10}   {'Mamba':>10}   {'Increase':>9}   "
          f"{'Widened (kwargs)':<28} {'Matched Params':>16}")
    print("-" * 100)

    for base_name, mamba_name in MODEL_PAIRS:
        try:
            # Standard base model (defaults — typically base_features=64 or
            # the model's default backbone)
            base_model = get_model(base_name, in_channels=1, num_classes=4)
            base_params = count_params(base_model)
            del base_model

            # Mamba-enhanced model
            mamba_model = get_model(mamba_name, in_channels=1, num_classes=4,
                                   mamba_type=args.mamba_type)
            mamba_params = count_params(mamba_model)
            del mamba_model

            increase_pct = ((mamba_params - base_params) / base_params) * 100

            # Find parameter-matched widened baseline
            matched_kwargs, matched_params, status = find_matched_widening(
                base_name, mamba_params
            )

            label = _kwargs_label(matched_kwargs)
            if status == 'N/A':
                print(f"{base_name:<20} {base_params:>10.2f}M  {mamba_params:>10.2f}M  "
                      f"{increase_pct:>+8.1f}%   {'N/A (pretrained-tied)':<28} "
                      f"{'—':>16}  (N/A)")
            else:
                print(f"{base_name:<20} {base_params:>10.2f}M  {mamba_params:>10.2f}M  "
                      f"{increase_pct:>+8.1f}%   {label:<28} "
                      f"{matched_params:>10.2f}M  ({status})")

            results.append({
                'base_name': base_name,
                'mamba_name': mamba_name,
                'mamba_type': args.mamba_type,
                'base_params_M': round(base_params, 2),
                'mamba_params_M': round(mamba_params, 2),
                'param_increase_pct': round(increase_pct, 1),
                'matched_kwargs': matched_kwargs,        # dict or None
                'matched_label': label,
                'matched_params_M': round(matched_params, 2),
                'match_status': status,
            })

        except Exception as e:
            print(f"{base_name:<20} ERROR: {e}")

        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print summary
    avg_increase = sum(r['param_increase_pct'] for r in results) / len(results)
    print("-" * 90)
    print(f"\nAverage parameter increase from Mamba: {avg_increase:+.1f}%")

    # Save config
    if args.output_json:
        config = {
            'mamba_type': args.mamba_type,
            'models': results,
            # Only emit training entries for models we can actually widen.
            # Models marked N/A (pretrained-Transformer) are documented in the
            # paper as "param-matching not applicable" rather than retrained.
            'param_matched_training': [
                {
                    'model': r['base_name'],
                    'override_kwargs': r['matched_kwargs'],
                    'display_name': f"{r['base_name']}_wide",
                    'target_params_M': r['mamba_params_M'],
                    'matched_params_M': r['matched_params_M'],
                    'match_status': r['match_status'],
                }
                for r in results
                if r['match_status'] != 'N/A' and r['matched_kwargs'] is not None
            ]
        }
        with open(args.output_json, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nConfig saved to {args.output_json}")

    print("\n" + "=" * 80)
    print("PAPER TABLE FORMAT (copy-paste for LaTeX)")
    print("=" * 80)
    print(f"{'Model':<35} & {'Params (M)':>12} & {'Dice':>8} & {'HD95':>8} \\\\")
    print("\\midrule")
    for r in results:
        # Base row
        print(f"{r['base_name']:<35} & {r['base_params_M']:>10.2f}  &          &          \\\\")
        # Widened (or N/A) row
        if r['match_status'] == 'N/A':
            print(f"{r['base_name'] + '_wide (N/A)':<35} & "
                  f"{'—':>10}    &          &          \\\\  % pretrained-tied")
        else:
            wide_label = f"{r['base_name']}_wide ({r['matched_label']})"
            print(f"{wide_label:<35} & {r['matched_params_M']:>10.2f}  &          &          \\\\")
        # Mamba row
        mamba_label = f"{r['mamba_name']}_{r['mamba_type']}"
        print(f"{mamba_label:<35} & {r['mamba_params_M']:>10.2f}  &          &          \\\\")
        print("\\midrule")


if __name__ == '__main__':
    main()
