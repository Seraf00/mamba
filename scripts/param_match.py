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
    ('gudu', 'mamba_gudu'),
]


def count_params(model):
    """Count total model parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def find_matched_base_features(base_name, target_params_m, min_bf=32, max_bf=256):
    """
    Binary search for base_features that gives target param count.

    Args:
        base_name: Base model name (e.g., 'unet_v1')
        target_params_m: Target parameter count in millions
        min_bf: Minimum base_features to try
        max_bf: Maximum base_features to try

    Returns:
        Best base_features value and actual param count
    """
    best_bf = 64
    best_diff = float('inf')
    best_params = 0

    # Try powers of 2 and midpoints for clean architecture
    candidates = sorted(set(
        list(range(min_bf, max_bf + 1, 8)) +  # every 8
        [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]  # key values
    ))

    for bf in candidates:
        try:
            kwargs = {'in_channels': 1, 'num_classes': 4, 'base_features': bf}
            model = get_model(base_name, **kwargs)
            params = count_params(model)
            diff = abs(params - target_params_m)

            if diff < best_diff:
                best_diff = diff
                best_bf = bf
                best_params = params

            del model
        except Exception:
            continue

    return best_bf, best_params


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

    print(f"\n{'Base Model':<20} {'Base (64)':<14} {'Mamba':>14} {'Increase':>10} "
          f"{'Matched BF':>12} {'Matched Params':>16}")
    print("-" * 90)

    for base_name, mamba_name in MODEL_PAIRS:
        try:
            # Standard base model (base_features=64)
            base_model = get_model(base_name, in_channels=1, num_classes=4)
            base_params = count_params(base_model)
            del base_model

            # Mamba-enhanced model
            mamba_model = get_model(mamba_name, in_channels=1, num_classes=4,
                                   mamba_type=args.mamba_type)
            mamba_params = count_params(mamba_model)
            del mamba_model

            increase_pct = ((mamba_params - base_params) / base_params) * 100

            # Find parameter-matched base_features
            matched_bf, matched_params = find_matched_base_features(
                base_name, mamba_params
            )

            match_error = abs(matched_params - mamba_params) / mamba_params * 100

            print(f"{base_name:<20} {base_params:>10.2f}M   {mamba_params:>10.2f}M  "
                  f"{increase_pct:>+8.1f}%   bf={matched_bf:<8d} {matched_params:>10.2f}M "
                  f"({'OK' if match_error < 10 else 'APPROX'})")

            results.append({
                'base_name': base_name,
                'mamba_name': mamba_name,
                'mamba_type': args.mamba_type,
                'base_params_M': round(base_params, 2),
                'mamba_params_M': round(mamba_params, 2),
                'param_increase_pct': round(increase_pct, 1),
                'matched_base_features': matched_bf,
                'matched_params_M': round(matched_params, 2),
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
            'param_matched_training': [
                {
                    'model': r['base_name'],
                    'base_features': r['matched_base_features'],
                    'display_name': f"{r['base_name']}_wide",
                    'target_params_M': r['mamba_params_M'],
                }
                for r in results
            ]
        }
        with open(args.output_json, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nConfig saved to {args.output_json}")

    print("\n" + "=" * 80)
    print("PAPER TABLE FORMAT (copy-paste for LaTeX)")
    print("=" * 80)
    print(f"{'Model':<25} & {'Params (M)':>12} & {'Dice':>8} & {'HD95':>8} \\\\")
    print("\\midrule")
    for r in results:
        print(f"{r['base_name']:<25} & {r['base_params_M']:>10.2f}  &          &          \\\\")
        print(f"{r['base_name']}_wide (bf={r['matched_base_features']}){'':<1} & "
              f"{r['matched_params_M']:>10.2f}  &          &          \\\\")
        print(f"{r['mamba_name']}_{r['mamba_type']:<6} & "
              f"{r['mamba_params_M']:>10.2f}  &          &          \\\\")
        print("\\midrule")


if __name__ == '__main__':
    main()
