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
# - Transformer models widen along the one axis that keeps their pretrained
#   weights loadable (see ``_TRANSUNET_SWEEP`` below).

_BF_SWEEP = [{'base_features': bf} for bf in
             sorted(set(list(range(32, 257, 8)) +
                        [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]))]

# resnet50 ~ 25M params, resnet101 ~ 44M, resnet152 ~ 60M
_RESNET_BACKBONES = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# TransUNet widens by DEPTH, not width. Width (vit_dim) would land just as close
# on parameters -- vit_dim=1128 is -0.56% against depth's -0.44% -- but it makes
# every ViT-B/16 tensor the wrong shape, so the control would be a randomly
# initialised transformer competing against a pretrained one and the extra
# parameters would not be what the comparison measured. Depth keeps the first 12
# blocks pretrained and adds fresh ones, which is how the Mamba variant adds its
# capacity too. It also leaves head_dim at 64, so the control does not perturb
# the head-dimension analysis behind the Mamba-2 shared-memory ceiling.
_TRANSUNET_SWEEP = [{'vit_layers': n} for n in range(12, 41)]

# SwinUNet widens by DEPTH of stage 3, which is how the Swin family itself
# scales (Swin-T [2,2,6,2] -> Swin-S [2,2,18,2]). An earlier version widened
# embed_dim (96 -> 114): it matched parameters, but Swin-Tiny's pretrained
# weights are 96-wide, so the model could not even be constructed with
# pretraining and the R3 run failed at load time. That was missed because this
# sweep counts parameters with pretrained=False -- see _verify_buildable().
_SWIN_SWEEP = [{'depths': [2, 2, d, 2]} for d in range(6, 32, 2)]

WIDEN_SEARCH_SPACE = {
    # Continuous-width models
    'unet_v1':            _BF_SWEEP,
    'unet_v2':            _BF_SWEEP,
    'nnunet':             _BF_SWEEP,
    'dense_context_unet': _BF_SWEEP,
    # Discrete pretrained-CNN backbones
    'unet_resnet':        [{'backbone': b} for b in _RESNET_BACKBONES],
    # fpn_channels used to stop at 512, which capped the search at 75.1M against
    # a 173.1M target and produced the "APPROX" row that missed by 57%.
    'fpn':                [{'backbone': b, 'fpn_channels': fpn}
                           for b in _RESNET_BACKBONES
                           for fpn in (128, 256, 384, 512, 768, 896,
                                       1024, 1152, 1280)],
    'deeplab_v3':         [{'backbone': 'resnet50'}, {'backbone': 'resnet101'}],
    # Pretrained-Transformer models
    'swin_unet':          _SWIN_SWEEP,
    'transunet':          _TRANSUNET_SWEEP,
}

# Below this parameter increase there is nothing for a widened control to
# control for: the question "is the gain from the extra parameters?" has the
# answer "there are no meaningful extra parameters". nnU-Net's Mamba variant is
# actually SMALLER than its baseline (-1.9%), and DeepLabV3+'s is +4.1%. Training
# a "wide" arm for those burns GPU hours to produce a model that is identical to
# the baseline, which is exactly what the previous param_config did for three of
# its seven entries.
MIN_INCREASE_PCT_FOR_CONTROL = 15.0


def _build_kwargs(base_name, trial_kwargs):
    """Kwargs for a parameter-count probe.

    ``pretrained=False`` because the count is identical either way and the
    alternative is re-downloading ViT-B/16 once per trial -- 29 times for the
    TransUNet depth sweep alone. SwinUNet is a fixed 224 px model.
    """
    kw = {'in_channels': 1, 'num_classes': 4, 'pretrained': False, **trial_kwargs}
    if base_name == 'swin_unet':
        kw.setdefault('img_size', 224)
    return kw


def find_matched_widening(base_name, target_params_m, base_params_m=None):
    """
    Search the per-model widening space for the kwargs that give the param
    count closest to ``target_params_m``.

    Returns (kwargs_dict_or_None, actual_params_M, status_str). ``status_str``
    is one of ``'OK'`` (within 10%), ``'APPROX'`` (best-effort), ``'N/A'`` (no
    widening axis for this model), or ``'UNNEEDED'`` (the Mamba variant adds
    too few parameters for a widened control to answer anything).
    """
    space = WIDEN_SEARCH_SPACE.get(base_name)
    if space is None:
        return None, 0.0, 'N/A'

    # A control only earns its GPU hours if there is a parameter gap to explain.
    if base_params_m:
        increase = (target_params_m - base_params_m) / base_params_m * 100
        if increase < MIN_INCREASE_PCT_FOR_CONTROL:
            return None, 0.0, 'UNNEEDED'

    best_kwargs = None
    best_diff = float('inf')
    best_params = 0.0

    for trial_kwargs in space:
        try:
            model = get_model(base_name, **_build_kwargs(base_name, trial_kwargs))
            params = count_params(model)
            diff = abs(params - target_params_m)

            if diff < best_diff:
                best_diff = diff
                best_kwargs = trial_kwargs
                best_params = params

            del model
        except Exception:
            # Some kwarg combos may be invalid (e.g. encoder/fpn mismatch, or a
            # SwinUNet embed_dim the PatchExpanding rearrange rejects); skip and
            # keep searching the rest of the space.
            continue

    if best_kwargs is None:
        return None, 0.0, 'N/A'

    # A "widening" that reproduces the model's own defaults is not a control --
    # it is the baseline under a second name. That is what shipped for
    # unet_resnet (resnet34), deeplab_v3 (resnet50) and nnunet (bf=32).
    if base_params_m and abs(best_params - base_params_m) < 1e-6:
        return None, best_params, 'UNNEEDED'

    problem = _verify_buildable(base_name, best_kwargs)
    if problem:
        return best_kwargs, best_params, f'BROKEN: {problem}'

    err_pct = (best_diff / target_params_m) * 100 if target_params_m > 0 else 0
    status = 'OK' if err_pct < 10 else 'APPROX'
    return best_kwargs, best_params, status


def _verify_buildable(base_name, kwargs):
    """Build the chosen control EXACTLY as training will: pretrained weights on.

    The sweep counts parameters with pretrained=False (fast, no downloads), and
    that is how embed_dim=114 for SwinUNet got through: it counts fine, but
    Swin-Tiny's 96-wide weights cannot load into it, so training failed at
    construction. Returns an error string, or None if the model builds.
    """
    kw = {'in_channels': 1, 'num_classes': 4, **kwargs}
    if base_name == 'swin_unet':
        kw.setdefault('img_size', 224)
    try:
        m = get_model(base_name, **kw)
        del m
        return None
    except Exception as e:  # noqa: BLE001
        return f'{type(e).__name__}: {str(e)[:120]}'


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
                base_name, mamba_params, base_params
            )

            label = _kwargs_label(matched_kwargs)
            if status in ('N/A', 'UNNEEDED'):
                why = ('no widening axis' if status == 'N/A'
                       else f'only {increase_pct:+.1f}% to explain')
                print(f"{base_name:<20} {base_params:>10.2f}M  {mamba_params:>10.2f}M  "
                      f"{increase_pct:>+8.1f}%   {status + ' (' + why + ')':<28} "
                      f"{'-':>16}")
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
            'min_increase_pct_for_control': MIN_INCREASE_PCT_FOR_CONTROL,
            # Only emit training entries for models that can be widened AND
            # where a widened control answers something. N/A means no widening
            # axis exists; UNNEEDED means the Mamba variant adds under
            # MIN_INCREASE_PCT_FOR_CONTROL, so the baseline already IS the
            # control. Both are reported in the paper as a sentence rather than
            # trained -- which is the difference between this config and the one
            # it replaces, where three of seven entries silently reproduced the
            # baseline's own defaults.
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
                if r['match_status'] not in ('N/A', 'UNNEEDED')
                and not str(r['match_status']).startswith('BROKEN')
                and r['matched_kwargs'] is not None
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
        if r['match_status'] in ('N/A', 'UNNEEDED'):
            note = ('no widening axis' if r['match_status'] == 'N/A'
                    else f"only {r['param_increase_pct']:+.1f}% to explain")
            print(f"{r['base_name'] + '_wide (' + r['match_status'] + ')':<35} & "
                  f"{'—':>10}    &          &          \\\\  % {note}")
        else:
            wide_label = f"{r['base_name']}_wide ({r['matched_label']})"
            print(f"{wide_label:<35} & {r['matched_params_M']:>10.2f}  &          &          \\\\")
        # Mamba row
        mamba_label = f"{r['mamba_name']}_{r['mamba_type']}"
        print(f"{mamba_label:<35} & {r['mamba_params_M']:>10.2f}  &          &          \\\\")
        print("\\midrule")


if __name__ == '__main__':
    main()
