#!/usr/bin/env python3
"""
Run explainability analysis on ALL trained models.

Discovers trained checkpoints across experiment directories and generates
Grad-CAM, uncertainty, Mamba state, and attention visualizations for each.
Produces a comparison grid figure for the paper.

Usage:
    python scripts/explain_all_models.py \
        --results_dirs /content/results/base_models /content/results/mamba_models \
        --data_dir /content/mamba/data/CAMUS/ \
        --output /content/results/explainability/
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model
from data import CAMUSDataset, get_transforms


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(results_dirs: List[str]) -> List[Dict]:
    """Find all best_model.pth checkpoints with their metadata."""
    checkpoints = []
    for results_dir in results_dirs:
        rdir = Path(results_dir)
        if not rdir.exists():
            print(f"  Skipping {rdir} (not found)")
            continue
        for ckpt_path in sorted(rdir.rglob('best_model.pth')):
            model_dir = ckpt_path.parent
            results_file = model_dir / 'results.json'
            meta = {}
            if results_file.exists():
                with open(results_file) as f:
                    meta = json.load(f)

            display_name = meta.get('display_name', model_dir.name)
            model_name = meta.get('model_name', display_name.split('_mamba')[0] if '_mamba' not in display_name else display_name)
            mamba_type = meta.get('mamba_type', None)

            checkpoints.append({
                'checkpoint': str(ckpt_path),
                'model_name': model_name,
                'display_name': display_name,
                'mamba_type': mamba_type,
                'model_dir': str(model_dir),
            })

    return checkpoints


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(ckpt_info: Dict, device: str):
    """Load a model from checkpoint info dict."""
    checkpoint = torch.load(ckpt_info['checkpoint'], map_location=device, weights_only=False)

    model_name = ckpt_info['model_name']
    mamba_type = ckpt_info['mamba_type']

    kwargs = {'in_channels': 1, 'num_classes': 4}
    if mamba_type:
        kwargs['mamba_type'] = mamba_type

    model = get_model(model_name, **kwargs)

    # Load weights (handle different checkpoint formats)
    state_dict = checkpoint.get('model_state_dict',
                  checkpoint.get('state_dict', checkpoint))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def get_test_samples(data_dir: str, n_samples: int = 3, seed: int = 42) -> List[Dict]:
    """Load a few test samples for explainability.

    NOTE: ``CAMUSDataset`` returns a ``(image, mask)`` tuple by default for
    training-loop compatibility, and only switches to a dict (with metadata
    such as ``patient_id``/``pixel_spacing``) when ``include_info=True``.
    Explainability needs metadata, so we request the dict form. We also
    fall back gracefully if a future version returns the tuple form.
    """
    transform = get_transforms(split='val', img_size=(256, 256))
    dataset = CAMUSDataset(
        root_dir=data_dir,
        split='test',
        transform=transform,
        include_info=True,
    )

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    samples = []
    for idx in indices:
        sample = dataset[idx]

        # Defensive: accept either dict (include_info=True) or (image, mask) tuple
        if isinstance(sample, dict):
            image = sample['image']
            mask = sample.get('mask', None)
            patient_id = sample.get('patient_id', f'sample_{idx}')
        elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
            image, mask = sample[0], sample[1]
            patient_id = f'sample_{idx}'
        else:
            raise TypeError(
                f"Unexpected sample type from CAMUSDataset[{idx}]: {type(sample)}"
            )

        # Add batch dim -> (1, 1, H, W)
        image = image.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

        samples.append({
            'image': image,
            'mask': mask,
            'patient_id': patient_id,
            'idx': int(idx),
        })

    return samples


# ---------------------------------------------------------------------------
# Per-model explainability
# ---------------------------------------------------------------------------

def run_gradcam_for_model(model, image: torch.Tensor, target_class: int = 1):
    """Run Grad-CAM and return the heatmap as numpy array."""
    from explainability import GradCAM

    # Find last conv layer
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        return None

    try:
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate(image, target_class=target_class)
        return cam
    except Exception as e:
        print(f"    Grad-CAM error: {e}")
        return None


def run_uncertainty_for_model(model, image: torch.Tensor, n_samples: int = 10):
    """Run MC Dropout uncertainty and return uncertainty map."""
    from explainability import UncertaintyEstimator

    try:
        estimator = UncertaintyEstimator(model, n_samples=n_samples)
        pred, probs, uncertainty = estimator.predict_with_uncertainty(image)
        return {
            'prediction': pred.squeeze().cpu().numpy(),
            'confidence': probs.max(dim=1)[0].squeeze().cpu().numpy(),
            'uncertainty': uncertainty.squeeze().cpu().numpy(),
        }
    except Exception as e:
        print(f"    Uncertainty error: {e}")
        return None


def run_mamba_states_for_model(model, image: torch.Tensor):
    """Run Mamba state visualization, return figure or None."""
    from explainability import MambaStateVisualizer

    try:
        visualizer = MambaStateVisualizer(model)
        fig = visualizer.visualize_state_evolution(image)
        return fig
    except Exception as e:
        print(f"    Mamba state error: {e}")
        return None


# ---------------------------------------------------------------------------
# Comparison grid figure
# ---------------------------------------------------------------------------

def create_gradcam_comparison_grid(
    all_results: List[Dict],
    samples: List[Dict],
    output_dir: Path,
    target_class: int = 1
):
    """
    Create a grid figure: rows=models, cols=samples.
    Each cell shows Grad-CAM overlay.
    """
    n_models = len(all_results)
    n_samples = len(samples)

    if n_models == 0 or n_samples == 0:
        return

    fig, axes = plt.subplots(
        n_models + 1, n_samples,
        figsize=(4 * n_samples, 3 * (n_models + 1)),
        squeeze=False
    )

    class_names = {1: 'LV', 2: 'MYO', 3: 'LA'}
    class_label = class_names.get(target_class, f'Class {target_class}')

    # First row: input images
    for j, sample in enumerate(samples):
        img_np = sample['image'].squeeze().cpu().numpy()
        axes[0][j].imshow(img_np, cmap='gray')
        axes[0][j].set_title(f"Patient {sample['patient_id']}", fontsize=9)
        axes[0][j].axis('off')
        if j == 0:
            axes[0][j].set_ylabel('Input', fontsize=10, fontweight='bold')

    # Model rows
    for i, result in enumerate(all_results):
        display_name = result['display_name']
        row = i + 1

        for j in range(n_samples):
            cam = result['gradcams'][j]
            img_np = samples[j]['image'].squeeze().cpu().numpy()

            if cam is not None:
                axes[row][j].imshow(img_np, cmap='gray')
                axes[row][j].imshow(cam, cmap='jet', alpha=0.5)
            else:
                axes[row][j].imshow(img_np, cmap='gray')
                axes[row][j].text(0.5, 0.5, 'N/A', transform=axes[row][j].transAxes,
                                  ha='center', va='center', fontsize=14, color='red')

            axes[row][j].axis('off')

        # Model name as row label
        axes[row][0].set_ylabel(display_name, fontsize=8, fontweight='bold', rotation=90)

    plt.suptitle(f'Grad-CAM Comparison — Target: {class_label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'gradcam_comparison_class{target_class}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / f'gradcam_comparison_class{target_class}.png'}")


def create_uncertainty_comparison_grid(
    all_results: List[Dict],
    samples: List[Dict],
    output_dir: Path
):
    """
    Create uncertainty comparison grid: rows=models, cols=samples.
    Each cell shows uncertainty heatmap.
    """
    n_models = len(all_results)
    n_samples = len(samples)

    if n_models == 0 or n_samples == 0:
        return

    fig, axes = plt.subplots(
        n_models + 1, n_samples,
        figsize=(4 * n_samples, 3 * (n_models + 1)),
        squeeze=False
    )

    # First row: input images
    for j, sample in enumerate(samples):
        img_np = sample['image'].squeeze().cpu().numpy()
        axes[0][j].imshow(img_np, cmap='gray')
        axes[0][j].set_title(f"Patient {sample['patient_id']}", fontsize=9)
        axes[0][j].axis('off')

    # Model rows
    for i, result in enumerate(all_results):
        display_name = result['display_name']
        row = i + 1

        for j in range(n_samples):
            unc_data = result['uncertainties'][j]

            if unc_data is not None:
                axes[row][j].imshow(unc_data['uncertainty'], cmap='hot')
            else:
                img_np = samples[j]['image'].squeeze().cpu().numpy()
                axes[row][j].imshow(img_np, cmap='gray')
                axes[row][j].text(0.5, 0.5, 'N/A', transform=axes[row][j].transAxes,
                                  ha='center', va='center', fontsize=14, color='red')

            axes[row][j].axis('off')

        axes[row][0].set_ylabel(display_name, fontsize=8, fontweight='bold', rotation=90)

    plt.suptitle('Uncertainty Comparison (MC Dropout)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'uncertainty_comparison.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Run explainability on all trained models')
    parser.add_argument('--results_dirs', type=str, nargs='+', required=True,
                        help='Directories containing trained model checkpoints')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to CAMUS dataset')
    parser.add_argument('--output', type=str, default='results/explainability',
                        help='Output directory')
    parser.add_argument('--n_samples', type=int, default=3,
                        help='Number of test samples to visualize')
    parser.add_argument('--n_mc_samples', type=int, default=10,
                        help='MC Dropout samples for uncertainty')
    parser.add_argument('--target_class', type=int, default=1,
                        help='Target class for Grad-CAM (1=LV, 2=MYO, 3=LA)')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['gradcam', 'uncertainty'],
                        choices=['gradcam', 'uncertainty', 'mamba_states', 'all'],
                        help='Explainability methods to run')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if 'all' in args.methods:
        args.methods = ['gradcam', 'uncertainty', 'mamba_states']

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Discover checkpoints
    print("Discovering trained models...")
    checkpoints = discover_checkpoints(args.results_dirs)
    print(f"Found {len(checkpoints)} trained models:")
    for ckpt in checkpoints:
        print(f"  - {ckpt['display_name']}")

    if not checkpoints:
        print("No checkpoints found!")
        return

    # Load test samples (same samples for all models)
    print(f"\nLoading {args.n_samples} test samples...")
    samples = get_test_samples(args.data_dir, n_samples=args.n_samples, seed=args.seed)
    for s in samples:
        s['image'] = s['image'].to(device)
    print(f"Loaded {len(samples)} samples")

    # Run explainability on each model
    all_results = []

    for i, ckpt_info in enumerate(checkpoints, 1):
        display_name = ckpt_info['display_name']
        is_mamba = ckpt_info['mamba_type'] is not None

        print(f"\n[{i}/{len(checkpoints)}] {display_name}")

        # Load model
        try:
            model = load_model_from_checkpoint(ckpt_info, str(device))
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        result = {
            'display_name': display_name,
            'model_name': ckpt_info['model_name'],
            'is_mamba': is_mamba,
            'gradcams': [],
            'uncertainties': [],
        }

        # Per-model output directory
        model_out = output_dir / display_name
        model_out.mkdir(parents=True, exist_ok=True)

        for j, sample in enumerate(samples):
            image = sample['image']

            # Grad-CAM
            if 'gradcam' in args.methods:
                cam = run_gradcam_for_model(model, image, target_class=args.target_class)
                result['gradcams'].append(cam)

                # Save individual
                if cam is not None:
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                    ax.imshow(image.squeeze().cpu().numpy(), cmap='gray')
                    ax.imshow(cam, cmap='jet', alpha=0.5)
                    ax.set_title(f'{display_name} — Patient {sample["patient_id"]}', fontsize=9)
                    ax.axis('off')
                    plt.savefig(model_out / f'gradcam_sample{j}.png', dpi=150, bbox_inches='tight')
                    plt.close()

            # Uncertainty
            if 'uncertainty' in args.methods:
                unc = run_uncertainty_for_model(model, image, n_samples=args.n_mc_samples)
                result['uncertainties'].append(unc)

                if unc is not None:
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(unc['prediction'], cmap='viridis')
                    axes[0].set_title('Prediction')
                    axes[0].axis('off')
                    axes[1].imshow(unc['confidence'], cmap='RdYlGn', vmin=0, vmax=1)
                    axes[1].set_title('Confidence')
                    axes[1].axis('off')
                    axes[2].imshow(unc['uncertainty'], cmap='hot')
                    axes[2].set_title('Uncertainty')
                    axes[2].axis('off')
                    plt.suptitle(f'{display_name} — Patient {sample["patient_id"]}', fontsize=10)
                    plt.savefig(model_out / f'uncertainty_sample{j}.png', dpi=150, bbox_inches='tight')
                    plt.close()

        # Mamba states (only for Mamba models, one sample)
        if 'mamba_states' in args.methods and is_mamba:
            fig = run_mamba_states_for_model(model, samples[0]['image'])
            if fig is not None:
                fig.savefig(model_out / 'mamba_states.png', dpi=150, bbox_inches='tight')
                plt.close(fig)

        all_results.append(result)

        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Create comparison grids
    print(f"\n{'='*60}")
    print("Creating comparison figures...")
    print(f"{'='*60}")

    if 'gradcam' in args.methods:
        create_gradcam_comparison_grid(all_results, samples, output_dir, args.target_class)

    if 'uncertainty' in args.methods:
        create_uncertainty_comparison_grid(all_results, samples, output_dir)

    # Save summary JSON
    summary = []
    for r in all_results:
        entry = {
            'display_name': r['display_name'],
            'model_name': r['model_name'],
            'is_mamba': r['is_mamba'],
            'gradcam_available': any(c is not None for c in r.get('gradcams', [])),
            'uncertainty_available': any(u is not None for u in r.get('uncertainties', [])),
        }
        # Mean uncertainty per model (if available)
        uncertainties = [u['uncertainty'].mean() for u in r.get('uncertainties', []) if u is not None]
        if uncertainties:
            entry['mean_uncertainty'] = float(np.mean(uncertainties))
        summary.append(entry)

    with open(output_dir / 'explainability_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! All results saved to {output_dir}")
    print(f"  - Per-model visualizations: {output_dir}/<model_name>/")
    print(f"  - Grad-CAM comparison grid: gradcam_comparison_class{args.target_class}.png")
    print(f"  - Uncertainty comparison grid: uncertainty_comparison.png")
    print(f"  - Summary: explainability_summary.json")


if __name__ == '__main__':
    main()
