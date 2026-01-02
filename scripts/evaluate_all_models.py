#!/usr/bin/env python3
"""
Comprehensive evaluation script for Q1 journal paper.

Evaluates all trained models on the test set and generates publication-ready
results including statistical analysis.

Usage:
    # Evaluate all models from a benchmark run
    python scripts/evaluate_all_models.py --checkpoint_dir ./results/benchmark_XXXXXX
    
    # Evaluate with ejection fraction analysis
    python scripts/evaluate_all_models.py --checkpoint_dir ./results/benchmark_XXXXXX --compute_ef
    
    # Save predictions for visualization
    python scripts/evaluate_all_models.py --checkpoint_dir ./results/benchmark_XXXXXX --save_predictions

Author: Research Team
Date: 2025
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from scipy import stats

from data import CAMUSDataset, get_transforms
from models import get_model
from metrics import SegmentationMetrics, EjectionFractionCalculator
from utils import set_seed, get_device


# Models requiring specific input sizes
SWIN_MODELS = ['swin_unet', 'mamba_swin_unet']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate all trained models for Q1 paper',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing trained model checkpoints')
    parser.add_argument('--data_dir', type=str, default='./data/CAMUS',
                        help='Path to CAMUS dataset')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Dataset split to evaluate')
    
    # Evaluation options
    parser.add_argument('--compute_ef', action='store_true',
                        help='Compute Ejection Fraction metrics')
    parser.add_argument('--per_class', action='store_true', default=True,
                        help='Compute per-class metrics')
    parser.add_argument('--robustness', action='store_true',
                        help='Evaluate robustness to image quality')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: checkpoint_dir/evaluation)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction visualizations')
    
    # System
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def find_model_checkpoints(checkpoint_dir: Path) -> List[Dict[str, Any]]:
    """Find all trained model checkpoints."""
    models = []
    
    for model_dir in sorted(checkpoint_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        
        # Look for best checkpoint
        checkpoint = model_dir / 'best_model.pth'
        if not checkpoint.exists():
            # Try other names
            for name in ['checkpoint_best.pth', 'model_best.pth', 'best.pth']:
                checkpoint = model_dir / name
                if checkpoint.exists():
                    break
        
        if not checkpoint.exists():
            # Find any .pth file
            pth_files = list(model_dir.glob('*.pth'))
            if pth_files:
                checkpoint = pth_files[0]
        
        if checkpoint.exists():
            # Load model info
            results_file = model_dir / 'results.json'
            if results_file.exists():
                with open(results_file) as f:
                    info = json.load(f)
            else:
                info = {'display_name': model_dir.name}
            
            models.append({
                'name': info.get('model_name', model_dir.name.split('_mamba')[0] if '_mamba' in model_dir.name else model_dir.name),
                'display_name': info.get('display_name', model_dir.name),
                'mamba_type': info.get('mamba_type'),
                'checkpoint': checkpoint,
                'dir': model_dir,
                'training_results': info
            })
    
    return models


def get_img_size(model_name: str, default: int = 256) -> int:
    """Get appropriate image size for model."""
    if any(s in model_name for s in SWIN_MODELS):
        return 224
    return default


def evaluate_model(
    model_config: Dict[str, Any],
    dataloader: DataLoader,
    device: torch.device,
    per_class: bool = True,
    compute_ef: bool = False
) -> Dict[str, Any]:
    """Evaluate a single model."""
    model_name = model_config['name']
    mamba_type = model_config['mamba_type']
    checkpoint_path = model_config['checkpoint']
    
    # Create model
    model_kwargs = {'in_channels': 1, 'num_classes': 4}
    if mamba_type:
        model_kwargs['mamba_type'] = mamba_type
    
    model = get_model(model_name, **model_kwargs)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Metrics
    metrics = SegmentationMetrics(num_classes=4, per_class=per_class)
    
    # Collect predictions
    all_preds = []
    all_targets = []
    all_images = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f"Evaluating {model_config['display_name']}", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            preds = outputs.argmax(dim=1)
            metrics.update(preds, masks)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            all_images.append(images.cpu().numpy())
    
    # Compute metrics
    results = metrics.compute()
    
    # Stack predictions
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute per-sample Dice for statistical analysis
    per_sample_dice = compute_per_sample_dice(all_preds, all_targets)
    results['per_sample_dice'] = per_sample_dice.tolist()
    results['dice_std'] = float(np.std(per_sample_dice))
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return results


def compute_per_sample_dice(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute Dice score for each sample."""
    dice_scores = []
    
    for pred, target in zip(preds, targets):
        # Compute mean Dice across classes (excluding background)
        class_dices = []
        for c in range(1, 4):  # LV_endo, LV_epi, LA
            pred_c = (pred == c).astype(float)
            target_c = (target == c).astype(float)
            
            intersection = np.sum(pred_c * target_c)
            union = np.sum(pred_c) + np.sum(target_c)
            
            if union > 0:
                dice = 2 * intersection / union
            else:
                dice = 1.0  # Both empty
            class_dices.append(dice)
        
        dice_scores.append(np.mean(class_dices))
    
    return np.array(dice_scores)


def statistical_comparison(results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Perform statistical comparison between models.
    
    Uses Wilcoxon signed-rank test with Bonferroni correction.
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    
    comparisons = {}
    
    # Pairwise comparisons
    for i in range(n_models):
        for j in range(i + 1, n_models):
            model_a = model_names[i]
            model_b = model_names[j]
            
            dice_a = np.array(results[model_a].get('per_sample_dice', []))
            dice_b = np.array(results[model_b].get('per_sample_dice', []))
            
            if len(dice_a) == len(dice_b) and len(dice_a) > 0:
                try:
                    stat, p_value = stats.wilcoxon(dice_a, dice_b)
                    
                    # Bonferroni correction
                    n_comparisons = n_models * (n_models - 1) // 2
                    p_corrected = min(p_value * n_comparisons, 1.0)
                    
                    comparisons[f"{model_a}_vs_{model_b}"] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'p_corrected': float(p_corrected),
                        'significant': p_corrected < 0.05,
                        'mean_diff': float(np.mean(dice_a) - np.mean(dice_b))
                    }
                except Exception as e:
                    comparisons[f"{model_a}_vs_{model_b}"] = {'error': str(e)}
    
    return comparisons


def create_latex_table(results: Dict[str, Dict], output_path: Path):
    """Create LaTeX table for paper."""
    # Define class names
    classes = ['LV Endo', 'LV Epi', 'LA']
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Segmentation performance on CAMUS test set.}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & " + " & ".join([f"Dice ({c})" for c in classes]) + r" & Mean Dice & Params (M) \\",
        r"\midrule",
    ]
    
    for model_name, metrics in sorted(results.items()):
        if 'error' in metrics:
            continue
        
        # Get per-class Dice
        dice_per_class = metrics.get('dice_per_class', [0, 0, 0])
        if isinstance(dice_per_class, dict):
            dice_per_class = [dice_per_class.get(f'class_{i}', 0) for i in range(1, 4)]
        
        mean_dice = metrics.get('mean_dice', metrics.get('dice', 0))
        params = metrics.get('params_M', 'N/A')
        
        # Format values
        dice_strs = [f"{d:.3f}" for d in dice_per_class[:3]]
        mean_str = f"{mean_dice:.3f}"
        params_str = f"{params:.2f}" if isinstance(params, (int, float)) else str(params)
        
        # Create row
        display_name = model_name.replace('_', r'\_')
        row = f"{display_name} & " + " & ".join(dice_strs) + f" & {mean_str} & {params_str} \\\\"
        lines.append(row)
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"LaTeX table saved to: {output_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Find checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    models = find_model_checkpoints(checkpoint_dir)
    
    if not models:
        print(f"No model checkpoints found in {checkpoint_dir}")
        return
    
    print(f"\nFound {len(models)} trained models:")
    for m in models:
        print(f"  - {m['display_name']}")
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Evaluate each model
    all_results = {}
    
    for model_config in models:
        display_name = model_config['display_name']
        print(f"\n{'='*60}")
        print(f"Evaluating: {display_name}")
        print(f"{'='*60}")
        
        # Get appropriate image size
        img_size = get_img_size(model_config['name'])
        
        # Create dataset
        transform = get_transforms(split='val', img_size=img_size)
        dataset = CAMUSDataset(
            root_dir=args.data_dir,
            split=args.split,
            transform=transform
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        try:
            results = evaluate_model(
                model_config,
                dataloader,
                device,
                per_class=args.per_class,
                compute_ef=args.compute_ef
            )
            
            # Add training info
            if 'training_results' in model_config:
                results['params_M'] = model_config['training_results'].get('num_params', 0) / 1e6
                results['inference_time_ms'] = model_config['training_results'].get('inference_time_ms')
            
            all_results[display_name] = results
            
            # Print results
            print(f"\n  Mean Dice: {results.get('mean_dice', results.get('dice', 0)):.4f} ± {results.get('dice_std', 0):.4f}")
            if 'dice_per_class' in results:
                print(f"  Per-class Dice: {results['dice_per_class']}")
            if 'iou' in results:
                print(f"  Mean IoU: {results['iou']:.4f}")
            if 'hd95' in results:
                print(f"  HD95: {results['hd95']:.2f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            all_results[display_name] = {'error': str(e)}
    
    # Statistical comparison
    print("\n" + "="*60)
    print("Statistical Comparison (Wilcoxon signed-rank test)")
    print("="*60)
    
    stat_results = statistical_comparison(all_results)
    
    significant_pairs = [(k, v) for k, v in stat_results.items() if v.get('significant')]
    if significant_pairs:
        print(f"\nSignificant differences found (p < 0.05, Bonferroni corrected):")
        for pair, res in significant_pairs:
            print(f"  {pair}: p={res['p_corrected']:.4f}, Δ={res['mean_diff']:.4f}")
    else:
        print("\nNo statistically significant differences found.")
    
    # Save all results
    output = {
        'evaluation_date': datetime.now().isoformat(),
        'split': args.split,
        'results': all_results,
        'statistical_comparison': stat_results
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        # Convert numpy arrays to lists
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(output), f, indent=2)
    
    # Create LaTeX table
    create_latex_table(all_results, output_dir / 'results_table.tex')
    
    # Create CSV
    create_csv(all_results, output_dir / 'results.csv')
    
    # Print summary table
    print("\n" + "="*90)
    print("RESULTS SUMMARY")
    print("="*90)
    print(f"\n{'Model':<35} {'Mean Dice':>12} {'Std':>8} {'IoU':>10} {'Params (M)':>12}")
    print("-"*90)
    
    for name, res in sorted(all_results.items(), key=lambda x: -x[1].get('mean_dice', x[1].get('dice', 0)) if 'error' not in x[1] else -999):
        if 'error' in res:
            print(f"{name:<35} {'ERROR':>12}")
        else:
            dice = res.get('mean_dice', res.get('dice', 0))
            std = res.get('dice_std', 0)
            iou = res.get('iou', 0)
            params = res.get('params_M', 'N/A')
            params_str = f"{params:.2f}" if isinstance(params, (int, float)) else str(params)
            print(f"{name:<35} {dice:>12.4f} {std:>8.4f} {iou:>10.4f} {params_str:>12}")
    
    print("-"*90)
    print(f"\n✅ Results saved to: {output_dir}")


def create_csv(results: Dict[str, Dict], output_path: Path):
    """Create CSV summary."""
    import csv
    
    if not results:
        return
    
    # Collect all keys
    all_keys = set()
    for r in results.values():
        if 'error' not in r:
            all_keys.update(k for k in r.keys() if not isinstance(r[k], (list, dict)))
    
    keys = sorted(all_keys)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model'] + keys)
        writer.writeheader()
        
        for name, res in results.items():
            if 'error' not in res:
                row = {'model': name}
                row.update({k: res.get(k, '') for k in keys})
                writer.writerow(row)
    
    print(f"CSV saved to: {output_path}")


if __name__ == '__main__':
    main()
