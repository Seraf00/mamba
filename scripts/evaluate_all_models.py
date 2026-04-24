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
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from scipy import stats

from data import CAMUSDataset, get_transforms
from data.camus_dataset import CAMUSPatient
from models import get_model
from metrics import SegmentationMetrics, EjectionFractionCalculator, CAMUSEFCalculator, compute_ejection_fraction
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
    """
    Evaluate a single model, with per-sample pixel spacing and ED/ES separation.

    Computes:
      - Per-class Dice / IoU (overall + ED + ES slices)
      - Per-class HD95 and ASSD in mm (using per-sample NIfTI spacing)
      - Per-sample Dice for bootstrap CI and Wilcoxon tests
    """
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

    # Separate metric accumulators for overall / ED / ES
    metrics_overall = SegmentationMetrics(num_classes=4)
    metrics_ed = SegmentationMetrics(num_classes=4)
    metrics_es = SegmentationMetrics(num_classes=4)

    # Collect per-sample predictions for stats + EF downstream
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {model_config['display_name']}", leave=False):
            # Batch is a dict when include_info=True via _eval_collate_fn
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            phases: List[str] = batch['phase']
            spacings: List[Tuple[float, float]] = batch['pixel_spacing']

            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['out']

            preds = outputs.argmax(dim=1)

            # Overall metrics with per-sample spacing
            metrics_overall.update(preds, masks, spacing=spacings)

            # Phase-stratified metrics
            ed_indices = [i for i, p in enumerate(phases) if p == 'ED']
            es_indices = [i for i, p in enumerate(phases) if p == 'ES']

            if ed_indices:
                ed_spacings = [spacings[i] for i in ed_indices]
                metrics_ed.update(
                    preds[ed_indices], masks[ed_indices], spacing=ed_spacings
                )
            if es_indices:
                es_spacings = [spacings[i] for i in es_indices]
                metrics_es.update(
                    preds[es_indices], masks[es_indices], spacing=es_spacings
                )

            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    # Aggregate
    results = metrics_overall.compute()

    # Attach ED / ES slices with suffixes
    results_ed = metrics_ed.compute()
    for k, v in results_ed.items():
        results[f'{k}_ed'] = v

    results_es = metrics_es.compute()
    for k, v in results_es.items():
        results[f'{k}_es'] = v

    # Stack predictions for per-sample stats
    all_preds_np = np.concatenate(all_preds, axis=0)
    all_targets_np = np.concatenate(all_targets, axis=0)

    per_sample_dice = compute_per_sample_dice(all_preds_np, all_targets_np)
    results['per_sample_dice'] = per_sample_dice.tolist()
    results['dice_std'] = float(np.std(per_sample_dice))

    # Bootstrap 95% confidence interval on mean Dice
    _, ci_lower, ci_upper = bootstrap_confidence_interval(per_sample_dice)
    results['dice_ci_lower'] = ci_lower
    results['dice_ci_upper'] = ci_upper

    # Clean up
    del model
    torch.cuda.empty_cache()

    return results


def _eval_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for evaluation that preserves per-sample pixel spacing
    and ED/ES phase labels. We cannot use the default collate because
    pixel_spacing is a per-sample tuple, and phase is a per-sample string.
    """
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])

    return {
        'image': images,
        'mask': masks,
        'patient_id': [item['patient_id'] for item in batch],
        'view': [item['view'] for item in batch],
        'phase': [item['phase'] for item in batch],
        'pixel_spacing': [tuple(item['pixel_spacing']) for item in batch],
    }


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


def evaluate_ef_biplane(
    model: torch.nn.Module,
    data_dir: str,
    split: str,
    device: torch.device,
    img_size: int = 256,
) -> Dict[str, Any]:
    """
    Evaluate Ejection Fraction using biplane Simpson's method.

    Groups predictions by patient across 2CH/4CH views and ED/ES phases,
    then computes biplane EF and compares against ground truth.

    Returns:
        Dictionary with EF metrics (MAE, RMSE, correlation, Bland-Altman).
    """
    # Create dataset with all info
    transform = get_transforms(split='val', img_size=(img_size, img_size))
    dataset = CAMUSDataset(
        root_dir=data_dir,
        split=split,
        views=['2CH', '4CH'],
        phases=['ED', 'ES'],
        transform=transform,
        include_info=True,
    )

    # Group samples by patient
    patient_samples: Dict[str, Dict] = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        pid = sample['patient_id']
        view = sample['view']
        phase = sample['phase']

        if pid not in patient_samples:
            patient_samples[pid] = {}

        key = f"{view}_{phase}"
        patient_samples[pid][key] = {
            'image': sample['image'],
            'mask': sample['mask'],
            'pixel_spacing': sample['pixel_spacing'],
            'ef_gt': sample['ef'],
        }

    # Compute EF for patients that have all 4 required masks
    ef_calculator = CAMUSEFCalculator(lv_label=1)
    skipped = 0

    model.eval()
    with torch.no_grad():
        for pid, samples in tqdm(patient_samples.items(), desc="Computing biplane EF"):
            required_keys = ['2CH_ED', '2CH_ES', '4CH_ED', '4CH_ES']
            if not all(k in samples for k in required_keys):
                skipped += 1
                continue

            # Get predictions for each view/phase
            preds = {}
            for key in required_keys:
                img = samples[key]['image'].unsqueeze(0).to(device)
                output = model(img)
                if isinstance(output, dict):
                    output = output['out']
                preds[key] = output.argmax(dim=1).squeeze(0).cpu().numpy()

            # Get per-view pixel spacing
            spacing_2ch = samples['2CH_ED']['pixel_spacing']
            spacing_4ch = samples['4CH_ED']['pixel_spacing']

            # Get ground truth EF (use 4CH as reference, fallback to 2CH)
            ef_gt = samples['4CH_ED']['ef_gt']
            if ef_gt <= 0:
                ef_gt = samples['2CH_ED']['ef_gt']
            ef_gt = ef_gt if ef_gt > 0 else None

            ef_calculator.compute_ef(
                a2c_ed=preds['2CH_ED'],
                a2c_es=preds['2CH_ES'],
                a2c_spacing=spacing_2ch,
                a4c_ed=preds['4CH_ED'],
                a4c_es=preds['4CH_ES'],
                a4c_spacing=spacing_4ch,
                ef_ground_truth=ef_gt,
                patient_id=pid,
            )

    stats = ef_calculator.compute_statistics()
    if skipped > 0:
        stats['patients_skipped'] = skipped
        stats['patients_evaluated'] = len(patient_samples) - skipped

    return stats


def bootstrap_confidence_interval(
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        scores: Array of per-sample scores.
        n_bootstrap: Number of bootstrap iterations.
        ci: Confidence level (default: 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap)
    n = len(scores)
    for i in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    return float(np.mean(scores)), lower, upper


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
    """
    Create LaTeX table for paper (CAMUS leaderboard format):
    per-class Dice, per-class HD95 (mm), EF MAE, EF correlation.
    """
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Segmentation and clinical performance on CAMUS. "
        r"Dice reported per cardiac structure (LV\textsubscript{endo}, "
        r"LV\textsubscript{epi}, LA); HD95 and ASSD in mm; EF MAE (\%) and "
        r"EF Pearson correlation.}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Model & Dice LV$_{\text{endo}}$ & Dice LV$_{\text{epi}}$ & Dice LA "
        r"& Mean Dice & HD95 (mm) & ASSD (mm) & EF MAE (\%) & EF $r$ \\",
        r"\midrule",
    ]

    def _sort_key(item):
        res = item[1]
        if 'error' in res:
            return 999.0
        return -res.get('dice_mean', 0)

    for model_name, metrics in sorted(results.items(), key=_sort_key):
        if 'error' in metrics:
            continue

        lv_endo = metrics.get('dice_lv_endocardium', 0)
        lv_epi = metrics.get('dice_lv_epicardium', 0)
        la = metrics.get('dice_left_atrium', 0)
        mean_dice = metrics.get('dice_mean', 0)
        hd95 = metrics.get('hd95_mean', float('nan'))
        assd = metrics.get('assd_mean', float('nan'))

        ef_metrics = metrics.get('ef_metrics', {}) or {}
        ef_mae = ef_metrics.get('ef_mae', float('nan'))
        ef_r = ef_metrics.get('ef_correlation', float('nan'))

        def _fmt(x, fmt=".3f"):
            if isinstance(x, float) and (x != x):  # NaN check
                return '--'
            return format(x, fmt)

        display_name = model_name.replace('_', r'\_')
        row = (
            f"{display_name} & "
            f"{_fmt(lv_endo)} & {_fmt(lv_epi)} & {_fmt(la)} & {_fmt(mean_dice)} & "
            f"{_fmt(hd95, '.2f')} & {_fmt(assd, '.2f')} & "
            f"{_fmt(ef_mae, '.2f')} & {_fmt(ef_r, '.3f')} \\\\"
        )
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
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

        # Create dataset with include_info=True so we get pixel spacing and phase
        # (required for HD95 in mm and ED/ES stratified reporting).
        transform = get_transforms(split='val', img_size=(img_size, img_size))
        dataset = CAMUSDataset(
            root_dir=args.data_dir,
            split=args.split,
            transform=transform,
            include_info=True,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=_eval_collate_fn,
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

            # Biplane EF evaluation
            if args.compute_ef:
                print(f"  Computing biplane EF...")
                try:
                    # Reload model for EF eval (need fresh instance)
                    model_kwargs = {'in_channels': 1, 'num_classes': 4}
                    if model_config['mamba_type']:
                        model_kwargs['mamba_type'] = model_config['mamba_type']
                    ef_model = get_model(model_config['name'], **model_kwargs)
                    checkpoint = torch.load(model_config['checkpoint'], map_location=device, weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        ef_model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        ef_model.load_state_dict(checkpoint['state_dict'])
                    else:
                        ef_model.load_state_dict(checkpoint)
                    ef_model = ef_model.to(device)

                    ef_results = evaluate_ef_biplane(
                        ef_model, args.data_dir, args.split, device, img_size
                    )
                    results['ef_metrics'] = ef_results

                    if 'ef_mae' in ef_results:
                        print(f"  EF MAE: {ef_results['ef_mae']:.2f}%")
                    if 'ef_correlation' in ef_results:
                        print(f"  EF Correlation: {ef_results['ef_correlation']:.4f}")
                    if 'bland_altman_bias' in ef_results:
                        print(f"  EF Bias: {ef_results['bland_altman_bias']:.2f}%")

                    del ef_model
                    torch.cuda.empty_cache()
                except Exception as ef_err:
                    print(f"  EF evaluation error: {ef_err}")
                    results['ef_metrics'] = {'error': str(ef_err)}

            all_results[display_name] = results

            # Print results
            dice_mean = results.get('dice_mean', 0)
            dice_std = results.get('dice_std', 0)
            print(f"\n  Mean Dice: {dice_mean:.4f} +/- {dice_std:.4f}")

            # Per-class Dice (new keys from SegmentationMetrics)
            for cls_key in ('lv_endocardium', 'lv_epicardium', 'left_atrium'):
                if f'dice_{cls_key}' in results:
                    name = cls_key.replace('_', ' ').title()
                    print(f"    Dice ({name}): {results[f'dice_{cls_key}']:.4f}")

            if 'iou_mean' in results:
                print(f"  Mean IoU: {results['iou_mean']:.4f}")
            if 'hd95_mean' in results:
                print(f"  HD95 (mean, mm): {results['hd95_mean']:.2f}")
            if 'assd_mean' in results:
                print(f"  ASSD (mean, mm): {results['assd_mean']:.2f}")
            if 'hd95_mean_ed' in results and 'hd95_mean_es' in results:
                print(
                    f"  HD95 ED/ES (mm): {results['hd95_mean_ed']:.2f} / "
                    f"{results['hd95_mean_es']:.2f}"
                )

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
        # Convert numpy arrays/scalars to native Python types so json.dump works.
        # numpy.bool_, numpy.int64, numpy.float32 etc. are NOT JSON-serializable
        # by default; the previous version only handled arrays + containers.
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                # NaN/Inf become null per json spec when allow_nan=False; keep as float
                f_val = float(obj)
                if np.isnan(f_val) or np.isinf(f_val):
                    return None
                return f_val
            if isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            if isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return obj
            return obj
        json.dump(convert(output), f, indent=2)
    
    # Create LaTeX table
    create_latex_table(all_results, output_dir / 'results_table.tex')
    
    # Create CSV
    create_csv(all_results, output_dir / 'results.csv')
    
    # Print summary table with per-class Dice + HD95 for leaderboard comparison
    print("\n" + "=" * 120)
    print("RESULTS SUMMARY")
    print("=" * 120)
    header = (
        f"{'Model':<30} "
        f"{'Dice':>8} {'LV_endo':>9} {'LV_epi':>9} {'LA':>7} "
        f"{'HD95':>8} {'EF MAE':>8} {'EF r':>7} {'Params':>9}"
    )
    print("\n" + header)
    print("-" * 120)

    def _sort_key(item):
        res = item[1]
        if 'error' in res:
            return 999.0
        return -res.get('dice_mean', 0)

    for name, res in sorted(all_results.items(), key=_sort_key):
        if 'error' in res:
            print(f"{name:<30} ERROR ({res.get('error', '')[:80]})")
            continue

        dice = res.get('dice_mean', 0)
        lv_endo = res.get('dice_lv_endocardium', 0)
        lv_epi = res.get('dice_lv_epicardium', 0)
        la = res.get('dice_left_atrium', 0)
        hd95 = res.get('hd95_mean', float('nan'))
        params = res.get('params_M', 'N/A')
        params_str = f"{params:.1f}M" if isinstance(params, (int, float)) else str(params)

        ef_metrics = res.get('ef_metrics', {}) or {}
        ef_mae = ef_metrics.get('ef_mae', float('nan'))
        ef_r = ef_metrics.get('ef_correlation', float('nan'))

        # Guard format specifiers against None / non-numeric values
        def _fmt(v, spec):
            try:
                return format(float(v), spec)
            except (TypeError, ValueError):
                return f"{'N/A':>{spec.split('.')[0].lstrip('>').lstrip('<')}}" if '.' in spec else 'N/A'

        print(
            f"{name:<30} "
            f"{_fmt(dice,'>8.4f')} {_fmt(lv_endo,'>9.4f')} {_fmt(lv_epi,'>9.4f')} {_fmt(la,'>7.4f')} "
            f"{_fmt(hd95,'>8.2f')} {_fmt(ef_mae,'>8.2f')} {_fmt(ef_r,'>7.3f')} {params_str:>9}"
        )

    print("-" * 120)
    print(f"\nResults saved to: {output_dir}")


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
