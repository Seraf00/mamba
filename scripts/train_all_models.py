#!/usr/bin/env python3
"""
Comprehensive training script for Q1 journal paper benchmarking.

This script trains all 19 models (9 base + 10 Mamba-enhanced) with multiple
Mamba variants for systematic comparison.

Usage:
    # Train all models with all Mamba variants
    python scripts/train_all_models.py --data_dir ./data/CAMUS
    
    # Train only base models
    python scripts/train_all_models.py --base_only --data_dir ./data/CAMUS
    
    # Train only Mamba-enhanced models
    python scripts/train_all_models.py --mamba_only --data_dir ./data/CAMUS
    
    # Train specific models
    python scripts/train_all_models.py --models unet_v1 mamba_unet_v1 --data_dir ./data/CAMUS
    
    # Resume from a specific model
    python scripts/train_all_models.py --resume_from mamba_unet_v2 --data_dir ./data/CAMUS

Author: Research Team
Date: 2025
"""

import argparse
import sys
import os
import json
import time
import gc
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from data import CAMUSDataset, get_transforms
from models import get_model, list_models
from training import Trainer, TrainingConfig, CombinedLoss
from training.callbacks import EarlyStopping, ModelCheckpoint, TensorBoardLogger, CSVLogger
from metrics import SegmentationMetrics, EfficiencyBenchmark
from utils import set_seed, get_device


def cleanup_gpu_memory():
    """Aggressively clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset memory stats for accurate tracking
        torch.cuda.reset_peak_memory_stats()


def get_batch_size_for_model(model_name: str, default_batch_size: int, num_params: int) -> int:
    """
    Get appropriate batch size based on model size to avoid OOM.
    
    Args:
        model_name: Name of the model
        default_batch_size: Default batch size from args
        num_params: Number of model parameters
        
    Returns:
        Adjusted batch size
    """
    # Models that are known to be memory-hungry
    large_models = ['mamba', 'vmamba', 'fpn', 'gudu', 'deeplab']
    
    # Check if model is memory-intensive
    is_mamba = any(m in model_name.lower() for m in ['mamba', 'vmamba'])
    is_large = any(m in model_name.lower() for m in large_models)
    
    # Adjust based on parameter count
    if num_params > 80e6:  # >80M params
        return min(default_batch_size, 2)
    elif num_params > 50e6 or is_mamba:  # >50M or any Mamba model
        return min(default_batch_size, 4)
    elif num_params > 30e6 or is_large:  # >30M or known large models
        return min(default_batch_size, 6)
    else:
        return default_batch_size


def check_mamba_installation():
    """Check if mamba-ssm is properly installed for fast training."""
    print("\n" + "="*70)
    print("CHECKING MAMBA-SSM INSTALLATION")
    print("="*70)
    
    try:
        # Check if mamba_ssm is available
        try:
            import mamba_ssm
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            print("✓ mamba-ssm is installed with CUDA kernels")
            mamba_available = True
        except ImportError:
            print("✗ mamba-ssm NOT installed or CUDA kernels unavailable")
            mamba_available = False
        
        # Check our MambaBlock
        from models.modules import MambaBlock
        test_block = MambaBlock(dim=64, d_state=16)
        
        if test_block.use_fast_path:
            print("✓ Fast CUDA kernels will be used (100-1000x faster)")
        else:
            print("\n" + "!"*70)
            print("⚠️  WARNING: SLOW PYTHON IMPLEMENTATION WILL BE USED!")
            print("!"*70)
            print("\nThis will cause EXTREME slowdown (~100-1000x slower)!")
            print("Expected time per batch: 40+ minutes instead of 1-2 seconds!")
            print("\nTo fix, install mamba-ssm:")
            print("  pip install mamba-ssm")
            print("\nOr from source:")
            print("  pip install git+https://github.com/state-spaces/mamba.git")
            print("\nPress Ctrl+C to cancel, or wait 10 seconds to continue anyway...")
            print("!"*70 + "\n")
            
            import time
            for i in range(10, 0, -1):
                print(f"  Continuing in {i}...", end='\r')
                time.sleep(1)
            print("\n")
        
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"Warning: Could not check mamba installation: {e}")
        print("="*70 + "\n")


# ============================================================================
# Model Definitions
# ============================================================================

BASE_MODELS = [
    'unet_v1',
    'unet_v2', 
    'unet_resnet',
    'deeplab_v3',
    'nnunet',
    'gudu',
    'swin_unet',
    'transunet',
    'fpn',
]

MAMBA_MODELS = [
    'mamba_unet_v1',
    'mamba_unet_v2',
    'mamba_unet_resnet',
    'mamba_deeplab',
    'mamba_nnunet',
    'mamba_gudu',
    'mamba_swin_unet',
    'mamba_transunet',
    'mamba_fpn',
    'pure_mamba_unet',
]

MAMBA_VARIANTS = ['mamba', 'mamba2', 'vmamba']

# Models requiring specific input sizes
SWIN_MODELS = ['swin_unet', 'mamba_swin_unet']  # Require 224x224


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train all models for Q1 journal paper benchmarking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        help='Specific models to train (overrides --base_only/--mamba_only)')
    parser.add_argument('--base_only', action='store_true',
                        help='Train only base models (no Mamba)')
    parser.add_argument('--mamba_only', action='store_true',
                        help='Train only Mamba-enhanced models')
    parser.add_argument('--mamba_variants', nargs='+', type=str, 
                        default=['mamba'],
                        choices=MAMBA_VARIANTS,
                        help='Mamba variants to test for Mamba models (default: mamba only - fully optimized)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from this model (skip models before it)')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to CAMUS dataset')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size (Swin models will use 224)')
    parser.add_argument('--include_sequences', action='store_true',
                        help='Include half-sequence frames for more training data')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'poly', 'step', 'warmup_cosine'],
                        help='LR scheduler')
    parser.add_argument('--early_stopping', type=int, default=20,
                        help='Early stopping patience (0 to disable)')
    
    # Loss
    parser.add_argument('--dice_weight', type=float, default=1.0,
                        help='Dice loss weight')
    parser.add_argument('--ce_weight', type=float, default=1.0,
                        help='Cross entropy loss weight')
    
    # Cross-validation
    parser.add_argument('--cross_val', action='store_true',
                        help='Run 5-fold cross-validation')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./results/benchmark',
                        help='Base output directory')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    
    # System
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use automatic mixed precision training')
    
    # Debugging
    parser.add_argument('--dry_run', action='store_true',
                        help='Print training plan without executing')
    parser.add_argument('--skip_benchmark', action='store_true',
                        help='Skip efficiency benchmarking after training')
    
    return parser.parse_args()


def get_models_to_train(args) -> List[Dict[str, Any]]:
    """
    Get list of models to train with their configurations.
    
    Returns:
        List of dicts with 'name' and 'mamba_type' keys
    """
    models = []
    
    if args.models:
        # Specific models requested
        model_names = args.models
    elif args.base_only:
        model_names = BASE_MODELS
    elif args.mamba_only:
        model_names = MAMBA_MODELS
    else:
        # All models
        model_names = BASE_MODELS + MAMBA_MODELS
    
    for model_name in model_names:
        if model_name in MAMBA_MODELS:
            # Add each Mamba variant
            for mamba_type in args.mamba_variants:
                models.append({
                    'name': model_name,
                    'mamba_type': mamba_type,
                    'display_name': f"{model_name}_{mamba_type}"
                })
        else:
            # Base model (no Mamba variant)
            models.append({
                'name': model_name,
                'mamba_type': None,
                'display_name': model_name
            })
    
    # Handle resume_from
    if args.resume_from:
        skip_until = args.resume_from
        found = False
        filtered_models = []
        for m in models:
            if m['display_name'] == skip_until or m['name'] == skip_until:
                found = True
            if found:
                filtered_models.append(m)
        
        if filtered_models:
            models = filtered_models
        else:
            print(f"Warning: --resume_from model '{skip_until}' not found in training list")
    
    return models


def get_img_size_for_model(model_name: str, default_size: int) -> int:
    """Get appropriate image size for model."""
    if model_name in SWIN_MODELS:
        return 224  # Swin models need 224x224
    return default_size


def train_single_model(
    model_config: Dict[str, Any],
    args,
    exp_dir: Path,
    device: torch.device
) -> Dict[str, Any]:
    """
    Train a single model and return results.
    
    Args:
        model_config: Dict with 'name', 'mamba_type', 'display_name'
        args: Command line arguments
        exp_dir: Experiment directory
        device: Training device
        
    Returns:
        Dictionary with training results and metrics
    """
    # Clean GPU memory before starting new model
    cleanup_gpu_memory()
    
    model_name = model_config['name']
    mamba_type = model_config['mamba_type']
    display_name = model_config['display_name']
    
    print(f"\n{'='*70}")
    print(f"Training: {display_name}")
    if torch.cuda.is_available():
        print(f"GPU memory before: {torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated")
    print(f"{'='*70}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Determine image size (as tuple for transforms)
    img_size = get_img_size_for_model(model_name, args.img_size)
    img_size_tuple = (img_size, img_size)  # Convert to (H, W) tuple
    
    # Create transforms
    train_transform = get_transforms(split='train', img_size=img_size_tuple)
    val_transform = get_transforms(split='val', img_size=img_size_tuple)
    
    # Create datasets
    train_dataset = CAMUSDataset(
        root_dir=args.data_dir,
        split='train',
        transform=train_transform,
        include_sequences=args.include_sequences
    )
    val_dataset = CAMUSDataset(
        root_dir=args.data_dir,
        split='val',
        transform=val_transform
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Image size: {img_size}x{img_size}")
    
    # Create model first to determine appropriate batch size
    model_kwargs = {'in_channels': 1, 'num_classes': 4}
    if mamba_type:
        model_kwargs['mamba_type'] = mamba_type
    
    model = get_model(model_name, **model_kwargs)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get adjusted batch size based on model size
    batch_size = get_batch_size_for_model(model_name, args.batch_size, num_params)
    if batch_size != args.batch_size:
        print(f"  Adjusted batch size: {args.batch_size} -> {batch_size} (model has {num_params/1e6:.1f}M params)")
    
    print(f"  Parameters: {num_params/1e6:.2f}M ({trainable_params/1e6:.2f}M trainable)")
    print(f"  Batch size: {batch_size}")
    
    # Create data loaders with adjusted batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model save directory
    model_dir = exp_dir / display_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss function
    criterion = CombinedLoss(
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight
    )
    
    # Training config - use adjusted batch size
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=batch_size,  # Use adjusted batch size
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        save_dir=str(model_dir),
        device=str(device)
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(save_dir=str(model_dir), monitor='val_dice', mode='max'),
        TensorBoardLogger(log_dir=str(model_dir / 'logs')),
        CSVLogger(filename=str(model_dir / 'training_log.csv'))
    ]
    
    if args.early_stopping > 0:
        callbacks.insert(0, EarlyStopping(patience=args.early_stopping, mode='max'))
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config,
        callbacks=callbacks
    )
    
    # Train
    start_time = time.time()
    history = trainer.train()
    training_time = time.time() - start_time
    
    # Get best metrics
    best_val_dice = trainer.best_val_dice if hasattr(trainer, 'best_val_dice') else max(history.get('val_dice', [0]))
    
    print(f"\n  Training complete!")
    print(f"  Best Val Dice: {best_val_dice:.4f}")
    print(f"  Training time: {timedelta(seconds=int(training_time))}")
    
    # Benchmark efficiency (if not skipped)
    efficiency_results = {}
    if not args.skip_benchmark:
        try:
            benchmark = EfficiencyBenchmark(
                input_size=(1, 1, img_size, img_size),
                device=str(device),
                num_iterations=50,
                warmup_iterations=10
            )
            result = benchmark.benchmark(model, display_name)
            efficiency_results = {
                'params_M': result.num_parameters / 1e6,
                'inference_time_ms': result.inference_time_ms,
                'memory_MB': result.memory_mb,
                'flops_G': getattr(result, 'flops', 0) / 1e9 if hasattr(result, 'flops') else None
            }
        except Exception as e:
            print(f"  Warning: Benchmark failed - {e}")
    
    # Save model info
    results = {
        'model_name': model_name,
        'display_name': display_name,
        'mamba_type': mamba_type,
        'img_size': img_size,
        'num_params': num_params,
        'trainable_params': trainable_params,
        'best_val_dice': float(best_val_dice),
        'training_time_seconds': training_time,
        'epochs_trained': len(history.get('train_loss', [])),
        'final_train_loss': float(history.get('train_loss', [0])[-1]) if history.get('train_loss') else None,
        'final_val_loss': float(history.get('val_loss', [0])[-1]) if history.get('val_loss') else None,
        **efficiency_results
    }
    
    # Save individual results
    with open(model_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Aggressively clean up GPU memory
    # Delete all objects that may hold GPU references
    del model
    del trainer
    del train_loader
    del val_loader
    del train_dataset
    del val_dataset
    
    # Force garbage collection and clear CUDA cache
    cleanup_gpu_memory()
    
    print(f"  Memory cleaned. GPU memory after cleanup: "
          f"{torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated, "
          f"{torch.cuda.memory_reserved() / 1024**2:.1f}MB reserved" if torch.cuda.is_available() else "")
    
    return results


def main():
    args = parse_args()
    
    # Check if mamba-ssm is properly configured for Mamba models
    if not args.base_only or (args.models and any('mamba' in m.lower() for m in args.models)):
        check_mamba_installation()
    
    # Get models to train
    models_to_train = get_models_to_train(args)
    
    if not models_to_train:
        print("No models to train!")
        return
    
    # Create experiment directory
    exp_name = args.exp_name or datetime.now().strftime('benchmark_%Y%m%d_%H%M%S')
    exp_dir = Path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Print training plan
    print("\n" + "="*70)
    print("Q1 PAPER BENCHMARKING - TRAINING PLAN")
    print("="*70)
    print(f"\nExperiment: {exp_name}")
    print(f"Output directory: {exp_dir}")
    print(f"Device: {device}")
    print(f"Models to train: {len(models_to_train)}")
    print(f"\nModels:")
    for i, m in enumerate(models_to_train, 1):
        print(f"  {i:2d}. {m['display_name']}")
    print(f"\nTraining settings:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Scheduler: {args.scheduler}")
    print(f"  Early stopping: {args.early_stopping if args.early_stopping > 0 else 'Disabled'}")
    print(f"  Include sequences: {args.include_sequences}")
    if args.cross_val:
        print(f"  Cross-validation: {args.n_folds}-fold")
    
    if args.dry_run:
        print("\n[DRY RUN] Training not executed.")
        return
    
    # Save experiment config
    config = vars(args).copy()
    config['models_to_train'] = [m['display_name'] for m in models_to_train]
    config['start_time'] = datetime.now().isoformat()
    with open(exp_dir / 'experiment_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train all models
    all_results = []
    total_start = time.time()
    
    for i, model_config in enumerate(models_to_train, 1):
        print(f"\n\n{'#'*70}")
        print(f"# MODEL {i}/{len(models_to_train)}: {model_config['display_name']}")
        print(f"{'#'*70}")
        
        try:
            results = train_single_model(model_config, args, exp_dir, device)
            all_results.append(results)
            
            # Save aggregated results after each model
            with open(exp_dir / 'all_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
                
        except Exception as e:
            print(f"\n  ERROR: Training failed - {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'display_name': model_config['display_name'],
                'error': str(e)
            })
        finally:
            # Always clean up GPU memory between models
            cleanup_gpu_memory()
    
    total_time = time.time() - total_start
    
    # Print summary
    print("\n\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    print(f"\nTotal training time: {timedelta(seconds=int(total_time))}")
    print(f"Results saved to: {exp_dir}")
    
    # Print results table
    print("\n" + "-"*90)
    print(f"{'Model':<35} {'Params':>10} {'Val Dice':>10} {'Time':>12} {'Inf. (ms)':>10}")
    print("-"*90)
    
    for r in all_results:
        if 'error' in r:
            print(f"{r['display_name']:<35} {'ERROR':>10}")
        else:
            params = f"{r['num_params']/1e6:.2f}M"
            dice = f"{r['best_val_dice']:.4f}"
            time_str = str(timedelta(seconds=int(r['training_time_seconds'])))
            inf_time = f"{r.get('inference_time_ms', 0):.2f}" if r.get('inference_time_ms') else 'N/A'
            print(f"{r['display_name']:<35} {params:>10} {dice:>10} {time_str:>12} {inf_time:>10}")
    
    print("-"*90)
    
    # Create CSV summary
    create_csv_summary(all_results, exp_dir / 'summary.csv')
    
    print(f"\n✅ All results saved to: {exp_dir}")
    print("\nNext steps:")
    print("  1. Evaluate on test set: python scripts/evaluate.py --checkpoint_dir", exp_dir)
    print("  2. Run benchmark: python scripts/benchmark.py --all")
    print("  3. Generate figures for paper")


def create_csv_summary(results: List[Dict], output_path: Path):
    """Create CSV summary of all results."""
    import csv
    
    if not results:
        return
    
    # Get all keys
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    
    # Remove 'error' from keys and sort
    keys = sorted([k for k in all_keys if k != 'error'])
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['display_name'] + keys)
        writer.writeheader()
        for r in results:
            if 'error' not in r:
                writer.writerow({k: r.get(k, '') for k in ['display_name'] + keys})
    
    print(f"\nCSV summary saved to: {output_path}")


if __name__ == '__main__':
    main()
