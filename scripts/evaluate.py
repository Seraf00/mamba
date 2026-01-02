"""
Evaluation script for cardiac segmentation models.

Usage:
    python scripts/evaluate.py --model mamba_unet_v1 --checkpoint best_model.pth
    python scripts/evaluate.py --checkpoint_dir ./checkpoints/exp1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from data import CAMUSDataset, get_transforms
from models import get_model
from metrics import SegmentationMetrics, EjectionFractionCalculator
from utils import set_seed, get_device, load_model
from utils.visualization import save_predictions, plot_segmentation


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate cardiac segmentation model')
    
    # Model
    parser.add_argument('--model', type=str, default='mamba_unet_v1',
                        help='Model name')
    parser.add_argument('--mamba_type', type=str, default='mamba',
                        help='Mamba variant')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/CAMUS',
                        help='Path to CAMUS dataset')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction visualizations')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    
    return parser.parse_args()


def evaluate(model, dataloader, device, metrics, ef_calculator=None):
    """
    Run evaluation.
    
    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device
        metrics: SegmentationMetrics instance
        ef_calculator: Optional EF calculator
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics.reset()
    
    all_predictions = []
    all_targets = []
    all_images = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            # Update metrics
            metrics.update(outputs, masks)
            
            # Store for visualization
            all_predictions.append(outputs.argmax(dim=1).cpu())
            all_targets.append(masks.cpu())
            all_images.append(images.cpu())
    
    # Compute final metrics
    results = metrics.compute()
    
    # Concatenate all predictions
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    images = torch.cat(all_images, dim=0)
    
    return results, predictions, targets, images


def main():
    args = parse_args()
    
    set_seed(args.seed)
    device = get_device() if args.device == 'cuda' else torch.device(args.device)
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    transform = get_transforms(split='test')
    dataset = CAMUSDataset(
        root_dir=args.data_dir,
        split=args.split,
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Evaluating on {len(dataset)} samples")
    
    # Model
    model = get_model(
        args.model,
        in_channels=1,
        num_classes=4,
        mamba_type=args.mamba_type
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded checkpoint from {args.checkpoint}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    
    # Metrics
    metrics = SegmentationMetrics(num_classes=4)
    
    # Evaluate
    results, predictions, targets, images = evaluate(
        model, dataloader, device, metrics
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    
    for key, value in sorted(results.items()):
        print(f"{key}: {value:.4f}")
    
    print("=" * 50)
    
    # Save results
    results_file = output_dir / f'results_{args.model}_{args.split}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Save predictions
    if args.save_predictions:
        pred_dir = output_dir / 'predictions'
        save_predictions(
            images, targets, predictions,
            output_dir=str(pred_dir),
            prefix=f'{args.model}_{args.split}'
        )
        print(f"Predictions saved to {pred_dir}")
    
    return results


if __name__ == '__main__':
    main()
