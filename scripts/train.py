"""
Training script for cardiac segmentation models.

Usage:
    python scripts/train.py --model mamba_unet_v1 --epochs 100
    python scripts/train.py --config configs/experiment_configs.yaml
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from data import CAMUSDataset, get_transforms
from models import get_model
from training import Trainer, TrainingConfig, CombinedLoss
from training.callbacks import EarlyStopping, ModelCheckpoint, TensorBoardLogger, CSVLogger
from utils import set_seed, get_device, load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train cardiac segmentation model')
    
    # Model
    parser.add_argument('--model', type=str, default='mamba_unet_v1',
                        help='Model name')
    parser.add_argument('--mamba_type', type=str, default='mamba',
                        choices=['mamba', 'mamba2', 'vmamba'],
                        help='Mamba variant to use')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/CAMUS',
                        help='Path to CAMUS dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
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
    
    # Loss
    parser.add_argument('--dice_weight', type=float, default=1.0,
                        help='Dice loss weight')
    parser.add_argument('--ce_weight', type=float, default=1.0,
                        help='Cross entropy loss weight')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Checkpoint save directory')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (overrides other args)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = get_device() if args.device == 'cuda' else torch.device(args.device)
    
    # Experiment name
    exp_name = args.exp_name or f"{args.model}_{args.mamba_type}"
    save_dir = Path(args.save_dir) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment: {exp_name}")
    print(f"Save directory: {save_dir}")
    
    # Data transforms
    train_transform = get_transforms(split='train')
    val_transform = get_transforms(split='val')
    
    # Datasets
    train_dataset = CAMUSDataset(
        root_dir=args.data_dir,
        split='train',
        transform=train_transform
    )
    val_dataset = CAMUSDataset(
        root_dir=args.data_dir,
        split='val',
        transform=val_transform
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    model = get_model(
        args.model,
        in_channels=1,
        num_classes=4,
        mamba_type=args.mamba_type
    )
    
    print(f"Model: {args.model}")
    print(f"Mamba type: {args.mamba_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    criterion = CombinedLoss(
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight
    )
    
    # Training config
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        save_dir=str(save_dir),
        device=args.device
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=20, mode='max'),
        ModelCheckpoint(save_dir=str(save_dir), monitor='val_dice', mode='max'),
        TensorBoardLogger(log_dir=str(save_dir / 'logs')),
        CSVLogger(filename=str(save_dir / 'training_log.csv'))
    ]
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config,
        callbacks=callbacks
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    history = trainer.train()
    
    print("\nTraining complete!")
    print(f"Best validation Dice: {trainer.best_val_dice:.4f}")
    
    return history


if __name__ == '__main__':
    main()
