"""
Training loop and trainer class for cardiac segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Dict, Callable, List
from dataclasses import dataclass, field
from pathlib import Path
import time
import gc
from tqdm import tqdm

from metrics import SegmentationMetrics


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Optimization
    optimizer: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    scheduler: str = 'cosine'  # 'cosine', 'poly', 'step'
    warmup_epochs: int = 5
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    save_dir: str = './checkpoints'
    save_every: int = 10
    save_best: bool = True
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 0.001
    
    # Logging
    log_every: int = 10
    use_tensorboard: bool = True
    
    # Device
    device: str = 'cuda'
    num_workers: int = 4

    # Gradient accumulation
    gradient_accumulation_steps: int = 1


class Trainer:
    """
    Trainer for cardiac segmentation models.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: TrainingConfig,
        callbacks: Optional[List] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.callbacks = callbacks or []
        
        # Setup
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision - use new API
        if config.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Metrics
        self.metrics = SegmentationMetrics(num_classes=4)
        
        # State
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'lr': []
        }
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        params = self.model.parameters()
        
        if self.config.optimizer == 'adam':
            return torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = len(self.train_loader) * self.config.warmup_epochs

        if self.config.scheduler == 'warmup_cosine':
            from training.scheduler import WarmupCosineScheduler
            return WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=self.config.warmup_epochs,
                total_epochs=self.config.epochs,
                min_lr=1e-7
            )
        elif self.config.scheduler == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler == 'poly':
            from torch.optim.lr_scheduler import PolynomialLR
            return PolynomialLR(
                self.optimizer,
                total_iters=self.config.epochs,
                power=0.9
            )
        elif self.config.scheduler == 'step':
            from torch.optim.lr_scheduler import StepLR
            return StepLR(self.optimizer, step_size=30, gamma=0.1)
        else:
            return None
    
    def train(self) -> Dict:
        """
        Run full training loop.
        
        Returns:
            Training history
        """
        print(f"Starting training for {self.config.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Callbacks - on_train_begin
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(self)
        
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Callbacks - on_epoch_begin
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_begin'):
                    callback.on_epoch_begin(self, epoch)
            
            # Train epoch
            train_loss = self._train_epoch()
            
            # Validate
            val_loss, val_metrics = self._validate()
            val_dice = val_metrics.get('dice_mean', 0)
            
            # Update history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_dice)
            self.history['lr'].append(current_lr)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Print progress
            print(f"Epoch {epoch + 1}/{self.config.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Dice: {val_dice:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            # Save best model
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
                self.best_val_loss = val_loss
                patience_counter = 0
                
                if self.config.save_best:
                    self._save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Callbacks - on_epoch_end
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(self, epoch, {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_dice': val_dice,
                        'is_best': is_best
                    })

            # Early stopping — check callback signal OR internal counter
            should_stop = False
            for callback in self.callbacks:
                if hasattr(callback, 'should_stop') and callback.should_stop:
                    should_stop = True
                    break
            if should_stop or (self.config.early_stopping and patience_counter >= self.config.patience):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Callbacks - on_train_end
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(self)
        
        # Save final model
        self._save_checkpoint('final_model.pth')
        
        print(f"Training complete. Best Val Dice: {self.best_val_dice:.4f}")
        
        return self.history
    
    def _compute_loss_with_deep_supervision(
        self, outputs, masks: torch.Tensor
    ) -> tuple:
        """
        Compute loss, handling deep supervision if model returns aux outputs.

        Args:
            outputs: Model output — tensor, or dict with 'out' and optional 'aux'.
            masks: Ground truth masks (B, H, W).

        Returns:
            (loss, main_output_tensor) — loss is the scalar to backprop,
            main_output_tensor is the primary prediction for logging.
        """
        if isinstance(outputs, dict) and 'aux' in outputs:
            # Deep supervision: main output + weighted auxiliary losses
            main_output = outputs['out']
            loss = self.criterion(main_output, masks)

            aux_outputs = outputs['aux']
            n_aux = len(aux_outputs)
            for i, aux in enumerate(aux_outputs):
                # Exponentially decreasing weights: 0.5, 0.25, 0.125, ...
                weight = 0.5 ** (n_aux - i)
                # Resize masks to match auxiliary output resolution
                if aux.shape[2:] != masks.shape[1:]:
                    aux_masks = F.interpolate(
                        masks.unsqueeze(1).float(),
                        size=aux.shape[2:],
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    aux_masks = masks
                loss = loss + weight * self.criterion(aux, aux_masks)

            return loss, main_output
        else:
            if isinstance(outputs, dict):
                outputs = outputs['out']
            loss = self.criterion(outputs, masks)
            return loss, outputs

    def _train_epoch(self) -> float:
        """Train for one epoch with optional gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accum_steps = self.config.gradient_accumulation_steps

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass with mixed precision
            if self.config.use_amp and self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss, _ = self._compute_loss_with_deep_supervision(outputs, masks)
                    if accum_steps > 1:
                        loss = loss / accum_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss, _ = self._compute_loss_with_deep_supervision(outputs, masks)
                if accum_steps > 1:
                    loss = loss / accum_steps

                loss.backward()

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * (accum_steps if accum_steps > 1 else 1)
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item() * (accum_steps if accum_steps > 1 else 1):.4f}'})

        return total_loss / num_batches
    
    def _validate(self) -> tuple:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        self.metrics.reset()
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                num_batches += 1
                
                # Update metrics
                self.metrics.update(outputs, masks)
        
        avg_loss = total_loss / num_batches
        metrics = self.metrics.compute()
        
        return avg_loss, metrics
    
    def _save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_dice': self.best_val_dice,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        path = Path(self.config.save_dir) / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_dice = checkpoint['best_val_dice']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate on test set."""
        self.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                
                self.metrics.update(outputs, masks)
        
        return self.metrics.compute()


if __name__ == '__main__':
    print("Trainer module loaded successfully")
