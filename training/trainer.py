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
    # 'float16' (default, with loss scaling) or 'bfloat16'. bf16 has fp32's
    # exponent range, so activations cannot overflow the way they do in fp16 --
    # which is what turned the widened DenseContextU-Net and FPN controls to NaN
    # mid-training. No GradScaler is needed or used with bf16.
    amp_dtype: str = 'float16'
    
    # Checkpointing
    save_dir: str = './checkpoints'
    save_every: int = 10          # periodic checkpoint_epoch_N.pth; <= 0 disables
    save_best: bool = True
    # Rolling resume state (last.pth): full optimizer/scheduler/scaler/RNG state,
    # overwritten every `resume_every` epochs and deleted when training finishes.
    # 0 disables. Lets an interrupted run continue at the next epoch instead of
    # restarting a 100-epoch model from zero.
    resume_every: int = 0
    
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

    # Gradient clipping (0 = disabled)
    max_grad_norm: float = 0.0


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
        if config.amp_dtype not in ('float16', 'bfloat16'):
            raise ValueError(f"amp_dtype must be 'float16' or 'bfloat16', got {config.amp_dtype!r}")
        self._bf16 = bool(config.use_amp and config.amp_dtype == 'bfloat16')
        if config.use_amp and not self._bf16:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        self.diverged_epoch = None    # set when a loss goes non-finite
        
        # Metrics
        self.metrics = SegmentationMetrics(num_classes=4)
        
        # State
        self.current_epoch = 0
        self.start_epoch = 0          # > 0 only after load_checkpoint()
        self.patience_counter = 0
        # Wall time spent training, carried across a resume so that
        # training_time_seconds describes the whole run, not the last session.
        self.elapsed_seconds = 0.0
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

        if self.start_epoch > 0:
            # best_model.pth can be NEWER than the resume state: a crash after
            # an improving epoch saved it, but before last.pth caught up. Adopt
            # its score so the file on disk and the best_val_dice reported in
            # results.json describe the same model.
            best_file = Path(self.config.save_dir) / 'best_model.pth'
            if best_file.exists():
                try:
                    b = torch.load(best_file, map_location='cpu', weights_only=False)
                    on_disk = b.get('best_score', b.get('best_val_dice'))
                    if on_disk is not None and float(on_disk) > self.best_val_dice:
                        self.best_val_dice = float(on_disk)
                except Exception:
                    pass

            # A resumed run must not forget what it has already achieved. A
            # checkpoint callback that starts from best_score=None treats the
            # first resumed epoch as an improvement and overwrites
            # best_model.pth with whatever that epoch produced, even if it is
            # worse than the model it replaces.
            for callback in self.callbacks:
                if (hasattr(callback, 'best_score')
                        and getattr(callback, 'monitor', 'val_dice') == 'val_dice'):
                    callback.best_score = self.best_val_dice
            print(f"Resuming at epoch {self.start_epoch + 1}/{self.config.epochs} "
                  f"(best val Dice so far {self.best_val_dice:.4f})")

        t_session = time.time()
        elapsed_at_start = self.elapsed_seconds

        for epoch in range(self.start_epoch, self.config.epochs):
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
            
            # Divergence guard. A non-finite loss does not recover: in the R3
            # run the widened DenseContextU-Net went NaN at epoch 14 and FPN at
            # epoch 4, then trained on for hours producing nothing. Stop here,
            # BEFORE this epoch can write best_model.pth or last.pth, so no NaN
            # state is saved or resumed. Not early stopping: a finite run is
            # never affected.
            if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
                self.diverged_epoch = epoch + 1
                print(f"DIVERGED: non-finite loss at epoch {epoch + 1} "
                      f"(train {train_loss}, val {val_loss}) -- stopping this model")
                break

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
                self.patience_counter = 0

                if self.config.save_best:
                    self._save_checkpoint('best_model.pth')
            else:
                self.patience_counter += 1

            self.elapsed_seconds = elapsed_at_start + (time.time() - t_session)

            # Save periodic checkpoint
            if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
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

            # Resume state goes AFTER the callbacks, so the CSV row and the
            # best-model file for this epoch already exist when it is written:
            # a crash between the two can then only lose work, never record an
            # epoch the resume state does not know about.
            if (self.config.resume_every > 0
                    and (epoch + 1) % self.config.resume_every == 0
                    and epoch + 1 < self.config.epochs):
                self._save_checkpoint('last.pth')

            # Early stopping — check callback signal OR internal counter
            should_stop = False
            for callback in self.callbacks:
                if hasattr(callback, 'should_stop') and callback.should_stop:
                    should_stop = True
                    break
            if should_stop or (self.config.early_stopping
                               and self.patience_counter >= self.config.patience):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Callbacks - on_train_end
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(self)
        
        # Save final model
        self._save_checkpoint('final_model.pth')

        # A finished run has no use for its resume state, and on Drive it is
        # the largest file in the directory.
        last = Path(self.config.save_dir) / 'last.pth'
        if last.exists():
            last.unlink()

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
                    self.scaler.unscale_(self.optimizer)
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # fp32, or bf16 autocast. With enabled=False the context is a
                # no-op, so the fp32 path is exactly what it was before.
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self._bf16):
                    outputs = self.model(images)
                    loss, _ = self._compute_loss_with_deep_supervision(outputs, masks)
                    if accum_steps > 1:
                        loss = loss / accum_steps

                loss.backward()

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
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
        """Save checkpoint.

        Holds everything a resumed run needs to continue as if uninterrupted:
        optimizer moments, scheduler position, the AMP loss scale, the early-
        stopping counter, and every RNG the next epoch will draw from (the
        DataLoader shuffle comes from torch's CPU generator, dropout from CUDA's).
        """
        import random as _random
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_dice': self.best_val_dice,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'elapsed_seconds': self.elapsed_seconds,
            'history': self.history,
            'rng': {
                'torch': torch.get_rng_state(),
                'cuda': (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
                'numpy': np.random.get_state(),
                'python': _random.getstate(),
            },
            'config': self.config
        }

        # Write to a temporary name and rename into place. torch.save is not
        # atomic, and a runtime killed mid-write -- which is exactly when a
        # resume file matters -- would otherwise leave a truncated last.pth
        # that fails to load and takes the good one with it.
        path = Path(self.config.save_dir) / filename
        tmp = path.with_suffix(path.suffix + '.tmp')
        torch.save(checkpoint, tmp)
        tmp.replace(path)

    def load_checkpoint(self, filepath: str):
        """Load a checkpoint and arrange for train() to continue after it."""
        import random as _random
        checkpoint = torch.load(filepath, map_location=self.device,
                                weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_dice = checkpoint['best_val_dice']
        self.best_val_loss = checkpoint['best_val_loss']
        self.patience_counter = checkpoint.get('patience_counter', 0)
        self.elapsed_seconds = checkpoint.get('elapsed_seconds', 0.0)
        self.history = checkpoint['history']

        # Restore RNG last, after everything above that might draw from it.
        rng = checkpoint.get('rng')
        if rng:
            # map_location above moves every tensor to the training device,
            # including these generator states -- which must be CPU ByteTensors.
            torch.set_rng_state(rng['torch'].cpu())
            if rng.get('cuda') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng['cuda']])
            np.random.set_state(rng['numpy'])
            _random.setstate(rng['python'])

        print(f"Loaded checkpoint from epoch {self.current_epoch + 1}; "
              f"continuing at epoch {self.start_epoch + 1}")
    
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
