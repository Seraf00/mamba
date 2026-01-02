"""
Training callbacks for monitoring and checkpointing.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class EarlyStopping:
    """
    Early stopping callback.
    
    Stops training when monitored metric stops improving.
    
    Args:
        patience: Number of epochs with no improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max'
        verbose: Print messages
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        """Check if should stop."""
        score = logs.get('val_dice', logs.get('val_loss', 0))
        
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print("Early stopping triggered")
    
    def _is_improvement(self, score: float) -> bool:
        """Check if score is an improvement."""
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


class ModelCheckpoint:
    """
    Save model checkpoints.
    
    Args:
        save_dir: Directory to save checkpoints
        monitor: Metric to monitor
        mode: 'min' or 'max'
        save_best_only: Only save when metric improves
        save_every: Save every N epochs
    """
    
    def __init__(
        self,
        save_dir: str = './checkpoints',
        monitor: str = 'val_dice',
        mode: str = 'max',
        save_best_only: bool = True,
        save_every: Optional[int] = None,
        verbose: bool = True
    ):
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_every = save_every
        self.verbose = verbose
        
        self.best_score = None
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        """Save checkpoint if needed."""
        score = logs.get(self.monitor, 0)
        
        # Save periodic
        if self.save_every and (epoch + 1) % self.save_every == 0:
            self._save(trainer, f'checkpoint_epoch_{epoch + 1}.pth')
        
        # Save best
        if self.best_score is None or self._is_improvement(score):
            self.best_score = score
            if self.save_best_only:
                self._save(trainer, 'best_model.pth')
                if self.verbose:
                    print(f"Saved best model with {self.monitor}: {score:.4f}")
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'max':
            return score > self.best_score
        return score < self.best_score
    
    def _save(self, trainer, filename: str):
        """Save model."""
        path = self.save_dir / filename
        torch.save({
            'epoch': trainer.current_epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'best_score': self.best_score,
            'history': trainer.history
        }, path)


class TensorBoardLogger:
    """
    TensorBoard logging callback.
    
    Args:
        log_dir: Directory for TensorBoard logs
    """
    
    def __init__(self, log_dir: str = './logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = None
    
    def on_train_begin(self, trainer):
        """Initialize writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(self.log_dir / timestamp)
        except ImportError:
            print("TensorBoard not available")
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        """Log metrics."""
        if self.writer is None:
            return
        
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)
        
        # Log learning rate
        lr = trainer.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, epoch)
    
    def on_train_end(self, trainer):
        """Close writer."""
        if self.writer is not None:
            self.writer.close()


class CSVLogger:
    """
    Log training metrics to CSV file.
    """
    
    def __init__(self, filename: str = 'training_log.csv'):
        self.filename = filename
        self.file = None
        self.header_written = False
    
    def on_train_begin(self, trainer):
        """Open file."""
        self.file = open(self.filename, 'w')
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        """Write metrics."""
        if self.file is None:
            return
        
        logs['epoch'] = epoch
        logs['lr'] = trainer.optimizer.param_groups[0]['lr']
        
        if not self.header_written:
            self.file.write(','.join(logs.keys()) + '\n')
            self.header_written = True
        
        values = [str(v) if not isinstance(v, float) else f'{v:.6f}' 
                  for v in logs.values()]
        self.file.write(','.join(values) + '\n')
        self.file.flush()
    
    def on_train_end(self, trainer):
        """Close file."""
        if self.file is not None:
            self.file.close()


class LearningRateMonitor:
    """
    Monitor and log learning rate changes.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.lr_history = []
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        """Log current learning rate."""
        lr = trainer.optimizer.param_groups[0]['lr']
        self.lr_history.append(lr)
        
        if self.verbose and epoch > 0:
            prev_lr = self.lr_history[-2] if len(self.lr_history) > 1 else lr
            if lr != prev_lr:
                print(f"Learning rate changed: {prev_lr:.2e} -> {lr:.2e}")


class ProgressPrinter:
    """
    Print training progress.
    """
    
    def __init__(self, print_every: int = 1):
        self.print_every = print_every
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        """Print progress."""
        if (epoch + 1) % self.print_every != 0:
            return
        
        msg = f"Epoch {epoch + 1}"
        for key, value in logs.items():
            if isinstance(value, float):
                msg += f" | {key}: {value:.4f}"
            elif isinstance(value, bool):
                if key == 'is_best' and value:
                    msg += " | *BEST*"
        
        print(msg)


if __name__ == '__main__':
    print("Callbacks module loaded successfully")
    print("Available: EarlyStopping, ModelCheckpoint, TensorBoardLogger, CSVLogger")
