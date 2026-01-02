"""
Learning rate schedulers for training.

Includes warmup cosine, polynomial, and custom schedulers.
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Optional


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing with linear warmup.
    
    LR increases linearly during warmup, then follows cosine decay.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        min_lr: Minimum learning rate
        warmup_start_lr: Starting LR for warmup
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-7,
        warmup_start_lr: float = 1e-7,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            return [
                self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )
                for base_lr in self.base_lrs
            ]


class PolyScheduler(_LRScheduler):
    """
    Polynomial learning rate decay.
    
    LR = base_lr * (1 - epoch/total_epochs)^power
    
    Args:
        optimizer: PyTorch optimizer
        total_epochs: Total training epochs
        power: Polynomial power (typically 0.9)
        min_lr: Minimum learning rate
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        power: float = 0.9,
        min_lr: float = 1e-7,
        last_epoch: int = -1
    ):
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        factor = (1 - self.last_epoch / self.total_epochs) ** self.power
        return [
            max(self.min_lr, base_lr * factor)
            for base_lr in self.base_lrs
        ]


class WarmupPolyScheduler(_LRScheduler):
    """
    Polynomial decay with warmup.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        power: float = 0.9,
        min_lr: float = 1e-7,
        warmup_start_lr: float = 1e-7,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Poly decay
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            factor = (1 - progress) ** self.power
            return [
                max(self.min_lr, base_lr * factor)
                for base_lr in self.base_lrs
            ]


class OneCycleLR(_LRScheduler):
    """
    1cycle learning rate policy.
    
    Increases LR from min to max, then decreases to min again.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.total_steps * self.pct_start:
            # Increase phase
            progress = step / (self.total_steps * self.pct_start)
            return [
                self.initial_lr + progress * (self.max_lr - self.initial_lr)
                for _ in self.base_lrs
            ]
        else:
            # Decrease phase
            progress = (step - self.total_steps * self.pct_start) / (
                self.total_steps * (1 - self.pct_start)
            )
            return [
                self.max_lr - progress * (self.max_lr - self.final_lr)
                for _ in self.base_lrs
            ]


class LinearWarmupScheduler(_LRScheduler):
    """
    Simple linear warmup followed by constant LR.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        warmup_start_lr: float = 1e-7,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        return self.base_lrs


def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 5,
    min_lr: float = 1e-7,
    **kwargs
) -> _LRScheduler:
    """
    Factory function for schedulers.
    
    Args:
        name: Scheduler name ('cosine', 'poly', 'step', 'warmup_cosine')
        optimizer: PyTorch optimizer
        total_epochs: Total training epochs
        warmup_epochs: Warmup epochs
        min_lr: Minimum learning rate
        
    Returns:
        Learning rate scheduler
    """
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=min_lr
        )
    
    elif name == 'warmup_cosine':
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr=min_lr
        )
    
    elif name == 'poly':
        return PolyScheduler(
            optimizer,
            total_epochs=total_epochs,
            power=kwargs.get('power', 0.9),
            min_lr=min_lr
        )
    
    elif name == 'warmup_poly':
        return WarmupPolyScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            power=kwargs.get('power', 0.9),
            min_lr=min_lr
        )
    
    elif name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif name == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get('milestones', [30, 60, 90]),
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10),
            min_lr=min_lr
        )
    
    elif name == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', optimizer.param_groups[0]['lr']),
            total_steps=total_epochs
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {name}")


if __name__ == '__main__':
    # Test schedulers
    import matplotlib.pyplot as plt
    
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    schedulers = {
        'warmup_cosine': WarmupCosineScheduler(optimizer, 10, 100),
        'warmup_poly': WarmupPolyScheduler(optimizer, 10, 100),
    }
    
    for name, scheduler in schedulers.items():
        lrs = []
        for epoch in range(100):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        plt.plot(lrs, label=name)
        
        # Reset optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.title('Learning Rate Schedules')
    plt.savefig('lr_schedules.png')
    print("Saved lr_schedules.png")
