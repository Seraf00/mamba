"""
Miscellaneous utility functions.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed set to {seed}")


def pin_determinism(seed: int = 42, mode: str = "full") -> dict:
    """Pin arithmetic precision and algorithm choice, and report what was set.

    ``set_seed`` above pins the RNG and cuDNN's algorithm *selection*, but not
    arithmetic *precision* -- and precision is the larger effect by far. TF32 is
    on by default from Ampere onward; ablated on FPN-UNet in
    ``results/hardware/ef_precision_ablation.json``, turning it off moved the
    EF by 0.88 points, more than the gaps between adjacent architectures in the
    leaderboard. cuDNN determinism accounted for none of the shift and
    deterministic algorithms for 0.28 of it; TF32 was the whole remainder.

    TF32 off is full FP32 and the more accurate computation, so it is what
    ships. Training a session without this means its numbers are not comparable
    to the pinned EF tables, and moving to a different GPU generation changes
    the default -- which is exactly the confound this revision exists to remove.

    Must be called before any CUDA work. ``mode`` mirrors
    ``scripts/yolo/eval_baseline_ef.py`` so a flag can be ablated on its own;
    ``full`` is the shipping configuration.
    """
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('PYTHONHASHSEED', str(seed))

    set_seed(seed)

    want_cudnn = mode in ('full', 'cudnn')
    want_tf32 = mode in ('full', 'tf32')
    want_algos = mode in ('full', 'algos')

    torch.backends.cudnn.benchmark = not want_cudnn   # autotuning off when pinned
    torch.backends.cudnn.deterministic = want_cudnn
    torch.backends.cuda.matmul.allow_tf32 = not want_tf32
    torch.backends.cudnn.allow_tf32 = not want_tf32

    # warn_only: a few upsampling kernels have no deterministic implementation,
    # and that must not abort a 100-epoch run.
    torch.use_deterministic_algorithms(want_algos, warn_only=True)

    return {
        'mode': mode,
        'seed': seed,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'torch': torch.__version__,
        'gpu': (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else 'cpu'),
    }


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Get available device.
    
    Args:
        gpu_id: Specific GPU ID, None for auto-detect
        
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        count = sum(p.numel() for p in model.parameters())
    
    return count


def print_model_summary(model: nn.Module, input_size: tuple = (1, 1, 256, 256)):
    """
    Print model summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
    """
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print("=" * 60)
    
    # Layer breakdown
    print("\nLayer breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,} params")


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate."""
    return optimizer.param_groups[0]['lr']


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    """Set learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def freeze_layers(model: nn.Module, layer_names: list):
    """
    Freeze specified layers.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                break


def unfreeze_layers(model: nn.Module, layer_names: Optional[list] = None):
    """
    Unfreeze specified layers (or all if None).
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze (None = all)
    """
    for name, param in model.named_parameters():
        if layer_names is None:
            param.requires_grad = True
        else:
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = True
                    break


def get_memory_usage() -> dict:
    """Get GPU memory usage."""
    if not torch.cuda.is_available():
        return {'available': False}
    
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
        'cached_mb': torch.cuda.memory_reserved() / 1024 / 1024,
        'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


def clear_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


class AverageMeter:
    """
    Compute and store the average and current value.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """Simple timer for profiling."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        import time
        self.start_time = time.time()
    
    def stop(self):
        import time
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


if __name__ == '__main__':
    # Test utilities
    set_seed(42)
    device = get_device()
    
    model = nn.Linear(10, 10)
    print(f"Parameters: {count_parameters(model):,}")
    
    print("Misc utilities loaded successfully")
