"""
I/O utilities for saving and loading models and configs.
"""

import torch
import torch.nn as nn
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import asdict, is_dataclass


def save_model(
    model: nn.Module,
    filepath: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict] = None,
    config: Optional[Any] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        filepath: Save path
        optimizer: Optional optimizer state
        epoch: Current epoch
        metrics: Training metrics
        config: Training configuration
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if config is not None:
        if is_dataclass(config):
            config = asdict(config)
        checkpoint['config'] = config
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(
    model: nn.Module,
    filepath: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cuda',
    strict: bool = True
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model (architecture must match)
        filepath: Checkpoint path
        optimizer: Optional optimizer to load state
        device: Device to load to
        strict: Strict state dict loading
        
    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {filepath}")
    
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"  Metrics: {checkpoint['metrics']}")
    
    return checkpoint


def save_config(config: Union[Dict, Any], filepath: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dict or dataclass
        filepath: Save path (.yaml or .json)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if is_dataclass(config):
        config = asdict(config)
    
    if filepath.suffix == '.yaml' or filepath.suffix == '.yml':
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unknown config format: {filepath.suffix}")
    
    print(f"Config saved to {filepath}")


def load_config(filepath: str) -> Dict:
    """
    Load configuration from file.
    
    Args:
        filepath: Config file path (.yaml or .json)
        
    Returns:
        Configuration dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    if filepath.suffix == '.yaml' or filepath.suffix == '.yml':
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unknown config format: {filepath.suffix}")
    
    return config


def export_onnx(
    model: nn.Module,
    filepath: str,
    input_shape: tuple = (1, 1, 256, 256),
    opset_version: int = 11,
    dynamic_axes: Optional[Dict] = None
):
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        filepath: Output path
        input_shape: Input tensor shape
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes for variable batch size
    """
    model.eval()
    
    dummy_input = torch.randn(*input_shape)
    
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
    print(f"ONNX model exported to {filepath}")


def export_torchscript(
    model: nn.Module,
    filepath: str,
    input_shape: tuple = (1, 1, 256, 256),
    method: str = 'trace'
):
    """
    Export model to TorchScript format.
    
    Args:
        model: PyTorch model
        filepath: Output path
        input_shape: Input tensor shape
        method: 'trace' or 'script'
    """
    model.eval()
    
    if method == 'trace':
        dummy_input = torch.randn(*input_shape)
        scripted = torch.jit.trace(model, dummy_input)
    else:
        scripted = torch.jit.script(model)
    
    scripted.save(filepath)
    print(f"TorchScript model exported to {filepath}")


class ModelIO:
    """
    Unified model I/O handler.
    """
    
    def __init__(self, save_dir: str = './checkpoints'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        model: nn.Module,
        name: str,
        **kwargs
    ):
        """Save model with given name."""
        filepath = self.save_dir / f'{name}.pth'
        save_model(model, str(filepath), **kwargs)
    
    def load(
        self,
        model: nn.Module,
        name: str,
        **kwargs
    ) -> Dict:
        """Load model with given name."""
        filepath = self.save_dir / f'{name}.pth'
        return load_model(model, str(filepath), **kwargs)
    
    def list_checkpoints(self) -> list:
        """List available checkpoints."""
        return sorted(self.save_dir.glob('*.pth'))
    
    def get_best(self) -> Optional[Path]:
        """Get best model checkpoint if exists."""
        best_path = self.save_dir / 'best_model.pth'
        return best_path if best_path.exists() else None


if __name__ == '__main__':
    # Test I/O
    model = torch.nn.Linear(10, 10)
    
    save_model(model, 'test_model.pth', epoch=1, metrics={'dice': 0.9})
    load_model(model, 'test_model.pth')
    
    config = {'lr': 0.001, 'epochs': 100}
    save_config(config, 'test_config.yaml')
    loaded = load_config('test_config.yaml')
    
    print("I/O module loaded successfully")
