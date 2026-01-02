"""
Feature map extraction and visualization.

Tools for extracting and visualizing intermediate feature maps
from any layer in the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
import matplotlib.pyplot as plt


class FeatureExtractor:
    """
    Extract intermediate feature maps from any layer.
    
    Registers hooks to capture activations during forward pass.
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
        return_input: bool = False
    ):
        self.model = model
        self.layer_names = layer_names or []
        self.return_input = return_input
        
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks = []
        
        if layer_names:
            self._register_hooks()
    
    def _get_module(self, name: str) -> nn.Module:
        """Get module by name."""
        parts = name.split('.')
        module = self.model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def _register_hooks(self):
        """Register forward hooks."""
        def get_hook(name):
            def hook(module, input, output):
                if self.return_input:
                    self.features[name] = input[0].detach() if isinstance(input, tuple) else input.detach()
                else:
                    self.features[name] = output.detach() if isinstance(output, torch.Tensor) else output[0].detach()
            return hook
        
        for name in self.layer_names:
            try:
                module = self._get_module(name)
                hook = module.register_forward_hook(get_hook(name))
                self.hooks.append(hook)
            except AttributeError:
                print(f"Warning: Could not find layer {name}")
    
    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features for an input.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            
        Returns:
            Dictionary of layer names to feature tensors
        """
        self.features = {}
        
        with torch.no_grad():
            self.model.eval()
            _ = self.model(input_tensor)
        
        return self.features
    
    def extract_all_layers(
        self,
        input_tensor: torch.Tensor,
        layer_types: Optional[Tuple] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from all layers of specified types.
        
        Args:
            input_tensor: Input tensor
            layer_types: Tuple of layer types to extract (e.g., (nn.Conv2d, nn.BatchNorm2d))
            
        Returns:
            Dictionary of all layer features
        """
        if layer_types is None:
            layer_types = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Linear)
        
        # Find all layers of specified types
        layer_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, layer_types):
                layer_names.append(name)
        
        # Register hooks
        self.layer_names = layer_names
        self._register_hooks()
        
        # Extract
        features = self.extract(input_tensor)
        
        return features
    
    def get_layer_info(self) -> Dict[str, Dict]:
        """Get information about each extracted layer."""
        info = {}
        for name, feat in self.features.items():
            info[name] = {
                'shape': list(feat.shape),
                'mean': feat.mean().item(),
                'std': feat.std().item(),
                'min': feat.min().item(),
                'max': feat.max().item()
            }
        return info
    
    def __del__(self):
        self.remove_hooks()


class FeatureVisualizer:
    """
    Visualize feature maps in various formats.
    """
    
    @staticmethod
    def visualize_channels(
        feature: torch.Tensor,
        num_channels: int = 16,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Visualize individual channels as a grid.
        
        Args:
            feature: Feature tensor (B, C, H, W) or (C, H, W)
            num_channels: Number of channels to visualize
            normalize: Whether to normalize each channel
            
        Returns:
            Grid visualization as numpy array
        """
        if feature.dim() == 4:
            feature = feature[0]  # Take first batch
        
        C, H, W = feature.shape
        num_channels = min(num_channels, C)
        
        # Create grid
        cols = int(np.ceil(np.sqrt(num_channels)))
        rows = int(np.ceil(num_channels / cols))
        
        grid = np.zeros((rows * H, cols * W))
        
        for i in range(num_channels):
            row = i // cols
            col = i % cols
            
            channel = feature[i].cpu().numpy()
            
            if normalize:
                channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            
            grid[row * H:(row + 1) * H, col * W:(col + 1) * W] = channel
        
        return grid
    
    @staticmethod
    def visualize_mean_activation(
        feature: torch.Tensor
    ) -> np.ndarray:
        """Visualize mean activation across channels."""
        if feature.dim() == 4:
            feature = feature[0]
        
        mean_act = feature.mean(dim=0).cpu().numpy()
        mean_act = (mean_act - mean_act.min()) / (mean_act.max() - mean_act.min() + 1e-8)
        
        return mean_act
    
    @staticmethod
    def visualize_max_activation(
        feature: torch.Tensor
    ) -> np.ndarray:
        """Visualize max activation across channels."""
        if feature.dim() == 4:
            feature = feature[0]
        
        max_act = feature.max(dim=0)[0].cpu().numpy()
        max_act = (max_act - max_act.min()) / (max_act.max() - max_act.min() + 1e-8)
        
        return max_act
    
    @staticmethod
    def visualize_activation_histogram(
        feature: torch.Tensor,
        bins: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get activation histogram."""
        values = feature.cpu().numpy().flatten()
        hist, edges = np.histogram(values, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, hist
    
    @staticmethod
    def plot_feature_maps(
        features: Dict[str, torch.Tensor],
        input_image: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ):
        """Plot feature maps from multiple layers."""
        num_layers = len(features)
        
        if input_image is not None:
            num_layers += 1
        
        cols = min(4, num_layers)
        rows = (num_layers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flatten() if num_layers > 1 else [axes]
        
        idx = 0
        
        # Plot input image if provided
        if input_image is not None:
            if input_image.dim() == 4:
                img = input_image[0, 0].cpu().numpy()
            elif input_image.dim() == 3:
                img = input_image[0].cpu().numpy()
            else:
                img = input_image.cpu().numpy()
            
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title('Input')
            axes[idx].axis('off')
            idx += 1
        
        # Plot feature maps
        for name, feat in features.items():
            vis = FeatureVisualizer.visualize_mean_activation(feat)
            axes[idx].imshow(vis, cmap='viridis')
            axes[idx].set_title(name[:30])  # Truncate long names
            axes[idx].axis('off')
            idx += 1
        
        # Hide unused axes
        for i in range(idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_feature_pyramid(
        features: Dict[str, torch.Tensor],
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create a feature pyramid visualization.
        
        Resizes all features to same size and stacks as channels.
        """
        resized = []
        
        for name, feat in features.items():
            if feat.dim() == 4:
                feat = feat[0]
            
            # Mean across channels
            mean_feat = feat.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)
            
            # Resize
            resized_feat = F.interpolate(
                mean_feat,
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze().cpu().numpy()
            
            # Normalize
            resized_feat = (resized_feat - resized_feat.min()) / \
                          (resized_feat.max() - resized_feat.min() + 1e-8)
            
            resized.append(resized_feat)
        
        # Stack as RGB-like (take first 3 or pad)
        if len(resized) >= 3:
            pyramid = np.stack(resized[:3], axis=-1)
        else:
            pyramid = np.stack(resized + [resized[0]] * (3 - len(resized)), axis=-1)
        
        return pyramid


class DeepFeatureAnalyzer:
    """
    Analyze deep features for interpretability.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.extractor = None
    
    def analyze_receptive_field(
        self,
        layer_name: str,
        input_size: Tuple[int, int] = (256, 256)
    ) -> Dict:
        """Estimate effective receptive field of a layer."""
        # Create gradient-based receptive field estimation
        input_tensor = torch.zeros(1, 1, *input_size, requires_grad=True)
        
        self.extractor = FeatureExtractor(self.model, [layer_name])
        features = self.extractor.extract(input_tensor)
        
        if layer_name not in features:
            return {}
        
        feat = features[layer_name]
        
        # Get center activation
        if feat.dim() == 4:
            _, _, h, w = feat.shape
            center_h, center_w = h // 2, w // 2
            target = feat[0, :, center_h, center_w].sum()
        else:
            target = feat[0, feat.shape[1] // 2, :].sum()
        
        # Backward
        target.backward()
        
        # Gradient magnitude shows receptive field
        grad = input_tensor.grad[0, 0].abs().numpy()
        
        # Estimate RF size
        threshold = grad.max() * 0.01
        rf_mask = grad > threshold
        
        # Find bounding box
        rows = np.any(rf_mask, axis=1)
        cols = np.any(rf_mask, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            rf_height = rmax - rmin + 1
            rf_width = cmax - cmin + 1
        else:
            rf_height = rf_width = 0
        
        return {
            'gradient_map': grad,
            'rf_height': rf_height,
            'rf_width': rf_width,
            'rf_area': rf_height * rf_width
        }
    
    def compare_layer_activations(
        self,
        input_tensor: torch.Tensor,
        layer_names: List[str]
    ) -> Dict[str, Dict]:
        """Compare activation statistics across layers."""
        self.extractor = FeatureExtractor(self.model, layer_names)
        features = self.extractor.extract(input_tensor)
        
        comparison = {}
        
        for name, feat in features.items():
            flat = feat.cpu().numpy().flatten()
            
            comparison[name] = {
                'mean': float(np.mean(flat)),
                'std': float(np.std(flat)),
                'min': float(np.min(flat)),
                'max': float(np.max(flat)),
                'sparsity': float(np.mean(np.abs(flat) < 0.01)),
                'shape': list(feat.shape)
            }
        
        return comparison


if __name__ == '__main__':
    print("Feature map visualization module loaded successfully")
    print("Available classes: FeatureExtractor, FeatureVisualizer, DeepFeatureAnalyzer")
