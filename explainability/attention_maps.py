"""
Attention map visualization for Transformer and Mamba models.

Provides tools to visualize and interpret attention patterns
in transformer-based and hybrid architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
import matplotlib.pyplot as plt


class AttentionVisualizer:
    """
    Visualize attention maps from transformer models.
    
    Extracts and visualizes attention weights from multi-head
    self-attention layers in transformer-based models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        attention_layer_names: Optional[List[str]] = None
    ):
        self.model = model
        self.attention_layer_names = attention_layer_names or self._find_attention_layers()
        self.attention_maps: Dict[str, torch.Tensor] = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _find_attention_layers(self) -> List[str]:
        """Auto-detect attention layers in the model."""
        attention_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                attention_layers.append(name)
            elif 'attn' in name.lower() and hasattr(module, 'forward'):
                attention_layers.append(name)
        return attention_layers
    
    def _register_hooks(self):
        """Register hooks to capture attention weights."""
        def get_attention_hook(name):
            def hook(module, input, output):
                # Handle different attention output formats
                if isinstance(output, tuple) and len(output) >= 2:
                    # MultiheadAttention returns (output, attention_weights)
                    self.attention_maps[name] = output[1].detach()
                elif hasattr(module, 'attention_weights'):
                    self.attention_maps[name] = module.attention_weights.detach()
            return hook
        
        for name in self.attention_layer_names:
            try:
                module = self._get_module(name)
                hook = module.register_forward_hook(get_attention_hook(name))
                self.hooks.append(hook)
            except AttributeError:
                print(f"Warning: Could not find layer {name}")
    
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
    
    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_attention_maps(
        self,
        input_image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps for an input.
        
        Args:
            input_image: Input tensor (B, C, H, W)
            
        Returns:
            Dictionary mapping layer names to attention weights
        """
        self.attention_maps = {}
        
        with torch.no_grad():
            self.model.eval()
            _ = self.model(input_image)
        
        return self.attention_maps
    
    def visualize_attention(
        self,
        attention: torch.Tensor,
        input_size: Tuple[int, int],
        head_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Visualize attention as a heatmap.
        
        Args:
            attention: Attention weights (B, num_heads, N, N) or (B, N, N)
            input_size: Original input size (H, W)
            head_idx: Specific head to visualize. If None, averages all heads
            
        Returns:
            Attention heatmap (H, W)
        """
        attention = attention.cpu().numpy()
        
        if attention.ndim == 4:
            if head_idx is not None:
                attention = attention[:, head_idx, :, :]
            else:
                attention = attention.mean(axis=1)
        
        # Average over batch
        attention = attention.mean(axis=0)
        
        # CLS token attention or average
        if attention.shape[0] > 1:
            # Use attention from CLS token (first) or average
            attention = attention[0, 1:]  # Attention from CLS to patches
        else:
            attention = attention.mean(axis=0)
        
        # Reshape to spatial
        num_patches = attention.shape[0]
        patch_size = int(np.sqrt(num_patches))
        
        if patch_size * patch_size == num_patches:
            attention = attention.reshape(patch_size, patch_size)
        else:
            # Handle non-square
            attention = attention.reshape(-1, 1).squeeze()
            attention = np.resize(attention, input_size)
        
        # Resize to input size
        attention = np.array(
            F.interpolate(
                torch.from_numpy(attention).unsqueeze(0).unsqueeze(0).float(),
                size=input_size,
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
        )
        
        # Normalize
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        return attention
    
    def plot_attention_heads(
        self,
        input_image: torch.Tensor,
        layer_name: str,
        save_path: Optional[str] = None
    ):
        """Plot attention from all heads in a layer."""
        attention_maps = self.get_attention_maps(input_image)
        
        if layer_name not in attention_maps:
            print(f"Layer {layer_name} not found in attention maps")
            return
        
        attention = attention_maps[layer_name]
        num_heads = attention.shape[1]
        
        # Create grid
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flatten() if num_heads > 1 else [axes]
        
        input_size = input_image.shape[2:]
        
        for i in range(num_heads):
            attn_map = self.visualize_attention(attention, input_size, head_idx=i)
            axes[i].imshow(attn_map, cmap='viridis')
            axes[i].set_title(f'Head {i}')
            axes[i].axis('off')
        
        # Hide unused axes
        for i in range(num_heads, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def __del__(self):
        self.remove_hooks()


class AttentionRollout:
    """
    Attention Rollout for computing global attention flow.
    
    Recursively multiplies attention matrices through layers
    to track how information flows from input to output.
    """
    
    def __init__(
        self,
        model: nn.Module,
        attention_layers: Optional[List[str]] = None,
        discard_ratio: float = 0.9
    ):
        self.model = model
        self.visualizer = AttentionVisualizer(model, attention_layers)
        self.discard_ratio = discard_ratio
    
    def compute_rollout(
        self,
        input_image: torch.Tensor,
        start_layer: int = 0
    ) -> np.ndarray:
        """
        Compute attention rollout.
        
        Args:
            input_image: Input tensor (B, C, H, W)
            start_layer: Layer to start rollout from
            
        Returns:
            Attention rollout map (H, W)
        """
        attention_maps = self.visualizer.get_attention_maps(input_image)
        
        # Get sorted attention layers
        layers = list(attention_maps.keys())
        
        if len(layers) == 0:
            raise ValueError("No attention maps captured")
        
        # Start with identity
        result = None
        
        for i, layer_name in enumerate(layers[start_layer:]):
            attention = attention_maps[layer_name]
            
            # Average over heads
            attention = attention.mean(dim=1).cpu().numpy()  # (B, N, N)
            
            # Add residual connection
            attention = 0.5 * attention + 0.5 * np.eye(attention.shape[-1])
            
            # Normalize rows
            attention = attention / attention.sum(axis=-1, keepdims=True)
            
            if result is None:
                result = attention
            else:
                result = np.matmul(attention, result)
        
        # Get attention from CLS token
        if result.ndim == 3:
            result = result.mean(axis=0)  # Average over batch
        
        mask = result[0, 1:]  # CLS to patches
        
        # Reshape to spatial
        num_patches = mask.shape[0]
        size = int(np.sqrt(num_patches))
        
        if size * size == num_patches:
            mask = mask.reshape(size, size)
        
        # Resize
        input_size = input_image.shape[2:]
        mask = F.interpolate(
            torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float(),
            size=input_size,
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        # Normalize
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        
        return mask
    
    def compute_rollout_per_class(
        self,
        input_image: torch.Tensor,
        prediction: torch.Tensor,
        num_classes: int = 4
    ) -> Dict[int, np.ndarray]:
        """Compute class-specific attention rollout."""
        base_rollout = self.compute_rollout(input_image)
        
        # Weight rollout by class prediction
        pred_probs = F.softmax(prediction, dim=1).cpu().numpy()
        
        class_rollouts = {}
        for c in range(num_classes):
            class_prob = pred_probs[0, c, :, :]
            
            # Resize class prob to match rollout
            if class_prob.shape != base_rollout.shape:
                class_prob = F.interpolate(
                    torch.from_numpy(class_prob).unsqueeze(0).unsqueeze(0).float(),
                    size=base_rollout.shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
            
            weighted_rollout = base_rollout * class_prob
            weighted_rollout = (weighted_rollout - weighted_rollout.min()) / \
                              (weighted_rollout.max() - weighted_rollout.min() + 1e-8)
            
            class_rollouts[c] = weighted_rollout
        
        return class_rollouts


class SwinAttentionVisualizer(AttentionVisualizer):
    """
    Specialized attention visualizer for Swin Transformer.
    
    Handles window-based attention and shifted windows.
    """
    
    def visualize_window_attention(
        self,
        attention: torch.Tensor,
        window_size: int,
        input_size: Tuple[int, int],
        shift_size: int = 0
    ) -> np.ndarray:
        """Visualize window-based attention from Swin Transformer."""
        attention = attention.cpu().numpy()
        
        # attention shape: (num_windows, num_heads, window_size^2, window_size^2)
        num_windows = attention.shape[0]
        H, W = input_size
        
        # Average over heads
        attention = attention.mean(axis=1)  # (num_windows, ws^2, ws^2)
        
        # Compute attention per patch (diagonal elements or average)
        patch_attention = attention.sum(axis=-1)  # (num_windows, ws^2)
        
        # Reshape windows back to image
        nH = H // window_size
        nW = W // window_size
        
        attention_map = patch_attention.reshape(nH, nW, window_size, window_size)
        attention_map = attention_map.transpose(0, 2, 1, 3).reshape(H, W)
        
        # Reverse shift if applied
        if shift_size > 0:
            attention_map = np.roll(attention_map, shift_size, axis=(0, 1))
        
        # Normalize
        attention_map = (attention_map - attention_map.min()) / \
                        (attention_map.max() - attention_map.min() + 1e-8)
        
        return attention_map


if __name__ == '__main__':
    print("Attention visualization module loaded successfully")
    print("Available classes: AttentionVisualizer, AttentionRollout, SwinAttentionVisualizer")
