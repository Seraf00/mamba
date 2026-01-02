"""
Mamba State Space Model visualization.

Provides tools to visualize and interpret the internal state
dynamics of Mamba SSM blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


class MambaStateVisualizer:
    """
    Visualize Mamba SSM state evolution.
    
    Captures and visualizes the hidden state dynamics of Mamba
    blocks to understand how the model processes sequences.
    """
    
    def __init__(
        self,
        model: nn.Module,
        mamba_layer_names: Optional[List[str]] = None
    ):
        self.model = model
        self.mamba_layer_names = mamba_layer_names or self._find_mamba_layers()
        
        self.state_history: Dict[str, List[torch.Tensor]] = {}
        self.delta_values: Dict[str, torch.Tensor] = {}
        self.B_values: Dict[str, torch.Tensor] = {}
        self.C_values: Dict[str, torch.Tensor] = {}
        
        self.hooks = []
        self._register_hooks()
    
    def _find_mamba_layers(self) -> List[str]:
        """Auto-detect Mamba layers in the model."""
        mamba_layers = []
        for name, module in self.model.named_modules():
            if 'mamba' in name.lower():
                if hasattr(module, 'forward'):
                    mamba_layers.append(name)
        return mamba_layers
    
    def _register_hooks(self):
        """Register hooks to capture Mamba states."""
        def get_state_hook(name):
            def hook(module, input, output):
                # Store output for visualization
                if isinstance(output, torch.Tensor):
                    self.state_history[name] = [output.detach()]
                elif isinstance(output, tuple):
                    self.state_history[name] = [o.detach() for o in output if isinstance(o, torch.Tensor)]
                
                # Try to capture SSM parameters if available
                if hasattr(module, 'delta'):
                    self.delta_values[name] = module.delta.detach() if isinstance(module.delta, torch.Tensor) else None
                if hasattr(module, 'B'):
                    self.B_values[name] = module.B.detach() if isinstance(module.B, torch.Tensor) else None
                if hasattr(module, 'C'):
                    self.C_values[name] = module.C.detach() if isinstance(module.C, torch.Tensor) else None
            return hook
        
        for name in self.mamba_layer_names:
            try:
                module = self._get_module(name)
                hook = module.register_forward_hook(get_state_hook(name))
                self.hooks.append(hook)
            except AttributeError:
                pass
    
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
    
    def visualize_states(
        self,
        input_image: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Visualize state evolution for each Mamba layer.
        
        Args:
            input_image: Input tensor (B, C, H, W)
            
        Returns:
            Dictionary of state visualizations per layer
        """
        self.state_history = {}
        self.delta_values = {}
        self.B_values = {}
        self.C_values = {}
        
        with torch.no_grad():
            self.model.eval()
            _ = self.model(input_image)
        
        visualizations = {}
        
        for name, states in self.state_history.items():
            if len(states) > 0:
                state = states[0]
                
                if state.dim() == 4:  # (B, C, H, W)
                    # Visualize as feature map
                    vis = self._visualize_feature_map(state)
                elif state.dim() == 3:  # (B, L, D)
                    # Visualize as sequence
                    vis = self._visualize_sequence_state(state, input_image.shape[2:])
                else:
                    vis = state.mean(dim=0).cpu().numpy()
                
                visualizations[name] = vis
        
        return visualizations
    
    def _visualize_feature_map(self, feature: torch.Tensor) -> np.ndarray:
        """Convert feature map to visualization."""
        # Average over batch and channels
        vis = feature.mean(dim=(0, 1)).cpu().numpy()
        
        # Normalize
        vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
        
        return vis
    
    def _visualize_sequence_state(
        self,
        state: torch.Tensor,
        spatial_size: Tuple[int, int]
    ) -> np.ndarray:
        """Reshape sequence state to spatial visualization."""
        B, L, D = state.shape
        H, W = spatial_size
        
        # Average over batch and features
        state = state.mean(dim=(0, 2)).cpu().numpy()  # (L,)
        
        # Try to reshape to spatial
        if L == H * W:
            vis = state.reshape(H, W)
        else:
            # Interpolate
            vis = np.interp(
                np.linspace(0, L - 1, H * W),
                np.arange(L),
                state
            ).reshape(H, W)
        
        # Normalize
        vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
        
        return vis
    
    def visualize_scan_patterns(
        self,
        input_image: torch.Tensor,
        layer_name: str
    ) -> Dict[str, np.ndarray]:
        """
        Visualize different scan patterns in VM-Mamba.
        
        For Visual Mamba with cross-scan (SS2D), visualizes
        the four scan directions.
        """
        _ = self.visualize_states(input_image)
        
        if layer_name not in self.state_history:
            return {}
        
        states = self.state_history[layer_name]
        
        if len(states) < 4:
            return {'combined': self._visualize_feature_map(states[0])}
        
        scan_names = ['left_right', 'right_left', 'top_bottom', 'bottom_top']
        visualizations = {}
        
        for i, (name, state) in enumerate(zip(scan_names, states[:4])):
            visualizations[name] = self._visualize_feature_map(state)
        
        return visualizations
    
    def visualize_state_evolution(
        self,
        input_image: torch.Tensor,
        layer_name: str,
        position: Tuple[int, int] = (0, 0)
    ) -> np.ndarray:
        """
        Visualize how state evolves at a specific spatial position.
        
        Args:
            input_image: Input tensor
            layer_name: Target Mamba layer
            position: (x, y) position to track
            
        Returns:
            State evolution over channels/time
        """
        _ = self.visualize_states(input_image)
        
        if layer_name not in self.state_history:
            return np.array([])
        
        state = self.state_history[layer_name][0]
        
        if state.dim() == 4:  # (B, C, H, W)
            x, y = position
            evolution = state[0, :, y, x].cpu().numpy()
        else:
            evolution = state[0, position[0], :].cpu().numpy()
        
        return evolution
    
    def plot_state_dynamics(
        self,
        input_image: torch.Tensor,
        layer_name: str,
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive state dynamics visualization.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Get visualizations
        vis = self.visualize_states(input_image)
        
        if layer_name not in vis:
            print(f"Layer {layer_name} not found")
            return fig
        
        # State activation map
        axes[0, 0].imshow(vis[layer_name], cmap='viridis')
        axes[0, 0].set_title(f'State Activation: {layer_name}')
        axes[0, 0].axis('off')
        
        # Original image
        img = input_image[0, 0].cpu().numpy() if input_image.dim() == 4 else input_image[0].cpu().numpy()
        axes[0, 1].imshow(img, cmap='gray')
        axes[0, 1].set_title('Input Image')
        axes[0, 1].axis('off')
        
        # Overlay
        overlay = 0.5 * (img - img.min()) / (img.max() - img.min() + 1e-8)
        overlay += 0.5 * vis[layer_name]
        axes[1, 0].imshow(overlay, cmap='viridis')
        axes[1, 0].set_title('State Overlay')
        axes[1, 0].axis('off')
        
        # State histogram
        state = self.state_history[layer_name][0].cpu().numpy().flatten()
        axes[1, 1].hist(state, bins=50, color='blue', alpha=0.7)
        axes[1, 1].set_title('State Value Distribution')
        axes[1, 1].set_xlabel('State Value')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def compute_state_importance(
        self,
        input_image: torch.Tensor,
        output: torch.Tensor,
        layer_name: str
    ) -> np.ndarray:
        """
        Compute importance of each state dimension for the output.
        
        Uses gradient-based importance estimation.
        """
        self.model.eval()
        
        # Get state through forward pass
        _ = self.model(input_image.requires_grad_(True))
        
        if layer_name not in self.state_history:
            return np.array([])
        
        state = self.state_history[layer_name][0]
        
        # Compute gradient of output w.r.t. state
        grad = torch.autograd.grad(
            output.sum(),
            state,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        if grad is None:
            return np.array([])
        
        # Importance is magnitude of gradient * activation
        importance = (grad.abs() * state.abs()).mean(dim=0)
        
        return importance.cpu().numpy()
    
    def __del__(self):
        self.remove_hooks()


class CrossScanVisualizer:
    """
    Visualize cross-scan patterns in Visual Mamba (SS2D).
    
    Shows how the 2D cross-scan unfolds the image into sequences
    and processes them in different directions.
    """
    
    def __init__(self, window_size: int = 8):
        self.window_size = window_size
    
    def visualize_scan_order(
        self,
        image_size: Tuple[int, int],
        scan_direction: str = 'left_right'
    ) -> np.ndarray:
        """
        Visualize the order in which pixels are processed.
        
        Args:
            image_size: (H, W) of the image
            scan_direction: One of 'left_right', 'right_left', 'top_bottom', 'bottom_top'
            
        Returns:
            Order map showing processing sequence
        """
        H, W = image_size
        order = np.zeros((H, W))
        
        if scan_direction == 'left_right':
            order = np.arange(H * W).reshape(H, W)
        elif scan_direction == 'right_left':
            order = np.arange(H * W).reshape(H, W)[:, ::-1]
        elif scan_direction == 'top_bottom':
            order = np.arange(H * W).reshape(W, H).T
        elif scan_direction == 'bottom_top':
            order = np.arange(H * W).reshape(W, H).T[::-1, :]
        else:
            # Cross-scan pattern
            order = self._cross_scan_order(H, W)
        
        return order
    
    def _cross_scan_order(self, H: int, W: int) -> np.ndarray:
        """Generate cross-scan order."""
        order = np.zeros((H, W))
        idx = 0
        
        for d in range(H + W - 1):
            if d % 2 == 0:
                # Go down
                for i in range(min(d, H - 1), max(0, d - W + 1) - 1, -1):
                    j = d - i
                    if 0 <= j < W:
                        order[i, j] = idx
                        idx += 1
            else:
                # Go up
                for i in range(max(0, d - W + 1), min(d, H - 1) + 1):
                    j = d - i
                    if 0 <= j < W:
                        order[i, j] = idx
                        idx += 1
        
        return order
    
    def plot_all_scans(
        self,
        image_size: Tuple[int, int] = (32, 32),
        save_path: Optional[str] = None
    ):
        """Plot all four scan directions."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        directions = ['left_right', 'right_left', 'top_bottom', 'bottom_top']
        titles = ['Left → Right', 'Right → Left', 'Top → Bottom', 'Bottom → Top']
        
        for ax, direction, title in zip(axes.flatten(), directions, titles):
            order = self.visualize_scan_order(image_size, direction)
            im = ax.imshow(order, cmap='viridis')
            ax.set_title(title)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.suptitle('Visual Mamba Cross-Scan Patterns', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


if __name__ == '__main__':
    print("Mamba State Visualization module loaded successfully")
    print("Available classes: MambaStateVisualizer, CrossScanVisualizer")
