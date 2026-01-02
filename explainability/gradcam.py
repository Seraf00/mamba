"""
Grad-CAM and variants for segmentation model explainability.

Provides gradient-weighted class activation mapping for understanding
which regions of the input contribute most to predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Union, Dict
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Generates heatmaps showing which regions of the input image
    contributed most to the model's prediction for a specific class.
    
    Args:
        model: The segmentation model
        target_layer: Name of the layer to compute CAM for
        use_cuda: Whether to use GPU
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        use_cuda: bool = True
    ):
        self.model = model
        self.target_layer = target_layer
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
        
        if self.use_cuda:
            self.model = self.model.cuda()
    
    def _get_target_layer(self) -> nn.Module:
        """Get the target layer module from the model."""
        layers = self.target_layer.split('.')
        module = self.model
        for layer in layers:
            if layer.isdigit():
                module = module[int(layer)]
            else:
                module = getattr(module, layer)
        return module
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        target = self._get_target_layer()
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        self.hooks.append(target.register_forward_hook(forward_hook))
        self.hooks.append(target.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate(
        self,
        input_image: torch.Tensor,
        class_idx: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_image: Input tensor (B, C, H, W) or (C, H, W)
            class_idx: Target class index. If None, uses predicted class
            normalize: Whether to normalize output to [0, 1]
            
        Returns:
            Heatmap as numpy array (H, W)
        """
        # Handle input dimensions
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        
        if self.use_cuda:
            input_image = input_image.cuda()
        
        input_image.requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Handle different output formats
        if isinstance(output, dict):
            output = output['out']
        
        B, C, H, W = output.shape
        
        # Get target class
        if class_idx is None:
            # Use the class with highest average probability
            class_idx = output.mean(dim=(0, 2, 3)).argmax().item()
        
        # Create target for backprop
        target = torch.zeros_like(output)
        target[:, class_idx, :, :] = output[:, class_idx, :, :]
        
        # Backward pass
        self.model.zero_grad()
        target.sum().backward(retain_graph=True)
        
        # Compute CAM
        gradients = self.gradients  # (B, C, h, w)
        activations = self.activations  # (B, C, h, w)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, h, w)
        cam = F.relu(cam)  # Only positive contributions
        
        # Resize to input size
        cam = F.interpolate(
            cam,
            size=input_image.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Convert to numpy
        cam = cam.squeeze().cpu().numpy()
        
        if normalize and cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def generate_all_classes(
        self,
        input_image: torch.Tensor,
        num_classes: int = 4
    ) -> Dict[int, np.ndarray]:
        """Generate CAM for all classes."""
        cams = {}
        for class_idx in range(num_classes):
            cams[class_idx] = self.generate(input_image, class_idx)
        return cams
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on image.
        
        Args:
            image: Original image (H, W) or (H, W, 3)
            heatmap: CAM heatmap (H, W) normalized to [0, 1]
            alpha: Transparency for overlay
            colormap: OpenCV colormap
            
        Returns:
            Overlaid image (H, W, 3)
        """
        # Normalize image
        if image.max() > 1:
            image = image / 255.0
        
        # Handle grayscale
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Apply colormap to heatmap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap), colormap
        )
        heatmap_colored = heatmap_colored[:, :, ::-1] / 255.0  # BGR to RGB
        
        # Overlay
        overlay = alpha * heatmap_colored + (1 - alpha) * image
        overlay = np.clip(overlay, 0, 1)
        
        return (overlay * 255).astype(np.uint8)
    
    def __del__(self):
        self.remove_hooks()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ with improved localization.
    
    Uses second-order gradients for better localization of multiple
    instances of the same class and smaller objects.
    """
    
    def generate(
        self,
        input_image: torch.Tensor,
        class_idx: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap."""
        # Handle input dimensions
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        
        if self.use_cuda:
            input_image = input_image.cuda()
        
        input_image.requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        if isinstance(output, dict):
            output = output['out']
        
        B, C, H, W = output.shape
        
        if class_idx is None:
            class_idx = output.mean(dim=(0, 2, 3)).argmax().item()
        
        # Create target
        target = output[:, class_idx, :, :]
        
        # First backward pass
        self.model.zero_grad()
        target.sum().backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        # Compute weights using Grad-CAM++
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        
        # Global sum of activations
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        
        # Alpha coefficients
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        # Weights
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        
        # CAM
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Resize
        cam = F.interpolate(
            cam,
            size=input_image.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        cam = cam.squeeze().cpu().numpy()
        
        if normalize and cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam


class LayerCAM(GradCAM):
    """
    Layer-CAM for fine-grained spatial attention.
    
    Computes importance at each spatial location separately,
    providing more precise localization.
    """
    
    def generate(
        self,
        input_image: torch.Tensor,
        class_idx: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate Layer-CAM heatmap."""
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        
        if self.use_cuda:
            input_image = input_image.cuda()
        
        input_image.requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        if isinstance(output, dict):
            output = output['out']
        
        if class_idx is None:
            class_idx = output.mean(dim=(0, 2, 3)).argmax().item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[:, class_idx, :, :]
        target.sum().backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        # Layer-CAM: element-wise product with ReLU on gradients
        cam = F.relu(gradients) * activations
        cam = cam.sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Resize
        cam = F.interpolate(
            cam,
            size=input_image.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        cam = cam.squeeze().cpu().numpy()
        
        if normalize and cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam


class SegmentationCAM:
    """
    CAM specifically designed for segmentation models.
    
    Generates per-class activation maps for semantic segmentation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layers: List[str],
        use_cuda: bool = True
    ):
        self.model = model
        self.target_layers = target_layers
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        self.cams = {layer: GradCAM(model, layer, use_cuda) for layer in target_layers}
    
    def generate_multiscale(
        self,
        input_image: torch.Tensor,
        class_idx: int
    ) -> Dict[str, np.ndarray]:
        """Generate CAM from multiple layers for multi-scale attention."""
        results = {}
        for layer_name, cam in self.cams.items():
            results[layer_name] = cam.generate(input_image, class_idx)
        return results
    
    def generate_fused(
        self,
        input_image: torch.Tensor,
        class_idx: int,
        fusion: str = 'mean'
    ) -> np.ndarray:
        """Generate fused CAM from multiple layers."""
        cams = self.generate_multiscale(input_image, class_idx)
        
        # Stack and fuse
        cam_stack = np.stack(list(cams.values()), axis=0)
        
        if fusion == 'mean':
            fused = cam_stack.mean(axis=0)
        elif fusion == 'max':
            fused = cam_stack.max(axis=0)
        elif fusion == 'weighted':
            # Weight by layer depth (deeper = more weight)
            weights = np.arange(1, len(cams) + 1)
            weights = weights / weights.sum()
            fused = (cam_stack * weights[:, None, None]).sum(axis=0)
        else:
            fused = cam_stack.mean(axis=0)
        
        return fused
    
    def remove_hooks(self):
        """Remove all hooks."""
        for cam in self.cams.values():
            cam.remove_hooks()


if __name__ == '__main__':
    # Example usage
    print("Grad-CAM module loaded successfully")
    print("Available classes: GradCAM, GradCAMPlusPlus, LayerCAM, SegmentationCAM")
