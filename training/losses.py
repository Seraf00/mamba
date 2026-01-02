"""
Loss functions for cardiac segmentation.

Includes Dice, Focal, Boundary, Tversky, and combined losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    L_dice = 1 - (2 * |A ∩ B| + ε) / (|A| + |B| + ε)
    
    Args:
        smooth: Smoothing factor
        reduction: 'none', 'mean', 'sum'
        ignore_index: Class to ignore
        softmax: Apply softmax to predictions
    """
    
    def __init__(
        self,
        smooth: float = 1e-6,
        reduction: str = 'mean',
        ignore_index: Optional[int] = None,
        softmax: bool = True
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.softmax = softmax
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W)
            target: Ground truth (B, H, W) class indices
            
        Returns:
            Dice loss
        """
        num_classes = pred.shape[1]
        
        # Apply softmax
        if self.softmax:
            pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_onehot = F.one_hot(target.long(), num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        
        # Compute Dice per class
        dice_losses = []
        
        for c in range(num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue
            
            pred_c = pred[:, c]
            target_c = target_onehot[:, c]
            
            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_losses.append(1 - dice)
        
        dice_losses = torch.stack(dice_losses, dim=1)
        
        if self.reduction == 'none':
            return dice_losses
        elif self.reduction == 'mean':
            return dice_losses.mean()
        else:
            return dice_losses.sum()


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    
    FL(p) = -α(1-p)^γ log(p)
    
    Args:
        alpha: Class weights (C,) or scalar
        gamma: Focusing parameter
        reduction: 'none', 'mean', 'sum'
        ignore_index: Class to ignore
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W)
            target: Ground truth (B, H, W)
            
        Returns:
            Focal loss
        """
        # Cross entropy
        ce_loss = F.cross_entropy(
            pred, target.long(),
            weight=self.alpha,
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        # Focal term
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class TverskyLoss(nn.Module):
    """
    Tversky Loss for handling class imbalance.
    
    Generalizes Dice loss with α and β controlling FP/FN penalty.
    
    TL = 1 - (TP + ε) / (TP + α*FP + β*FN + ε)
    
    Args:
        alpha: False positive penalty (lower = less penalty)
        beta: False negative penalty (higher = more penalty)
        smooth: Smoothing factor
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        num_classes = pred.shape[1]
        pred = F.softmax(pred, dim=1)
        
        target_onehot = F.one_hot(target.long(), num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        
        tversky_losses = []
        
        for c in range(1, num_classes):  # Skip background
            pred_c = pred[:, c]
            target_c = target_onehot[:, c]
            
            tp = (pred_c * target_c).sum(dim=(1, 2))
            fp = (pred_c * (1 - target_c)).sum(dim=(1, 2))
            fn = ((1 - pred_c) * target_c).sum(dim=(1, 2))
            
            tversky = (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth
            )
            tversky_losses.append(1 - tversky)
        
        tversky_losses = torch.stack(tversky_losses, dim=1)
        
        if self.reduction == 'mean':
            return tversky_losses.mean()
        return tversky_losses.sum()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss using distance transform.
    
    Penalizes predictions based on distance to ground truth boundary.
    
    Args:
        theta0: Threshold for boundary region
    """
    
    def __init__(self, theta0: float = 3):
        super().__init__()
        self.theta0 = theta0
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dist_map: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W)
            target: Ground truth (B, H, W)
            dist_map: Pre-computed distance map (optional)
            
        Returns:
            Boundary loss
        """
        pred_probs = F.softmax(pred, dim=1)
        
        if dist_map is None:
            dist_map = self._compute_distance_map(target)
        
        dist_map = dist_map.to(pred.device)
        
        # Boundary loss as dot product with distance map
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target.long(), num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        
        boundary_loss = (pred_probs * dist_map).sum() / pred.numel()
        
        return boundary_loss
    
    def _compute_distance_map(self, target: torch.Tensor) -> torch.Tensor:
        """Compute signed distance map from target."""
        from scipy.ndimage import distance_transform_edt
        
        target_np = target.cpu().numpy()
        dist_maps = []
        
        for b in range(target_np.shape[0]):
            mask = target_np[b]
            dist = np.zeros_like(mask, dtype=np.float32)
            
            for c in range(1, mask.max() + 1):
                class_mask = (mask == c)
                if class_mask.any():
                    # Distance from boundary
                    pos_dist = distance_transform_edt(class_mask)
                    neg_dist = distance_transform_edt(~class_mask)
                    signed_dist = neg_dist - pos_dist
                    dist = np.maximum(dist, np.abs(signed_dist))
            
            dist_maps.append(dist)
        
        dist_map = np.stack(dist_maps, axis=0)
        return torch.from_numpy(dist_map).float()


class CombinedLoss(nn.Module):
    """
    Combined loss function.
    
    Combines multiple losses with weights.
    
    Args:
        losses: List of (loss_fn, weight) tuples
    """
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        focal_weight: float = 0.0,
        boundary_weight: float = 0.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
        self.dice_loss = DiceLoss() if dice_weight > 0 else None
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.focal_loss = FocalLoss(alpha=class_weights) if focal_weight > 0 else None
        self.boundary_loss = BoundaryLoss() if boundary_weight > 0 else None
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss."""
        total_loss = 0.0
        
        # Dice loss
        if self.dice_loss is not None and self.dice_weight > 0:
            total_loss += self.dice_weight * self.dice_loss(pred, target)
        
        # Cross entropy
        if self.ce_weight > 0:
            total_loss += self.ce_weight * self.ce_loss(pred, target.long())
        
        # Focal loss
        if self.focal_loss is not None and self.focal_weight > 0:
            total_loss += self.focal_weight * self.focal_loss(pred, target)
        
        # Boundary loss
        if self.boundary_loss is not None and self.boundary_weight > 0:
            total_loss += self.boundary_weight * self.boundary_loss(pred, target)
        
        return total_loss


class DeepSupervisionLoss(nn.Module):
    """
    Deep supervision loss for multi-scale outputs.
    
    Args:
        base_loss: Base loss function
        weights: Weights for each output level
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights
    
    def forward(
        self,
        outputs: List[torch.Tensor],
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            outputs: List of predictions at different scales
            target: Ground truth (B, H, W)
        """
        if self.weights is None:
            weights = [1.0 / len(outputs)] * len(outputs)
        else:
            weights = self.weights
        
        total_loss = 0.0
        
        for out, weight in zip(outputs, weights):
            # Resize target to match output size
            if out.shape[2:] != target.shape[1:]:
                resized_target = F.interpolate(
                    target.unsqueeze(1).float(),
                    size=out.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
            else:
                resized_target = target
            
            total_loss += weight * self.base_loss(out, resized_target)
        
        return total_loss


def get_loss_function(
    name: str = 'combined',
    num_classes: int = 4,
    class_weights: Optional[List[float]] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function for loss functions.
    
    Args:
        name: Loss function name
        num_classes: Number of classes
        class_weights: Optional class weights
        
    Returns:
        Loss function module
    """
    if class_weights is not None:
        class_weights = torch.tensor(class_weights)
    
    if name == 'dice':
        return DiceLoss(**kwargs)
    elif name == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    elif name == 'focal':
        return FocalLoss(alpha=class_weights, **kwargs)
    elif name == 'tversky':
        return TverskyLoss(**kwargs)
    elif name == 'boundary':
        return BoundaryLoss(**kwargs)
    elif name == 'combined':
        return CombinedLoss(class_weights=class_weights, **kwargs)
    else:
        raise ValueError(f"Unknown loss function: {name}")


if __name__ == '__main__':
    # Test losses
    pred = torch.randn(2, 4, 256, 256)
    target = torch.randint(0, 4, (2, 256, 256))
    
    # Test each loss
    print("Testing loss functions...")
    
    dice = DiceLoss()
    print(f"Dice Loss: {dice(pred, target).item():.4f}")
    
    focal = FocalLoss()
    print(f"Focal Loss: {focal(pred, target).item():.4f}")
    
    tversky = TverskyLoss()
    print(f"Tversky Loss: {tversky(pred, target).item():.4f}")
    
    combined = CombinedLoss()
    print(f"Combined Loss: {combined(pred, target).item():.4f}")
