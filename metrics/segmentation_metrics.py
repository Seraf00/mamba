"""
Segmentation metrics for cardiac image segmentation.

Includes Dice, IoU, Hausdorff distance, and surface distance metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff


class DiceScore(nn.Module):
    """
    Dice Similarity Coefficient (DSC).
    
    DSC = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        smooth: Smoothing factor to avoid division by zero
        reduction: 'none', 'mean', or 'sum'
        ignore_index: Class index to ignore (usually background)
    """
    
    def __init__(
        self,
        smooth: float = 1e-6,
        reduction: str = 'mean',
        ignore_index: Optional[int] = None
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_per_class: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute Dice score.
        
        Args:
            pred: Predictions (B, C, H, W) logits or (B, H, W) class indices
            target: Ground truth (B, H, W) class indices
            return_per_class: Return per-class scores
            
        Returns:
            Dice score (scalar or per-class dict)
        """
        # Convert logits to class predictions if needed
        if pred.dim() == 4:
            num_classes = pred.shape[1]
            pred_probs = F.softmax(pred, dim=1)
        else:
            num_classes = target.max().item() + 1
            pred_probs = F.one_hot(pred.long(), num_classes).permute(0, 3, 1, 2).float()
        
        # One-hot encode target
        target_onehot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
        
        dice_scores = []
        
        for c in range(num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue
            
            pred_c = pred_probs[:, c]
            target_c = target_onehot[:, c]
            
            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        dice_scores = torch.stack(dice_scores, dim=1)
        
        if return_per_class:
            return {
                'per_class': dice_scores,
                'mean': dice_scores.mean()
            }
        
        if self.reduction == 'none':
            return dice_scores
        elif self.reduction == 'mean':
            return dice_scores.mean()
        else:  # sum
            return dice_scores.sum()


class IoUScore(nn.Module):
    """
    Intersection over Union (Jaccard Index).
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        smooth: Smoothing factor
        reduction: 'none', 'mean', or 'sum'
        ignore_index: Class to ignore
    """
    
    def __init__(
        self,
        smooth: float = 1e-6,
        reduction: str = 'mean',
        ignore_index: Optional[int] = None
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_per_class: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute IoU score."""
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        
        num_classes = max(pred.max().item(), target.max().item()) + 1
        
        iou_scores = []
        
        for c in range(num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue
            
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2)) - intersection
            
            iou = (intersection + self.smooth) / (union + self.smooth)
            iou_scores.append(iou)
        
        iou_scores = torch.stack(iou_scores, dim=1)
        
        if return_per_class:
            return {
                'per_class': iou_scores,
                'mean': iou_scores.mean()
            }
        
        if self.reduction == 'none':
            return iou_scores
        elif self.reduction == 'mean':
            return iou_scores.mean()
        else:
            return iou_scores.sum()


class HausdorffDistance:
    """
    Hausdorff Distance between segmentation boundaries.
    
    Measures the maximum distance between boundary points.
    HD = max(h(A, B), h(B, A))
    where h(A, B) = max_{a ∈ A} min_{b ∈ B} ||a - b||
    
    Args:
        percentile: Use percentile HD (e.g., 95) instead of max
        spacing: Pixel spacing in mm (H, W)
    """
    
    def __init__(
        self,
        percentile: Optional[float] = 95,
        spacing: Tuple[float, float] = (1.0, 1.0)
    ):
        self.percentile = percentile
        self.spacing = spacing
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute Hausdorff distance.
        
        Args:
            pred: Predictions (B, H, W) or (H, W)
            target: Ground truth (B, H, W) or (H, W)
            class_idx: Specific class to compute (None = all)
            
        Returns:
            Dictionary with HD values per class
        """
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        if pred.ndim == 2:
            pred = pred[np.newaxis, ...]
            target = target[np.newaxis, ...]
        
        num_classes = max(pred.max(), target.max()) + 1
        classes = [class_idx] if class_idx is not None else range(1, num_classes)
        
        results = {}
        
        for c in classes:
            hd_values = []
            
            for b in range(pred.shape[0]):
                pred_mask = (pred[b] == c)
                target_mask = (target[b] == c)
                
                if not pred_mask.any() or not target_mask.any():
                    continue
                
                hd = self._compute_hd(pred_mask, target_mask)
                hd_values.append(hd)
            
            if hd_values:
                results[f'class_{c}_hd'] = np.mean(hd_values)
                results[f'class_{c}_hd_std'] = np.std(hd_values)
        
        if results:
            all_hd = [v for k, v in results.items() if 'std' not in k]
            results['mean_hd'] = np.mean(all_hd)
        
        return results
    
    def _compute_hd(
        self,
        pred_mask: np.ndarray,
        target_mask: np.ndarray
    ) -> float:
        """Compute HD between two binary masks."""
        # Get boundary points
        pred_boundary = self._get_boundary(pred_mask)
        target_boundary = self._get_boundary(target_mask)
        
        if len(pred_boundary) == 0 or len(target_boundary) == 0:
            return float('inf')
        
        # Apply spacing
        pred_boundary = pred_boundary * np.array(self.spacing)
        target_boundary = target_boundary * np.array(self.spacing)
        
        # Compute directed Hausdorff distances
        hd1 = directed_hausdorff(pred_boundary, target_boundary)[0]
        hd2 = directed_hausdorff(target_boundary, pred_boundary)[0]
        
        if self.percentile is not None:
            # Compute percentile HD
            from scipy.spatial.distance import cdist
            distances = cdist(pred_boundary, target_boundary)
            hd1 = np.percentile(distances.min(axis=1), self.percentile)
            hd2 = np.percentile(distances.min(axis=0), self.percentile)
        
        return max(hd1, hd2)
    
    def _get_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Extract boundary points from binary mask."""
        from scipy.ndimage import binary_erosion
        
        eroded = binary_erosion(mask)
        boundary = mask & ~eroded
        
        return np.array(np.where(boundary)).T


class SurfaceDistance:
    """
    Average Surface Distance (ASD) and related metrics.
    
    ASD = (1/|S_A| + 1/|S_B|) * (Σ d(a, S_B) + Σ d(b, S_A))
    
    Args:
        spacing: Pixel spacing in mm
    """
    
    def __init__(self, spacing: Tuple[float, float] = (1.0, 1.0)):
        self.spacing = spacing
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> Dict[str, float]:
        """Compute surface distance metrics."""
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        if pred.ndim == 2:
            pred = pred[np.newaxis, ...]
            target = target[np.newaxis, ...]
        
        num_classes = max(pred.max(), target.max()) + 1
        classes = [class_idx] if class_idx is not None else range(1, num_classes)
        
        results = {}
        
        for c in classes:
            asd_values = []
            
            for b in range(pred.shape[0]):
                pred_mask = (pred[b] == c)
                target_mask = (target[b] == c)
                
                if not pred_mask.any() or not target_mask.any():
                    continue
                
                asd = self._compute_asd(pred_mask, target_mask)
                asd_values.append(asd)
            
            if asd_values:
                results[f'class_{c}_asd'] = np.mean(asd_values)
        
        if results:
            all_asd = list(results.values())
            results['mean_asd'] = np.mean(all_asd)
        
        return results
    
    def _compute_asd(
        self,
        pred_mask: np.ndarray,
        target_mask: np.ndarray
    ) -> float:
        """Compute ASD between two masks."""
        # Distance transforms
        pred_dist = distance_transform_edt(~pred_mask, sampling=self.spacing)
        target_dist = distance_transform_edt(~target_mask, sampling=self.spacing)
        
        # Get boundaries
        pred_boundary = self._get_boundary(pred_mask)
        target_boundary = self._get_boundary(target_mask)
        
        # Surface distances
        pred_to_target = target_dist[pred_boundary]
        target_to_pred = pred_dist[target_boundary]
        
        if len(pred_to_target) == 0 or len(target_to_pred) == 0:
            return float('inf')
        
        asd = (pred_to_target.mean() + target_to_pred.mean()) / 2
        return asd
    
    def _get_boundary(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get boundary as indices."""
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask)
        boundary = mask & ~eroded
        return boundary


class SegmentationMetrics:
    """
    Comprehensive segmentation metrics calculator.
    
    Combines all metrics for easy evaluation.
    """
    
    CLASS_NAMES = {
        0: 'Background',
        1: 'LV Endocardium',
        2: 'LV Epicardium',
        3: 'Left Atrium'
    }
    
    def __init__(
        self,
        num_classes: int = 4,
        spacing: Tuple[float, float] = (1.0, 1.0),
        ignore_background: bool = True
    ):
        self.num_classes = num_classes
        self.spacing = spacing
        self.ignore_background = ignore_background
        
        self.dice = DiceScore(ignore_index=0 if ignore_background else None)
        self.iou = IoUScore(ignore_index=0 if ignore_background else None)
        self.hausdorff = HausdorffDistance(percentile=95, spacing=spacing)
        self.surface_distance = SurfaceDistance(spacing=spacing)
        
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.dice_scores = []
        self.iou_scores = []
        self.hd_scores = []
        self.asd_scores = []
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ):
        """
        Update metrics with new predictions.
        
        Args:
            pred: Predictions (B, C, H, W) or (B, H, W)
            target: Ground truth (B, H, W)
        """
        # Dice and IoU
        dice = self.dice(pred, target, return_per_class=True)
        iou = self.iou(pred, target, return_per_class=True)
        
        self.dice_scores.append(dice['per_class'])
        self.iou_scores.append(iou['per_class'])
        
        # Hausdorff and ASD (more expensive)
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        
        hd = self.hausdorff(pred, target)
        asd = self.surface_distance(pred, target)
        
        self.hd_scores.append(hd)
        self.asd_scores.append(asd)
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        results = {}
        
        # Dice
        if self.dice_scores:
            dice_tensor = torch.cat(self.dice_scores, dim=0)
            results['dice_mean'] = dice_tensor.mean().item()
            
            for c in range(dice_tensor.shape[1]):
                class_name = self.CLASS_NAMES.get(c + 1, f'Class_{c + 1}')
                results[f'dice_{class_name.lower().replace(" ", "_")}'] = dice_tensor[:, c].mean().item()
        
        # IoU
        if self.iou_scores:
            iou_tensor = torch.cat(self.iou_scores, dim=0)
            results['iou_mean'] = iou_tensor.mean().item()
            
            for c in range(iou_tensor.shape[1]):
                class_name = self.CLASS_NAMES.get(c + 1, f'Class_{c + 1}')
                results[f'iou_{class_name.lower().replace(" ", "_")}'] = iou_tensor[:, c].mean().item()
        
        # HD and ASD
        if self.hd_scores:
            hd_values = [s.get('mean_hd', 0) for s in self.hd_scores if 'mean_hd' in s]
            if hd_values:
                results['hd95_mean'] = np.mean(hd_values)
        
        if self.asd_scores:
            asd_values = [s.get('mean_asd', 0) for s in self.asd_scores if 'mean_asd' in s]
            if asd_values:
                results['asd_mean'] = np.mean(asd_values)
        
        return results
    
    def __str__(self) -> str:
        """String representation of metrics."""
        metrics = self.compute()
        lines = ['Segmentation Metrics:', '-' * 40]
        for key, value in metrics.items():
            lines.append(f'{key}: {value:.4f}')
        return '\n'.join(lines)


if __name__ == '__main__':
    # Test metrics
    pred = torch.randint(0, 4, (2, 256, 256))
    target = torch.randint(0, 4, (2, 256, 256))
    
    metrics = SegmentationMetrics()
    metrics.update(pred, target)
    print(metrics)
