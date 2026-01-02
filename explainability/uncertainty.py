"""
Uncertainty estimation for segmentation predictions.

Provides methods to quantify prediction uncertainty, which is
critical for clinical applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Callable
from copy import deepcopy


class MCDropout:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Enables dropout during inference and runs multiple forward
    passes to estimate uncertainty.
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 20,
        dropout_rate: Optional[float] = None
    ):
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        
        # Store original dropout states
        self.original_dropout_training = {}
        
    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                self.original_dropout_training[name] = module.training
                module.train()
                
                if self.dropout_rate is not None:
                    module.p = self.dropout_rate
    
    def _restore_dropout(self):
        """Restore original dropout states."""
        for name, module in self.model.named_modules():
            if name in self.original_dropout_training:
                if not self.original_dropout_training[name]:
                    module.eval()
    
    def predict_with_uncertainty(
        self,
        input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make prediction with uncertainty estimation.
        
        Args:
            input_tensor: Input image (B, C, H, W)
            
        Returns:
            mean_prediction: Mean prediction across samples
            uncertainty_map: Uncertainty (entropy or variance)
            all_predictions: All MC samples
        """
        self._enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                output = self.model(input_tensor)
                
                if isinstance(output, dict):
                    output = output['out']
                
                # Softmax for probabilities
                probs = F.softmax(output, dim=1)
                predictions.append(probs)
        
        self._restore_dropout()
        
        # Stack predictions: (n_samples, B, C, H, W)
        all_predictions = torch.stack(predictions, dim=0)
        
        # Mean prediction
        mean_prediction = all_predictions.mean(dim=0)
        
        # Uncertainty: predictive entropy
        entropy = self._compute_entropy(mean_prediction)
        
        # Also compute variance
        variance = all_predictions.var(dim=0).mean(dim=1)  # Variance across classes
        
        return mean_prediction, entropy, all_predictions
    
    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute predictive entropy."""
        # probs: (B, C, H, W)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=1)  # (B, H, W)
        return entropy
    
    def compute_mutual_information(
        self,
        all_predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mutual information (epistemic uncertainty).
        
        MI = H[y|x] - E[H[y|x,w]]
        """
        # all_predictions: (n_samples, B, C, H, W)
        
        # Predictive entropy (aleatoric + epistemic)
        mean_pred = all_predictions.mean(dim=0)
        predictive_entropy = self._compute_entropy(mean_pred)
        
        # Expected entropy (aleatoric only)
        sample_entropies = []
        for sample in all_predictions:
            sample_entropies.append(self._compute_entropy(sample))
        expected_entropy = torch.stack(sample_entropies, dim=0).mean(dim=0)
        
        # Mutual information (epistemic uncertainty)
        mutual_info = predictive_entropy - expected_entropy
        
        return mutual_info


class EnsembleUncertainty:
    """
    Ensemble-based uncertainty estimation.
    
    Uses multiple trained models to estimate uncertainty.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        device: str = 'cuda'
    ):
        self.models = models
        self.device = device
        
        for model in self.models:
            model.to(device)
            model.eval()
    
    def predict_with_uncertainty(
        self,
        input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Make prediction using ensemble.
        
        Returns:
            mean_prediction: Ensemble mean
            uncertainty: Ensemble disagreement (variance/entropy)
            all_predictions: Individual model predictions
        """
        input_tensor = input_tensor.to(self.device)
        
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                output = model(input_tensor)
                
                if isinstance(output, dict):
                    output = output['out']
                
                probs = F.softmax(output, dim=1)
                predictions.append(probs)
        
        # Stack predictions
        all_predictions = torch.stack(predictions, dim=0)
        
        # Mean prediction
        mean_prediction = all_predictions.mean(dim=0)
        
        # Uncertainty: variance across ensemble
        variance = all_predictions.var(dim=0).mean(dim=1)
        
        return mean_prediction, variance, predictions
    
    def compute_disagreement(
        self,
        predictions: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute pairwise disagreement between models."""
        n_models = len(predictions)
        disagreements = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pred_i = predictions[i].argmax(dim=1)
                pred_j = predictions[j].argmax(dim=1)
                disagreement = (pred_i != pred_j).float()
                disagreements.append(disagreement)
        
        # Average disagreement
        avg_disagreement = torch.stack(disagreements, dim=0).mean(dim=0)
        
        return avg_disagreement


class UncertaintyEstimator:
    """
    Unified uncertainty estimation interface.
    
    Supports multiple uncertainty estimation methods.
    """
    
    METHODS = ['mc_dropout', 'ensemble', 'deep_ensemble', 'evidential']
    
    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        method: str = 'mc_dropout',
        n_samples: int = 20,
        **kwargs
    ):
        self.method = method
        self.n_samples = n_samples
        
        if method == 'mc_dropout':
            self.estimator = MCDropout(model, n_samples, **kwargs)
        elif method == 'ensemble':
            if isinstance(model, list):
                self.estimator = EnsembleUncertainty(model)
            else:
                raise ValueError("Ensemble method requires list of models")
        else:
            raise ValueError(f"Unknown method: {method}. Available: {self.METHODS}")
    
    def predict_with_uncertainty(
        self,
        input_tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Make prediction with uncertainty.
        
        Returns:
            Dictionary with 'prediction', 'uncertainty', 'samples'
        """
        mean_pred, uncertainty, samples = self.estimator.predict_with_uncertainty(input_tensor)
        
        return {
            'prediction': mean_pred,
            'uncertainty': uncertainty,
            'samples': samples
        }
    
    def get_confidence_mask(
        self,
        uncertainty: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """Get mask of confident vs uncertain regions."""
        # Normalize uncertainty to [0, 1]
        norm_uncertainty = (uncertainty - uncertainty.min()) / \
                          (uncertainty.max() - uncertainty.min() + 1e-8)
        
        confident_mask = norm_uncertainty < threshold
        
        return confident_mask
    
    def get_uncertainty_statistics(
        self,
        uncertainty: torch.Tensor,
        segmentation: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute uncertainty statistics."""
        flat_uncertainty = uncertainty.cpu().numpy().flatten()
        
        stats = {
            'mean': float(np.mean(flat_uncertainty)),
            'std': float(np.std(flat_uncertainty)),
            'max': float(np.max(flat_uncertainty)),
            'min': float(np.min(flat_uncertainty)),
            'median': float(np.median(flat_uncertainty))
        }
        
        # Per-class uncertainty if segmentation provided
        if segmentation is not None:
            seg = segmentation.cpu().numpy()
            unc = uncertainty.cpu().numpy()
            
            num_classes = int(seg.max()) + 1
            for c in range(num_classes):
                mask = seg == c
                if mask.sum() > 0:
                    stats[f'class_{c}_mean'] = float(unc[mask].mean())
        
        return stats


class BoundaryUncertainty:
    """
    Specialized uncertainty analysis for segmentation boundaries.
    
    Boundaries typically have higher uncertainty, which this
    class helps analyze and visualize.
    """
    
    @staticmethod
    def extract_boundaries(
        segmentation: torch.Tensor,
        width: int = 3
    ) -> torch.Tensor:
        """Extract boundary regions from segmentation."""
        # segmentation: (B, H, W) or (B, C, H, W)
        if segmentation.dim() == 4:
            segmentation = segmentation.argmax(dim=1)
        
        # Use Sobel-like edge detection
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        if segmentation.is_cuda:
            kernel = kernel.cuda()
        
        # Convert to float for convolution
        seg_float = segmentation.unsqueeze(1).float()
        
        edges = F.conv2d(seg_float, kernel, padding=1)
        boundaries = (edges.abs() > 0).float()
        
        # Dilate boundaries
        if width > 1:
            dilate_kernel = torch.ones(1, 1, width, width, device=boundaries.device)
            boundaries = F.conv2d(boundaries, dilate_kernel, padding=width // 2)
            boundaries = (boundaries > 0).float()
        
        return boundaries.squeeze(1)
    
    @staticmethod
    def analyze_boundary_uncertainty(
        uncertainty: torch.Tensor,
        segmentation: torch.Tensor
    ) -> Dict[str, float]:
        """Compare uncertainty at boundaries vs interior."""
        boundaries = BoundaryUncertainty.extract_boundaries(segmentation)
        
        unc = uncertainty.cpu().numpy()
        bound = boundaries.cpu().numpy().astype(bool)
        
        boundary_uncertainty = unc[bound].mean() if bound.sum() > 0 else 0
        interior_uncertainty = unc[~bound].mean() if (~bound).sum() > 0 else 0
        
        return {
            'boundary_mean': float(boundary_uncertainty),
            'interior_mean': float(interior_uncertainty),
            'boundary_ratio': float(boundary_uncertainty / (interior_uncertainty + 1e-8)),
            'boundary_fraction': float(bound.sum() / bound.size)
        }


class CalibrationAnalysis:
    """
    Analyze model calibration (reliability of confidence estimates).
    """
    
    @staticmethod
    def compute_ece(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 15
    ) -> float:
        """
        Compute Expected Calibration Error.
        
        Args:
            predictions: Softmax predictions (B, C, H, W)
            targets: Ground truth labels (B, H, W)
            n_bins: Number of calibration bins
            
        Returns:
            ECE value
        """
        # Get confidence and predictions
        confidence, predicted = predictions.max(dim=1)
        
        conf = confidence.cpu().numpy().flatten()
        pred = predicted.cpu().numpy().flatten()
        true = targets.cpu().numpy().flatten()
        
        correct = (pred == true).astype(float)
        
        # Compute ECE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            in_bin = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
            
            if in_bin.sum() > 0:
                avg_confidence = conf[in_bin].mean()
                avg_accuracy = correct[in_bin].mean()
                
                ece += in_bin.sum() * np.abs(avg_accuracy - avg_confidence)
        
        ece /= len(conf)
        
        return float(ece)
    
    @staticmethod
    def reliability_diagram(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute data for reliability diagram.
        
        Returns:
            bin_centers, accuracies, bin_counts
        """
        confidence, predicted = predictions.max(dim=1)
        
        conf = confidence.cpu().numpy().flatten()
        pred = predicted.cpu().numpy().flatten()
        true = targets.cpu().numpy().flatten()
        
        correct = (pred == true).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        accuracies = []
        counts = []
        
        for i in range(n_bins):
            in_bin = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
            
            if in_bin.sum() > 0:
                accuracies.append(correct[in_bin].mean())
                counts.append(in_bin.sum())
            else:
                accuracies.append(0)
                counts.append(0)
        
        return bin_centers, np.array(accuracies), np.array(counts)


if __name__ == '__main__':
    print("Uncertainty estimation module loaded successfully")
    print("Available classes: MCDropout, EnsembleUncertainty, UncertaintyEstimator")
    print("Additional: BoundaryUncertainty, CalibrationAnalysis")
