"""
Model inference and prediction utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path


class Predictor:
    """
    Model predictor for cardiac segmentation.
    
    Handles model loading, preprocessing, inference, and postprocessing.
    
    Args:
        model: PyTorch model or path to checkpoint
        device: Device for inference
        tta: Use test-time augmentation
    """
    
    def __init__(
        self,
        model: Union[nn.Module, str],
        device: str = 'cuda',
        tta: bool = False,
        postprocess: bool = True
    ):
        self.device = torch.device(device)
        self.tta = tta
        self.postprocess = postprocess
        
        # Load model
        if isinstance(model, str):
            self.model = self._load_model(model)
        else:
            self.model = model
        
        self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Assume model architecture is known
        # In practice, store model class in checkpoint
        from models import get_model
        
        config = checkpoint.get('config', {})
        model_name = config.get('model', 'mamba_unet_v1')
        
        model = get_model(model_name, in_channels=1, num_classes=4)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def predict(
        self,
        image: Union[torch.Tensor, np.ndarray],
        return_probs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict segmentation for single image.
        
        Args:
            image: Input image (H, W), (C, H, W), or (B, C, H, W)
            return_probs: Return probability maps
            
        Returns:
            Segmentation mask (H, W) or (B, H, W)
            If return_probs: (mask, probabilities)
        """
        # Prepare input
        image = self._prepare_input(image)
        
        with torch.no_grad():
            if self.tta:
                probs = self._tta_predict(image)
            else:
                outputs = self.model(image)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                probs = F.softmax(outputs, dim=1)
        
        # Get class predictions
        preds = probs.argmax(dim=1)
        
        # Postprocess
        if self.postprocess:
            preds = self._postprocess(preds)
        
        if return_probs:
            return preds, probs
        return preds
    
    def _prepare_input(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Prepare input tensor."""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        elif image.dim() == 3:
            image = image.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        
        return image.to(self.device)
    
    def _tta_predict(self, image: torch.Tensor) -> torch.Tensor:
        """Test-time augmentation prediction."""
        probs_list = []
        
        # Original
        outputs = self.model(image)
        if isinstance(outputs, dict):
            outputs = outputs['out']
        probs_list.append(F.softmax(outputs, dim=1))
        
        # Horizontal flip
        flipped = torch.flip(image, dims=[3])
        outputs = self.model(flipped)
        if isinstance(outputs, dict):
            outputs = outputs['out']
        probs_list.append(torch.flip(F.softmax(outputs, dim=1), dims=[3]))
        
        # Vertical flip
        flipped = torch.flip(image, dims=[2])
        outputs = self.model(flipped)
        if isinstance(outputs, dict):
            outputs = outputs['out']
        probs_list.append(torch.flip(F.softmax(outputs, dim=1), dims=[2]))
        
        # Average
        probs = torch.stack(probs_list, dim=0).mean(dim=0)
        
        return probs
    
    def _postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Apply postprocessing."""
        from .postprocessing import remove_small_components
        
        # Remove small components for each class
        preds_np = preds.cpu().numpy()
        processed = np.zeros_like(preds_np)
        
        for b in range(preds_np.shape[0]):
            processed[b] = remove_small_components(preds_np[b], min_size=100)
        
        return torch.from_numpy(processed).to(preds.device)
    
    def predict_with_uncertainty(
        self,
        image: Union[torch.Tensor, np.ndarray],
        n_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation using MC Dropout.
        
        Args:
            image: Input image
            n_samples: Number of forward passes
            
        Returns:
            (prediction, mean_probs, uncertainty_map)
        """
        image = self._prepare_input(image)
        
        # Enable dropout
        self._enable_dropout()
        
        probs_samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.model(image)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                probs_samples.append(F.softmax(outputs, dim=1))
        
        # Disable dropout
        self.model.eval()
        
        # Stack and compute statistics
        probs_stack = torch.stack(probs_samples, dim=0)
        mean_probs = probs_stack.mean(dim=0)
        uncertainty = probs_stack.var(dim=0).sum(dim=1)  # Sum variance across classes
        
        prediction = mean_probs.argmax(dim=1)
        
        return prediction, mean_probs, uncertainty
    
    def _enable_dropout(self):
        """Enable dropout layers for MC Dropout."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()


class BatchPredictor:
    """
    Batch inference for multiple images.
    """
    
    def __init__(
        self,
        model: Union[nn.Module, str],
        batch_size: int = 8,
        device: str = 'cuda',
        num_workers: int = 4
    ):
        self.predictor = Predictor(model, device)
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def predict_dataset(
        self,
        dataset,
        return_metrics: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Predict on entire dataset.
        
        Args:
            dataset: PyTorch Dataset
            return_metrics: Compute metrics during prediction
            
        Returns:
            Dictionary with predictions and optionally metrics
        """
        from torch.utils.data import DataLoader
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        all_preds = []
        all_probs = []
        
        for images, _ in loader:
            preds, probs = self.predictor.predict(images, return_probs=True)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        return {
            'predictions': np.concatenate(all_preds, axis=0),
            'probabilities': np.concatenate(all_probs, axis=0)
        }


if __name__ == '__main__':
    print("Predictor module loaded successfully")
