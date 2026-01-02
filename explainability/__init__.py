"""
Explainability module for Mamba-enhanced cardiac segmentation.

This package provides interpretability tools for medical image segmentation:
- Grad-CAM and variants for visual explanations
- Attention map visualization
- Mamba state space visualization
- Uncertainty estimation
- Clinical report generation
"""

from .gradcam import GradCAM, GradCAMPlusPlus, LayerCAM
from .attention_maps import AttentionVisualizer, AttentionRollout
from .mamba_state_viz import MambaStateVisualizer
from .feature_maps import FeatureExtractor, FeatureVisualizer
from .uncertainty import UncertaintyEstimator, MCDropout, EnsembleUncertainty
from .clinical_report import ClinicalReport, ReportGenerator

__all__ = [
    # Grad-CAM
    'GradCAM',
    'GradCAMPlusPlus', 
    'LayerCAM',
    # Attention
    'AttentionVisualizer',
    'AttentionRollout',
    # Mamba-specific
    'MambaStateVisualizer',
    # Feature maps
    'FeatureExtractor',
    'FeatureVisualizer',
    # Uncertainty
    'UncertaintyEstimator',
    'MCDropout',
    'EnsembleUncertainty',
    # Clinical
    'ClinicalReport',
    'ReportGenerator',
]
