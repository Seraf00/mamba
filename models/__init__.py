"""
Models module for cardiac segmentation.

Provides unified access to all base models and Mamba-enhanced variants.
"""

# Base models
from .base.unet_v1 import UNetV1 as UNet
from .base.unet_v2 import UNetV2
from .base.unet_resnet import UNetResNet
from .base.deeplab_v3 import DeepLabV3
from .base.nnunet import nnUNet
from .base.gudu import GUDU
from .base.swin_unet import SwinUNet
from .base.transunet import TransUNet
from .base.fpn import FPNUNet as FPN, FPNUNet as FPNSegmentation

# Mamba-enhanced models
from .mamba_enhanced.mamba_unet_v1 import MambaUNetV1 as MambaUNet
from .mamba_enhanced.mamba_unet_v2 import MambaUNetV2
from .mamba_enhanced.mamba_unet_resnet import MambaUNetResNet
from .mamba_enhanced.mamba_deeplab import MambaDeepLab
from .mamba_enhanced.mamba_nnunet import MambaNNUNet
from .mamba_enhanced.mamba_gudu import MambaGUDU
from .mamba_enhanced.mamba_swin_unet import MambaSwinUNet
from .mamba_enhanced.mamba_transunet import MambaTransUNet
from .mamba_enhanced.mamba_fpn import MambaFPN
from .mamba_enhanced.pure_mamba_unet import PureMambaUNet

# Modules
from .modules import (
    MambaBlock,
    Mamba2Block,
    VMMambaBlock,
    create_mamba_block,
    MambaEncoder,
    MambaDecoder,
    MambaBottleneck,
    MambaSkipConnection,
)

# Model registry
MODEL_REGISTRY = {
    # Base models
    'unet': UNet,
    'unet_v1': UNet,
    'unet_v2': UNetV2,
    'unet_resnet': UNetResNet,
    'deeplab_v3': DeepLabV3,
    'nnunet': nnUNet,
    'gudu': GUDU,
    'swin_unet': SwinUNet,
    'transunet': TransUNet,
    'fpn': FPNSegmentation,
    
    # Mamba-enhanced models
    'mamba_unet': MambaUNet,
    'mamba_unet_v1': MambaUNet,
    'mamba_unet_v2': MambaUNetV2,
    'mamba_unet_resnet': MambaUNetResNet,
    'mamba_deeplab': MambaDeepLab,
    'mamba_nnunet': MambaNNUNet,
    'mamba_gudu': MambaGUDU,
    'mamba_swin_unet': MambaSwinUNet,
    'mamba_transunet': MambaTransUNet,
    'mamba_fpn': MambaFPN,
    'pure_mamba_unet': PureMambaUNet,
}


def get_model(
    name: str,
    in_channels: int = 1,
    num_classes: int = 4,
    **kwargs
):
    """
    Factory function to create models.
    
    Args:
        name: Model name (see MODEL_REGISTRY)
        in_channels: Number of input channels
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
            - mamba_type: 'mamba', 'mamba2', or 'vmamba'
            - pretrained: Use pretrained backbone
            - etc.
            
    Returns:
        Instantiated model
        
    Example:
        >>> model = get_model('mamba_unet_v1', in_channels=1, num_classes=4)
        >>> model = get_model('mamba_unet_v2', mamba_type='vmamba')
    """
    name = name.lower()
    
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: {name}. Available: {available}"
        )
    
    model_class = MODEL_REGISTRY[name]
    
    # Create model with appropriate arguments
    try:
        model = model_class(
            in_channels=in_channels,
            num_classes=num_classes,
            **kwargs
        )
    except TypeError:
        # Some models may have different arg names
        model = model_class(
            in_chans=in_channels,
            num_classes=num_classes,
            **kwargs
        )
    
    return model


def list_models():
    """List all available models."""
    print("Available models:")
    print("-" * 40)
    print("\nBase models:")
    base_models = [k for k in MODEL_REGISTRY.keys() if 'mamba' not in k]
    for m in base_models:
        print(f"  - {m}")
    
    print("\nMamba-enhanced models:")
    mamba_models = [k for k in MODEL_REGISTRY.keys() if 'mamba' in k]
    for m in mamba_models:
        print(f"  - {m}")


__all__ = [
    # Factory
    'get_model',
    'list_models',
    'MODEL_REGISTRY',
    
    # Base models
    'UNet',
    'UNetV2',
    'UNetResNet',
    'DeepLabV3',
    'nnUNet',
    'GUDU',
    'SwinUNet',
    'TransUNet',
    'FPN',
    'FPNSegmentation',
    
    # Mamba-enhanced models
    'MambaUNet',
    'MambaUNetV2',
    'MambaUNetResNet',
    'MambaDeepLab',
    'MambaUNetNN',
    'MambaGUDU',
    'MambaSwinUNet',
    'MambaTransUNet',
    'MambaFPN',
    'PureMambaUNet',
    
    # Modules
    'MambaBlock',
    'Mamba2Block',
    'VMMambaBlock',
    'create_mamba_block',
    'MambaEncoder',
    'MambaDecoder',
    'MambaBottleneck',
    'MambaSkipConnection',
]
