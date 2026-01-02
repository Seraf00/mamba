"""
Mamba-nnUNet - nnUNet Enhanced with Mamba State Space Models

nnUNet architecture with Mamba integration for enhanced
long-range dependency modeling while maintaining efficiency.

Integration points:
- Encoder stages: Mamba after conv blocks
- Bottleneck: Dual-path Mamba
- Skip connections: Mamba-based refinement
- Deep supervision with Mamba
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Literal

import sys
sys.path.append('..')

from models.modules import (
    create_mamba_block,
    MambaBottleneck,
    DualPathMambaBottleneck,
    MambaSkipConnection
)


class ConvBlock(nn.Module):
    """nnUNet-style conv block with instance norm and leaky ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        
        self.lrelu = nn.LeakyReLU(0.01, inplace=True)
        
        if use_residual and (in_channels != out_channels or stride != 1):
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True)
            )
            self.use_residual = True
        elif self.use_residual:
            self.residual = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.lrelu(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        if self.use_residual:
            x = x + self.residual(residual)
        
        return self.lrelu(x)


class MambaEncoderStageNN(nn.Module):
    """nnUNet encoder stage with optional Mamba enhancement."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        first_stride: int = 1,
        use_mamba: bool = False,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                ConvBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride=first_stride if i == 0 else 1
                )
            )
        self.blocks = nn.Sequential(*blocks)
        
        # Optional Mamba enhancement
        self.mamba = None
        if use_mamba:
            self.mamba = create_mamba_block(
                variant=mamba_type,
                dim=out_channels,
                d_state=d_state
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        if self.mamba is not None:
            x = x + self.mamba(x)
        return x


class MambaDecoderStageNN(nn.Module):
    """nnUNet decoder stage with Mamba skip enhancement."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        use_mamba_skip: bool = True,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # Mamba skip connection
        self.mamba_skip = None
        if use_mamba_skip:
            self.mamba_skip = MambaSkipConnection(
                encoder_channels=skip_channels,
                decoder_channels=out_channels,
                mamba_type=mamba_type,
                d_state=d_state
            )
        
        # Conv blocks after fusion
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                ConvBlock(
                    out_channels + skip_channels if i == 0 else out_channels,
                    out_channels
                )
            )
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Mamba skip enhancement
        if self.mamba_skip is not None:
            skip = self.mamba_skip(skip, x)
        
        x = torch.cat([x, skip], dim=1)
        return self.blocks(x)


class MambaNNUNet(nn.Module):
    """
    Mamba-Enhanced nnUNet.
    
    nnUNet with Mamba integration:
    - Instance normalization + LeakyReLU (nnUNet style)
    - Mamba in deeper encoder stages
    - Dual-path Mamba bottleneck
    - Mamba-enhanced skip connections
    - Deep supervision support
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        base_features: Base number of features
        num_stages: Number of encoder/decoder stages
        features_per_stage: Optional custom features per stage
        mamba_type: Type of Mamba ('mamba', 'mamba2', 'vmamba')
        mamba_in_encoder: Add Mamba to encoder (deeper stages)
        mamba_in_skip: Add Mamba to skip connections
        use_dual_bottleneck: Use dual-path Mamba bottleneck
        deep_supervision: Enable deep supervision
        d_state: SSM state dimension
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_features: int = 32,
        num_stages: int = 5,
        features_per_stage: Optional[List[int]] = None,
        mamba_type: Literal['mamba', 'mamba2', 'vmamba'] = 'vmamba',
        mamba_in_encoder: bool = True,
        mamba_in_skip: bool = True,
        use_dual_bottleneck: bool = True,
        deep_supervision: bool = True,
        d_state: int = 16
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.mamba_type = mamba_type
        
        # Feature dimensions
        if features_per_stage is None:
            features_per_stage = [
                min(base_features * (2 ** i), 320)
                for i in range(num_stages)
            ]
        
        self.features_per_stage = features_per_stage
        
        # Encoder
        self.encoder_stages = nn.ModuleList()
        for i in range(num_stages):
            in_ch = in_channels if i == 0 else features_per_stage[i - 1]
            out_ch = features_per_stage[i]
            stride = 1 if i == 0 else 2
            
            # Add Mamba to deeper encoder stages
            use_mamba = mamba_in_encoder and i >= num_stages // 2
            
            self.encoder_stages.append(
                MambaEncoderStageNN(
                    in_ch, out_ch,
                    num_blocks=2,
                    first_stride=stride,
                    use_mamba=use_mamba,
                    mamba_type=mamba_type,
                    d_state=d_state
                )
            )
        
        # Dual-path Mamba bottleneck
        if use_dual_bottleneck:
            self.bottleneck = DualPathMambaBottleneck(
                dim=features_per_stage[-1],
                mamba_types=('mamba', mamba_type),  # Use mamba + specified type
                d_state=d_state
            )
        else:
            self.bottleneck = MambaBottleneck(
                dim=features_per_stage[-1],
                mamba_type=mamba_type,
                d_state=d_state
            )
        
        # Decoder
        self.decoder_stages = nn.ModuleList()
        self.seg_heads = nn.ModuleList()
        
        for i in range(num_stages - 1, 0, -1):
            in_ch = features_per_stage[i]
            skip_ch = features_per_stage[i - 1]
            out_ch = features_per_stage[i - 1]
            
            self.decoder_stages.append(
                MambaDecoderStageNN(
                    in_ch, skip_ch, out_ch,
                    num_blocks=2,
                    use_mamba_skip=mamba_in_skip,
                    mamba_type=mamba_type,
                    d_state=d_state
                )
            )
            
            # Segmentation head for deep supervision
            if deep_supervision:
                self.seg_heads.append(
                    nn.Conv2d(out_ch, num_classes, kernel_size=1)
                )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            If training with deep supervision: Dict with 'out' and 'aux'
            Else: Output tensor (B, num_classes, H, W)
        """
        target_size = x.shape[2:]
        
        # Encoder
        encoder_features = []
        for stage in self.encoder_stages:
            x = stage(x)
            encoder_features.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        skips = encoder_features[:-1][::-1]
        outputs = []
        
        for i, (stage, skip) in enumerate(zip(self.decoder_stages, skips)):
            x = stage(x, skip)
            
            if self.deep_supervision and self.training:
                seg = self.seg_heads[i](x)
                seg = F.interpolate(seg, size=target_size, mode='bilinear', align_corners=True)
                outputs.append(seg)
        
        if self.deep_supervision and self.training:
            return {'out': outputs[-1], 'aux': outputs[:-1]}
        
        # Final output (use last decoder output)
        out = self.seg_heads[-1](x) if self.deep_supervision else nn.Conv2d(
            self.features_per_stage[0], self.num_classes, 1
        ).to(x.device)(x)
        
        if out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=True)
        
        return out
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience functions
def mamba_nnunet_small(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaNNUNet:
    """Small Mamba-nnUNet."""
    return MambaNNUNet(
        in_channels, num_classes,
        base_features=16,
        features_per_stage=[16, 32, 64, 128, 256],
        mamba_type=mamba_type
    )


def mamba_nnunet_base(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaNNUNet:
    """Standard Mamba-nnUNet."""
    return MambaNNUNet(
        in_channels, num_classes,
        base_features=32,
        features_per_stage=[32, 64, 128, 256, 320],
        mamba_type=mamba_type
    )


def mamba_nnunet_large(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaNNUNet:
    """Large Mamba-nnUNet."""
    return MambaNNUNet(
        in_channels, num_classes,
        base_features=32,
        num_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        mamba_type=mamba_type
    )


if __name__ == '__main__':
    # Test the model
    model = MambaNNUNet(
        in_channels=1, num_classes=4,
        mamba_type='vmamba',
        deep_supervision=True
    )
    print(f"Mamba-nnUNet Parameters: {model.count_parameters():,}")
    
    # Test forward pass (training mode)
    model.train()
    x = torch.randn(2, 1, 256, 256)
    outputs = model(x)
    
    if isinstance(outputs, dict):
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {outputs['out'].shape}")
        print(f"Auxiliary outputs: {len(outputs['aux'])}")
    
    # Test eval mode
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"\nEval output shape: {y.shape}")
