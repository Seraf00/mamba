"""
Pure-Mamba-UNet - Pure State Space Model UNet Architecture

A UNet architecture built entirely with Mamba state space models,
replacing all traditional convolutions and attention with Mamba blocks.

This is an experimental architecture exploring the limits of
Mamba for medical image segmentation.

Key features:
- Mamba-based encoder blocks
- Mamba-based decoder blocks
- Mamba skip connections
- No convolutional layers in the main path (only in stem/head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Literal, Tuple

import sys
sys.path.append('..')

from models.modules import (
    create_mamba_block,
    MambaBottleneck,
    MambaSkipConnection,
    MambaEncoder,
    MambaDecoder
)


class MambaStemBlock(nn.Module):
    """Initial stem block to convert image to feature representation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 2
    ):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, patch_size, stride=patch_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class PureMambaEncoderBlock(nn.Module):
    """Pure Mamba encoder block with downsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_mamba_layers: int = 2,
        mamba_type: str = 'vmamba',
        d_state: int = 16,
        downsample: bool = True
    ):
        super().__init__()
        
        # Channel projection if needed
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.BatchNorm2d(out_channels)
        
        # Stacked Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            create_mamba_block(
                variant=mamba_type,
                dim=out_channels,
                d_state=d_state
            )
            for _ in range(num_mamba_layers)
        ])
        
        # Downsampling via Mamba with stride (using pooling for compatibility)
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.MaxPool2d(2),
                create_mamba_block(
                    variant=mamba_type,
                    dim=out_channels,
                    d_state=d_state
                )
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project channels
        x = self.norm(self.proj(x))
        
        # Mamba processing
        for mamba in self.mamba_blocks:
            x = x + mamba(x)
        
        skip = x
        
        # Downsample
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x, skip


class PureMambaDecoderBlock(nn.Module):
    """Pure Mamba decoder block with upsampling and skip fusion."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_mamba_layers: int = 2,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        # Upsample via transposed conv then Mamba
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels)
        )
        
        # Mamba skip connection
        self.mamba_skip = MambaSkipConnection(
            encoder_channels=skip_channels,
            decoder_channels=out_channels,
            mamba_type=mamba_type,
            d_state=d_state
        )
        
        # Skip fusion projection
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Stacked Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            create_mamba_block(
                variant=mamba_type,
                dim=out_channels,
                d_state=d_state
            )
            for _ in range(num_mamba_layers)
        ])
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Mamba skip enhancement
        skip_enhanced = self.mamba_skip(skip, x)
        
        # Fuse
        x = self.fusion(torch.cat([x, skip_enhanced], dim=1))
        
        # Mamba processing
        for mamba in self.mamba_blocks:
            x = x + mamba(x)
        
        return x


class MambaBottleneckDeep(nn.Module):
    """Deep Mamba bottleneck with multiple paths."""
    
    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        # Multi-layer Mamba
        self.mamba_layers = nn.ModuleList([
            create_mamba_block(
                variant=mamba_type,
                dim=dim,
                d_state=d_state
            )
            for _ in range(num_layers)
        ])
        
        # Global context path
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        
        # Layer norm
        self.norm = nn.BatchNorm2d(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Sequential Mamba processing
        for mamba in self.mamba_layers:
            x = x + mamba(x)
        
        # Global context
        global_ctx = self.global_path(x)
        x = x * global_ctx
        
        return self.norm(x + residual)


class BidirectionalMambaBlock(nn.Module):
    """Bidirectional Mamba for capturing both forward and backward dependencies."""
    
    def __init__(
        self,
        dim: int,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        # Forward Mamba
        self.mamba_fwd = create_mamba_block(
            variant=mamba_type,
            dim=dim,
            d_state=d_state
        )
        
        # Backward Mamba (flip, process, flip back)
        self.mamba_bwd = create_mamba_block(
            variant=mamba_type,
            dim=dim,
            d_state=d_state
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        fwd = self.mamba_fwd(x)
        
        # Backward pass (flip spatial dims)
        x_flip = torch.flip(x, dims=[2, 3])
        bwd = self.mamba_bwd(x_flip)
        bwd = torch.flip(bwd, dims=[2, 3])
        
        # Fuse
        return self.fusion(torch.cat([fwd, bwd], dim=1))


class PureMambaUNet(nn.Module):
    """
    Pure-Mamba-UNet: A UNet built entirely with Mamba state space models.
    
    This architecture replaces all convolutional and attention operations
    in the main path with Mamba blocks, exploring pure SSM-based
    medical image segmentation.
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        base_features: Base number of features
        num_stages: Number of encoder/decoder stages
        num_mamba_layers: Mamba layers per encoder/decoder block
        mamba_type: Type of Mamba ('mamba', 'mamba2', 'vmamba')
        use_bidirectional: Use bidirectional Mamba blocks
        d_state: SSM state dimension
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_features: int = 64,
        num_stages: int = 4,
        num_mamba_layers: int = 2,
        mamba_type: Literal['mamba', 'mamba2', 'vmamba'] = 'vmamba',
        use_bidirectional: bool = True,
        d_state: int = 16
    ):
        super().__init__()
        
        self.num_stages = num_stages
        self.mamba_type = mamba_type
        
        # Feature dimensions
        self.features = [base_features * (2 ** i) for i in range(num_stages + 1)]
        
        # Stem
        self.stem = MambaStemBlock(in_channels, self.features[0], patch_size=2)
        
        # Bidirectional Mamba after stem (optional)
        self.stem_mamba = BidirectionalMambaBlock(
            self.features[0], mamba_type, d_state
        ) if use_bidirectional else None
        
        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(num_stages):
            self.encoders.append(
                PureMambaEncoderBlock(
                    self.features[i],
                    self.features[i + 1],
                    num_mamba_layers=num_mamba_layers,
                    mamba_type=mamba_type,
                    d_state=d_state,
                    downsample=True
                )
            )
        
        # Bottleneck
        self.bottleneck = MambaBottleneckDeep(
            self.features[-1],
            num_layers=4,
            mamba_type=mamba_type,
            d_state=d_state
        )
        
        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(num_stages - 1, -1, -1):
            self.decoders.append(
                PureMambaDecoderBlock(
                    self.features[i + 1],
                    self.features[i + 1],  # Skip comes from encoder with same channels
                    self.features[i],
                    num_mamba_layers=num_mamba_layers,
                    mamba_type=mamba_type,
                    d_state=d_state
                )
            )
        
        # Final upsampling and segmentation head
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(self.features[0], self.features[0] // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.features[0] // 2),
            nn.ReLU(inplace=True)
        )
        
        # Final Mamba refinement
        self.final_mamba = create_mamba_block(
            variant=mamba_type,
            dim=self.features[0] // 2,
            d_state=d_state
        )
        
        self.seg_head = nn.Conv2d(self.features[0] // 2, num_classes, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, num_classes, H, W)
        """
        target_size = x.shape[2:]
        
        # Stem
        x = self.stem(x)
        
        # Optional bidirectional Mamba
        if self.stem_mamba is not None:
            x = x + self.stem_mamba(x)
        
        # Encoder
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skips = skips[::-1]
        for decoder, skip in zip(self.decoders, skips):
            x = decoder(x, skip)
        
        # Final upsampling
        x = self.final_up(x)
        x = x + self.final_mamba(x)
        
        # Segmentation
        out = self.seg_head(x)
        
        if out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=True)
        
        return out
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience functions
def pure_mamba_unet_tiny(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> PureMambaUNet:
    """Tiny Pure-Mamba-UNet."""
    return PureMambaUNet(
        in_channels, num_classes,
        base_features=32,
        num_stages=3,
        num_mamba_layers=1,
        mamba_type=mamba_type
    )


def pure_mamba_unet_small(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> PureMambaUNet:
    """Small Pure-Mamba-UNet."""
    return PureMambaUNet(
        in_channels, num_classes,
        base_features=48,
        num_stages=4,
        num_mamba_layers=2,
        mamba_type=mamba_type
    )


def pure_mamba_unet_base(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> PureMambaUNet:
    """Standard Pure-Mamba-UNet."""
    return PureMambaUNet(
        in_channels, num_classes,
        base_features=64,
        num_stages=4,
        num_mamba_layers=2,
        mamba_type=mamba_type
    )


def pure_mamba_unet_large(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> PureMambaUNet:
    """Large Pure-Mamba-UNet."""
    return PureMambaUNet(
        in_channels, num_classes,
        base_features=96,
        num_stages=5,
        num_mamba_layers=3,
        mamba_type=mamba_type
    )


if __name__ == '__main__':
    # Test the model
    model = PureMambaUNet(
        in_channels=1, num_classes=4,
        base_features=64,
        num_stages=4,
        mamba_type='vmamba'
    )
    print(f"Pure-Mamba-UNet Parameters: {model.count_parameters():,}")
    
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
