"""
Mamba-UNet V1 - Classic UNet Enhanced with Mamba Blocks

UNet V1 architecture with Mamba state space models integrated at:
- Bottleneck: Global context modeling
- Skip connections: Feature refinement
- Optional: Encoder/Decoder enhancement

Supports Mamba, Mamba2, and VM-Mamba variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Literal

import sys
sys.path.append('..')

from models.modules import (
    create_mamba_block,
    MambaBottleneck,
    MambaSkipConnection,
    MambaEncoderStage,
    MambaDecoderStage
)


class DoubleConv(nn.Module):
    """Double convolution block: (Conv -> BN -> ReLU) x 2"""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """Encoder block with optional Mamba enhancement."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_mamba: bool = False,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)
        
        # Optional Mamba enhancement
        self.mamba = None
        if use_mamba:
            self.mamba = create_mamba_block(
                variant=mamba_type,
                dim=out_channels,
                d_state=d_state
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        if self.mamba is not None:
            x = x + self.mamba(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block with Mamba-enhanced skip connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_mamba_skip: bool = True,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Mamba-enhanced skip connection
        self.mamba_skip = None
        if use_mamba_skip:
            self.mamba_skip = MambaSkipConnection(
                encoder_channels=in_channels // 2,
                decoder_channels=in_channels // 2,
                mamba_type=mamba_type,
                d_state=d_state
            )
        
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Handle size mismatch
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        
        # Apply Mamba skip enhancement
        if self.mamba_skip is not None:
            skip = self.mamba_skip(skip, x)
        
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class MambaUNetV1(nn.Module):
    """
    Mamba-Enhanced UNet V1.
    
    Classic UNet with Mamba blocks integrated for:
    1. Bottleneck: Global context via Mamba
    2. Skip connections: Feature refinement via Mamba
    3. Optional encoder/decoder Mamba layers
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        base_features: Base number of features
        depth: Number of encoder/decoder levels
        mamba_type: Type of Mamba ('mamba', 'mamba2', 'vmamba')
        mamba_in_encoder: Add Mamba to encoder blocks
        mamba_in_skip: Add Mamba to skip connections
        mamba_in_bottleneck: Add Mamba bottleneck
        d_state: SSM state dimension
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_features: int = 64,
        depth: int = 4,
        mamba_type: Literal['mamba', 'mamba2', 'vmamba'] = 'vmamba',
        mamba_in_encoder: bool = False,
        mamba_in_skip: bool = True,
        mamba_in_bottleneck: bool = True,
        d_state: int = 16
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth
        self.mamba_type = mamba_type
        
        # Feature dimensions
        features = [base_features * (2 ** i) for i in range(depth + 1)]
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, features[0])
        
        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(
                EncoderBlock(
                    features[i], features[i + 1],
                    use_mamba=mamba_in_encoder,
                    mamba_type=mamba_type,
                    d_state=d_state
                )
            )
        
        # Mamba Bottleneck
        self.bottleneck = None
        if mamba_in_bottleneck:
            self.bottleneck = MambaBottleneck(
                dim=features[-1],
                mamba_type=mamba_type,
                d_state=d_state,
                depth=2
            )
        
        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.decoders.append(
                DecoderBlock(
                    features[i + 1], features[i],
                    use_mamba_skip=mamba_in_skip,
                    mamba_type=mamba_type,
                    d_state=d_state
                )
            )
        
        # Output
        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)
        
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
        # Initial conv
        x1 = self.inc(x)
        
        # Encoder
        skips = [x1]
        enc = x1
        for encoder in self.encoders:
            enc = encoder(enc)
            skips.append(enc)
        
        # Bottleneck
        if self.bottleneck is not None:
            enc = self.bottleneck(enc)
        
        # Remove bottleneck from skips
        skips = skips[:-1]
        
        # Decoder
        dec = enc
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            dec = decoder(dec, skip)
        
        return self.outc(dec)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_mamba_parameters(self) -> int:
        """Count Mamba-specific parameters."""
        mamba_params = 0
        if self.bottleneck is not None:
            mamba_params += sum(p.numel() for p in self.bottleneck.parameters())
        for decoder in self.decoders:
            if decoder.mamba_skip is not None:
                mamba_params += sum(p.numel() for p in decoder.mamba_skip.parameters())
        for encoder in self.encoders:
            if encoder.mamba is not None:
                mamba_params += sum(p.numel() for p in encoder.mamba.parameters())
        return mamba_params


# Convenience functions for different configurations
def mamba_unet_v1_small(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaUNetV1:
    """Small Mamba-UNet with 32 base features."""
    return MambaUNetV1(
        in_channels, num_classes,
        base_features=32, depth=4,
        mamba_type=mamba_type
    )


def mamba_unet_v1_base(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaUNetV1:
    """Standard Mamba-UNet with 64 base features."""
    return MambaUNetV1(
        in_channels, num_classes,
        base_features=64, depth=4,
        mamba_type=mamba_type
    )


def mamba_unet_v1_full(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaUNetV1:
    """Full Mamba-UNet with Mamba in encoder, skip, and bottleneck."""
    return MambaUNetV1(
        in_channels, num_classes,
        base_features=64, depth=4,
        mamba_type=mamba_type,
        mamba_in_encoder=True,
        mamba_in_skip=True,
        mamba_in_bottleneck=True
    )


if __name__ == '__main__':
    # Test the model
    model = MambaUNetV1(in_channels=1, num_classes=4, mamba_type='vmamba')
    print(f"Mamba-UNet V1 Parameters: {model.count_parameters():,}")
    print(f"Mamba Parameters: {model.get_mamba_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test different Mamba types
    for mamba_type in ['mamba', 'mamba2', 'vmamba']:
        m = MambaUNetV1(1, 4, mamba_type=mamba_type)
        print(f"\n{mamba_type}: {m.count_parameters():,} params")
