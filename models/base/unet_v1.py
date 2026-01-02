"""
UNet V1 - Classic UNet Architecture

Original U-Net from "U-Net: Convolutional Networks for Biomedical Image Segmentation"
by Ronneberger et al., 2015.

Features:
- Symmetric encoder-decoder with skip connections
- Double convolution blocks
- Max pooling for downsampling
- Transposed convolution for upsampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class DoubleConv(nn.Module):
    """
    Double convolution block: (Conv -> BN -> ReLU) x 2
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        mid_channels: Middle channels (default: same as out_channels)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None
    ):
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
    """
    Encoder block: MaxPool -> DoubleConv
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class DecoderBlock(nn.Module):
    """
    Decoder block: Upsample -> Concat Skip -> DoubleConv
    
    Args:
        in_channels: Input channels (from previous decoder stage)
        out_channels: Output channels
        bilinear: Use bilinear upsampling instead of transposed conv
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = False
    ):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Handle size mismatch
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        
        x = F.pad(x, [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2
        ])
        
        # Concatenate skip connection
        x = torch.cat([skip, x], dim=1)
        
        return self.conv(x)


class UNetV1(nn.Module):
    """
    Classic U-Net Architecture.
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output classes
        base_features: Base number of features (doubled at each level)
        depth: Number of encoder/decoder levels
        bilinear: Use bilinear upsampling instead of transposed conv
        
    Architecture:
        Encoder: in -> 64 -> 128 -> 256 -> 512
        Bottleneck: 512 -> 1024
        Decoder: 1024 -> 512 -> 256 -> 128 -> 64 -> num_classes
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_features: int = 64,
        depth: int = 4,
        bilinear: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth
        self.bilinear = bilinear
        
        # Calculate feature dimensions
        features = [base_features * (2 ** i) for i in range(depth + 1)]
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, features[0])
        
        # Encoder path
        self.encoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(
                EncoderBlock(features[i], features[i + 1])
            )
        
        # Decoder path
        self.decoders = nn.ModuleList()
        factor = 2 if bilinear else 1
        
        for i in range(depth - 1, -1, -1):
            in_ch = features[i + 1] + features[i] if i == depth - 1 else features[i + 1]
            if i == depth - 1:
                in_ch = features[i + 1]
            self.decoders.append(
                DecoderBlock(features[i + 1], features[i] // factor if bilinear else features[i], bilinear)
            )
        
        # Output convolution
        self.outc = nn.Conv2d(features[0] // factor if bilinear else features[0], num_classes, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
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
        
        # Encoder path - collect skip connections
        skips = [x1]
        x = x1
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        
        # Remove bottleneck from skips (it's the current x)
        skips = skips[:-1]
        
        # Decoder path - use skip connections in reverse
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            x = decoder(x, skip)
        
        # Output
        return self.outc(x)
    
    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features for analysis.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            List of feature tensors at each scale
        """
        features = []
        
        x = self.inc(x)
        features.append(x)
        
        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)
        
        return features
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience functions for common configurations
def unet_v1_small(in_channels: int = 1, num_classes: int = 4) -> UNetV1:
    """Small UNet with 32 base features."""
    return UNetV1(in_channels, num_classes, base_features=32, depth=4)


def unet_v1_base(in_channels: int = 1, num_classes: int = 4) -> UNetV1:
    """Standard UNet with 64 base features."""
    return UNetV1(in_channels, num_classes, base_features=64, depth=4)


def unet_v1_large(in_channels: int = 1, num_classes: int = 4) -> UNetV1:
    """Large UNet with 64 base features and 5 levels."""
    return UNetV1(in_channels, num_classes, base_features=64, depth=5)


if __name__ == '__main__':
    # Test the model
    model = UNetV1(in_channels=1, num_classes=4)
    print(f"UNet V1 Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test feature extraction
    features = model.get_features(x)
    print("\nFeature shapes:")
    for i, f in enumerate(features):
        print(f"  Level {i}: {f.shape}")
