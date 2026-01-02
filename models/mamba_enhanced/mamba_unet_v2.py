"""
Mamba-UNet V2 - Enhanced UNet V2 with Mamba Integration

UNet V2 (with attention gates, SE blocks, residual connections)
enhanced with Mamba state space models.

Integration points:
- Bottleneck: Multi-scale Mamba for global context
- Skip connections: Cross-Mamba fusion
- Attention replacement: Mamba-attention hybrid
- Deep supervision with Mamba refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Literal, Union

import sys
sys.path.append('..')

from models.modules import (
    create_mamba_block,
    MambaBottleneck,
    MultiscaleMambaBottleneck,
    CrossMambaFusion,
    GatedMambaSkip,
    HybridAttentionMamba
)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class ResidualConvBlock(nn.Module):
    """Residual convolution block with SE."""
    
    def __init__(self, in_channels: int, out_channels: int, use_se: bool = True):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.residual = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.conv(x)
        x = self.se(x)
        return self.relu(x + residual)


class MambaEncoderBlockV2(nn.Module):
    """Encoder block with hybrid Mamba-attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_mamba: bool = True,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        self.pool = nn.MaxPool2d(2)
        self.conv = ResidualConvBlock(in_channels, out_channels)
        
        # Hybrid Mamba-Attention block
        self.hybrid = None
        if use_mamba:
            self.hybrid = HybridAttentionMamba(
                dim=out_channels,
                num_heads=4,
                mamba_type=mamba_type,
                fusion_mode='parallel',
                d_state=d_state
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        if self.hybrid is not None:
            x = self.hybrid(x)
        return x


class MambaDecoderBlockV2(nn.Module):
    """Decoder block with Cross-Mamba skip fusion."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_mamba_skip: bool = True,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Cross-Mamba fusion for skip connection
        self.mamba_skip = None
        if use_mamba_skip:
            self.mamba_skip = CrossMambaFusion(
                dim=in_channels // 2,
                mamba_type=mamba_type,
                d_state=d_state
            )
        
        self.conv = ResidualConvBlock(in_channels // 2 + skip_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Cross-Mamba fusion
        if self.mamba_skip is not None:
            # CrossMambaFusion returns a single fused feature
            fused = self.mamba_skip(x, skip)
            x = torch.cat([fused, x], dim=1)
        else:
            x = torch.cat([skip, x], dim=1)
        
        return self.conv(x)


class MambaUNetV2(nn.Module):
    """
    Mamba-Enhanced UNet V2.
    
    Enhanced UNet with:
    - Residual blocks + SE attention
    - Hybrid Mamba-Attention in encoder
    - Cross-Mamba skip connections
    - Multi-scale Mamba bottleneck
    - Deep supervision with Mamba refinement
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        base_features: Base number of features
        depth: Number of encoder/decoder levels
        mamba_type: Type of Mamba ('mamba', 'mamba2', 'vmamba')
        mamba_in_encoder: Add hybrid Mamba-attention to encoder
        mamba_in_skip: Add Cross-Mamba to skip connections
        use_multiscale_bottleneck: Use multi-scale Mamba bottleneck
        deep_supervision: Enable deep supervision
        d_state: SSM state dimension
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_features: int = 64,
        depth: int = 4,
        mamba_type: Literal['mamba', 'mamba2', 'vmamba'] = 'vmamba',
        mamba_in_encoder: bool = True,
        mamba_in_skip: bool = True,
        use_multiscale_bottleneck: bool = True,
        deep_supervision: bool = False,
        d_state: int = 16
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth
        self.mamba_type = mamba_type
        self.deep_supervision = deep_supervision
        
        # Feature dimensions
        features = [base_features * (2 ** i) for i in range(depth + 1)]
        
        # Initial convolution
        self.inc = ResidualConvBlock(in_channels, features[0])
        
        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(
                MambaEncoderBlockV2(
                    features[i], features[i + 1],
                    use_mamba=mamba_in_encoder and i >= depth // 2,  # Mamba in deeper layers
                    mamba_type=mamba_type,
                    d_state=d_state
                )
            )
        
        # Multi-scale Mamba bottleneck
        if use_multiscale_bottleneck:
            self.bottleneck = MultiscaleMambaBottleneck(
                dim=features[-1],
                mamba_type=mamba_type,
                scales=[1, 2, 4],
                d_state=d_state
            )
        else:
            self.bottleneck = MambaBottleneck(
                dim=features[-1],
                mamba_type=mamba_type,
                d_state=d_state
            )
        
        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.decoders.append(
                MambaDecoderBlockV2(
                    features[i + 1], features[i], features[i],
                    use_mamba_skip=mamba_in_skip,
                    mamba_type=mamba_type,
                    d_state=d_state
                )
            )
        
        # Output
        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)
        
        # Deep supervision heads
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(features[i], num_classes, kernel_size=1)
                for i in range(1, depth)
            ])
        
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
        
        # Initial conv
        x1 = self.inc(x)
        
        # Encoder
        skips = [x1]
        enc = x1
        for encoder in self.encoders:
            enc = encoder(enc)
            skips.append(enc)
        
        # Bottleneck
        enc = self.bottleneck(enc)
        
        # Remove bottleneck from skips
        skips = skips[:-1]
        
        # Decoder
        dec = enc
        dec_features = []
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            dec = decoder(dec, skip)
            dec_features.append(dec)
        
        # Output
        out = self.outc(dec)
        
        # Deep supervision
        if self.deep_supervision and self.training:
            aux_outputs = []
            for i, (feat, head) in enumerate(zip(dec_features[:-1], self.ds_heads)):
                aux = head(feat)
                aux = F.interpolate(aux, size=target_size, mode='bilinear', align_corners=True)
                aux_outputs.append(aux)
            return {'out': out, 'aux': aux_outputs}
        
        return out
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience functions
def mamba_unet_v2_base(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaUNetV2:
    """Standard Mamba-UNet V2."""
    return MambaUNetV2(
        in_channels, num_classes,
        base_features=64,
        mamba_type=mamba_type
    )


def mamba_unet_v2_deep(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaUNetV2:
    """Mamba-UNet V2 with deep supervision."""
    return MambaUNetV2(
        in_channels, num_classes,
        base_features=64,
        mamba_type=mamba_type,
        deep_supervision=True
    )


if __name__ == '__main__':
    # Test the model
    model = MambaUNetV2(in_channels=1, num_classes=4, mamba_type='vmamba')
    print(f"Mamba-UNet V2 Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
