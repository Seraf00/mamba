"""
UNet V2 - Improved UNet Architecture

Enhanced U-Net with modern improvements:
- Residual connections within blocks
- Attention gates for skip connections
- Deep supervision
- Squeeze-and-Excitation blocks

Based on "UNet++: A Nested U-Net Architecture" and "Attention U-Net" concepts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck
    """
    
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
    """
    Residual convolution block with SE attention.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        use_se: Use Squeeze-and-Excitation attention
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_se: bool = True
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()
        
        # SE attention
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.conv(x)
        x = self.se(x)
        x = self.relu(x + residual)
        return x


class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections.
    
    Learns to focus on relevant spatial regions from encoder.
    
    Args:
        gate_channels: Channels from decoder (gating signal)
        skip_channels: Channels from encoder (skip connection)
        inter_channels: Intermediate channels
    """
    
    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None
    ):
        super().__init__()
        
        inter_channels = inter_channels or skip_channels // 2
        
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: Gating signal from decoder (B, C_g, H, W)
            skip: Skip connection from encoder (B, C_s, H, W)
            
        Returns:
            Attended skip connection (B, C_s, H, W)
        """
        # Align spatial dimensions
        g = self.W_g(gate)
        x = self.W_x(skip)
        
        # Handle size mismatch
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Compute attention
        psi = self.relu(g + x)
        psi = self.psi(psi)
        
        return skip * psi


class EncoderBlockV2(nn.Module):
    """
    Encoder block with residual connections.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        use_se: Use Squeeze-and-Excitation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_se: bool = True
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ResidualConvBlock(in_channels, out_channels, use_se)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class DecoderBlockV2(nn.Module):
    """
    Decoder block with attention gate.
    
    Args:
        in_channels: Input channels from previous decoder
        skip_channels: Channels from skip connection
        out_channels: Output channels
        use_attention: Use attention gate for skip
        use_se: Use Squeeze-and-Excitation
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = True,
        use_se: bool = True
    ):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        self.attention = AttentionGate(
            gate_channels=in_channels // 2,
            skip_channels=skip_channels
        ) if use_attention else None
        
        self.conv = ResidualConvBlock(
            in_channels // 2 + skip_channels,
            out_channels,
            use_se
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Handle size mismatch
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        
        # Apply attention to skip connection
        if self.attention is not None:
            skip = self.attention(x, skip)
        
        # Concatenate and convolve
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class DeepSupervision(nn.Module):
    """
    Deep supervision module for auxiliary outputs.
    
    Args:
        in_channels_list: List of input channels at each scale
        num_classes: Number of output classes
    """
    
    def __init__(self, in_channels_list: List[int], num_classes: int):
        super().__init__()
        
        self.heads = nn.ModuleList([
            nn.Conv2d(ch, num_classes, kernel_size=1)
            for ch in in_channels_list
        ])
    
    def forward(
        self,
        features: List[torch.Tensor],
        target_size: Tuple[int, int]
    ) -> List[torch.Tensor]:
        """
        Args:
            features: List of decoder features at different scales
            target_size: Target spatial size (H, W)
            
        Returns:
            List of upsampled predictions
        """
        outputs = []
        for feat, head in zip(features, self.heads):
            out = head(feat)
            if out.shape[2:] != target_size:
                out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=True)
            outputs.append(out)
        return outputs


class UNetV2(nn.Module):
    """
    Improved U-Net V2 Architecture.
    
    Features:
    - Residual connections within blocks
    - Squeeze-and-Excitation attention
    - Attention gates for skip connections
    - Optional deep supervision
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        base_features: Base number of features
        depth: Number of encoder/decoder levels
        use_attention: Use attention gates
        use_se: Use Squeeze-and-Excitation blocks
        deep_supervision: Enable deep supervision
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_features: int = 64,
        depth: int = 4,
        use_attention: bool = True,
        use_se: bool = True,
        deep_supervision: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth
        self.deep_supervision = deep_supervision
        
        # Feature dimensions
        features = [base_features * (2 ** i) for i in range(depth + 1)]
        
        # Initial convolution
        self.inc = ResidualConvBlock(in_channels, features[0], use_se)
        
        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(
                EncoderBlockV2(features[i], features[i + 1], use_se)
            )
        
        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.decoders.append(
                DecoderBlockV2(
                    features[i + 1],
                    features[i],
                    features[i],
                    use_attention,
                    use_se
                )
            )
        
        # Output
        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)
        
        # Deep supervision
        if deep_supervision:
            self.ds = DeepSupervision(features[:depth], num_classes)
        
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
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            return_features: Return intermediate features
            
        Returns:
            Dictionary with 'out' and optionally 'aux' (deep supervision),
            'features' (encoder features)
        """
        target_size = x.shape[2:]
        
        # Encoder
        x1 = self.inc(x)
        skips = [x1]
        enc = x1
        
        for encoder in self.encoders:
            enc = encoder(enc)
            skips.append(enc)
        
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
        
        result = {'out': out}
        
        # Deep supervision
        if self.deep_supervision and self.training:
            aux_outputs = self.ds(dec_features[:-1], target_size)
            result['aux'] = aux_outputs
        
        if return_features:
            result['features'] = skips + [enc]
        
        return result
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience functions
def unet_v2_small(in_channels: int = 1, num_classes: int = 4) -> UNetV2:
    """Small UNet V2 with 32 base features."""
    return UNetV2(in_channels, num_classes, base_features=32)


def unet_v2_base(in_channels: int = 1, num_classes: int = 4) -> UNetV2:
    """Standard UNet V2 with 64 base features."""
    return UNetV2(in_channels, num_classes, base_features=64)


def unet_v2_large(in_channels: int = 1, num_classes: int = 4) -> UNetV2:
    """Large UNet V2 with deep supervision."""
    return UNetV2(in_channels, num_classes, base_features=64, depth=5, deep_supervision=True)


if __name__ == '__main__':
    # Test the model
    model = UNetV2(in_channels=1, num_classes=4, deep_supervision=True)
    print(f"UNet V2 Parameters: {model.count_parameters():,}")
    
    # Test forward pass (training mode for deep supervision)
    model.train()
    x = torch.randn(2, 1, 256, 256)
    outputs = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs['out'].shape}")
    
    if 'aux' in outputs:
        print(f"Auxiliary outputs: {len(outputs['aux'])}")
        for i, aux in enumerate(outputs['aux']):
            print(f"  Aux {i}: {aux.shape}")
