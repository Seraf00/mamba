"""
GUDU - Global-context U-Net with Dense Skip Connections

U-Net variant with dense skip connections and global context module,
inspired by DenseNet connectivity for better feature reuse.

Features:
- Dense skip connections between encoder and decoder
- Global context aggregation module
- Multi-scale feature fusion
- Efficient channel attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class DenseConvBlock(nn.Module):
    """
    Dense convolution block with growth rate.
    
    Each layer receives features from all preceding layers.
    
    Args:
        in_channels: Input channels
        growth_rate: Number of output channels per layer
        num_layers: Number of conv layers in block
    """
    
    def __init__(
        self,
        in_channels: int,
        growth_rate: int = 32,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_in = in_channels + i * growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(layer_in),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(layer_in, growth_rate, kernel_size=3, padding=1, bias=False)
                )
            )
        
        self.out_channels = in_channels + num_layers * growth_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class TransitionDown(nn.Module):
    """
    Transition layer for downsampling.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transition(x)


class TransitionUp(nn.Module):
    """
    Transition layer for upsampling.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class GlobalContextModule(nn.Module):
    """
    Global Context aggregation module.
    
    Captures long-range dependencies using attention mechanism.
    
    Args:
        in_channels: Input channels
        reduction: Channel reduction ratio
    """
    
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        
        inter_channels = in_channels // reduction
        
        # Global context extraction
        self.context = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Softmax(dim=2)  # Spatial softmax
        )
        
        # Channel transform
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.LayerNorm([inter_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, bias=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Compute attention weights
        attn = self.context(x)  # (B, 1, H, W)
        attn = attn.view(B, 1, H * W)  # (B, 1, HW)
        
        # Compute weighted sum of all positions
        x_flat = x.view(B, C, H * W)  # (B, C, HW)
        context = torch.bmm(x_flat, attn.transpose(1, 2))  # (B, C, 1)
        context = context.view(B, C, 1, 1)  # (B, C, 1, 1)
        
        # Transform and add back
        context = self.transform(context)
        
        return x + context


class ChannelAttention(nn.Module):
    """
    Channel attention module.
    
    Args:
        in_channels: Input channels
        reduction: Reduction ratio
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class DenseSkipConnection(nn.Module):
    """
    Dense skip connection fusing multiple encoder features.
    
    Args:
        encoder_channels: List of encoder channels to fuse
        out_channels: Output channels after fusion
    """
    
    def __init__(
        self,
        encoder_channels: List[int],
        out_channels: int,
        target_size: Optional[Tuple[int, int]] = None
    ):
        super().__init__()
        
        self.target_size = target_size
        
        # Project each encoder feature to same channels
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels // len(encoder_channels), kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels // len(encoder_channels)),
                nn.ReLU(inplace=True)
            )
            for ch in encoder_channels
        ])
        
        # Fuse projected features
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Channel attention
        self.attention = ChannelAttention(out_channels)
    
    def forward(
        self,
        features: List[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        target_size = target_size or self.target_size
        
        projected = []
        for feat, proj in zip(features, self.projections):
            p = proj(feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=True)
            projected.append(p)
        
        fused = torch.cat(projected, dim=1)
        fused = self.fuse(fused)
        fused = self.attention(fused)
        
        return fused


class GUDU(nn.Module):
    """
    Global-context U-Net with Dense skip connections.
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        base_features: Base number of features
        growth_rate: Growth rate for dense blocks
        num_layers_per_block: Layers in each dense block
        depth: Number of encoding levels
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_features: int = 48,
        growth_rate: int = 16,
        num_layers_per_block: int = 4,
        depth: int = 4
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.transitions_down = nn.ModuleList()
        
        current_channels = base_features
        encoder_channels = [base_features]
        
        for i in range(depth):
            # Dense block
            block = DenseConvBlock(
                current_channels, growth_rate, num_layers_per_block
            )
            self.encoder_blocks.append(block)
            current_channels = block.out_channels
            encoder_channels.append(current_channels)
            
            # Transition down (compress and downsample)
            if i < depth - 1:
                out_ch = current_channels // 2
                self.transitions_down.append(TransitionDown(current_channels, out_ch))
                current_channels = out_ch
        
        # Bottleneck with global context
        self.bottleneck = nn.Sequential(
            DenseConvBlock(current_channels, growth_rate, num_layers_per_block),
            GlobalContextModule(current_channels + num_layers_per_block * growth_rate)
        )
        bottleneck_channels = current_channels + num_layers_per_block * growth_rate
        
        # Decoder
        self.transitions_up = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.dense_skips = nn.ModuleList()
        
        current_channels = bottleneck_channels
        
        for i in range(depth - 1, -1, -1):
            # Transition up
            out_ch = encoder_channels[i + 1]
            self.transitions_up.append(TransitionUp(current_channels, out_ch))
            
            # Dense skip connection (fuse multiple encoder levels)
            skip_channels = encoder_channels[:i + 2]  # All encoder features up to this level
            self.dense_skips.append(
                DenseSkipConnection(skip_channels, out_ch)
            )
            
            # Decoder dense block
            block = DenseConvBlock(out_ch * 2, growth_rate, num_layers_per_block)
            self.decoder_blocks.append(block)
            current_channels = block.out_channels
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(current_channels, base_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, num_classes, kernel_size=1)
        )
        
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
        input_size = x.shape[2:]
        
        # Initial
        x = self.initial(x)
        
        # Encoder
        encoder_features = [x]
        for i, (block, trans) in enumerate(zip(
            self.encoder_blocks[:-1], self.transitions_down
        )):
            x = block(x)
            encoder_features.append(x)
            x = trans(x)
        
        # Last encoder block (no transition)
        x = self.encoder_blocks[-1](x)
        encoder_features.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with dense skip connections
        for i, (up, skip_module, block) in enumerate(zip(
            self.transitions_up, self.dense_skips, self.decoder_blocks
        )):
            x = up(x)
            
            # Get target size for skip connection
            target_size = x.shape[2:]
            
            # Dense skip: fuse relevant encoder features
            level = self.depth - 1 - i
            skip_features = encoder_features[:level + 2]
            skip = skip_module(skip_features, target_size)
            
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        
        # Output
        x = self.output(x)
        
        # Ensure output matches input size
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience functions
def gudu_small(in_channels: int = 1, num_classes: int = 4) -> GUDU:
    """Small GUDU model."""
    return GUDU(in_channels, num_classes, base_features=32, growth_rate=12, depth=4)


def gudu_base(in_channels: int = 1, num_classes: int = 4) -> GUDU:
    """Standard GUDU model."""
    return GUDU(in_channels, num_classes, base_features=48, growth_rate=16, depth=4)


def gudu_large(in_channels: int = 1, num_classes: int = 4) -> GUDU:
    """Large GUDU model."""
    return GUDU(in_channels, num_classes, base_features=64, growth_rate=24, depth=5)


if __name__ == '__main__':
    # Test the model
    model = GUDU(in_channels=1, num_classes=4)
    print(f"GUDU Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
