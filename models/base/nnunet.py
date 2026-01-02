"""
nnUNet - Self-Configuring UNet Architecture

Simplified implementation inspired by "nnU-Net: Self-adapting Framework 
for U-Net-Based Medical Image Segmentation" by Isensee et al., 2021.

Features:
- Instance normalization (better for varying image intensities)
- Leaky ReLU activations
- Residual connections in encoder
- Deep supervision
- Configurable based on dataset properties
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Union


class ConvBlock(nn.Module):
    """
    nnUNet convolution block with instance norm and leaky ReLU.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        use_residual: Add residual connection
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size,
            padding=padding, bias=False
        )
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        # Residual projection if needed
        if use_residual and (in_channels != out_channels or stride != 1):
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True)
            )
            self.use_residual = True
        elif self.use_residual:
            self.residual = nn.Identity()
        else:
            self.residual = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.lrelu(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        if self.use_residual:
            x = x + self.residual(residual)
        
        x = self.lrelu(x)
        return x


class StackedConvBlocks(nn.Module):
    """
    Stack of convolution blocks.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        num_blocks: Number of conv blocks
        first_stride: Stride for first block (for downsampling)
        use_residual: Use residual connections
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        first_stride: int = 1,
        use_residual: bool = True
    ):
        super().__init__()
        
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                ConvBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride=first_stride if i == 0 else 1,
                    use_residual=use_residual
                )
            )
        
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class nnUNetEncoder(nn.Module):
    """
    nnUNet encoder with strided convolutions for downsampling.
    
    Args:
        in_channels: Input channels
        base_features: Base number of features
        num_stages: Number of encoding stages
        features_per_stage: Features at each stage (overrides base_features)
        blocks_per_stage: Number of conv blocks per stage
        use_residual: Use residual connections
    """
    
    def __init__(
        self,
        in_channels: int,
        base_features: int = 32,
        num_stages: int = 5,
        features_per_stage: Optional[List[int]] = None,
        blocks_per_stage: Optional[List[int]] = None,
        use_residual: bool = True
    ):
        super().__init__()
        
        # Default features: double each stage up to max
        if features_per_stage is None:
            features_per_stage = [
                min(base_features * (2 ** i), 320)
                for i in range(num_stages)
            ]
        
        # Default: 2 blocks per stage
        if blocks_per_stage is None:
            blocks_per_stage = [2] * num_stages
        
        self.stages = nn.ModuleList()
        
        for i in range(num_stages):
            in_ch = in_channels if i == 0 else features_per_stage[i - 1]
            out_ch = features_per_stage[i]
            stride = 1 if i == 0 else 2  # First stage no downsampling
            
            self.stages.append(
                StackedConvBlocks(
                    in_ch, out_ch,
                    num_blocks=blocks_per_stage[i],
                    first_stride=stride,
                    use_residual=use_residual
                )
            )
        
        self.features_per_stage = features_per_stage
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns features at each stage."""
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class nnUNetDecoder(nn.Module):
    """
    nnUNet decoder with transposed convolutions.
    
    Args:
        encoder_features: List of encoder feature channels
        num_classes: Number of output classes
        blocks_per_stage: Number of conv blocks per stage
        use_residual: Use residual connections
        deep_supervision: Enable deep supervision outputs
    """
    
    def __init__(
        self,
        encoder_features: List[int],
        num_classes: int,
        blocks_per_stage: Optional[List[int]] = None,
        use_residual: bool = True,
        deep_supervision: bool = True
    ):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        num_stages = len(encoder_features) - 1  # Decoder stages = encoder - 1
        
        if blocks_per_stage is None:
            blocks_per_stage = [2] * num_stages
        
        # Reverse for decoder
        decoder_features = encoder_features[:-1][::-1]  # Exclude bottleneck, reverse
        
        self.upsample_layers = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        self.seg_heads = nn.ModuleList()
        
        for i in range(num_stages):
            # Upsampling
            in_ch = encoder_features[-(i + 1)]  # Start from bottleneck
            out_ch = decoder_features[i]
            
            self.upsample_layers.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            
            # Conv after concatenation
            self.conv_blocks.append(
                StackedConvBlocks(
                    out_ch * 2,  # After concat with skip
                    out_ch,
                    num_blocks=blocks_per_stage[i],
                    use_residual=use_residual
                )
            )
            
            # Segmentation head for deep supervision
            if deep_supervision or i == num_stages - 1:
                self.seg_heads.append(
                    nn.Conv2d(out_ch, num_classes, kernel_size=1)
                )
            else:
                self.seg_heads.append(None)
    
    def forward(
        self,
        encoder_features: List[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            encoder_features: List of encoder features (low to high level)
            target_size: Target output size for final upsampling
            
        Returns:
            If deep_supervision: List of predictions at each scale
            Else: Single prediction
        """
        # Reverse encoder features for decoder
        skips = encoder_features[:-1][::-1]
        x = encoder_features[-1]  # Bottleneck
        
        outputs = []
        
        for i, (up, conv, head) in enumerate(zip(
            self.upsample_layers, self.conv_blocks, self.seg_heads
        )):
            # Upsample
            x = up(x)
            
            # Handle size mismatch with skip
            skip = skips[i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            # Concatenate and convolve
            x = torch.cat([x, skip], dim=1)
            x = conv(x)
            
            # Segmentation head
            if head is not None:
                seg = head(x)
                if target_size is not None and seg.shape[2:] != target_size:
                    seg = F.interpolate(seg, size=target_size, mode='bilinear', align_corners=True)
                outputs.append(seg)
        
        if self.deep_supervision and self.training:
            return outputs
        else:
            return outputs[-1] if outputs else x


class nnUNet(nn.Module):
    """
    nnUNet - Self-Configuring U-Net for Medical Image Segmentation.
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        base_features: Base number of features
        num_stages: Number of encoder/decoder stages
        features_per_stage: Optional custom features per stage
        blocks_per_stage: Number of conv blocks per stage
        use_residual: Use residual connections
        deep_supervision: Enable deep supervision (for training)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_features: int = 32,
        num_stages: int = 5,
        features_per_stage: Optional[List[int]] = None,
        blocks_per_stage: Optional[List[int]] = None,
        use_residual: bool = True,
        deep_supervision: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        # Encoder
        self.encoder = nnUNetEncoder(
            in_channels=in_channels,
            base_features=base_features,
            num_stages=num_stages,
            features_per_stage=features_per_stage,
            blocks_per_stage=blocks_per_stage,
            use_residual=use_residual
        )
        
        # Decoder
        self.decoder = nnUNetDecoder(
            encoder_features=self.encoder.features_per_stage,
            num_classes=num_classes,
            blocks_per_stage=blocks_per_stage[1:] if blocks_per_stage else None,
            use_residual=use_residual,
            deep_supervision=deep_supervision
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
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
        
        # Encode
        encoder_features = self.encoder(x)
        
        # Decode
        outputs = self.decoder(encoder_features, target_size)
        
        if isinstance(outputs, list) and self.training:
            return {
                'out': outputs[-1],
                'aux': outputs[:-1]
            }
        
        return outputs if not isinstance(outputs, list) else outputs[-1]
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience functions for common configurations
def nnunet_small(in_channels: int = 1, num_classes: int = 4) -> nnUNet:
    """Small nnUNet with 16 base features."""
    return nnUNet(
        in_channels, num_classes,
        base_features=16,
        num_stages=5,
        features_per_stage=[16, 32, 64, 128, 256]
    )


def nnunet_base(in_channels: int = 1, num_classes: int = 4) -> nnUNet:
    """Standard nnUNet with 32 base features."""
    return nnUNet(
        in_channels, num_classes,
        base_features=32,
        num_stages=5,
        features_per_stage=[32, 64, 128, 256, 320]
    )


def nnunet_large(in_channels: int = 1, num_classes: int = 4) -> nnUNet:
    """Large nnUNet with more features."""
    return nnUNet(
        in_channels, num_classes,
        base_features=32,
        num_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320]
    )


if __name__ == '__main__':
    # Test the model
    model = nnUNet(in_channels=1, num_classes=4, deep_supervision=True)
    print(f"nnUNet Parameters: {model.count_parameters():,}")
    
    # Test forward pass (training mode)
    model.train()
    x = torch.randn(2, 1, 256, 256)
    outputs = model(x)
    
    if isinstance(outputs, dict):
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {outputs['out'].shape}")
        print(f"Auxiliary outputs: {len(outputs['aux'])}")
        for i, aux in enumerate(outputs['aux']):
            print(f"  Aux {i}: {aux.shape}")
    
    # Test eval mode
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"\nEval output shape: {y.shape}")
