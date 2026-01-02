"""
Mamba-DeepLab - DeepLabV3 Enhanced with Mamba State Space Models

DeepLabV3 with Mamba integration for enhanced global context modeling.

Integration points:
- ASPP replacement/enhancement with Mamba
- Mamba bottleneck after ASPP
- Decoder enhancement with Mamba
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Literal
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet101_Weights

import sys
sys.path.append('..')

from models.modules import (
    create_mamba_block,
    MambaBottleneck,
    ASPPMambaBottleneck
)


class ASPPConv(nn.Module):
    """ASPP convolution with specified dilation rate."""
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ASPPPooling(nn.Module):
    """ASPP global average pooling branch."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        x = self.conv(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class MambaASPP(nn.Module):
    """
    ASPP module enhanced with Mamba for global context.
    
    Combines traditional ASPP with Mamba branch for
    enhanced long-range dependency modeling.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels per branch
        atrous_rates: Dilation rates for ASPP
        mamba_type: Type of Mamba block
        d_state: SSM state dimension
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        atrous_rates: Tuple[int, ...] = (6, 12, 18),
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Dilated convolutions
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        
        # Global pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
        # Mamba branch for global context
        self.mamba_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.mamba = create_mamba_block(
            variant=mamba_type,
            dim=out_channels,
            d_state=d_state
        )
        
        self.convs = nn.ModuleList(modules)
        
        # Project concatenated features (6 branches: 1x1 + 3 dilated + pool + mamba)
        num_branches = len(atrous_rates) + 3  # 1x1, dilated, pool, mamba
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, out_channels, 
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        
        # Standard ASPP branches
        for conv in self.convs:
            features.append(conv(x))
        
        # Mamba branch
        mamba_feat = self.mamba_branch(x)
        mamba_feat = self.mamba(mamba_feat)
        features.append(mamba_feat)
        
        features = torch.cat(features, dim=1)
        return self.project(features)


class MambaDeepLabHead(nn.Module):
    """
    DeepLabV3 head with Mamba enhancement.
    
    Args:
        in_channels: Input channels from backbone
        low_level_channels: Channels from low-level features
        num_classes: Number of output classes
        aspp_channels: Output channels of ASPP
        mamba_type: Type of Mamba block
        d_state: SSM state dimension
    """
    
    def __init__(
        self,
        in_channels: int,
        low_level_channels: int,
        num_classes: int,
        aspp_channels: int = 256,
        atrous_rates: Tuple[int, ...] = (6, 12, 18),
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        # Mamba-enhanced ASPP
        self.aspp = MambaASPP(
            in_channels, aspp_channels, atrous_rates,
            mamba_type=mamba_type, d_state=d_state
        )
        
        # Low-level feature processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Mamba for decoder fusion
        self.decoder_mamba = create_mamba_block(
            variant=mamba_type,
            dim=aspp_channels + 48,
            d_state=d_state
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_channels + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def forward(
        self,
        high_level: torch.Tensor,
        low_level: torch.Tensor
    ) -> torch.Tensor:
        # ASPP on high-level features
        x = self.aspp(high_level)
        
        # Upsample to low-level feature size
        x = F.interpolate(x, size=low_level.shape[2:], mode='bilinear', align_corners=True)
        
        # Process low-level features
        low_level = self.low_level_conv(low_level)
        
        # Concatenate
        x = torch.cat([x, low_level], dim=1)
        
        # Mamba enhancement
        x = x + self.decoder_mamba(x)
        
        # Decode
        x = self.decoder(x)
        
        return x


class MambaDeepLab(nn.Module):
    """
    Mamba-Enhanced DeepLabV3.
    
    DeepLabV3 with Mamba integration:
    - Mamba branch in ASPP for global context
    - Mamba enhancement in decoder
    - Optional Mamba after backbone stages
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        backbone: ResNet variant ('resnet50', 'resnet101')
        pretrained: Use ImageNet pretrained weights
        output_stride: Output stride (8 or 16)
        mamba_type: Type of Mamba ('mamba', 'mamba2', 'vmamba')
        d_state: SSM state dimension
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        backbone: Literal['resnet50', 'resnet101'] = 'resnet50',
        pretrained: bool = True,
        output_stride: Literal[8, 16] = 16,
        mamba_type: Literal['mamba', 'mamba2', 'vmamba'] = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.mamba_type = mamba_type
        
        # Atrous rates based on output stride
        if output_stride == 16:
            atrous_rates = (6, 12, 18)
        else:
            atrous_rates = (12, 24, 36)
        
        # Load backbone
        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
            high_level_channels = 2048
        else:
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet101(weights=weights)
            high_level_channels = 2048
        
        low_level_channels = 256
        
        # Modify first conv
        if in_channels != 3:
            self.input_conv = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                with torch.no_grad():
                    self.input_conv.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
                    if in_channels > 1:
                        self.input_conv.weight.data = self.input_conv.weight.data.repeat(1, in_channels, 1, 1)
        else:
            self.input_conv = resnet.conv1
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        
        # Modify layer3 and layer4 for dilated convolutions
        if output_stride == 16:
            self.layer3 = resnet.layer3
            self.layer4 = self._make_dilated(resnet.layer4, dilation=2)
        else:
            self.layer3 = self._make_dilated(resnet.layer3, dilation=2)
            self.layer4 = self._make_dilated(resnet.layer4, dilation=4)
        
        # Mamba-enhanced DeepLab head
        self.head = MambaDeepLabHead(
            in_channels=high_level_channels,
            low_level_channels=low_level_channels,
            num_classes=num_classes,
            aspp_channels=256,
            atrous_rates=atrous_rates,
            mamba_type=mamba_type,
            d_state=d_state
        )
        
        self._init_head_weights()
    
    def _make_dilated(self, layer: nn.Module, dilation: int) -> nn.Module:
        """Convert layer to use dilated convolutions."""
        for name, module in layer.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.kernel_size == (3, 3):
                    module.dilation = (dilation, dilation)
                    module.padding = (dilation, dilation)
                    module.stride = (1, 1)
                # Also fix stride in 1x1 convs for the first block (downsample)
                elif module.stride == (2, 2):
                    module.stride = (1, 1)
        return layer
    
    def _init_head_weights(self):
        """Initialize head weights."""
        for m in self.head.modules():
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
        
        # Backbone
        x = self.input_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        high_level = self.layer4(x)
        
        # Mamba-DeepLab head
        x = self.head(high_level, low_level)
        
        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Convenience functions
def mamba_deeplabv3_resnet50(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba',
    pretrained: bool = True
) -> MambaDeepLab:
    """Mamba-DeepLabV3 with ResNet50."""
    return MambaDeepLab(
        in_channels, num_classes,
        backbone='resnet50',
        pretrained=pretrained,
        mamba_type=mamba_type
    )


def mamba_deeplabv3_resnet101(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba',
    pretrained: bool = True
) -> MambaDeepLab:
    """Mamba-DeepLabV3 with ResNet101."""
    return MambaDeepLab(
        in_channels, num_classes,
        backbone='resnet101',
        pretrained=pretrained,
        mamba_type=mamba_type
    )


if __name__ == '__main__':
    # Test the model
    model = MambaDeepLab(
        in_channels=1, num_classes=4,
        backbone='resnet50', pretrained=False,
        mamba_type='vmamba'
    )
    print(f"Mamba-DeepLabV3 Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
