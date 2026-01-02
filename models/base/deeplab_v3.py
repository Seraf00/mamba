"""
DeepLab V3 - Atrous Spatial Pyramid Pooling for Semantic Segmentation

Implementation of DeepLabV3 from "Rethinking Atrous Convolution for 
Semantic Image Segmentation" by Chen et al., 2017.

Features:
- ASPP (Atrous Spatial Pyramid Pooling) module
- Multi-scale feature extraction with dilated convolutions
- ResNet backbone with dilated convolutions
- Global context via image-level features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Literal
from torchvision import models
from torchvision.models import (
    ResNet50_Weights, ResNet101_Weights
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


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    
    Captures multi-scale context using parallel dilated convolutions
    with different rates plus global image pooling.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels for each branch
        atrous_rates: Tuple of dilation rates for ASPP convolutions
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        atrous_rates: Tuple[int, ...] = (6, 12, 18)
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
        
        self.convs = nn.ModuleList(modules)
        
        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, 
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for conv in self.convs:
            features.append(conv(x))
        features = torch.cat(features, dim=1)
        return self.project(features)


class DeepLabHead(nn.Module):
    """
    DeepLabV3 head with ASPP and decoder.
    
    Args:
        in_channels: Input channels from backbone
        low_level_channels: Channels from low-level features
        num_classes: Number of output classes
        aspp_channels: Output channels of ASPP
    """
    
    def __init__(
        self,
        in_channels: int,
        low_level_channels: int,
        num_classes: int,
        aspp_channels: int = 256,
        atrous_rates: Tuple[int, ...] = (6, 12, 18)
    ):
        super().__init__()
        
        # ASPP
        self.aspp = ASPP(in_channels, aspp_channels, atrous_rates)
        
        # Low-level feature processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
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
        """
        Args:
            high_level: High-level features from backbone (1/16 or 1/8)
            low_level: Low-level features for skip (1/4)
            
        Returns:
            Predictions at 1/4 resolution
        """
        # ASPP on high-level features
        x = self.aspp(high_level)
        
        # Upsample to low-level feature size
        x = F.interpolate(
            x, size=low_level.shape[2:],
            mode='bilinear', align_corners=True
        )
        
        # Process low-level features
        low_level = self.low_level_conv(low_level)
        
        # Concatenate and decode
        x = torch.cat([x, low_level], dim=1)
        x = self.decoder(x)
        
        return x


class DeepLabV3(nn.Module):
    """
    DeepLabV3 with ResNet backbone.
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        backbone: ResNet variant ('resnet50', 'resnet101')
        pretrained: Use ImageNet pretrained weights
        output_stride: Output stride (8 or 16)
        atrous_rates: Dilation rates for ASPP
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        backbone: Literal['resnet50', 'resnet101'] = 'resnet50',
        pretrained: bool = True,
        output_stride: Literal[8, 16] = 16,
        atrous_rates: Optional[Tuple[int, ...]] = None
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.output_stride = output_stride
        
        # Default atrous rates based on output stride
        if atrous_rates is None:
            if output_stride == 16:
                atrous_rates = (6, 12, 18)
            else:  # output_stride == 8
                atrous_rates = (12, 24, 36)
        
        # Load backbone
        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
            high_level_channels = 2048
        else:  # resnet101
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet101(weights=weights)
            high_level_channels = 2048
        
        low_level_channels = 256  # From layer1
        
        # Modify first conv for different input channels
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
        
        self.layer1 = resnet.layer1  # 1/4, 256 channels
        self.layer2 = resnet.layer2  # 1/8, 512 channels
        
        # Modify layer3 and layer4 for dilated convolutions
        if output_stride == 16:
            self.layer3 = resnet.layer3  # 1/16, 1024 channels
            self.layer4 = self._make_dilated(resnet.layer4, dilation=2)  # 1/16, 2048 channels
        else:  # output_stride == 8
            self.layer3 = self._make_dilated(resnet.layer3, dilation=2)  # 1/8
            self.layer4 = self._make_dilated(resnet.layer4, dilation=4)  # 1/8
        
        # DeepLab head
        self.head = DeepLabHead(
            in_channels=high_level_channels,
            low_level_channels=low_level_channels,
            num_classes=num_classes,
            aspp_channels=256,
            atrous_rates=atrous_rates
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
        
        low_level = self.layer1(x)  # 1/4
        x = self.layer2(low_level)  # 1/8
        x = self.layer3(x)          # 1/16 or 1/8
        high_level = self.layer4(x) # 1/16 or 1/8
        
        # DeepLab head
        x = self.head(high_level, low_level)
        
        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x
    
    def get_backbone_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract backbone features for analysis."""
        features = []
        
        x = self.relu(self.bn1(self.input_conv(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)
        
        return features
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Convenience functions
def deeplabv3_resnet50(in_channels: int = 1, num_classes: int = 4, pretrained: bool = True) -> DeepLabV3:
    """DeepLabV3 with ResNet50 backbone."""
    return DeepLabV3(in_channels, num_classes, backbone='resnet50', pretrained=pretrained)


def deeplabv3_resnet101(in_channels: int = 1, num_classes: int = 4, pretrained: bool = True) -> DeepLabV3:
    """DeepLabV3 with ResNet101 backbone."""
    return DeepLabV3(in_channels, num_classes, backbone='resnet101', pretrained=pretrained)


if __name__ == '__main__':
    # Test the model
    model = DeepLabV3(in_channels=1, num_classes=4, backbone='resnet50', pretrained=False)
    print(f"DeepLabV3-ResNet50 Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test backbone features
    features = model.get_backbone_features(x)
    print("\nBackbone feature shapes:")
    for i, f in enumerate(features):
        print(f"  Layer {i+1}: {f.shape}")
