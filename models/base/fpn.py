"""
FPN - Feature Pyramid Network for Semantic Segmentation

Implementation of FPN from "Feature Pyramid Networks for Object Detection"
by Lin et al., 2017, adapted for semantic segmentation.

Features:
- Top-down pathway with lateral connections
- Multi-scale feature fusion
- Can be used with any encoder backbone
- Panoptic FPN-style segmentation head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Literal, Dict
from torchvision import models
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
)


class FPNBlock(nn.Module):
    """
    FPN building block for top-down pathway.
    
    Args:
        in_channels: Input channels from encoder
        out_channels: Output channels (pyramid channels)
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Lateral connection (1x1 conv to reduce channels)
        self.lateral = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Smooth layer after addition
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(
        self,
        encoder_feature: torch.Tensor,
        upsampled_feature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            encoder_feature: Feature from encoder at this level
            upsampled_feature: Upsampled feature from higher level (coarser)
            
        Returns:
            FPN feature at this level
        """
        lateral = self.lateral(encoder_feature)
        
        if upsampled_feature is not None:
            # Upsample and add
            if lateral.shape[2:] != upsampled_feature.shape[2:]:
                upsampled_feature = F.interpolate(
                    upsampled_feature, size=lateral.shape[2:],
                    mode='bilinear', align_corners=True
                )
            lateral = lateral + upsampled_feature
        
        return self.smooth(lateral)


class FPNNeck(nn.Module):
    """
    FPN Neck module - can be attached to any encoder.
    
    Creates a multi-scale feature pyramid from encoder features.
    
    Args:
        encoder_channels: List of encoder feature channels (low to high level)
        fpn_channels: Number of channels in FPN features
        num_outs: Number of output levels (can add extra levels via pooling)
    """
    
    def __init__(
        self,
        encoder_channels: List[int],
        fpn_channels: int = 256,
        num_outs: Optional[int] = None
    ):
        super().__init__()
        
        self.num_ins = len(encoder_channels)
        self.num_outs = num_outs or self.num_ins
        
        # FPN blocks for each encoder level
        self.fpn_blocks = nn.ModuleList([
            FPNBlock(ch, fpn_channels)
            for ch in encoder_channels
        ])
        
        # Extra levels via pooling if needed
        if self.num_outs > self.num_ins:
            self.extra_levels = nn.ModuleList([
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=2, padding=1)
                for _ in range(self.num_outs - self.num_ins)
            ])
        else:
            self.extra_levels = None
    
    def forward(self, encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            encoder_features: List of encoder features [P2, P3, P4, P5] (low to high)
            
        Returns:
            List of FPN features [P2, P3, P4, P5, ...] (same order)
        """
        assert len(encoder_features) == self.num_ins
        
        # Top-down pathway
        fpn_features = [None] * self.num_ins
        
        # Start from highest level (coarsest)
        fpn_features[-1] = self.fpn_blocks[-1](encoder_features[-1], None)
        
        # Build top-down
        for i in range(self.num_ins - 2, -1, -1):
            fpn_features[i] = self.fpn_blocks[i](
                encoder_features[i],
                fpn_features[i + 1]
            )
        
        # Add extra levels if needed
        if self.extra_levels is not None:
            extra = fpn_features[-1]
            for level in self.extra_levels:
                extra = level(F.relu(extra))
                fpn_features.append(extra)
        
        return fpn_features


class FPNSegmentationHead(nn.Module):
    """
    Panoptic FPN-style segmentation head.
    
    Fuses all FPN levels and produces segmentation output.
    
    Args:
        fpn_channels: Number of channels in FPN features
        num_classes: Number of output classes
        num_levels: Number of FPN levels to fuse
    """
    
    def __init__(
        self,
        fpn_channels: int = 256,
        num_classes: int = 4,
        num_levels: int = 4
    ):
        super().__init__()
        
        self.num_levels = num_levels
        
        # Convs for each level before fusion
        self.level_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_levels)
        ])
        
        # Fusion and output
        self.fusion = nn.Sequential(
            nn.Conv2d(fpn_channels * num_levels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        
        self.output = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)
    
    def forward(
        self,
        fpn_features: List[torch.Tensor],
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
            fpn_features: List of FPN features
            target_size: Target spatial size (H, W)
            
        Returns:
            Segmentation output (B, num_classes, H, W)
        """
        # Process each level and upsample to target size
        processed = []
        for i, (feat, conv) in enumerate(zip(fpn_features[:self.num_levels], self.level_convs)):
            feat = conv(feat)
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            processed.append(feat)
        
        # Concatenate and fuse
        fused = torch.cat(processed, dim=1)
        fused = self.fusion(fused)
        
        return self.output(fused)


class FPNUNet(nn.Module):
    """
    FPN-UNet: Feature Pyramid Network with UNet-style architecture.
    
    Combines FPN's multi-scale feature fusion with UNet's
    encoder-decoder structure.
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        backbone: Encoder backbone ('resnet18', 'resnet34', 'resnet50', 'resnet101')
        pretrained: Use pretrained backbone
        fpn_channels: Number of FPN pyramid channels
    """
    
    ENCODER_CHANNELS = {
        'resnet18': [64, 64, 128, 256, 512],
        'resnet34': [64, 64, 128, 256, 512],
        'resnet50': [64, 256, 512, 1024, 2048],
        'resnet101': [64, 256, 512, 1024, 2048],
    }
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        backbone: Literal['resnet18', 'resnet34', 'resnet50', 'resnet101'] = 'resnet50',
        pretrained: bool = True,
        fpn_channels: int = 256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        encoder_channels = self.ENCODER_CHANNELS[backbone]
        
        # Load backbone
        resnet = self._get_resnet(backbone, pretrained)
        
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
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # FPN neck
        self.fpn = FPNNeck(
            encoder_channels=encoder_channels[1:],  # Skip first conv output
            fpn_channels=fpn_channels
        )
        
        # Segmentation head
        self.seg_head = FPNSegmentationHead(
            fpn_channels=fpn_channels,
            num_classes=num_classes,
            num_levels=4
        )
        
        self._init_fpn_weights()
    
    def _get_resnet(self, backbone: str, pretrained: bool):
        """Get ResNet model."""
        weights_map = {
            'resnet18': ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
            'resnet34': ResNet34_Weights.IMAGENET1K_V1 if pretrained else None,
            'resnet50': ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
            'resnet101': ResNet101_Weights.IMAGENET1K_V1 if pretrained else None,
        }
        model_map = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
        }
        return model_map[backbone](weights=weights_map[backbone])
    
    def _init_fpn_weights(self):
        """Initialize FPN and head weights."""
        for module in [self.fpn, self.seg_head]:
            for m in module.modules():
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
        
        # Encoder
        x = self.input_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = x  # /2
        
        x = self.maxpool(x)
        c2 = self.layer1(x)  # /4
        c3 = self.layer2(c2)  # /8
        c4 = self.layer3(c3)  # /16
        c5 = self.layer4(c4)  # /32
        
        # FPN
        fpn_features = self.fpn([c2, c3, c4, c5])
        
        # Segmentation
        out = self.seg_head(fpn_features, target_size=input_size)
        
        return out
    
    def get_fpn_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract FPN features for analysis."""
        x = self.relu(self.bn1(self.input_conv(x)))
        x = self.maxpool(x)
        
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        return self.fpn([c2, c3, c4, c5])
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Convenience functions
def fpn_resnet18(in_channels: int = 1, num_classes: int = 4, pretrained: bool = True) -> FPNUNet:
    """FPN with ResNet18 backbone."""
    return FPNUNet(in_channels, num_classes, backbone='resnet18', pretrained=pretrained)


def fpn_resnet34(in_channels: int = 1, num_classes: int = 4, pretrained: bool = True) -> FPNUNet:
    """FPN with ResNet34 backbone."""
    return FPNUNet(in_channels, num_classes, backbone='resnet34', pretrained=pretrained)


def fpn_resnet50(in_channels: int = 1, num_classes: int = 4, pretrained: bool = True) -> FPNUNet:
    """FPN with ResNet50 backbone."""
    return FPNUNet(in_channels, num_classes, backbone='resnet50', pretrained=pretrained)


if __name__ == '__main__':
    # Test the model
    model = FPNUNet(in_channels=1, num_classes=4, backbone='resnet50', pretrained=False)
    print(f"FPN-ResNet50 Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test FPN features
    fpn_feats = model.get_fpn_features(x)
    print("\nFPN feature shapes:")
    for i, f in enumerate(fpn_feats):
        print(f"  P{i+2}: {f.shape}")
