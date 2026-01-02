"""
Mamba-FPN - Feature Pyramid Network Enhanced with Mamba State Space Models

FPN architecture with Mamba integration for improved
multi-scale feature fusion and long-range dependencies.

Integration strategy:
- Mamba in lateral connections for enhanced feature transformation
- Mamba in top-down pathway for better context propagation
- Mamba fusion module for multi-scale aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Literal, Dict, Union
from torchvision import models

import sys
sys.path.append('..')

from models.modules import (
    create_mamba_block,
    MambaBottleneck,
    MambaSkipConnection
)


class MambaLateralConnection(nn.Module):
    """Lateral connection with Mamba enhancement for FPN."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        # Standard 1x1 conv for channel reduction
        self.lateral = nn.Conv2d(in_channels, out_channels, 1)
        
        # Mamba for enhanced feature transformation
        self.mamba = create_mamba_block(
            variant=mamba_type,
            dim=out_channels,
            d_state=d_state
        )
        
        # Refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lateral(x)
        x = x + self.mamba(x)
        return self.refine(x)


class MambaTopDownPath(nn.Module):
    """Top-down pathway with Mamba for context propagation."""
    
    def __init__(
        self,
        channels: int,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        # Mamba for temporal/spatial context in top-down flow
        self.mamba = create_mamba_block(
            variant=mamba_type,
            dim=channels,
            d_state=d_state
        )
        
        # Fusion conv
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, top: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        # Upsample top feature
        top_up = F.interpolate(top, size=lateral.shape[2:], mode='nearest')
        
        # Mamba enhancement on upsampled features
        top_mamba = top_up + self.mamba(top_up)
        
        # Fusion
        fused = self.fusion(torch.cat([top_mamba, lateral], dim=1))
        
        return fused


class MambaFPNFusion(nn.Module):
    """Multi-scale fusion module with Mamba."""
    
    def __init__(
        self,
        channels: int,
        num_levels: int = 4,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        self.num_levels = num_levels
        
        # Per-level Mamba processing
        self.level_mamba = nn.ModuleList([
            create_mamba_block(
                variant=mamba_type,
                dim=channels,
                d_state=d_state
            )
            for _ in range(num_levels)
        ])
        
        # Cross-scale Mamba after concatenation
        self.cross_mamba = create_mamba_block(
            variant=mamba_type,
            dim=channels * num_levels,
            d_state=d_state
        )
        
        # Final projection
        self.proj = nn.Sequential(
            nn.Conv2d(channels * num_levels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: List[torch.Tensor], target_size: tuple) -> torch.Tensor:
        # Process each level with Mamba
        processed = []
        for feat, mamba in zip(features, self.level_mamba):
            feat = feat + mamba(feat)
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            processed.append(feat)
        
        # Concatenate and cross-scale Mamba
        x = torch.cat(processed, dim=1)
        x = x + self.cross_mamba(x)
        
        return self.proj(x)


class MambaFPNNeck(nn.Module):
    """FPN Neck with Mamba enhancement."""
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels
        
        # Lateral connections with Mamba
        self.laterals = nn.ModuleList([
            MambaLateralConnection(
                in_ch, out_channels,
                mamba_type=mamba_type,
                d_state=d_state
            )
            for in_ch in in_channels_list
        ])
        
        # Top-down pathways with Mamba
        self.top_downs = nn.ModuleList([
            MambaTopDownPath(
                out_channels,
                mamba_type=mamba_type,
                d_state=d_state
            )
            for _ in range(self.num_levels - 1)
        ])
        
        # Output convs
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(self.num_levels)
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps [C2, C3, C4, C5] from backbone
            
        Returns:
            List of FPN feature maps [P2, P3, P4, P5]
        """
        # Lateral connections
        laterals = [lat(feat) for lat, feat in zip(self.laterals, features)]
        
        # Top-down pathway
        for i in range(self.num_levels - 1, 0, -1):
            laterals[i - 1] = self.top_downs[self.num_levels - 1 - i](
                laterals[i], laterals[i - 1]
            )
        
        # Output
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        
        return outputs


class MambaFPNHead(nn.Module):
    """Segmentation head for Mamba-FPN."""
    
    def __init__(
        self,
        in_channels: int,
        num_levels: int,
        num_classes: int,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        # Mamba fusion across scales
        self.fusion = MambaFPNFusion(
            in_channels,
            num_levels=num_levels,
            mamba_type=mamba_type,
            d_state=d_state
        )
        
        # Upsampling and refinement
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Final Mamba for semantic consistency
        self.final_mamba = create_mamba_block(
            variant=mamba_type,
            dim=in_channels // 4,
            d_state=d_state
        )
        
        self.seg_head = nn.Conv2d(in_channels // 4, num_classes, 1)
    
    def forward(
        self,
        features: List[torch.Tensor],
        target_size: tuple
    ) -> torch.Tensor:
        # Get the highest resolution feature size
        finest_size = features[0].shape[2:]
        
        # Fuse all scales
        x = self.fusion(features, finest_size)
        
        # Upsample
        x = self.upsample(x)
        
        # Final Mamba
        x = x + self.final_mamba(x)
        
        # Segmentation
        out = self.seg_head(x)
        
        if out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=True)
        
        return out


class MambaFPN(nn.Module):
    """
    Mamba-FPN: Feature Pyramid Network with Mamba Enhancement.
    
    Architecture:
    - ResNet backbone for multi-scale features
    - FPN neck with Mamba lateral and top-down connections
    - Mamba fusion across pyramid levels
    - Mamba-enhanced segmentation head
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        backbone: ResNet backbone ('resnet18', 'resnet34', 'resnet50', 'resnet101')
        fpn_channels: FPN output channels
        mamba_type: Type of Mamba ('mamba', 'mamba2', 'vmamba')
        d_state: SSM state dimension
        pretrained: Use pretrained backbone
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        backbone: str = 'resnet50',
        fpn_channels: int = 256,
        mamba_type: Literal['mamba', 'mamba2', 'vmamba'] = 'vmamba',
        d_state: int = 16,
        pretrained: bool = True
    ):
        super().__init__()
        
        self.mamba_type = mamba_type
        
        # Input adapter for single channel
        self.input_adapter = nn.Conv2d(in_channels, 3, 1) if in_channels != 3 else nn.Identity()
        
        # Backbone
        if backbone == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.backbone_channels = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
            self.backbone_channels = [256, 512, 1024, 2048]
        elif backbone == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            self.backbone_channels = [64, 128, 256, 512]
        else:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.backbone_channels = [64, 128, 256, 512]
        
        # Extract backbone stages
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Mamba bottleneck
        self.bottleneck = MambaBottleneck(
            dim=self.backbone_channels[-1],
            mamba_type=mamba_type,
            num_layers=2,
            d_state=d_state
        )
        
        # FPN Neck with Mamba
        self.fpn = MambaFPNNeck(
            in_channels_list=self.backbone_channels,
            out_channels=fpn_channels,
            mamba_type=mamba_type,
            d_state=d_state
        )
        
        # Segmentation Head with Mamba fusion
        self.head = MambaFPNHead(
            in_channels=fpn_channels,
            num_levels=4,
            num_classes=num_classes,
            mamba_type=mamba_type,
            d_state=d_state
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize non-pretrained weights."""
        for m in [self.fpn, self.head]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, num_classes, H, W)
        """
        target_size = x.shape[2:]
        
        # Input adaptation
        x = self.input_adapter(x)
        
        # Backbone
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        # Mamba bottleneck
        c4 = self.bottleneck(c4)
        
        # FPN
        fpn_features = self.fpn([c1, c2, c3, c4])
        
        # Segmentation head
        out = self.head(fpn_features, target_size)
        
        return out
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience functions
def mamba_fpn_resnet18(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaFPN:
    """Mamba-FPN with ResNet18 backbone."""
    return MambaFPN(
        in_channels, num_classes,
        backbone='resnet18',
        fpn_channels=128,
        mamba_type=mamba_type
    )


def mamba_fpn_resnet34(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaFPN:
    """Mamba-FPN with ResNet34 backbone."""
    return MambaFPN(
        in_channels, num_classes,
        backbone='resnet34',
        fpn_channels=128,
        mamba_type=mamba_type
    )


def mamba_fpn_resnet50(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaFPN:
    """Mamba-FPN with ResNet50 backbone."""
    return MambaFPN(
        in_channels, num_classes,
        backbone='resnet50',
        fpn_channels=256,
        mamba_type=mamba_type
    )


def mamba_fpn_resnet101(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaFPN:
    """Mamba-FPN with ResNet101 backbone."""
    return MambaFPN(
        in_channels, num_classes,
        backbone='resnet101',
        fpn_channels=256,
        mamba_type=mamba_type
    )


if __name__ == '__main__':
    # Test the model
    model = MambaFPN(
        in_channels=1, num_classes=4,
        backbone='resnet50',
        mamba_type='vmamba',
        pretrained=False
    )
    print(f"Mamba-FPN Parameters: {model.count_parameters():,}")
    
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
