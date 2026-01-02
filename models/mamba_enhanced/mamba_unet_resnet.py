"""
Mamba-UNet-ResNet - UNet with ResNet Encoder Enhanced by Mamba

Combines pretrained ResNet encoder with Mamba-enhanced decoder
for efficient feature extraction and global context modeling.

Integration points:
- Bottleneck: Mamba after ResNet encoder
- Decoder: Mamba-enhanced upsampling with skip fusion
- Optional: Mamba layers after specific ResNet stages
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Literal
from torchvision import models
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
)

import sys
sys.path.append('..')

from models.modules import (
    create_mamba_block,
    MambaBottleneck,
    GlobalContextMambaBottleneck,
    MambaSkipConnection,
    GatedMambaSkip
)


class MambaDecoderBlock(nn.Module):
    """Decoder block with Mamba-enhanced skip connection."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        mamba_type: str = 'vmamba',
        use_gated_skip: bool = True,
        d_state: int = 16
    ):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Mamba-enhanced skip
        if use_gated_skip:
            self.mamba_skip = GatedMambaSkip(
                encoder_channels=skip_channels,
                decoder_channels=in_channels // 2,
                mamba_type=mamba_type,
                d_state=d_state
            )
            fused_channels = in_channels // 2  # GatedMambaSkip outputs fused
        else:
            self.mamba_skip = MambaSkipConnection(
                encoder_channels=skip_channels,
                decoder_channels=in_channels // 2,
                mamba_type=mamba_type,
                d_state=d_state
            )
            fused_channels = skip_channels + in_channels // 2
        
        self.use_gated = use_gated_skip
        
        self.conv = nn.Sequential(
            nn.Conv2d(fused_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Mamba skip fusion
        if self.use_gated:
            fused = self.mamba_skip(skip, x)
        else:
            skip = self.mamba_skip(skip, x)
            fused = torch.cat([skip, x], dim=1)
        
        return self.conv(fused)


class MambaUNetResNet(nn.Module):
    """
    Mamba-Enhanced UNet with ResNet Encoder.
    
    Uses pretrained ResNet for encoding with Mamba enhancement at:
    - Post-encoder stages (optional)
    - Global context bottleneck
    - Skip connection refinement
    - Decoder upsampling
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101')
        pretrained: Use ImageNet pretrained weights
        mamba_type: Type of Mamba ('mamba', 'mamba2', 'vmamba')
        mamba_after_encoder: Add Mamba after encoder stages
        use_global_bottleneck: Use global context Mamba bottleneck
        use_gated_skip: Use gated Mamba skip connections
        freeze_encoder: Freeze encoder weights
        d_state: SSM state dimension
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
        backbone: Literal['resnet18', 'resnet34', 'resnet50', 'resnet101'] = 'resnet34',
        pretrained: bool = True,
        mamba_type: Literal['mamba', 'mamba2', 'vmamba'] = 'vmamba',
        mamba_after_encoder: bool = False,
        use_global_bottleneck: bool = True,
        use_gated_skip: bool = True,
        freeze_encoder: bool = False,
        d_state: int = 16
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mamba_type = mamba_type
        
        encoder_channels = self.ENCODER_CHANNELS[backbone]
        
        # Load ResNet backbone
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
        
        # Optional Mamba after encoder stages
        self.encoder_mamba = nn.ModuleList()
        if mamba_after_encoder:
            for ch in encoder_channels[2:]:  # After layer2, layer3, layer4
                self.encoder_mamba.append(
                    create_mamba_block(variant=mamba_type, dim=ch, d_state=d_state)
                )
        
        # Global context Mamba bottleneck
        if use_global_bottleneck:
            self.bottleneck = GlobalContextMambaBottleneck(
                dim=encoder_channels[4],
                mamba_type=mamba_type,
                d_state=d_state
            )
        else:
            self.bottleneck = MambaBottleneck(
                dim=encoder_channels[4],
                mamba_type=mamba_type,
                d_state=d_state
            )
        
        # Decoder
        decoder_channels = [256, 128, 64, 32]
        
        self.decoder4 = MambaDecoderBlock(
            encoder_channels[4], encoder_channels[3], decoder_channels[0],
            mamba_type=mamba_type, use_gated_skip=use_gated_skip, d_state=d_state
        )
        self.decoder3 = MambaDecoderBlock(
            decoder_channels[0], encoder_channels[2], decoder_channels[1],
            mamba_type=mamba_type, use_gated_skip=use_gated_skip, d_state=d_state
        )
        self.decoder2 = MambaDecoderBlock(
            decoder_channels[1], encoder_channels[1], decoder_channels[2],
            mamba_type=mamba_type, use_gated_skip=use_gated_skip, d_state=d_state
        )
        self.decoder1 = MambaDecoderBlock(
            decoder_channels[2], encoder_channels[0], decoder_channels[3],
            mamba_type=mamba_type, use_gated_skip=use_gated_skip, d_state=d_state
        )
        
        # Final output
        self.final_up = nn.ConvTranspose2d(decoder_channels[3], decoder_channels[3], kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()
        
        self._init_decoder_weights()
    
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
    
    def _freeze_encoder(self):
        """Freeze encoder weights."""
        for param in self.input_conv.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def _init_decoder_weights(self):
        """Initialize decoder weights."""
        for module in [self.decoder4, self.decoder3, self.decoder2, 
                       self.decoder1, self.final_conv]:
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
        x0 = self.input_conv(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)  # /2
        
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)  # /4
        
        x2 = self.layer2(x1)  # /8
        if len(self.encoder_mamba) > 0:
            x2 = x2 + self.encoder_mamba[0](x2)
        
        x3 = self.layer3(x2)  # /16
        if len(self.encoder_mamba) > 1:
            x3 = x3 + self.encoder_mamba[1](x3)
        
        x4 = self.layer4(x3)  # /32
        if len(self.encoder_mamba) > 2:
            x4 = x4 + self.encoder_mamba[2](x4)
        
        # Bottleneck
        x4 = self.bottleneck(x4)
        
        # Decoder
        d4 = self.decoder4(x4, x3)  # /16
        d3 = self.decoder3(d4, x2)  # /8
        d2 = self.decoder2(d3, x1)  # /4
        d1 = self.decoder1(d2, x0)  # /2
        
        # Final
        out = self.final_up(d1)  # /1
        out = self.final_conv(out)
        
        # Ensure output matches input size
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        return out
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Convenience functions
def mamba_unet_resnet18(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba',
    pretrained: bool = True
) -> MambaUNetResNet:
    """Mamba-UNet with ResNet18 encoder."""
    return MambaUNetResNet(
        in_channels, num_classes,
        backbone='resnet18',
        pretrained=pretrained,
        mamba_type=mamba_type
    )


def mamba_unet_resnet34(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba',
    pretrained: bool = True
) -> MambaUNetResNet:
    """Mamba-UNet with ResNet34 encoder."""
    return MambaUNetResNet(
        in_channels, num_classes,
        backbone='resnet34',
        pretrained=pretrained,
        mamba_type=mamba_type
    )


def mamba_unet_resnet50(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba',
    pretrained: bool = True
) -> MambaUNetResNet:
    """Mamba-UNet with ResNet50 encoder."""
    return MambaUNetResNet(
        in_channels, num_classes,
        backbone='resnet50',
        pretrained=pretrained,
        mamba_type=mamba_type
    )


if __name__ == '__main__':
    # Test the model
    model = MambaUNetResNet(
        in_channels=1, num_classes=4,
        backbone='resnet34', pretrained=False,
        mamba_type='vmamba'
    )
    print(f"Mamba-UNet-ResNet34 Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
