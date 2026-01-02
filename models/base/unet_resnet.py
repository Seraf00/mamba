"""
UNet-ResNet - UNet with ResNet Encoder

U-Net architecture using pre-trained ResNet as encoder backbone.
Supports ResNet18, ResNet34, ResNet50, ResNet101, ResNet152.

Features:
- Pre-trained ImageNet weights for encoder
- Strong feature extraction from ResNet
- Flexible decoder matching encoder feature maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Literal
from torchvision import models
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights
)


class DecoderBlock(nn.Module):
    """
    Decoder block for UNet-ResNet.
    
    Args:
        in_channels: Input channels from previous decoder
        skip_channels: Channels from skip connection
        out_channels: Output channels
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int
    ):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        
        if skip is not None:
            # Handle size mismatch
            diff_h = skip.size(2) - x.size(2)
            diff_w = skip.size(3) - x.size(3)
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
            x = torch.cat([skip, x], dim=1)
        
        return self.conv(x)


class CenterBlock(nn.Module):
    """
    Center/bottleneck block.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetResNet(nn.Module):
    """
    UNet with ResNet encoder backbone.
    
    Uses pre-trained ResNet for encoding and custom decoder
    for upsampling with skip connections.
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        pretrained: Use ImageNet pretrained weights
        freeze_encoder: Freeze encoder weights
        
    Feature maps from ResNet:
        - conv1: H/2, 64 channels
        - layer1: H/4, 64/256 channels (resnet18/34 vs 50/101/152)
        - layer2: H/8, 128/512 channels
        - layer3: H/16, 256/1024 channels
        - layer4: H/32, 512/2048 channels
    """
    
    ENCODER_CHANNELS = {
        'resnet18': [64, 64, 128, 256, 512],
        'resnet34': [64, 64, 128, 256, 512],
        'resnet50': [64, 256, 512, 1024, 2048],
        'resnet101': [64, 256, 512, 1024, 2048],
        'resnet152': [64, 256, 512, 1024, 2048],
    }
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        backbone: Literal['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'] = 'resnet34',
        pretrained: bool = True,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Get encoder channels for this backbone
        encoder_channels = self.ENCODER_CHANNELS[backbone]
        
        # Load pretrained ResNet
        resnet = self._get_resnet(backbone, pretrained)
        
        # Modify first conv for different input channels
        if in_channels != 3:
            self.input_conv = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Initialize from pretrained weights if available
            if pretrained:
                with torch.no_grad():
                    # Average RGB weights for grayscale
                    self.input_conv.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
                    if in_channels > 1:
                        self.input_conv.weight.data = self.input_conv.weight.data.repeat(1, in_channels, 1, 1)
        else:
            self.input_conv = resnet.conv1
        
        self.input_bn = resnet.bn1
        self.input_relu = resnet.relu
        self.input_maxpool = resnet.maxpool
        
        # Encoder layers
        self.encoder1 = resnet.layer1  # H/4
        self.encoder2 = resnet.layer2  # H/8
        self.encoder3 = resnet.layer3  # H/16
        self.encoder4 = resnet.layer4  # H/32
        
        # Center block
        self.center = CenterBlock(encoder_channels[4], encoder_channels[4])
        
        # Decoder
        decoder_channels = [256, 128, 64, 32]
        
        self.decoder4 = DecoderBlock(
            encoder_channels[4], encoder_channels[3], decoder_channels[0]
        )
        self.decoder3 = DecoderBlock(
            decoder_channels[0], encoder_channels[2], decoder_channels[1]
        )
        self.decoder2 = DecoderBlock(
            decoder_channels[1], encoder_channels[1], decoder_channels[2]
        )
        self.decoder1 = DecoderBlock(
            decoder_channels[2], encoder_channels[0], decoder_channels[3]
        )
        
        # Final upsampling and output
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
        """Get ResNet model with appropriate weights."""
        weights_map = {
            'resnet18': ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
            'resnet34': ResNet34_Weights.IMAGENET1K_V1 if pretrained else None,
            'resnet50': ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
            'resnet101': ResNet101_Weights.IMAGENET1K_V1 if pretrained else None,
            'resnet152': ResNet152_Weights.IMAGENET1K_V1 if pretrained else None,
        }
        
        model_map = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
        }
        
        return model_map[backbone](weights=weights_map[backbone])
    
    def _freeze_encoder(self):
        """Freeze encoder weights for transfer learning."""
        for param in self.input_conv.parameters():
            param.requires_grad = False
        for param in self.input_bn.parameters():
            param.requires_grad = False
        for encoder in [self.encoder1, self.encoder2, self.encoder3, self.encoder4]:
            for param in encoder.parameters():
                param.requires_grad = False
    
    def _init_decoder_weights(self):
        """Initialize decoder weights."""
        for module in [self.center, self.decoder4, self.decoder3, 
                       self.decoder2, self.decoder1, self.final_conv]:
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
        # Store input size for final resize if needed
        input_size = x.shape[2:]
        
        # Encoder
        # Initial block: H -> H/2
        x0 = self.input_conv(x)
        x0 = self.input_bn(x0)
        x0 = self.input_relu(x0)
        
        # H/2 -> H/4
        x1 = self.input_maxpool(x0)
        x1 = self.encoder1(x1)
        
        # H/4 -> H/8
        x2 = self.encoder2(x1)
        
        # H/8 -> H/16
        x3 = self.encoder3(x2)
        
        # H/16 -> H/32
        x4 = self.encoder4(x3)
        
        # Center
        center = self.center(x4)
        
        # Decoder with skip connections
        d4 = self.decoder4(center, x3)  # H/16
        d3 = self.decoder3(d4, x2)       # H/8
        d2 = self.decoder2(d3, x1)       # H/4
        d1 = self.decoder1(d2, x0)       # H/2
        
        # Final upsampling
        out = self.final_up(d1)          # H
        out = self.final_conv(out)
        
        # Ensure output matches input size
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        return out
    
    def get_encoder_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract encoder features for analysis.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            List of encoder feature tensors
        """
        features = []
        
        x0 = self.input_relu(self.input_bn(self.input_conv(x)))
        features.append(x0)
        
        x1 = self.encoder1(self.input_maxpool(x0))
        features.append(x1)
        
        x2 = self.encoder2(x1)
        features.append(x2)
        
        x3 = self.encoder3(x2)
        features.append(x3)
        
        x4 = self.encoder4(x3)
        features.append(x4)
        
        return features
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Convenience functions for common configurations
def unet_resnet18(in_channels: int = 1, num_classes: int = 4, pretrained: bool = True) -> UNetResNet:
    """UNet with ResNet18 encoder."""
    return UNetResNet(in_channels, num_classes, backbone='resnet18', pretrained=pretrained)


def unet_resnet34(in_channels: int = 1, num_classes: int = 4, pretrained: bool = True) -> UNetResNet:
    """UNet with ResNet34 encoder."""
    return UNetResNet(in_channels, num_classes, backbone='resnet34', pretrained=pretrained)


def unet_resnet50(in_channels: int = 1, num_classes: int = 4, pretrained: bool = True) -> UNetResNet:
    """UNet with ResNet50 encoder."""
    return UNetResNet(in_channels, num_classes, backbone='resnet50', pretrained=pretrained)


def unet_resnet101(in_channels: int = 1, num_classes: int = 4, pretrained: bool = True) -> UNetResNet:
    """UNet with ResNet101 encoder."""
    return UNetResNet(in_channels, num_classes, backbone='resnet101', pretrained=pretrained)


if __name__ == '__main__':
    # Test the model
    model = UNetResNet(in_channels=1, num_classes=4, backbone='resnet34', pretrained=False)
    print(f"UNet-ResNet34 Parameters: {model.count_parameters():,}")
    print(f"Total Parameters: {model.count_parameters(trainable_only=False):,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test feature extraction
    features = model.get_encoder_features(x)
    print("\nEncoder feature shapes:")
    for i, f in enumerate(features):
        print(f"  Level {i}: {f.shape}")
    
    # Test different backbones
    for backbone in ['resnet18', 'resnet50']:
        m = UNetResNet(1, 4, backbone=backbone, pretrained=False)
        print(f"\n{backbone} params: {m.count_parameters():,}")
