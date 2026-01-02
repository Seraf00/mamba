"""
Mamba-TransUNet - TransUNet Enhanced with Mamba State Space Models

Hybrid CNN-Transformer-Mamba architecture combining:
- CNN encoder (ResNet) for local feature extraction
- Vision Transformer for global attention
- Mamba for efficient long-range dependencies

Integration strategy:
- Mamba in ViT blocks (hybrid attention-Mamba)
- Mamba skip connections
- Mamba-enhanced cascaded upsampler
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Literal
from torchvision import models
import math

import sys
sys.path.append('..')

from models.modules import (
    create_mamba_block,
    MambaBottleneck,
    MambaSkipConnection,
    HybridAttentionMamba
)


class MambaViTBlock(nn.Module):
    """Vision Transformer block enhanced with Mamba."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        mamba_type: str = 'vmamba',
        d_state: int = 16,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        
        # Multi-head self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=True
        )
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop)
        )
        
        # Mamba path
        self.norm3 = nn.LayerNorm(dim)
        self.mamba = create_mamba_block(
            variant=mamba_type,
            dim=dim,
            d_state=d_state
        )
        
        # Fusion gate
        self.gate = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, C) where N = H * W
            H, W: Spatial dimensions
        """
        # Self-attention path
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        # Mamba path (reshape to spatial format)
        B, N, C = x.shape
        x_spatial = self.norm3(x).view(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W
        mamba_out = self.mamba(x_spatial)
        mamba_out = mamba_out.permute(0, 2, 3, 1).view(B, N, C)  # B, N, C
        
        # Gated fusion
        gate = torch.sigmoid(self.gate)
        x = x + gate * mamba_out
        
        return x


class MambaViTEncoder(nn.Module):
    """Vision Transformer encoder with Mamba enhancement."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        mamba_type: str = 'vmamba',
        d_state: int = 16,
        drop: float = 0.0
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            MambaViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                mamba_type=mamba_type,
                d_state=d_state,
                drop=drop
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, H, W)
        return self.norm(x)


class MambaCascadedUpsampler(nn.Module):
    """Cascaded Upsampler with Mamba skip connections."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: List[int],
        out_channels: int = 256,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        self.num_stages = len(skip_channels)
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.mamba_skips = nn.ModuleList()
        
        current_channels = in_channels
        
        for i, skip_ch in enumerate(skip_channels):
            # Upsample
            self.ups.append(
                nn.ConvTranspose2d(current_channels, out_channels, kernel_size=2, stride=2)
            )
            
            # Mamba skip connection
            self.mamba_skips.append(
                MambaSkipConnection(
                    encoder_channels=skip_ch,
                    decoder_channels=out_channels,
                    mamba_type=mamba_type,
                    d_state=d_state
                )
            )
            
            # Fusion conv
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels + skip_ch, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
            current_channels = out_channels
    
    def forward(
        self,
        x: torch.Tensor,
        skips: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: Feature map from transformer (B, C, H, W)
            skips: Skip connections from CNN encoder (high to low resolution)
        """
        for up, conv, mamba_skip, skip in zip(
            self.ups, self.convs, self.mamba_skips, skips
        ):
            x = up(x)
            
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            # Mamba-enhanced skip
            skip = mamba_skip(skip, x)
            
            x = torch.cat([x, skip], dim=1)
            x = conv(x)
        
        return x


class MambaTransUNet(nn.Module):
    """
    Mamba-TransUNet: Hybrid CNN-Transformer-Mamba for segmentation.
    
    Architecture:
    - ResNet CNN encoder for local feature extraction
    - ViT + Mamba for global context
    - Cascaded upsampler with Mamba skip connections
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        img_size: Input image size (H, W) or int
        backbone: ResNet backbone ('resnet50', 'resnet101')
        embed_dim: Transformer embedding dimension
        num_heads: Number of attention heads
        num_transformer_layers: Number of transformer layers
        patch_size: Patch size for tokenization from CNN features
        mamba_type: Type of Mamba ('mamba', 'mamba2', 'vmamba')
        d_state: SSM state dimension
        pretrained: Use pretrained backbone
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        img_size: int = 256,
        backbone: str = 'resnet50',
        embed_dim: int = 768,
        num_heads: int = 12,
        num_transformer_layers: int = 12,
        patch_size: int = 1,
        mamba_type: Literal['mamba', 'mamba2', 'vmamba'] = 'vmamba',
        d_state: int = 16,
        pretrained: bool = True
    ):
        super().__init__()
        
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.embed_dim = embed_dim
        self.mamba_type = mamba_type
        
        # Input adaptation for single-channel
        self.input_adapter = nn.Conv2d(in_channels, 3, 1) if in_channels != 3 else nn.Identity()
        
        # CNN Encoder (ResNet)
        if backbone == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.cnn_channels = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
            self.cnn_channels = [256, 512, 1024, 2048]
        elif backbone == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            self.cnn_channels = [64, 128, 256, 512]
        else:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.cnn_channels = [64, 128, 256, 512]
        
        # CNN encoder stages
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
        
        # Patch embedding from CNN features
        cnn_out_channels = self.cnn_channels[-1]
        self.patch_embed = nn.Conv2d(cnn_out_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Calculate number of patches
        feat_size = self.img_size[0] // 32  # After ResNet downsampling
        num_patches = (feat_size // patch_size) ** 2
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Mamba-ViT Encoder
        self.transformer = MambaViTEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            mamba_type=mamba_type,
            d_state=d_state
        )
        
        # Additional Mamba bottleneck
        self.mamba_bottleneck = MambaBottleneck(
            dim=embed_dim,
            mamba_type=mamba_type,
            num_layers=2,
            d_state=d_state
        )
        
        # Reshape for decoder
        self.feat_size = feat_size // patch_size
        
        # Cascaded upsampler with Mamba skip connections
        skip_channels = self.cnn_channels[:-1][::-1]  # Reverse, exclude last
        self.decoder = MambaCascadedUpsampler(
            in_channels=embed_dim,
            skip_channels=skip_channels,
            out_channels=256,
            mamba_type=mamba_type,
            d_state=d_state
        )
        
        # Final upsampling and segmentation head
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
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
        
        # CNN Encoder
        x = self.stem(x)
        
        c1 = self.layer1(x)   # 1/4
        c2 = self.layer2(c1)  # 1/8
        c3 = self.layer3(c2)  # 1/16
        c4 = self.layer4(c3)  # 1/32
        
        # Patch embedding
        x = self.patch_embed(c4)  # B, embed_dim, H', W'
        B, C, H, W = x.shape
        
        # Flatten and add position embedding
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Mamba-ViT Transformer
        x = self.transformer(x, H, W)
        
        # Reshape back to spatial
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # Mamba bottleneck
        x = self.mamba_bottleneck(x)
        
        # Cascaded upsampler with skip connections
        skips = [c3, c2, c1]  # High to low resolution relative to decoder
        x = self.decoder(x, skips)
        
        # Final upsampling
        x = self.final_up(x)
        out = self.seg_head(x)
        
        if out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=True)
        
        return out
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience functions
def mamba_transunet_small(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaTransUNet:
    """Small Mamba-TransUNet with ResNet18 backbone."""
    return MambaTransUNet(
        in_channels, num_classes,
        backbone='resnet18',
        embed_dim=512,
        num_heads=8,
        num_transformer_layers=6,
        mamba_type=mamba_type
    )


def mamba_transunet_base(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaTransUNet:
    """Standard Mamba-TransUNet with ResNet50 backbone."""
    return MambaTransUNet(
        in_channels, num_classes,
        backbone='resnet50',
        embed_dim=768,
        num_heads=12,
        num_transformer_layers=12,
        mamba_type=mamba_type
    )


def mamba_transunet_large(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaTransUNet:
    """Large Mamba-TransUNet with ResNet101 backbone."""
    return MambaTransUNet(
        in_channels, num_classes,
        backbone='resnet101',
        embed_dim=1024,
        num_heads=16,
        num_transformer_layers=12,
        mamba_type=mamba_type
    )


if __name__ == '__main__':
    # Test the model
    model = MambaTransUNet(
        in_channels=1, num_classes=4,
        img_size=256,
        backbone='resnet50',
        embed_dim=768,
        mamba_type='vmamba',
        pretrained=False  # For testing
    )
    print(f"Mamba-TransUNet Parameters: {model.count_parameters():,}")
    
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
