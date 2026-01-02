"""
TransUNet - Transformers for Medical Image Segmentation

Implementation based on "TransUNet: Transformers Make Strong Encoders for 
Medical Image Segmentation" by Chen et al., 2021.

Features:
- CNN encoder (ResNet) for initial feature extraction
- Vision Transformer for global context modeling
- CNN decoder for upsampling
- Skip connections from CNN encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Literal
from torchvision import models
from torchvision.models import ResNet50_Weights
from einops import rearrange


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for transformer.
    
    Args:
        seq_len: Maximum sequence length
        embed_dim: Embedding dimension
    """
    
    def __init__(self, seq_len: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed[:, :x.size(1)]


class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer encoder block.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        drop: Dropout rate
        attn_drop: Attention dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=attn_drop,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for feature encoding.
    
    Args:
        in_channels: Input feature channels (from CNN encoder)
        embed_dim: Transformer embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        drop: Dropout rate
        attn_drop: Attention dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        max_seq_len: int = 256
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Linear projection if dimensions don't match
        self.proj = nn.Linear(in_channels, embed_dim) if in_channels != embed_dim else nn.Identity()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(max_seq_len, embed_dim)
        self.pos_drop = nn.Dropout(p=drop)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim, num_heads, mlp_ratio, drop, attn_drop
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, H*W, C)
            
        Returns:
            Output (B, H*W, embed_dim)
        """
        # Project to embedding dimension
        x = self.proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.pos_drop(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        return x


class DecoderBlock(nn.Module):
    """
    CNN decoder block for TransUNet.
    
    Args:
        in_channels: Input channels
        skip_channels: Skip connection channels
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
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


class TransUNet(nn.Module):
    """
    TransUNet: Hybrid CNN-Transformer for Medical Image Segmentation.
    
    Architecture:
    1. CNN encoder (ResNet) extracts hierarchical features
    2. Features are flattened and processed by Vision Transformer
    3. Transformer output is reshaped and upsampled by CNN decoder
    4. Skip connections from encoder to decoder
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        img_size: Input image size (must be divisible by 16)
        backbone: CNN backbone ('resnet50')
        pretrained: Use pretrained backbone
        vit_layers: Number of ViT layers
        vit_heads: Number of ViT attention heads
        vit_dim: ViT embedding dimension
        mlp_ratio: ViT MLP ratio
        drop_rate: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        img_size: int = 256,
        backbone: Literal['resnet50'] = 'resnet50',
        pretrained: bool = True,
        vit_layers: int = 12,
        vit_heads: int = 12,
        vit_dim: int = 768,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        
        # CNN Encoder (ResNet50)
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        
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
        
        self.layer1 = resnet.layer1  # 256 channels, /4
        self.layer2 = resnet.layer2  # 512 channels, /8
        self.layer3 = resnet.layer3  # 1024 channels, /16
        
        # Skip channels for decoder
        self.skip_channels = [256, 512, 1024]
        
        # Transformer encoder
        # Input to transformer: 1024 channels from layer3
        self.transformer = VisionTransformer(
            in_channels=1024,
            embed_dim=vit_dim,
            num_layers=vit_layers,
            num_heads=vit_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            max_seq_len=(img_size // 16) ** 2
        )
        
        # Project transformer output back to spatial features
        self.trans_proj = nn.Sequential(
            nn.Linear(vit_dim, 1024),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        # Start from 1024 channels at /16 resolution
        decoder_channels = [512, 256, 128, 64]
        
        self.decoder3 = DecoderBlock(1024, self.skip_channels[1], decoder_channels[0])  # /16 -> /8
        self.decoder2 = DecoderBlock(decoder_channels[0], self.skip_channels[0], decoder_channels[1])  # /8 -> /4
        self.decoder1 = DecoderBlock(decoder_channels[1], 64, decoder_channels[2])  # /4 -> /2
        self.decoder0 = DecoderBlock(decoder_channels[2], 0, decoder_channels[3])  # /2 -> /1
        
        # Final output
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
        self._init_decoder()
    
    def _init_decoder(self):
        """Initialize decoder weights."""
        for module in [self.decoder3, self.decoder2, self.decoder1, 
                       self.decoder0, self.final_conv, self.trans_proj]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, num_classes, H, W)
        """
        B, C, H_in, W_in = x.shape
        
        # CNN Encoder
        # Initial block: /2
        x0 = self.input_conv(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        
        # /4
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)  # 256 channels
        
        # /8
        x2 = self.layer2(x1)  # 512 channels
        
        # /16
        x3 = self.layer3(x2)  # 1024 channels
        
        # Flatten for transformer
        B, C_feat, H_feat, W_feat = x3.shape
        x_flat = rearrange(x3, 'b c h w -> b (h w) c')
        
        # Transformer
        x_trans = self.transformer(x_flat)
        
        # Project back and reshape
        x_trans = self.trans_proj(x_trans)
        x_trans = rearrange(x_trans, 'b (h w) c -> b c h w', h=H_feat, w=W_feat)
        
        # Decoder with skip connections
        d3 = self.decoder3(x_trans, x2)  # /8
        d2 = self.decoder2(d3, x1)       # /4
        d1 = self.decoder1(d2, x0)       # /2
        
        # Final decoder (no skip)
        d0 = self.decoder0(d1, None)     # /1
        
        # Output
        out = self.final_conv(d0)
        
        # Ensure output matches input size
        if out.shape[2:] != (H_in, W_in):
            out = F.interpolate(out, size=(H_in, W_in), mode='bilinear', align_corners=True)
        
        return out
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention maps from transformer layers.
        Useful for visualization and interpretability.
        
        Note: Requires modifying forward to store attention weights.
        """
        # This is a placeholder - would need to modify transformer
        # to return attention weights
        raise NotImplementedError("Attention map extraction not yet implemented")
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Convenience functions
def transunet_base(
    in_channels: int = 1,
    num_classes: int = 4,
    img_size: int = 224,
    pretrained: bool = True
) -> TransUNet:
    """Base TransUNet with 12 ViT layers."""
    return TransUNet(
        in_channels, num_classes, img_size,
        pretrained=pretrained,
        vit_layers=12, vit_heads=12, vit_dim=768
    )


def transunet_small(
    in_channels: int = 1,
    num_classes: int = 4,
    img_size: int = 224,
    pretrained: bool = True
) -> TransUNet:
    """Small TransUNet with 6 ViT layers."""
    return TransUNet(
        in_channels, num_classes, img_size,
        pretrained=pretrained,
        vit_layers=6, vit_heads=8, vit_dim=512
    )


def transunet_large(
    in_channels: int = 1,
    num_classes: int = 4,
    img_size: int = 224,
    pretrained: bool = True
) -> TransUNet:
    """Large TransUNet with 24 ViT layers."""
    return TransUNet(
        in_channels, num_classes, img_size,
        pretrained=pretrained,
        vit_layers=24, vit_heads=16, vit_dim=1024
    )


if __name__ == '__main__':
    # Test the model
    model = TransUNet(in_channels=1, num_classes=4, img_size=224, pretrained=False)
    print(f"TransUNet Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test different sizes
    x256 = torch.randn(2, 1, 256, 256)
    model256 = TransUNet(in_channels=1, num_classes=4, img_size=256, pretrained=False)
    y256 = model256(x256)
    print(f"\n256x256 - Input: {x256.shape}, Output: {y256.shape}")
