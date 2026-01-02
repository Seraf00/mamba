"""
Mamba-Swin-UNet - Swin Transformer UNet Enhanced with Mamba

Hybrid Swin Transformer and Mamba architecture combining:
- Swin Transformer's shifted window attention (local)
- Mamba's state space modeling (global/sequential)

Integration strategy:
- SwinMamba blocks: Alternating Swin attention and Mamba
- Mamba bottleneck for global context
- Hybrid skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Literal
import math

import sys
sys.path.append('..')

from models.modules import (
    create_mamba_block,
    MambaBottleneck,
    HybridAttentionMamba
)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinMambaBlock(nn.Module):
    """Swin Transformer block with Mamba enhancement."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        mamba_type: str = 'vmamba',
        d_state: int = 16,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        # Swin attention path
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(window_size, window_size),
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # MLP
        mlp_hidden = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop)
        )
        
        # Mamba for global context
        self.mamba = create_mamba_block(
            variant=mamba_type,
            dim=dim,
            d_state=d_state
        )
        
        # Fusion gate
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, H, W, C = x.shape
        residual = x
        
        # Swin attention path
        x_norm = self.norm1(x)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x_norm, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_norm
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Window attention
        attn_windows = self.attn(x_windows, mask=mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            attn_out = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_out = shifted_x
        
        # Mamba path (process in channel-first format)
        x_mamba = x_norm.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        mamba_out = self.mamba(x_mamba)
        mamba_out = mamba_out.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        
        # Gated fusion
        combined = torch.cat([attn_out, mamba_out], dim=-1)
        gate = self.gate(combined)
        x = residual + gate * attn_out + (1 - gate) * mamba_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    
    def __init__(
        self,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 96
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.view(B, H, W, C)
        return x, H, W


class PatchMerging(nn.Module):
    """Patch Merging Layer (downsampling)."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        
        # Padding if needed
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
        
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class PatchExpanding(nn.Module):
    """Patch Expanding Layer (upsampling)."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * 2, W * 2, C // 2)
        x = self.norm(x)
        return x


class MambaSwinEncoder(nn.Module):
    """Swin-Mamba Encoder."""
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mamba_type: str = 'vmamba',
        d_state: int = 16,
        downsample: bool = True
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            SwinMambaBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mamba_type=mamba_type,
                d_state=d_state
            )
            for i in range(depth)
        ])
        
        self.downsample = PatchMerging(dim) if downsample else None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x)
        
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x, skip


class MambaSwinDecoder(nn.Module):
    """Swin-Mamba Decoder."""
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mamba_type: str = 'vmamba',
        d_state: int = 16,
        upsample: bool = True
    ):
        super().__init__()
        
        self.upsample = PatchExpanding(dim * 2) if upsample else None
        
        # Concatenation projection
        self.concat_proj = nn.Linear(dim * 2, dim) if upsample else None
        
        self.blocks = nn.ModuleList([
            SwinMambaBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mamba_type=mamba_type,
                d_state=d_state
            )
            for i in range(depth)
        ])
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        
        if skip is not None and self.concat_proj is not None:
            # Handle size mismatch
            if x.shape[1:3] != skip.shape[1:3]:
                x = F.interpolate(
                    x.permute(0, 3, 1, 2), 
                    size=skip.shape[1:3], 
                    mode='bilinear', 
                    align_corners=True
                ).permute(0, 2, 3, 1)
            
            x = torch.cat([x, skip], dim=-1)
            x = self.concat_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        return x


class MambaSwinUNet(nn.Module):
    """
    Mamba-Swin-UNet: Hybrid Swin Transformer and Mamba for segmentation.
    
    Combines:
    - Swin Transformer's efficient shifted window attention
    - Mamba's state space modeling for global context
    - UNet-style encoder-decoder with skip connections
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        embed_dim: Initial embedding dimension
        depths: Depth at each stage
        num_heads: Number of attention heads per stage
        window_size: Window size for attention
        patch_size: Initial patch size
        mamba_type: Type of Mamba ('mamba', 'mamba2', 'vmamba')
        d_state: SSM state dimension
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 2, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        patch_size: int = 4,
        mamba_type: Literal['mamba', 'mamba2', 'vmamba'] = 'vmamba',
        d_state: int = 16,
        pretrained: bool = False,  # Ignored, included for API compatibility
        **kwargs  # Ignore other unknown arguments
    ):
        super().__init__()
        
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.mamba_type = mamba_type
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Encoder stages
        self.encoders = nn.ModuleList()
        for i in range(self.num_stages):
            dim = embed_dim * (2 ** i)
            self.encoders.append(
                MambaSwinEncoder(
                    dim=dim,
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mamba_type=mamba_type,
                    d_state=d_state,
                    downsample=(i < self.num_stages - 1)
                )
            )
        
        # Bottleneck with enhanced Mamba
        bottleneck_dim = embed_dim * (2 ** (self.num_stages - 1))
        self.bottleneck = MambaBottleneck(
            dim=bottleneck_dim,
            mamba_type=mamba_type,
            depth=2,
            d_state=d_state
        )
        
        # Decoder stages
        self.decoders = nn.ModuleList()
        for i in range(self.num_stages - 1, 0, -1):
            dim = embed_dim * (2 ** (i - 1))
            self.decoders.append(
                MambaSwinDecoder(
                    dim=dim,
                    depth=depths[i - 1],
                    num_heads=num_heads[i - 1],
                    window_size=window_size,
                    mamba_type=mamba_type,
                    d_state=d_state
                )
            )
        
        # Final projection
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        self.seg_head = nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1)
        
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
        
        # Patch embedding
        x, H, W = self.patch_embed(x)  # B, H, W, C
        
        # Encoder
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)
        
        # Bottleneck (convert to B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.bottleneck(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # Decoder
        skips = skips[:-1][::-1]
        for decoder, skip in zip(self.decoders, skips):
            x = decoder(x, skip)
        
        # Final upsampling
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.final_up(x)
        out = self.seg_head(x)
        
        if out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=True)
        
        return out
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience functions
def mamba_swin_unet_small(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaSwinUNet:
    """Small Mamba-Swin-UNet."""
    return MambaSwinUNet(
        in_channels, num_classes,
        embed_dim=48,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        mamba_type=mamba_type
    )


def mamba_swin_unet_base(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaSwinUNet:
    """Standard Mamba-Swin-UNet."""
    return MambaSwinUNet(
        in_channels, num_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        mamba_type=mamba_type
    )


def mamba_swin_unet_large(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaSwinUNet:
    """Large Mamba-Swin-UNet."""
    return MambaSwinUNet(
        in_channels, num_classes,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        mamba_type=mamba_type
    )


if __name__ == '__main__':
    # Test the model
    model = MambaSwinUNet(
        in_channels=1, num_classes=4,
        embed_dim=96,
        mamba_type='vmamba'
    )
    print(f"Mamba-Swin-UNet Parameters: {model.count_parameters():,}")
    
    x = torch.randn(2, 1, 224, 224)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
