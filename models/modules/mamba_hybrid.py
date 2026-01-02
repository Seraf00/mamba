"""
Hybrid Attention-Mamba Modules for Segmentation Networks.

This module provides hybrid architectures that combine:
1. Self-attention mechanisms (local/global)
2. Mamba State Space Models

The goal is to leverage the best of both worlds:
- Attention: Flexible, content-based interactions
- Mamba: Efficient long-range modeling with linear complexity

These hybrid modules can replace Transformer blocks in architectures
like Swin-UNet, TransUNet, etc.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .mamba_block import (
    MambaLayer,
    Mamba2Layer,
    VMMambaBlock,
    MambaBlock,
    DropPath,
    LayerNorm2d,
    RMSNorm,
    create_mamba_block
)


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention.
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projection
        attn_drop: Attention dropout rate
        proj_drop: Output projection dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, N, C) where N = H*W
            
        Returns:
            Output (B, N, C)
        """
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self-Attention (from Swin Transformer).
    
    Args:
        dim: Feature dimension
        window_size: Window size for local attention
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias
        attn_drop: Attention dropout
        proj_drop: Projection dropout
    """
    
    def __init__(
        self,
        dim: int,
        window_size: int = 7,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Get relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input (B*num_windows, window_size*window_size, C)
            mask: Attention mask for shifted windows
            
        Returns:
            Output (B*num_windows, window_size*window_size, C)
        """
        B_, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """Feed-forward MLP block."""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or dim * 4
        out_dim = out_dim or dim
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HybridAttentionMamba(nn.Module):
    """
    Hybrid block combining Self-Attention and Mamba.
    
    Processes features through both attention and Mamba in parallel
    or sequentially, then fuses the results.
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        mamba_type: Type of Mamba block
        fusion_mode: 'parallel', 'sequential', 'residual'
        attn_ratio: Ratio of attention path vs Mamba path (for parallel)
        window_size: Window size for local attention (None = global)
        d_state: SSM state dimension
        mlp_ratio: MLP expansion ratio
        drop_path: Stochastic depth rate
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mamba_type: str = 'vmamba',
        fusion_mode: str = 'parallel',
        attn_ratio: float = 0.5,
        window_size: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.fusion_mode = fusion_mode
        self.attn_ratio = attn_ratio
        self.window_size = window_size
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Attention branch
        if window_size is not None:
            self.attn = WindowAttention(
                dim=dim,
                window_size=window_size,
                num_heads=num_heads,
                attn_drop=drop,
                proj_drop=drop
            )
        else:
            self.attn = MultiHeadAttention(
                dim=dim,
                num_heads=num_heads,
                attn_drop=drop,
                proj_drop=drop
            )
        
        # Mamba branch
        self.mamba = create_mamba_block(
            variant=mamba_type,
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs
        )
        
        # Fusion layer for parallel mode
        if fusion_mode == 'parallel':
            self.fusion = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim)
            )
        
        # MLP
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = MLP(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            drop=drop
        )
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def _window_partition(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """Partition into windows for window attention."""
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        
        # Pad if needed
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        Hp, Wp = H + pad_h, W + pad_w
        
        # Partition
        x = x.view(B, Hp // self.window_size, self.window_size,
                   Wp // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, self.window_size * self.window_size, C)
        
        return x, Hp, Wp
    
    def _window_reverse(self, x: torch.Tensor, H: int, W: int, Hp: int, Wp: int) -> torch.Tensor:
        """Reverse window partition."""
        B = x.shape[0] // (Hp // self.window_size * Wp // self.window_size)
        
        x = x.view(B, Hp // self.window_size, Wp // self.window_size,
                   self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, Hp, Wp, -1)
        
        # Remove padding
        if Hp > H or Wp > W:
            x = x[:, :H, :W, :]
        
        x = x.view(B, H * W, -1)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, C, H, W)
            
        Returns:
            Output (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape to sequence
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Branch processing
        if self.fusion_mode == 'parallel':
            # Parallel: both branches process normalized input
            x_norm = self.norm1(x)
            
            # Attention branch
            if self.window_size is not None:
                x_win, Hp, Wp = self._window_partition(x_norm, H, W)
                attn_out = self.attn(x_win)
                attn_out = self._window_reverse(attn_out, H, W, Hp, Wp)
            else:
                attn_out = self.attn(x_norm)
            
            # Mamba branch
            x_2d = rearrange(x_norm, 'b (h w) c -> b c h w', h=H, w=W)
            mamba_out = self.mamba(x_2d)
            mamba_out = rearrange(mamba_out, 'b c h w -> b (h w) c')
            
            # Fuse
            combined = torch.cat([attn_out, mamba_out], dim=-1)
            fused = self.fusion(combined)
            x = x + self.drop_path(fused)
            
        elif self.fusion_mode == 'sequential':
            # Sequential: Attention first, then Mamba
            x_norm = self.norm1(x)
            
            # Attention
            if self.window_size is not None:
                x_win, Hp, Wp = self._window_partition(x_norm, H, W)
                attn_out = self.attn(x_win)
                attn_out = self._window_reverse(attn_out, H, W, Hp, Wp)
            else:
                attn_out = self.attn(x_norm)
            x = x + self.drop_path(attn_out)
            
            # Mamba
            x_norm = self.norm2(x)
            x_2d = rearrange(x_norm, 'b (h w) c -> b c h w', h=H, w=W)
            mamba_out = self.mamba(x_2d)
            mamba_out = rearrange(mamba_out, 'b c h w -> b (h w) c')
            x = x + self.drop_path(mamba_out)
            
        elif self.fusion_mode == 'residual':
            # Residual: Mamba as residual to attention
            x_norm = self.norm1(x)
            
            # Attention (main path)
            if self.window_size is not None:
                x_win, Hp, Wp = self._window_partition(x_norm, H, W)
                attn_out = self.attn(x_win)
                attn_out = self._window_reverse(attn_out, H, W, Hp, Wp)
            else:
                attn_out = self.attn(x_norm)
            
            # Mamba (residual)
            x_2d = rearrange(x_norm, 'b (h w) c -> b c h w', h=H, w=W)
            mamba_out = self.mamba(x_2d)
            mamba_out = rearrange(mamba_out, 'b c h w -> b (h w) c')
            
            # Combine with learned ratio
            x = x + self.drop_path(
                self.attn_ratio * attn_out + (1 - self.attn_ratio) * mamba_out
            )
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        
        # Reshape back
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x


class MambaAttentionBlock(nn.Module):
    """
    Mamba block with attention-style modulation.
    
    Uses attention to modulate Mamba's behavior, allowing
    content-dependent SSM processing.
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads for modulation
        mamba_type: Type of Mamba block
        d_state: SSM state dimension
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mamba_type: str = 'mamba',
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        drop_path: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # Compute attention weights for modulation
        self.norm1 = nn.LayerNorm(dim)
        self.attn_pool = nn.Sequential(
            nn.Linear(dim, num_heads),
            nn.Softmax(dim=1)  # Softmax over sequence dimension
        )
        
        # Mamba with modulation
        self.norm2 = nn.LayerNorm(dim)
        self.mamba = create_mamba_block(
            variant=mamba_type,
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs
        )
        
        # Modulation projection
        self.modulation = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )
        
        # MLP
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, hidden_dim=dim * 4)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, C, H, W)
            
        Returns:
            Output (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape to sequence
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Compute attention-based global context
        x_norm = self.norm1(x)
        attn_weights = self.attn_pool(x_norm)  # (B, L, num_heads)
        global_context = torch.einsum('bln,blc->bnc', attn_weights, x_norm)  # (B, num_heads, C)
        global_context = global_context.mean(dim=1)  # (B, C)
        
        # Compute modulation
        modulation = self.modulation(global_context)  # (B, C)
        
        # Mamba with modulation
        x_norm = self.norm2(x)
        x_2d = rearrange(x_norm, 'b (h w) c -> b c h w', h=H, w=W)
        mamba_out = self.mamba(x_2d)
        mamba_out = rearrange(mamba_out, 'b c h w -> b (h w) c')
        
        # Apply modulation
        mamba_out = mamba_out * (1 + modulation.unsqueeze(1))
        x = x + self.drop_path(mamba_out)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        
        # Reshape back
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x


class SwinMambaBlock(nn.Module):
    """
    Swin-style block with Mamba replacing window attention.
    
    Keeps the shifted window mechanism but uses Mamba
    for within-window processing.
    
    Args:
        dim: Feature dimension
        window_size: Window size
        shift_size: Shift size for SW-MSA
        mamba_type: Type of Mamba block
    """
    
    def __init__(
        self,
        dim: int,
        window_size: int = 7,
        shift_size: int = 0,
        mamba_type: str = 'mamba',
        d_state: int = 16,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        
        # Normalization
        self.norm1 = nn.LayerNorm(dim)
        
        # Mamba for window processing
        self.mamba = create_mamba_block(
            variant=mamba_type,
            dim=dim,
            d_state=d_state,
            **kwargs
        )
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio))
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, C, H, W)
            
        Returns:
            Output (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape
        x = rearrange(x, 'b c h w -> b (h w) c')
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        
        # Pad for windows
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        Hp, Wp = x.shape[1], x.shape[2]
        
        # Partition windows
        x = x.view(B, Hp // self.window_size, self.window_size,
                   Wp // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, self.window_size, self.window_size, C)
        
        # Mamba on windows (process each window as a mini-batch)
        windows = rearrange(windows, 'n h w c -> n c h w')
        windows = self.mamba(windows)
        windows = rearrange(windows, 'n c h w -> n h w c')
        
        # Merge windows
        x = windows.view(B, Hp // self.window_size, Wp // self.window_size,
                        self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, Hp, Wp, C)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        x = x.view(B, H * W, C)
        
        # Residual
        x = shortcut + self.drop_path(x)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # Reshape back
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x


class AlternatingHybridBlock(nn.Module):
    """
    Alternating attention and Mamba blocks.
    
    Pairs of blocks where first uses attention, second uses Mamba.
    Good for capturing both local and global patterns.
    
    Args:
        dim: Feature dimension
        depth: Number of block pairs (total blocks = depth * 2)
        num_heads: Attention heads
        mamba_type: Type of Mamba block
        window_size: Window size for local attention
    """
    
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        num_heads: int = 8,
        mamba_type: str = 'vmamba',
        window_size: int = 7,
        d_state: int = 16,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth * 2)]
        
        for i in range(depth):
            # Attention block (with window for local)
            self.blocks.append(
                self._make_attn_block(
                    dim, num_heads, window_size, mlp_ratio, dpr[i * 2]
                )
            )
            
            # Mamba block (for global)
            self.blocks.append(
                self._make_mamba_block(
                    dim, mamba_type, d_state, mlp_ratio, dpr[i * 2 + 1], **kwargs
                )
            )
    
    def _make_attn_block(self, dim, num_heads, window_size, mlp_ratio, drop_path):
        return nn.ModuleDict({
            'norm1': nn.LayerNorm(dim),
            'attn': WindowAttention(dim, window_size, num_heads),
            'norm2': nn.LayerNorm(dim),
            'mlp': MLP(dim, int(dim * mlp_ratio)),
            'drop_path': DropPath(drop_path) if drop_path > 0 else nn.Identity(),
            'window_size': nn.Identity(),  # Placeholder to store window_size
        })
    
    def _make_mamba_block(self, dim, mamba_type, d_state, mlp_ratio, drop_path, **kwargs):
        return nn.ModuleDict({
            'norm1': nn.LayerNorm(dim),
            'mamba': create_mamba_block(mamba_type, dim, d_state=d_state, **kwargs),
            'norm2': nn.LayerNorm(dim),
            'mlp': MLP(dim, int(dim * mlp_ratio)),
            'drop_path': DropPath(drop_path) if drop_path > 0 else nn.Identity(),
        })
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, C, H, W)
            
        Returns:
            Output (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:  # Attention block
                x = self._forward_attn(x, block, H, W)
            else:  # Mamba block
                x = self._forward_mamba(x, block)
        
        return x
    
    def _forward_attn(self, x, block, H, W):
        B, C, H, W = x.shape
        window_size = 7  # Default
        
        x = rearrange(x, 'b c h w -> b (h w) c')
        shortcut = x
        x = block['norm1'](x)
        
        # Window partition (simplified)
        x = x.view(B, H, W, C)
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]
        
        x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
        
        x = block['attn'](x)
        
        # Reverse
        x = x.view(B, Hp // window_size, Wp // window_size, window_size, window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]
        x = x.view(B, H * W, C)
        
        x = shortcut + block['drop_path'](x)
        x = x + block['drop_path'](block['mlp'](block['norm2'](x)))
        
        return rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
    
    def _forward_mamba(self, x, block):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        shortcut = x
        x = block['norm1'](x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        x = block['mamba'](x)
        
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = shortcut + block['drop_path'](x)
        x = x + block['drop_path'](block['mlp'](block['norm2'](x)))
        
        return rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)


if __name__ == "__main__":
    # Test hybrid modules
    print("Testing Hybrid Attention-Mamba modules...")
    
    B, C, H, W = 2, 96, 32, 32
    x = torch.randn(B, C, H, W)
    
    # Test HybridAttentionMamba (parallel)
    print("\n1. Testing HybridAttentionMamba (parallel)...")
    hybrid = HybridAttentionMamba(
        dim=C,
        num_heads=4,
        fusion_mode='parallel',
        window_size=8
    )
    out = hybrid(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test HybridAttentionMamba (sequential)
    print("\n2. Testing HybridAttentionMamba (sequential)...")
    hybrid_seq = HybridAttentionMamba(
        dim=C,
        num_heads=4,
        fusion_mode='sequential'
    )
    out = hybrid_seq(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test MambaAttentionBlock
    print("\n3. Testing MambaAttentionBlock...")
    mamba_attn = MambaAttentionBlock(dim=C, num_heads=4)
    out = mamba_attn(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test SwinMambaBlock
    print("\n4. Testing SwinMambaBlock...")
    swin_mamba = SwinMambaBlock(dim=C, window_size=8, shift_size=0)
    out = swin_mamba(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test SwinMambaBlock with shift
    print("\n5. Testing SwinMambaBlock (shifted)...")
    swin_mamba_shift = SwinMambaBlock(dim=C, window_size=8, shift_size=4)
    out = swin_mamba_shift(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test AlternatingHybridBlock
    print("\n6. Testing AlternatingHybridBlock...")
    alternating = AlternatingHybridBlock(
        dim=C,
        depth=2,
        num_heads=4,
        window_size=8
    )
    out = alternating(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    print("\nAll hybrid module tests passed!")
