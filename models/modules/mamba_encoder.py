"""
Mamba Encoder Modules for Segmentation Networks.

This module provides encoder architectures that can replace CNN or Transformer
encoders in segmentation networks like UNet, DeepLab, etc.

The encoders are designed to:
1. Extract hierarchical features at multiple scales
2. Capture long-range dependencies efficiently via SSM
3. Be drop-in replacements for existing encoders
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .mamba_block import (
    MambaLayer, 
    Mamba2Layer, 
    VMMambaBlock, 
    DropPath,
    LayerNorm2d,
    create_mamba_block
)


class PatchEmbed(nn.Module):
    """
    Patch Embedding layer that converts image to sequence of patch tokens.
    
    Similar to Vision Transformer but can use overlapping patches.
    
    Args:
        in_channels: Input image channels
        embed_dim: Output embedding dimension
        patch_size: Size of each patch
        stride: Stride for patch extraction (< patch_size for overlap)
        padding: Padding for convolution
        norm_layer: Normalization layer
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 96,
        patch_size: int = 4,
        stride: Optional[int] = None,
        padding: int = 0,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=self.stride,
            padding=padding
        )
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: Input image (B, C, H, W)
            
        Returns:
            Tuple of (embedded patches (B, C, H', W'), H', W')
        """
        x = self.proj(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = self.norm(x)
        
        return x, H, W


class PatchMerging(nn.Module):
    """
    Patch Merging layer for downsampling.
    
    Reduces spatial resolution by 2x while increasing channels.
    Similar to Swin Transformer's patch merging.
    
    Args:
        dim: Input dimension
        out_dim: Output dimension (default: 2*dim)
        norm_layer: Normalization layer
    """
    
    def __init__(
        self,
        dim: int,
        out_dim: Optional[int] = None,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        
        # Concatenate 2x2 patches then project
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, C, H, W)
            
        Returns:
            Downsampled output (B, out_dim, H/2, W/2)
        """
        B, C, H, W = x.shape
        
        # Ensure H, W are even
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2))
            H, W = x.shape[2], x.shape[3]
        
        # Reshape to (B, H/2, W/2, 4*C)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=2, p2=2)
        
        x = self.norm(x)
        x = self.reduction(x)
        
        # Reshape back to spatial
        x = rearrange(x, 'b (h w) c -> b c h w', h=H//2, w=W//2)
        
        return x


class ConvDownsample(nn.Module):
    """
    Convolutional downsampling layer.
    
    Alternative to patch merging using strided convolution.
    """
    
    def __init__(
        self,
        dim: int,
        out_dim: Optional[int] = None,
        kernel_size: int = 3,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__()
        
        self.out_dim = out_dim or 2 * dim
        
        self.conv = nn.Conv2d(
            dim,
            self.out_dim,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2
        )
        self.norm = norm_layer(self.out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class MambaEncoderStage(nn.Module):
    """
    Single stage of the Mamba encoder.
    
    Each stage contains multiple Mamba blocks at the same resolution,
    followed by downsampling (except for the last stage).
    
    Args:
        dim: Input/output dimension for this stage
        depth: Number of Mamba blocks
        mamba_type: Type of Mamba block ('mamba', 'mamba2', 'vmamba')
        downsample: Whether to downsample at the end
        out_dim: Output dimension after downsampling
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion ratio
        drop_path: Stochastic depth rate (list or float)
        downsample_type: 'patch_merge' or 'conv'
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        mamba_type: str = 'vmamba',
        downsample: bool = True,
        out_dim: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        drop_path: Union[float, List[float]] = 0.0,
        downsample_type: str = 'conv',
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        
        # Create drop path rates for each block
        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth
        
        # Build Mamba blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = create_mamba_block(
                variant=mamba_type,
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=drop_path[i],
                **kwargs
            )
            self.blocks.append(block)
        
        # Downsampling layer
        if downsample:
            out_dim = out_dim or 2 * dim
            if downsample_type == 'patch_merge':
                self.downsample = PatchMerging(dim, out_dim)
            else:
                self.downsample = ConvDownsample(dim, out_dim)
        else:
            self.downsample = None
    
    def forward(
        self, 
        x: torch.Tensor,
        return_before_downsample: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input (B, C, H, W)
            return_before_downsample: Whether to return features before downsampling
            
        Returns:
            Output features (and pre-downsample features if requested)
        """
        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        # Store pre-downsample features for skip connections
        x_before_ds = x
        
        # Downsample
        if self.downsample is not None:
            x = self.downsample(x)
        
        if return_before_downsample:
            return x, x_before_ds
        return x


class MambaEncoder(nn.Module):
    """
    Complete Mamba Encoder for image segmentation.
    
    Hierarchical encoder that extracts multi-scale features using Mamba blocks.
    Can be used as drop-in replacement for CNN or Transformer encoders.
    
    Args:
        in_channels: Number of input channels
        embed_dims: List of embedding dimensions for each stage
        depths: List of Mamba block depths for each stage
        mamba_type: Type of Mamba block ('mamba', 'mamba2', 'vmamba')
        patch_size: Initial patch embedding size
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion ratio
        drop_path_rate: Maximum stochastic depth rate
        downsample_type: Type of downsampling ('patch_merge' or 'conv')
        out_indices: Indices of stages to output (for multi-scale features)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        embed_dims: List[int] = [96, 192, 384, 768],
        depths: List[int] = [2, 2, 6, 2],
        mamba_type: str = 'vmamba',
        patch_size: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        drop_path_rate: float = 0.1,
        downsample_type: str = 'conv',
        out_indices: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__()
        
        self.num_stages = len(embed_dims)
        self.embed_dims = embed_dims
        self.out_indices = out_indices or list(range(self.num_stages))
        
        # Patch embedding stem
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dims[0],
            patch_size=patch_size,
            stride=patch_size,
            norm_layer=LayerNorm2d
        )
        
        # Calculate drop path rates
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        
        # Build encoder stages
        self.stages = nn.ModuleList()
        cur = 0
        
        for i in range(self.num_stages):
            # Get drop path rates for this stage
            stage_dpr = dpr[cur:cur + depths[i]]
            cur += depths[i]
            
            # Determine if downsampling needed
            downsample = i < self.num_stages - 1
            out_dim = embed_dims[i + 1] if downsample else None
            
            stage = MambaEncoderStage(
                dim=embed_dims[i],
                depth=depths[i],
                mamba_type=mamba_type,
                downsample=downsample,
                out_dim=out_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=stage_dpr,
                downsample_type=downsample_type,
                **kwargs
            )
            self.stages.append(stage)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input image (B, C, H, W)
            
        Returns:
            List of feature maps at different scales
        """
        # Patch embedding
        x, H, W = self.patch_embed(x)
        
        features = []
        
        for i, stage in enumerate(self.stages):
            # Get features before and after downsampling
            if i < len(self.stages) - 1:
                x, x_before_ds = stage(x, return_before_downsample=True)
                if i in self.out_indices:
                    features.append(x_before_ds)
            else:
                x = stage(x)
                if i in self.out_indices:
                    features.append(x)
        
        return features
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only final features.
        
        Args:
            x: Input image (B, C, H, W)
            
        Returns:
            Final feature map
        """
        features = self.forward(x)
        return features[-1]


class MambaEncoderWithCNN(nn.Module):
    """
    Hybrid encoder that combines CNN stem with Mamba stages.
    
    The CNN stem helps capture local features at high resolution,
    while Mamba stages handle long-range dependencies at lower resolutions.
    
    Args:
        in_channels: Number of input channels
        stem_channels: Channels for CNN stem
        embed_dims: Dimensions for Mamba stages
        stem_depth: Number of conv layers in stem
        depths: Depths of Mamba stages
        mamba_type: Type of Mamba block
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        stem_channels: int = 64,
        embed_dims: List[int] = [96, 192, 384, 768],
        stem_depth: int = 2,
        depths: List[int] = [2, 2, 6, 2],
        mamba_type: str = 'vmamba',
        **kwargs
    ):
        super().__init__()
        
        # CNN Stem
        stem_layers = []
        curr_channels = in_channels
        
        for i in range(stem_depth):
            out_channels = stem_channels if i < stem_depth - 1 else embed_dims[0]
            stride = 2 if i == 0 else 1
            
            stem_layers.extend([
                nn.Conv2d(curr_channels, out_channels, 3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            curr_channels = out_channels
        
        self.stem = nn.Sequential(*stem_layers)
        
        # Mamba encoder (without patch embedding since stem handles it)
        self.mamba_encoder = MambaEncoder(
            in_channels=embed_dims[0],
            embed_dims=embed_dims,
            depths=depths,
            mamba_type=mamba_type,
            patch_size=2,  # Smaller patch size since stem already downsamples
            **kwargs
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Input image (B, C, H, W)
            
        Returns:
            List of multi-scale feature maps
        """
        # CNN stem
        stem_feat = self.stem(x)
        
        # Mamba stages
        features = self.mamba_encoder(stem_feat)
        
        return features


if __name__ == "__main__":
    # Test encoder modules
    print("Testing Mamba Encoder modules...")
    
    B, C, H, W = 2, 1, 256, 256  # Typical medical image
    x = torch.randn(B, C, H, W)
    
    # Test MambaEncoderStage
    print("\n1. Testing MambaEncoderStage...")
    stage = MambaEncoderStage(
        dim=96,
        depth=2,
        mamba_type='vmamba',
        downsample=True
    )
    out, skip = stage(x.expand(-1, 96, -1, -1)[:, :, :64, :64], return_before_downsample=True)
    print(f"   Stage output: {out.shape}, Skip: {skip.shape}")
    
    # Test MambaEncoder
    print("\n2. Testing MambaEncoder...")
    encoder = MambaEncoder(
        in_channels=1,
        embed_dims=[96, 192, 384, 768],
        depths=[2, 2, 2, 2],
        mamba_type='vmamba'
    )
    features = encoder(x)
    print(f"   Input: {x.shape}")
    for i, f in enumerate(features):
        print(f"   Stage {i} output: {f.shape}")
    
    # Test with different Mamba types
    print("\n3. Testing with different Mamba types...")
    for mamba_type in ['mamba', 'mamba2', 'vmamba']:
        encoder = MambaEncoder(
            in_channels=1,
            embed_dims=[64, 128],
            depths=[2, 2],
            mamba_type=mamba_type
        )
        features = encoder(x)
        print(f"   {mamba_type}: {[f.shape for f in features]}")
    
    # Test hybrid encoder
    print("\n4. Testing MambaEncoderWithCNN...")
    hybrid_encoder = MambaEncoderWithCNN(
        in_channels=1,
        stem_channels=32,
        embed_dims=[64, 128, 256],
        depths=[2, 2, 2]
    )
    features = hybrid_encoder(x)
    print(f"   Hybrid encoder outputs: {[f.shape for f in features]}")
    
    print("\nAll encoder tests passed!")
