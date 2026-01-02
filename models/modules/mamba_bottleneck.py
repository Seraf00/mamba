"""
Mamba Bottleneck Modules for Segmentation Networks.

The bottleneck is the deepest part of encoder-decoder architectures
where the feature resolution is lowest and channel count is highest.

This module provides Mamba-based bottleneck designs that:
1. Capture long-range global context efficiently
2. Process features at the lowest resolution
3. Bridge encoder and decoder with rich representations
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
    RMSNorm,
    create_mamba_block
)


class MambaBottleneck(nn.Module):
    """
    Mamba-based bottleneck for UNet-like architectures.
    
    Replaces traditional conv blocks at the bottleneck with Mamba
    blocks for efficient global context modeling.
    
    Args:
        dim: Feature dimension
        depth: Number of Mamba blocks
        mamba_type: Type of Mamba block ('mamba', 'mamba2', 'vmamba')
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion ratio
        drop_path: Stochastic depth rate
        use_checkpoint: Use gradient checkpointing
    """
    
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        mamba_type: str = 'vmamba',
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        drop_path: float = 0.1,
        use_checkpoint: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # Input projection (optional channel adjustment)
        self.in_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        # Create drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        # Mamba blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = create_mamba_block(
                variant=mamba_type,
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=dpr[i],
                **kwargs
            )
            self.blocks.append(block)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
            
        Returns:
            Processed features (B, C, H, W)
        """
        # Input projection
        x = self.in_proj(x)
        
        # Apply Mamba blocks
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        # Output projection
        x = self.out_proj(x)
        
        return x


class GlobalContextMambaBottleneck(nn.Module):
    """
    Enhanced bottleneck with explicit global context aggregation.
    
    Combines Mamba's sequential processing with spatial pooling
    for better global understanding.
    
    Args:
        dim: Feature dimension
        depth: Number of Mamba blocks
        mamba_type: Type of Mamba block
        pool_sizes: Sizes for spatial pyramid pooling
    """
    
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        mamba_type: str = 'vmamba',
        pool_sizes: List[int] = [1, 2, 4],
        d_state: int = 16,
        expand: float = 2.0,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        
        # Spatial Pyramid Pooling for global context
        self.spp_branches = nn.ModuleList()
        for pool_size in pool_sizes:
            branch = nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(dim, dim // len(pool_sizes), 1),
                nn.BatchNorm2d(dim // len(pool_sizes)),
                nn.ReLU(inplace=True)
            )
            self.spp_branches.append(branch)
        
        # Fusion of pooled features
        self.spp_fuse = nn.Sequential(
            nn.Conv2d(dim + (dim // len(pool_sizes)) * len(pool_sizes), dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        # Mamba blocks for sequential processing
        self.mamba_blocks = nn.ModuleList()
        for _ in range(depth):
            block = create_mamba_block(
                variant=mamba_type,
                dim=dim,
                d_state=d_state,
                expand=expand,
                **kwargs
            )
            self.mamba_blocks.append(block)
        
        # Output projection
        self.output = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
            
        Returns:
            Enhanced features (B, C, H, W)
        """
        H, W = x.shape[2:]
        
        # Spatial Pyramid Pooling
        spp_outs = [x]
        for branch in self.spp_branches:
            pooled = branch(x)
            upsampled = F.interpolate(
                pooled, size=(H, W), mode='bilinear', align_corners=True
            )
            spp_outs.append(upsampled)
        
        # Fuse SPP features
        x = torch.cat(spp_outs, dim=1)
        x = self.spp_fuse(x)
        
        # Apply Mamba blocks
        for block in self.mamba_blocks:
            x = block(x)
        
        # Output
        x = self.output(x)
        
        return x


class MultiscaleMambaBottleneck(nn.Module):
    """
    Multi-scale bottleneck with parallel Mamba branches.
    
    Processes features at multiple dilations/scales in parallel,
    then fuses them together.
    
    Args:
        dim: Feature dimension
        depth: Number of Mamba blocks per branch
        num_branches: Number of parallel branches
        mamba_type: Type of Mamba block
    """
    
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        num_branches: int = 3,
        mamba_type: str = 'vmamba',
        d_state: int = 16,
        expand: float = 2.0,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.num_branches = num_branches
        branch_dim = dim // num_branches
        
        # Parallel Mamba branches with different configurations
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            # Vary d_state for different receptive fields
            branch_d_state = d_state * (i + 1)
            
            branch = nn.ModuleList()
            branch.append(nn.Conv2d(dim, branch_dim, 1))  # Input proj
            
            for _ in range(depth):
                block = create_mamba_block(
                    variant=mamba_type,
                    dim=branch_dim,
                    d_state=branch_d_state,
                    expand=expand,
                    **kwargs
                )
                branch.append(block)
            
            self.branches.append(branch)
        
        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_dim * num_branches, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
            
        Returns:
            Multi-scale processed features (B, C, H, W)
        """
        branch_outputs = []
        
        for branch in self.branches:
            # Input projection
            b = branch[0](x)
            
            # Mamba blocks
            for block in branch[1:]:
                b = block(b)
            
            branch_outputs.append(b)
        
        # Concatenate and fuse
        x = torch.cat(branch_outputs, dim=1)
        x = self.fuse(x)
        
        return x


class ASPPMambaBottleneck(nn.Module):
    """
    ASPP-style bottleneck with Mamba replacing dilated convolutions.
    
    Inspired by DeepLab's ASPP, but uses Mamba blocks with different
    configurations instead of dilated convolutions.
    
    Args:
        dim: Feature dimension
        out_dim: Output dimension
        mamba_type: Type of Mamba block
        rates: Different d_state values for multi-scale processing
    """
    
    def __init__(
        self,
        dim: int,
        out_dim: int = 256,
        mamba_type: str = 'vmamba',
        rates: List[int] = [8, 16, 32],
        **kwargs
    ):
        super().__init__()
        
        modules = []
        
        # 1x1 convolution branch
        modules.append(nn.Sequential(
            nn.Conv2d(dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        ))
        
        # Mamba branches with different d_state (receptive field analog)
        for rate in rates:
            branch = nn.Sequential(
                nn.Conv2d(dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            )
            # Add Mamba block with this d_state
            mamba_block = create_mamba_block(
                variant=mamba_type,
                dim=out_dim,
                d_state=rate,
                **kwargs
            )
            modules.append(nn.Sequential(branch, mamba_block))
        
        # Image-level features (global pooling)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        
        self.branches = nn.ModuleList(modules)
        
        # Final fusion
        self.project = nn.Sequential(
            nn.Conv2d(out_dim * (len(rates) + 2), out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
            
        Returns:
            ASPP features (B, out_dim, H, W)
        """
        H, W = x.shape[2:]
        
        # Process each branch
        outputs = []
        for branch in self.branches:
            outputs.append(branch(x))
        
        # Global pooling branch
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(
            global_feat, size=(H, W), mode='bilinear', align_corners=True
        )
        outputs.append(global_feat)
        
        # Concatenate and project
        x = torch.cat(outputs, dim=1)
        x = self.project(x)
        
        return x


class DualPathMambaBottleneck(nn.Module):
    """
    Dual-path bottleneck combining different Mamba variants.
    
    Uses two parallel paths with different Mamba types (e.g., Mamba1 + VMamba)
    to capture complementary information.
    
    Args:
        dim: Feature dimension
        depth: Number of blocks per path
        mamba_types: Tuple of two Mamba types for the two paths
    """
    
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        mamba_types: Tuple[str, str] = ('mamba', 'vmamba'),
        d_state: int = 16,
        expand: float = 2.0,
        **kwargs
    ):
        super().__init__()
        
        half_dim = dim // 2
        
        # Path 1
        self.path1_proj = nn.Conv2d(dim, half_dim, 1)
        self.path1_blocks = nn.ModuleList([
            create_mamba_block(
                variant=mamba_types[0],
                dim=half_dim,
                d_state=d_state,
                expand=expand,
                **kwargs
            )
            for _ in range(depth)
        ])
        
        # Path 2
        self.path2_proj = nn.Conv2d(dim, half_dim, 1)
        self.path2_blocks = nn.ModuleList([
            create_mamba_block(
                variant=mamba_types[1],
                dim=half_dim,
                d_state=d_state,
                expand=expand,
                **kwargs
            )
            for _ in range(depth)
        ])
        
        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
            
        Returns:
            Dual-path processed features (B, C, H, W)
        """
        # Path 1
        p1 = self.path1_proj(x)
        for block in self.path1_blocks:
            p1 = block(p1)
        
        # Path 2
        p2 = self.path2_proj(x)
        for block in self.path2_blocks:
            p2 = block(p2)
        
        # Concatenate and fuse
        x = torch.cat([p1, p2], dim=1)
        x = self.fuse(x)
        
        return x


def create_bottleneck(
    bottleneck_type: str,
    dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create bottleneck modules.
    
    Args:
        bottleneck_type: Type of bottleneck
        dim: Feature dimension
        **kwargs: Additional arguments
        
    Returns:
        Bottleneck module
    """
    bottlenecks = {
        'mamba': MambaBottleneck,
        'global_context': GlobalContextMambaBottleneck,
        'multiscale': MultiscaleMambaBottleneck,
        'aspp': ASPPMambaBottleneck,
        'dual_path': DualPathMambaBottleneck,
    }
    
    if bottleneck_type.lower() not in bottlenecks:
        raise ValueError(f"Unknown bottleneck: {bottleneck_type}. "
                        f"Choose from {list(bottlenecks.keys())}")
    
    return bottlenecks[bottleneck_type.lower()](dim=dim, **kwargs)


if __name__ == "__main__":
    # Test bottleneck modules
    print("Testing Mamba Bottleneck modules...")
    
    B, C, H, W = 2, 512, 16, 16  # Typical bottleneck size
    x = torch.randn(B, C, H, W)
    
    # Test basic MambaBottleneck
    print("\n1. Testing MambaBottleneck...")
    bottleneck = MambaBottleneck(dim=C, depth=2, mamba_type='vmamba')
    out = bottleneck(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test GlobalContextMambaBottleneck
    print("\n2. Testing GlobalContextMambaBottleneck...")
    gc_bottleneck = GlobalContextMambaBottleneck(
        dim=C, depth=2, pool_sizes=[1, 2, 4]
    )
    out = gc_bottleneck(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test MultiscaleMambaBottleneck
    print("\n3. Testing MultiscaleMambaBottleneck...")
    ms_bottleneck = MultiscaleMambaBottleneck(
        dim=C, depth=1, num_branches=3
    )
    out = ms_bottleneck(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test ASPPMambaBottleneck
    print("\n4. Testing ASPPMambaBottleneck...")
    aspp_bottleneck = ASPPMambaBottleneck(
        dim=C, out_dim=256, rates=[8, 16, 32]
    )
    out = aspp_bottleneck(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test DualPathMambaBottleneck
    print("\n5. Testing DualPathMambaBottleneck...")
    dual_bottleneck = DualPathMambaBottleneck(
        dim=C, depth=2, mamba_types=('mamba', 'vmamba')
    )
    out = dual_bottleneck(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test factory function
    print("\n6. Testing factory function...")
    for btype in ['mamba', 'global_context', 'multiscale', 'dual_path']:
        bn = create_bottleneck(btype, dim=256)
        out = bn(torch.randn(B, 256, H, W))
        print(f"   {btype}: {out.shape}")
    
    print("\nAll bottleneck tests passed!")
