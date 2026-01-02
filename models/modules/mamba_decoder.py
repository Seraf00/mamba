"""
Mamba Decoder Modules for Segmentation Networks.

This module provides decoder architectures that use Mamba blocks
for upsampling and feature aggregation in segmentation networks.

The decoders are designed to:
1. Progressively upsample features to full resolution
2. Fuse multi-scale encoder features via skip connections
3. Maintain long-range dependencies during decoding
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


class PatchExpanding(nn.Module):
    """
    Patch Expanding layer for upsampling.
    
    Increases spatial resolution by 2x while reducing channels.
    Inverse of PatchMerging.
    
    Args:
        dim: Input dimension
        out_dim: Output dimension (default: dim // 2)
        scale_factor: Upsampling scale factor
        norm_layer: Normalization layer
    """
    
    def __init__(
        self,
        dim: int,
        out_dim: Optional[int] = None,
        scale_factor: int = 2,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        
        self.dim = dim
        self.out_dim = out_dim or dim // 2
        self.scale_factor = scale_factor
        
        # Linear expansion
        self.expand = nn.Linear(dim, self.out_dim * scale_factor * scale_factor, bias=False)
        self.norm = norm_layer(self.out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, C, H, W)
            
        Returns:
            Upsampled output (B, out_dim, H*scale, W*scale)
        """
        B, C, H, W = x.shape
        
        # Reshape to sequence
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Expand
        x = self.expand(x)  # (B, H*W, out_dim * scale^2)
        
        # Reshape to spatial with pixel shuffle
        x = rearrange(
            x, 
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=H, w=W, p1=self.scale_factor, p2=self.scale_factor
        )
        
        # Apply normalization
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H*self.scale_factor, w=W*self.scale_factor)
        
        return x


class ConvUpsample(nn.Module):
    """
    Convolutional upsampling using transposed convolution or interpolation.
    
    Args:
        dim: Input dimension
        out_dim: Output dimension
        scale_factor: Upsampling factor
        mode: 'transpose' or 'bilinear'
    """
    
    def __init__(
        self,
        dim: int,
        out_dim: Optional[int] = None,
        scale_factor: int = 2,
        mode: str = 'bilinear',
        kernel_size: int = 3
    ):
        super().__init__()
        
        self.out_dim = out_dim or dim // 2
        self.scale_factor = scale_factor
        self.mode = mode
        
        if mode == 'transpose':
            self.up = nn.ConvTranspose2d(
                dim,
                self.out_dim,
                kernel_size=scale_factor,
                stride=scale_factor
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                nn.Conv2d(dim, self.out_dim, kernel_size=kernel_size, padding=kernel_size // 2)
            )
        
        self.norm = nn.BatchNorm2d(self.out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.up(x))


class SkipConnection(nn.Module):
    """
    Skip connection fusion module.
    
    Combines upsampled decoder features with encoder skip features.
    
    Args:
        decoder_dim: Decoder feature dimension
        encoder_dim: Encoder skip feature dimension
        out_dim: Output dimension
        mode: Fusion mode ('concat', 'add', 'attention')
    """
    
    def __init__(
        self,
        decoder_dim: int,
        encoder_dim: int,
        out_dim: Optional[int] = None,
        mode: str = 'concat'
    ):
        super().__init__()
        
        self.mode = mode
        self.out_dim = out_dim or decoder_dim
        
        if mode == 'concat':
            self.fuse = nn.Sequential(
                nn.Conv2d(decoder_dim + encoder_dim, self.out_dim, 1),
                nn.BatchNorm2d(self.out_dim),
                nn.ReLU(inplace=True)
            )
        elif mode == 'add':
            # Project encoder features to match decoder
            self.encoder_proj = nn.Conv2d(encoder_dim, decoder_dim, 1) if encoder_dim != decoder_dim else nn.Identity()
            self.fuse = nn.Sequential(
                nn.Conv2d(decoder_dim, self.out_dim, 1),
                nn.BatchNorm2d(self.out_dim),
                nn.ReLU(inplace=True)
            )
        elif mode == 'attention':
            self.query = nn.Conv2d(decoder_dim, decoder_dim // 4, 1)
            self.key = nn.Conv2d(encoder_dim, decoder_dim // 4, 1)
            self.value = nn.Conv2d(encoder_dim, encoder_dim, 1)
            self.fuse = nn.Sequential(
                nn.Conv2d(decoder_dim + encoder_dim, self.out_dim, 1),
                nn.BatchNorm2d(self.out_dim),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")
    
    def forward(
        self,
        decoder_feat: torch.Tensor,
        encoder_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            decoder_feat: Upsampled decoder features (B, C1, H, W)
            encoder_feat: Encoder skip features (B, C2, H, W)
            
        Returns:
            Fused features (B, out_dim, H, W)
        """
        # Ensure spatial dimensions match
        if decoder_feat.shape[2:] != encoder_feat.shape[2:]:
            encoder_feat = F.interpolate(
                encoder_feat,
                size=decoder_feat.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        
        if self.mode == 'concat':
            x = torch.cat([decoder_feat, encoder_feat], dim=1)
            return self.fuse(x)
        
        elif self.mode == 'add':
            encoder_feat = self.encoder_proj(encoder_feat)
            x = decoder_feat + encoder_feat
            return self.fuse(x)
        
        elif self.mode == 'attention':
            B, _, H, W = decoder_feat.shape
            
            q = self.query(decoder_feat).flatten(2)  # (B, C/4, H*W)
            k = self.key(encoder_feat).flatten(2)    # (B, C/4, H*W)
            v = self.value(encoder_feat).flatten(2)  # (B, C2, H*W)
            
            attn = torch.bmm(q.transpose(1, 2), k)   # (B, H*W, H*W)
            attn = F.softmax(attn / math.sqrt(q.shape[1]), dim=-1)
            
            attended = torch.bmm(v, attn.transpose(1, 2))  # (B, C2, H*W)
            attended = attended.view(B, -1, H, W)
            
            x = torch.cat([decoder_feat, attended], dim=1)
            return self.fuse(x)


class MambaDecoderStage(nn.Module):
    """
    Single stage of the Mamba decoder.
    
    Each stage:
    1. Upsamples the input
    2. Fuses with skip connection (if provided)
    3. Applies Mamba blocks
    
    Args:
        dim: Input dimension
        out_dim: Output dimension after upsampling
        skip_dim: Skip connection dimension (None if no skip)
        depth: Number of Mamba blocks
        mamba_type: Type of Mamba block
        upsample: Whether to upsample at the beginning
        skip_mode: Skip connection fusion mode
    """
    
    def __init__(
        self,
        dim: int,
        out_dim: int,
        skip_dim: Optional[int] = None,
        depth: int = 2,
        mamba_type: str = 'vmamba',
        upsample: bool = True,
        upsample_type: str = 'bilinear',
        skip_mode: str = 'concat',
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        drop_path: Union[float, List[float]] = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.out_dim = out_dim
        
        # Upsampling
        if upsample:
            self.upsample = ConvUpsample(dim, out_dim, mode=upsample_type)
        else:
            self.upsample = nn.Conv2d(dim, out_dim, 1) if dim != out_dim else nn.Identity()
        
        # Skip connection fusion
        if skip_dim is not None:
            self.skip_fuse = SkipConnection(
                decoder_dim=out_dim,
                encoder_dim=skip_dim,
                out_dim=out_dim,
                mode=skip_mode
            )
        else:
            self.skip_fuse = None
        
        # Create drop path rates
        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth
        
        # Mamba blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = create_mamba_block(
                variant=mamba_type,
                dim=out_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=drop_path[i],
                **kwargs
            )
            self.blocks.append(block)
    
    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
            skip: Skip connection features (B, C_skip, H*2, W*2)
            
        Returns:
            Decoded features (B, out_dim, H*2, W*2)
        """
        # Upsample
        x = self.upsample(x)
        
        # Fuse with skip connection
        if self.skip_fuse is not None and skip is not None:
            x = self.skip_fuse(x, skip)
        
        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        return x


class MambaDecoder(nn.Module):
    """
    Complete Mamba Decoder for image segmentation.
    
    Hierarchical decoder that progressively upsamples features using Mamba blocks
    while fusing with encoder skip connections.
    
    Args:
        encoder_dims: List of encoder feature dimensions (from deep to shallow)
        decoder_dims: List of decoder output dimensions for each stage
        depths: List of Mamba block depths for each stage
        num_classes: Number of output segmentation classes
        mamba_type: Type of Mamba block
        skip_mode: Skip connection fusion mode
        deep_supervision: Whether to add auxiliary outputs
    """
    
    def __init__(
        self,
        encoder_dims: List[int] = [768, 384, 192, 96],
        decoder_dims: List[int] = [384, 192, 96, 64],
        depths: List[int] = [2, 2, 2, 2],
        num_classes: int = 4,
        mamba_type: str = 'vmamba',
        skip_mode: str = 'concat',
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        drop_path_rate: float = 0.1,
        deep_supervision: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.num_stages = len(decoder_dims)
        self.deep_supervision = deep_supervision
        
        # Calculate drop path rates (decreasing as we go up in decoder)
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, total_depth)]
        
        # Build decoder stages
        self.stages = nn.ModuleList()
        cur = 0
        
        for i in range(self.num_stages):
            # Input dimension (from previous decoder stage or encoder)
            in_dim = encoder_dims[0] if i == 0 else decoder_dims[i - 1]
            
            # Skip dimension (from encoder, reversed order)
            skip_dim = encoder_dims[i + 1] if i < len(encoder_dims) - 1 else None
            
            # Get drop path rates for this stage
            stage_dpr = dpr[cur:cur + depths[i]]
            cur += depths[i]
            
            stage = MambaDecoderStage(
                dim=in_dim,
                out_dim=decoder_dims[i],
                skip_dim=skip_dim,
                depth=depths[i],
                mamba_type=mamba_type,
                skip_mode=skip_mode,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=stage_dpr,
                **kwargs
            )
            self.stages.append(stage)
        
        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_dims[-1], decoder_dims[-1], 3, padding=1),
            nn.BatchNorm2d(decoder_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dims[-1], num_classes, 1)
        )
        
        # Deep supervision heads (auxiliary outputs at intermediate scales)
        if deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(dim, num_classes, 1)
                for dim in decoder_dims[:-1]
            ])
        
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
    
    def forward(
        self,
        encoder_features: List[torch.Tensor],
        return_aux: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            encoder_features: List of encoder features from deep to shallow
                              [feat_stage4, feat_stage3, feat_stage2, feat_stage1]
            return_aux: Whether to return auxiliary outputs
            
        Returns:
            Segmentation output (B, num_classes, H, W)
            Optionally also auxiliary outputs for deep supervision
        """
        # Reverse encoder features for skip connections
        # encoder_features[0] is deepest, encoder_features[-1] is shallowest
        
        aux_outputs = []
        x = encoder_features[0]  # Start from deepest features
        
        for i, stage in enumerate(self.stages):
            # Get skip connection (from shallower encoder stage)
            skip = encoder_features[i + 1] if i + 1 < len(encoder_features) else None
            
            x = stage(x, skip)
            
            # Store for deep supervision
            if self.deep_supervision and i < len(self.stages) - 1:
                aux_outputs.append(self.aux_heads[i](x))
        
        # Final segmentation
        output = self.seg_head(x)
        
        if return_aux and self.deep_supervision:
            return output, aux_outputs
        return output


class LightweightMambaDecoder(nn.Module):
    """
    Lightweight decoder for efficient inference.
    
    Uses fewer Mamba blocks and simple upsampling for faster inference
    while maintaining reasonable accuracy.
    
    Args:
        encoder_dims: List of encoder feature dimensions
        hidden_dim: Single hidden dimension for all decoder stages
        num_classes: Number of output classes
        mamba_type: Type of Mamba block
    """
    
    def __init__(
        self,
        encoder_dims: List[int] = [768, 384, 192, 96],
        hidden_dim: int = 128,
        num_classes: int = 4,
        mamba_type: str = 'vmamba',
        **kwargs
    ):
        super().__init__()
        
        self.num_stages = len(encoder_dims)
        
        # Project all encoder features to same dimension
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            for dim in encoder_dims
        ])
        
        # Single Mamba block after fusion
        self.mamba = create_mamba_block(
            variant=mamba_type,
            dim=hidden_dim,
            d_state=8,  # Smaller state for efficiency
            expand=1.5,
            **kwargs
        )
        
        # Segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1)
        )
    
    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            encoder_features: List of encoder features
            
        Returns:
            Segmentation output
        """
        # Get target size (from shallowest features)
        target_size = encoder_features[-1].shape[2:]
        
        # Project and upsample all features
        projected = []
        for feat, proj in zip(encoder_features, self.proj_layers):
            p = proj(feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=True)
            projected.append(p)
        
        # Sum all features
        x = sum(projected)
        
        # Apply Mamba
        x = self.mamba(x)
        
        # Segmentation
        output = self.head(x)
        
        return output


if __name__ == "__main__":
    # Test decoder modules
    print("Testing Mamba Decoder modules...")
    
    # Simulate encoder outputs (from deep to shallow)
    B = 2
    encoder_features = [
        torch.randn(B, 768, 8, 8),    # Deepest (1/32)
        torch.randn(B, 384, 16, 16),  # 1/16
        torch.randn(B, 192, 32, 32),  # 1/8
        torch.randn(B, 96, 64, 64),   # Shallowest (1/4)
    ]
    
    # Test MambaDecoderStage
    print("\n1. Testing MambaDecoderStage...")
    stage = MambaDecoderStage(
        dim=768,
        out_dim=384,
        skip_dim=384,
        depth=2,
        mamba_type='vmamba'
    )
    out = stage(encoder_features[0], encoder_features[1])
    print(f"   Input: {encoder_features[0].shape}, Skip: {encoder_features[1].shape}")
    print(f"   Output: {out.shape}")
    
    # Test MambaDecoder
    print("\n2. Testing MambaDecoder...")
    decoder = MambaDecoder(
        encoder_dims=[768, 384, 192, 96],
        decoder_dims=[384, 192, 96, 64],
        depths=[2, 2, 2, 2],
        num_classes=4,
        mamba_type='vmamba',
        deep_supervision=True
    )
    output, aux = decoder(encoder_features, return_aux=True)
    print(f"   Main output: {output.shape}")
    print(f"   Aux outputs: {[a.shape for a in aux]}")
    
    # Test with different Mamba types
    print("\n3. Testing decoder with different Mamba types...")
    for mamba_type in ['mamba', 'mamba2', 'vmamba']:
        decoder = MambaDecoder(
            encoder_dims=[256, 128, 64],
            decoder_dims=[128, 64, 32],
            depths=[1, 1, 1],
            num_classes=4,
            mamba_type=mamba_type
        )
        small_feats = [
            torch.randn(B, 256, 8, 8),
            torch.randn(B, 128, 16, 16),
            torch.randn(B, 64, 32, 32),
        ]
        out = decoder(small_feats)
        print(f"   {mamba_type}: {out.shape}")
    
    # Test lightweight decoder
    print("\n4. Testing LightweightMambaDecoder...")
    light_decoder = LightweightMambaDecoder(
        encoder_dims=[768, 384, 192, 96],
        hidden_dim=128,
        num_classes=4
    )
    output = light_decoder(encoder_features)
    print(f"   Output: {output.shape}")
    
    print("\nAll decoder tests passed!")
