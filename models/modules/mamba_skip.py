"""
Mamba Skip Connection Modules for Segmentation Networks.

Skip connections are crucial in segmentation networks for:
1. Preserving fine-grained spatial details from encoder
2. Enabling gradient flow during training
3. Fusing multi-scale features effectively

This module provides Mamba-enhanced skip connection designs that:
1. Process encoder features before fusion with Mamba
2. Enable cross-attention between encoder and decoder features
3. Provide adaptive feature selection mechanisms
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


class MambaSkipConnection(nn.Module):
    """
    Mamba-enhanced skip connection processor.
    
    Processes encoder features through Mamba before fusion with decoder,
    allowing the model to selectively use relevant encoder information.
    
    Args:
        dim: Feature dimension (can also use encoder_channels as alias)
        encoder_channels: Alias for dim (encoder feature dimension)
        decoder_channels: Decoder feature dimension (accepted but not used, 
                         as this module only processes encoder features)
        depth: Number of Mamba blocks
        mamba_type: Type of Mamba block
        d_state: SSM state dimension
        refine: Whether to apply additional refinement after Mamba
    """
    
    def __init__(
        self,
        dim: int = None,
        encoder_channels: int = None,
        decoder_channels: int = None,  # Accepted for API compatibility, not used
        depth: int = 1,
        mamba_type: str = 'vmamba',
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        drop_path: float = 0.0,
        refine: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Support both 'dim' and 'encoder_channels' as the feature dimension
        if dim is not None:
            self.dim = dim
        elif encoder_channels is not None:
            self.dim = encoder_channels
        else:
            raise ValueError("Either 'dim' or 'encoder_channels' must be provided")
        
        dim = self.dim  # Local reference for use below
        
        # Mamba blocks for processing encoder features
        self.mamba_blocks = nn.ModuleList()
        for i in range(depth):
            block = create_mamba_block(
                variant=mamba_type,
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=drop_path,
                **kwargs
            )
            self.mamba_blocks.append(block)
        
        # Optional refinement
        if refine:
            self.refine = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.refine = nn.Identity()
    
    def forward(self, x: torch.Tensor, decoder_feat: torch.Tensor = None) -> torch.Tensor:
        """
        Process encoder skip features.
        
        Args:
            x: Encoder features (B, C, H, W)
            decoder_feat: Optional decoder features (B, C_dec, H, W) - currently unused,
                         but accepted for API compatibility with models that pass both.
            
        Returns:
            Processed features (B, C, H, W)
        """
        for block in self.mamba_blocks:
            x = block(x)
        
        x = self.refine(x)
        return x


class MambaSkipFusion(nn.Module):
    """
    Mamba-based feature fusion for skip connections.
    
    Fuses encoder skip features with decoder features using Mamba
    for adaptive, sequence-aware feature combination.
    
    Args:
        encoder_dim: Encoder feature dimension
        decoder_dim: Decoder feature dimension
        out_dim: Output dimension
        mamba_type: Type of Mamba block
        fusion_mode: 'concat', 'add', 'cross_mamba'
    """
    
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        out_dim: Optional[int] = None,
        mamba_type: str = 'vmamba',
        fusion_mode: str = 'concat',
        d_state: int = 16,
        **kwargs
    ):
        super().__init__()
        
        self.fusion_mode = fusion_mode
        self.out_dim = out_dim or decoder_dim
        
        if fusion_mode == 'concat':
            # Concatenate then process with Mamba
            self.fuse_proj = nn.Conv2d(encoder_dim + decoder_dim, self.out_dim, 1)
            self.mamba = create_mamba_block(
                variant=mamba_type,
                dim=self.out_dim,
                d_state=d_state,
                **kwargs
            )
            
        elif fusion_mode == 'add':
            # Project to same dim, add, then process
            self.enc_proj = nn.Conv2d(encoder_dim, self.out_dim, 1)
            self.dec_proj = nn.Conv2d(decoder_dim, self.out_dim, 1)
            self.mamba = create_mamba_block(
                variant=mamba_type,
                dim=self.out_dim,
                d_state=d_state,
                **kwargs
            )
            
        elif fusion_mode == 'cross_mamba':
            # Cross-Mamba: encoder conditions decoder processing
            self.enc_proj = nn.Conv2d(encoder_dim, decoder_dim, 1)
            self.cross_mamba = CrossMambaFusion(
                dim=decoder_dim,
                mamba_type=mamba_type,
                d_state=d_state,
                **kwargs
            )
            self.out_proj = nn.Conv2d(decoder_dim, self.out_dim, 1) if decoder_dim != self.out_dim else nn.Identity()
        
        self.norm = nn.BatchNorm2d(self.out_dim)
    
    def forward(
        self,
        encoder_feat: torch.Tensor,
        decoder_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse encoder and decoder features.
        
        Args:
            encoder_feat: Encoder skip features (B, C_enc, H, W)
            decoder_feat: Upsampled decoder features (B, C_dec, H, W)
            
        Returns:
            Fused features (B, out_dim, H, W)
        """
        # Ensure spatial dimensions match
        if encoder_feat.shape[2:] != decoder_feat.shape[2:]:
            encoder_feat = F.interpolate(
                encoder_feat,
                size=decoder_feat.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        
        if self.fusion_mode == 'concat':
            x = torch.cat([encoder_feat, decoder_feat], dim=1)
            x = self.fuse_proj(x)
            x = self.mamba(x)
            
        elif self.fusion_mode == 'add':
            enc = self.enc_proj(encoder_feat)
            dec = self.dec_proj(decoder_feat)
            x = enc + dec
            x = self.mamba(x)
            
        elif self.fusion_mode == 'cross_mamba':
            enc = self.enc_proj(encoder_feat)
            x = self.cross_mamba(decoder_feat, enc)
            x = self.out_proj(x)
        
        return self.norm(x)


class CrossMambaFusion(nn.Module):
    """
    Cross-Mamba module for feature interaction.
    
    Allows decoder features to be conditioned on encoder features
    through a cross-attention-like Mamba mechanism.
    
    Args:
        dim: Feature dimension
        mamba_type: Type of Mamba block
        d_state: SSM state dimension
    """
    
    def __init__(
        self,
        dim: int,
        mamba_type: str = 'vmamba',
        d_state: int = 16,
        expand: float = 2.0,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.d_inner = int(dim * expand)
        
        # Project decoder for query-like role
        self.dec_proj = nn.Linear(dim, self.d_inner * 2)
        
        # Project encoder for key/value-like role
        self.enc_proj = nn.Linear(dim, self.d_inner)
        
        # Mamba for processing combined features
        self.mamba = create_mamba_block(
            variant=mamba_type,
            dim=self.d_inner,
            d_state=d_state,
            expand=1.0,  # Already expanded
            **kwargs
        )
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        decoder_feat: torch.Tensor,
        encoder_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-attention-like Mamba fusion.
        
        Args:
            decoder_feat: Decoder features (B, C, H, W)
            encoder_feat: Encoder features (B, C, H, W)
            
        Returns:
            Fused features (B, C, H, W)
        """
        B, C, H, W = decoder_feat.shape
        
        # Reshape to sequence
        dec = rearrange(decoder_feat, 'b c h w -> b (h w) c')
        enc = rearrange(encoder_feat, 'b c h w -> b (h w) c')
        
        # Project
        dec_proj = self.dec_proj(dec)  # (B, L, 2*d_inner)
        dec_x, dec_gate = dec_proj.chunk(2, dim=-1)
        enc_proj = self.enc_proj(enc)  # (B, L, d_inner)
        
        # Combine: modulate decoder with encoder
        combined = dec_x * torch.sigmoid(enc_proj) + enc_proj
        
        # Process with Mamba
        combined = rearrange(combined, 'b (h w) c -> b c h w', h=H, w=W)
        processed = self.mamba(combined)
        processed = rearrange(processed, 'b c h w -> b (h w) c')
        
        # Gate and output
        output = processed * torch.sigmoid(dec_gate)
        output = self.out_proj(output)
        output = self.norm(output + dec)  # Residual
        
        # Reshape back
        output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
        
        return output


class GatedMambaSkip(nn.Module):
    """
    Gated skip connection with Mamba processing.
    
    Learns to gate how much encoder information to pass through,
    with Mamba processing for sequence-aware gating.
    
    Args:
        encoder_dim: Encoder feature dimension (or encoder_channels)
        decoder_dim: Decoder feature dimension (or decoder_channels)
        mamba_type: Type of Mamba block
    """
    
    def __init__(
        self,
        encoder_dim: int = None,
        decoder_dim: int = None,
        encoder_channels: int = None,  # Alias for encoder_dim
        decoder_channels: int = None,  # Alias for decoder_dim
        mamba_type: str = 'vmamba',
        d_state: int = 16,
        **kwargs
    ):
        super().__init__()
        
        # Support both naming conventions
        encoder_dim = encoder_dim or encoder_channels
        decoder_dim = decoder_dim or decoder_channels
        
        if encoder_dim is None or decoder_dim is None:
            raise ValueError("Must provide encoder_dim/encoder_channels and decoder_dim/decoder_channels")
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        # Project encoder features
        self.enc_proj = nn.Sequential(
            nn.Conv2d(encoder_dim, decoder_dim, 1),
            nn.BatchNorm2d(decoder_dim)
        )
        
        # Mamba for gate computation
        self.gate_mamba = create_mamba_block(
            variant=mamba_type,
            dim=decoder_dim * 2,
            d_state=d_state,
            expand=1.0,
            **kwargs
        )
        
        # Gate projection
        self.gate_proj = nn.Sequential(
            nn.Conv2d(decoder_dim * 2, decoder_dim, 1),
            nn.Sigmoid()
        )
        
        # Output refinement
        self.output = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        encoder_feat: torch.Tensor,
        decoder_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Gated skip fusion.
        
        Args:
            encoder_feat: Encoder features (B, C_enc, H, W)
            decoder_feat: Decoder features (B, C_dec, H, W)
            
        Returns:
            Gated fused features (B, C_dec, H, W)
        """
        # Match spatial dimensions
        if encoder_feat.shape[2:] != decoder_feat.shape[2:]:
            encoder_feat = F.interpolate(
                encoder_feat,
                size=decoder_feat.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        
        # Project encoder
        enc = self.enc_proj(encoder_feat)
        
        # Concatenate for gate computation
        combined = torch.cat([enc, decoder_feat], dim=1)
        
        # Compute gate using Mamba
        gate_feat = self.gate_mamba(combined)
        gate = self.gate_proj(gate_feat)
        
        # Apply gate
        gated = gate * enc + (1 - gate) * decoder_feat
        
        # Output
        output = self.output(gated)
        
        return output


class AttentionGateMamba(nn.Module):
    """
    Attention Gate enhanced with Mamba.
    
    Combines traditional attention gates (from Attention U-Net)
    with Mamba for improved spatial attention.
    
    Args:
        encoder_dim: Encoder feature dimension (F_l)
        decoder_dim: Decoder/gating feature dimension (F_g)
        inter_dim: Intermediate dimension
        mamba_type: Type of Mamba block
    """
    
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        inter_dim: Optional[int] = None,
        mamba_type: str = 'vmamba',
        d_state: int = 8,
        **kwargs
    ):
        super().__init__()
        
        inter_dim = inter_dim or encoder_dim // 2
        
        # Standard attention gate components
        self.W_g = nn.Sequential(
            nn.Conv2d(decoder_dim, inter_dim, 1, bias=False),
            nn.BatchNorm2d(inter_dim)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(encoder_dim, inter_dim, 1, bias=False),
            nn.BatchNorm2d(inter_dim)
        )
        
        # Mamba for attention refinement
        self.mamba = create_mamba_block(
            variant=mamba_type,
            dim=inter_dim,
            d_state=d_state,
            expand=1.5,
            **kwargs
        )
        
        # Attention coefficient
        self.psi = nn.Sequential(
            nn.Conv2d(inter_dim, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output = nn.Sequential(
            nn.Conv2d(encoder_dim, encoder_dim, 1),
            nn.BatchNorm2d(encoder_dim)
        )
    
    def forward(
        self,
        encoder_feat: torch.Tensor,  # x (skip connection)
        decoder_feat: torch.Tensor   # g (gating signal from decoder)
    ) -> torch.Tensor:
        """
        Attention-gated skip connection.
        
        Args:
            encoder_feat: Encoder skip features (B, C_enc, H_enc, W_enc)
            decoder_feat: Decoder gating signal (B, C_dec, H_dec, W_dec)
            
        Returns:
            Attended encoder features (B, C_enc, H_enc, W_enc)
        """
        # Upsample decoder features to match encoder spatial size
        g = F.interpolate(
            decoder_feat,
            size=encoder_feat.shape[2:],
            mode='bilinear',
            align_corners=True
        )
        
        # Compute attention
        g1 = self.W_g(g)
        x1 = self.W_x(encoder_feat)
        
        # Combine and process with Mamba
        psi = F.relu(g1 + x1, inplace=True)
        psi = self.mamba(psi)
        
        # Compute attention coefficients
        alpha = self.psi(psi)
        
        # Apply attention
        output = encoder_feat * alpha
        output = self.output(output)
        
        return output


class MultiScaleMambaSkip(nn.Module):
    """
    Multi-scale skip connection with Mamba.
    
    Processes skip connections at multiple scales and fuses them,
    useful for capturing both fine and coarse details.
    
    Args:
        dim: Feature dimension
        scales: Number of scales to process
        mamba_type: Type of Mamba block
    """
    
    def __init__(
        self,
        dim: int,
        scales: int = 3,
        mamba_type: str = 'vmamba',
        d_state: int = 16,
        **kwargs
    ):
        super().__init__()
        
        self.scales = scales
        
        # Multi-scale processing
        self.scale_branches = nn.ModuleList()
        for s in range(scales):
            pool_size = 2 ** s
            branch = nn.ModuleDict({
                'pool': nn.AvgPool2d(pool_size) if pool_size > 1 else nn.Identity(),
                'mamba': create_mamba_block(
                    variant=mamba_type,
                    dim=dim,
                    d_state=d_state,
                    **kwargs
                ),
            })
            self.scale_branches.append(branch)
        
        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * scales, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale skip processing.
        
        Args:
            x: Skip features (B, C, H, W)
            
        Returns:
            Multi-scale processed features (B, C, H, W)
        """
        H, W = x.shape[2:]
        
        outputs = []
        for branch in self.scale_branches:
            # Pool
            pooled = branch['pool'](x)
            
            # Mamba
            processed = branch['mamba'](pooled)
            
            # Upsample back
            if processed.shape[2:] != (H, W):
                processed = F.interpolate(
                    processed,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=True
                )
            
            outputs.append(processed)
        
        # Fuse
        x = torch.cat(outputs, dim=1)
        x = self.fuse(x)
        
        return x


if __name__ == "__main__":
    # Test skip connection modules
    print("Testing Mamba Skip Connection modules...")
    
    B, H, W = 2, 32, 32
    encoder_feat = torch.randn(B, 256, H, W)
    decoder_feat = torch.randn(B, 128, H, W)
    
    # Test MambaSkipConnection
    print("\n1. Testing MambaSkipConnection...")
    skip = MambaSkipConnection(dim=256, depth=1, mamba_type='vmamba')
    out = skip(encoder_feat)
    print(f"   Input: {encoder_feat.shape} -> Output: {out.shape}")
    
    # Test MambaSkipFusion (concat mode)
    print("\n2. Testing MambaSkipFusion (concat)...")
    fusion = MambaSkipFusion(
        encoder_dim=256,
        decoder_dim=128,
        out_dim=128,
        fusion_mode='concat'
    )
    out = fusion(encoder_feat, decoder_feat)
    print(f"   Encoder: {encoder_feat.shape}, Decoder: {decoder_feat.shape}")
    print(f"   Output: {out.shape}")
    
    # Test MambaSkipFusion (cross_mamba mode)
    print("\n3. Testing MambaSkipFusion (cross_mamba)...")
    fusion_cross = MambaSkipFusion(
        encoder_dim=256,
        decoder_dim=128,
        out_dim=128,
        fusion_mode='cross_mamba'
    )
    out = fusion_cross(encoder_feat, decoder_feat)
    print(f"   Output: {out.shape}")
    
    # Test GatedMambaSkip
    print("\n4. Testing GatedMambaSkip...")
    gated = GatedMambaSkip(encoder_dim=256, decoder_dim=128)
    out = gated(encoder_feat, decoder_feat)
    print(f"   Output: {out.shape}")
    
    # Test AttentionGateMamba
    print("\n5. Testing AttentionGateMamba...")
    attn_gate = AttentionGateMamba(
        encoder_dim=256,
        decoder_dim=128
    )
    out = attn_gate(encoder_feat, decoder_feat)
    print(f"   Output: {out.shape}")
    
    # Test MultiScaleMambaSkip
    print("\n6. Testing MultiScaleMambaSkip...")
    ms_skip = MultiScaleMambaSkip(dim=256, scales=3)
    out = ms_skip(encoder_feat)
    print(f"   Input: {encoder_feat.shape} -> Output: {out.shape}")
    
    print("\nAll skip connection tests passed!")
