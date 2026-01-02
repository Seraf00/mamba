"""
Mamba-GUDU - GUDU (Globally Guided Dense UNet) Enhanced with Mamba

GUDU architecture with Mamba integration for enhanced
global context modeling and dense feature refinement.

Integration points:
- Global context: Replace/augment with Mamba for better long-range
- Dense connections: Mamba-enhanced feature fusion
- Channel attention: Hybrid attention-Mamba
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Literal

import sys
sys.path.append('..')

from models.modules import (
    create_mamba_block,
    MambaBottleneck,
    MambaSkipConnection,
    HybridAttentionMamba
)


class MambaChannelAttention(nn.Module):
    """Channel attention enhanced with Mamba for global context."""
    
    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        # Standard channel attention path
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        # Mamba path for global context
        self.mamba = create_mamba_block(
            variant=mamba_type,
            dim=in_channels,
            d_state=d_state
        )
        
        # Fusion
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        channel_attn = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        
        # Mamba global context
        mamba_out = self.mamba(x)
        
        # Apply channel attention and add Mamba residual
        return x * channel_attn + 0.1 * mamba_out


class MambaGlobalContext(nn.Module):
    """Global context module with Mamba for long-range dependencies."""
    
    def __init__(
        self,
        in_channels: int,
        mamba_type: str = 'vmamba',
        d_state: int = 16,
        num_mamba_layers: int = 2
    ):
        super().__init__()
        
        # Multi-layer Mamba for global context
        self.mamba_layers = nn.ModuleList([
            create_mamba_block(
                variant=mamba_type,
                dim=in_channels,
                d_state=d_state
            )
            for _ in range(num_mamba_layers)
        ])
        
        # Global pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU(inplace=True)
        )
        
        # Fusion
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.norm = nn.BatchNorm2d(in_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Mamba global context
        mamba_out = x
        for mamba in self.mamba_layers:
            mamba_out = mamba_out + mamba(mamba_out)
        
        # Global pooling
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Fusion
        out = self.fusion(torch.cat([mamba_out, global_feat], dim=1))
        out = self.norm(out)
        
        return out + residual


class DenseBlock(nn.Module):
    """Dense block with optional Mamba enhancement."""
    
    def __init__(
        self,
        in_channels: int,
        growth_rate: int = 32,
        num_layers: int = 4,
        use_mamba: bool = False,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channels + i * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1)
                )
            )
        
        self.mamba = None
        if use_mamba:
            self.mamba = create_mamba_block(
                variant=mamba_type,
                dim=in_channels + num_layers * growth_rate,
                d_state=d_state
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        
        out = torch.cat(features, dim=1)
        
        if self.mamba is not None:
            out = out + self.mamba(out)
        
        return out


class TransitionDown(nn.Module):
    """Transition layer for downsampling."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transition(x)


class MambaDenseSkip(nn.Module):
    """Dense skip connection with Mamba enhancement for GUDU."""
    
    def __init__(
        self,
        channels_list: List[int],
        out_channels: int,
        mamba_type: str = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        # Project each skip to common channel
        self.projections = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1) for ch in channels_list
        ])
        
        # Mamba for fused features
        self.mamba = create_mamba_block(
            variant=mamba_type,
            dim=out_channels,
            d_state=d_state
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(channels_list), out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: List[torch.Tensor], target_size: tuple) -> torch.Tensor:
        projected = []
        for feat, proj in zip(features, self.projections):
            feat = proj(feat)
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            feat = feat + self.mamba(feat)  # Mamba refinement
            projected.append(feat)
        
        return self.fusion(torch.cat(projected, dim=1))


class MambaGUDU(nn.Module):
    """
    Mamba-Enhanced GUDU (Globally Guided Dense UNet).
    
    GUDU with Mamba integration for improved global context:
    - Dense blocks with Mamba in deeper stages
    - Mamba global context module
    - Mamba-enhanced channel attention
    - Dense skip connections with Mamba
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        base_features: Initial number of features
        growth_rate: Growth rate for dense blocks
        num_layers_per_block: Layers in each dense block
        mamba_type: Type of Mamba ('mamba', 'mamba2', 'vmamba')
        d_state: SSM state dimension
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_features: int = 64,
        growth_rate: int = 32,
        num_layers_per_block: List[int] = [4, 4, 4, 4],
        mamba_type: Literal['mamba', 'mamba2', 'vmamba'] = 'vmamba',
        d_state: int = 16
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mamba_type = mamba_type
        
        num_blocks = len(num_layers_per_block)
        
        # Initial conv
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_features, 3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True)
        )
        
        # Encoder: Dense blocks with transitions
        self.encoder_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.channel_attentions = nn.ModuleList()
        
        current_channels = base_features
        encoder_channels = [base_features]
        
        for i, num_layers in enumerate(num_layers_per_block):
            # Dense block (Mamba in deeper blocks)
            use_mamba = i >= num_blocks // 2
            block = DenseBlock(
                current_channels,
                growth_rate,
                num_layers,
                use_mamba=use_mamba,
                mamba_type=mamba_type,
                d_state=d_state
            )
            self.encoder_blocks.append(block)
            
            current_channels = current_channels + num_layers * growth_rate
            encoder_channels.append(current_channels)
            
            # Mamba channel attention
            self.channel_attentions.append(
                MambaChannelAttention(
                    current_channels,
                    mamba_type=mamba_type,
                    d_state=d_state
                )
            )
            
            # Transition (except last)
            if i < num_blocks - 1:
                trans_out = current_channels // 2
                self.transitions.append(TransitionDown(current_channels, trans_out))
                current_channels = trans_out
        
        # Mamba global context bottleneck
        self.global_context = MambaGlobalContext(
            encoder_channels[-1],
            mamba_type=mamba_type,
            d_state=d_state,
            num_mamba_layers=2
        )
        
        # Decoder with dense skip connections
        self.decoder_ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.dense_skips = nn.ModuleList()
        
        dec_channels = encoder_channels[-1]
        skip_channels = encoder_channels[:-1][::-1]
        
        for i in range(num_blocks - 1):
            # Upsample
            out_ch = skip_channels[i] // 2
            self.decoder_ups.append(
                nn.ConvTranspose2d(dec_channels, out_ch, kernel_size=2, stride=2)
            )
            
            # Dense skip (uses all previous encoder features)
            self.dense_skips.append(
                MambaDenseSkip(
                    [encoder_channels[j] for j in range(num_blocks - i)],
                    out_ch,
                    mamba_type=mamba_type,
                    d_state=d_state
                )
            )
            
            # Decoder conv
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            
            dec_channels = out_ch
        
        # Final segmentation head
        self.seg_head = nn.Conv2d(dec_channels, num_classes, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        target_size = x.shape[2:]
        
        # Initial conv
        x = self.init_conv(x)
        
        # Encoder
        encoder_features = [x]
        for i, (block, ca) in enumerate(zip(self.encoder_blocks, self.channel_attentions)):
            x = block(x)
            x = ca(x)
            encoder_features.append(x)
            
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        # Global context bottleneck
        x = self.global_context(x)
        
        # Decoder with dense skip connections
        for i, (up, dense_skip, dec) in enumerate(zip(
            self.decoder_ups, self.dense_skips, self.decoder_blocks
        )):
            x = up(x)
            
            # Get features for dense skip
            skip_feats = encoder_features[:len(encoder_features) - i]
            skip = dense_skip(skip_feats, x.shape[2:])
            
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        
        # Segmentation head
        out = self.seg_head(x)
        
        if out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=True)
        
        return out
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience functions
def mamba_gudu_small(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaGUDU:
    """Small Mamba-GUDU."""
    return MambaGUDU(
        in_channels, num_classes,
        base_features=32,
        growth_rate=16,
        num_layers_per_block=[3, 3, 3, 3],
        mamba_type=mamba_type
    )


def mamba_gudu_base(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaGUDU:
    """Standard Mamba-GUDU."""
    return MambaGUDU(
        in_channels, num_classes,
        base_features=64,
        growth_rate=32,
        num_layers_per_block=[4, 4, 4, 4],
        mamba_type=mamba_type
    )


def mamba_gudu_large(
    in_channels: int = 1,
    num_classes: int = 4,
    mamba_type: str = 'vmamba'
) -> MambaGUDU:
    """Large Mamba-GUDU."""
    return MambaGUDU(
        in_channels, num_classes,
        base_features=64,
        growth_rate=48,
        num_layers_per_block=[6, 6, 6, 6],
        mamba_type=mamba_type
    )


if __name__ == '__main__':
    # Test the model
    model = MambaGUDU(
        in_channels=1, num_classes=4,
        mamba_type='vmamba'
    )
    print(f"Mamba-GUDU Parameters: {model.count_parameters():,}")
    
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
