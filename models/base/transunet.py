"""
TransUNet - Transformers for Medical Image Segmentation

Implementation based on "TransUNet: Transformers Make Strong Encoders for
Medical Image Segmentation" by Chen et al., 2021.

Key architectural details matching the paper:
- ResNetV2 encoder with Weight Standardization (StdConv2d) and GroupNorm
- Layers (3, 4, 9) producing features at /4 (256ch), /8 (512ch), /16 (1024ch)
- ViT-B/16 transformer with 12 layers, 768-dim embeddings, 12 heads
- Conv1x1 projection (768->512) after transformer
- Cascaded upsampler decoder with channels (256, 128, 64, 16)
- Skip connections from encoder stages
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
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
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to reduce memory at cost of ~20% speed."""
        self.gradient_checkpointing = True

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
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
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


# ---------------------------------------------------------------------------
# Fix 1: Weight Standardization Conv2d (used in ResNetV2 per TransUNet paper)
# ---------------------------------------------------------------------------

class StdConv2d(nn.Conv2d):
    """Convolution with Weight Standardization (used in ResNetV2 per TransUNet paper)."""
    def forward(self, x):
        w = self.weight
        w = w - w.mean(dim=[1, 2, 3], keepdim=True)
        w = w / (w.std(dim=[1, 2, 3], keepdim=True) + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


# ---------------------------------------------------------------------------
# Fix 2: Custom ResNetV2 encoder with layers=(3,4,9) and StdConv2d + GroupNorm
# ---------------------------------------------------------------------------

class PreActBottleneck(nn.Module):
    """Pre-activation Bottleneck block with Weight Standardization (TransUNet paper)."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = out_channels // 4
        self.gn1 = nn.GroupNorm(32, in_channels)
        self.conv1 = StdConv2d(in_channels, mid_channels, 1, bias=False)
        self.gn2 = nn.GroupNorm(32, mid_channels)
        self.conv2 = StdConv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(32, mid_channels)
        self.conv3 = StdConv2d(mid_channels, out_channels, 1, bias=False)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = StdConv2d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x):
        residual = x
        y = F.relu(self.gn1(x))
        if self.downsample is not None:
            residual = self.downsample(y)
        y = self.conv1(y)
        y = self.conv2(F.relu(self.gn2(y)))
        y = self.conv3(F.relu(self.gn3(y)))
        return y + residual


class ResNetV2Encoder(nn.Module):
    """Custom ResNetV2 encoder matching TransUNet paper (layers 3, 4, 9)."""
    def __init__(self, in_channels=1, layers=(3, 4, 9)):
        super().__init__()
        # Stem
        self.conv1 = StdConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(32, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_stage(64, 256, layers[0], stride=1)    # /4, 256 ch
        self.layer2 = self._make_stage(256, 512, layers[1], stride=2)   # /8, 512 ch
        self.layer3 = self._make_stage(512, 1024, layers[2], stride=2)  # /16, 1024 ch

        self.gn_out = nn.GroupNorm(32, 1024)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        blocks = [PreActBottleneck(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            blocks.append(PreActBottleneck(out_channels, out_channels, 1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x0 = F.relu(self.gn1(self.conv1(x)))  # /2, 64 ch
        x0_pool = self.maxpool(x0)             # /4, 64 ch
        x1 = self.layer1(x0_pool)              # /4, 256 ch
        x2 = self.layer2(x1)                   # /8, 512 ch
        x3 = self.layer3(x2)                   # /16, 1024 ch
        x3 = F.relu(self.gn_out(x3))
        return x0, x1, x2, x3


class TransUNet(nn.Module):
    """
    TransUNet: Hybrid CNN-Transformer for Medical Image Segmentation.

    Architecture (matching Chen et al., 2021):
    1. ResNetV2 encoder (layers 3,4,9) with Weight Standardization + GroupNorm
    2. Features flattened and processed by ViT-B/16 (12 layers, 768-dim, 12 heads)
    3. Conv1x1 projection (768 -> 512) after transformer
    4. Cascaded upsampler decoder with channels (256, 128, 64, 16)
    5. Skip connections from encoder stages (x0=64ch, x1=256ch, x2=512ch)

    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        img_size: Input image size (must be divisible by 16)
        pretrained: Use pretrained ViT-B/16 weights from timm (ImageNet-21k)
        vit_layers: Number of ViT layers
        vit_heads: Number of ViT attention heads
        vit_dim: ViT embedding dimension
        mlp_ratio: ViT MLP ratio
        drop_rate: Dropout rate
        drop_path_rate: Stochastic depth rate (unused currently, reserved for future)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        img_size: int = 256,
        pretrained: bool = True,
        vit_layers: int = 12,
        vit_heads: int = 12,
        vit_dim: int = 768,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        drop_path_rate: float = 0.0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size

        # ---------------------------------------------------------------
        # Fix 2: Custom ResNetV2 encoder (replaces torchvision ResNet50)
        # ---------------------------------------------------------------
        self.encoder = ResNetV2Encoder(in_channels=in_channels, layers=(3, 4, 9))

        # Skip channels from ResNetV2: x0=64 (/2), x1=256 (/4), x2=512 (/8)
        self.skip_channels = [64, 256, 512]

        # ---------------------------------------------------------------
        # Transformer encoder
        # Input: 1024 channels from encoder layer3
        # ---------------------------------------------------------------
        self.transformer = VisionTransformer(
            in_channels=1024,
            embed_dim=vit_dim,
            num_layers=vit_layers,
            num_heads=vit_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            max_seq_len=(img_size // 16) ** 2
        )

        # ---------------------------------------------------------------
        # Fix 3: Conv1x1 projection (768 -> 512) instead of Linear(768 -> 1024)
        # Reshape to spatial BEFORE projection (Conv2d operates on spatial dims)
        # ---------------------------------------------------------------
        self.trans_proj = nn.Sequential(
            nn.Conv2d(vit_dim, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ---------------------------------------------------------------
        # Fix 4: Decoder channels (256, 128, 64, 16) matching paper
        # ---------------------------------------------------------------
        decoder_channels = [256, 128, 64, 16]

        # decoder3: input=512 (from trans_proj), skip=x2 (512ch), out=256
        self.decoder3 = DecoderBlock(512, self.skip_channels[2], decoder_channels[0])   # /16->/8, skip=x2(512ch)
        # decoder2: input=256, skip=x1 (256ch), out=128
        self.decoder2 = DecoderBlock(decoder_channels[0], self.skip_channels[1], decoder_channels[1])  # /8->/4, skip=x1(256ch)
        # decoder1: input=128, skip=x0 (64ch), out=64
        self.decoder1 = DecoderBlock(decoder_channels[1], self.skip_channels[0], decoder_channels[2])   # /4->/2, skip=x0(64ch)
        # decoder0: input=64, no skip, out=16
        self.decoder0 = DecoderBlock(decoder_channels[2], 0, decoder_channels[3])    # /2->/1, no skip

        # Final output head
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)
        )

        self._init_decoder()

        # ---------------------------------------------------------------
        # Fix 5: Load pretrained weights from timm
        # ---------------------------------------------------------------
        if pretrained:
            self._load_pretrained_resnet()
            self._load_pretrained_vit()

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

    def _load_pretrained_resnet(self):
        """
        Load pretrained ResNetV2 (BiT) encoder weights from timm (ImageNet-21k).

        Maps timm's resnetv2_50x1_bit.goog_in21k weights to our ResNetV2Encoder.

        Key mapping:
          timm:  stem.conv.weight           -> ours: encoder.conv1.weight
          timm:  stages.{s}.blocks.{b}.norm{n}  -> ours: encoder.layer{s+1}.{b}.gn{n}
          timm:  stages.{s}.blocks.{b}.conv{n}  -> ours: encoder.layer{s+1}.{b}.conv{n}
          timm:  stages.{s}.blocks.{b}.downsample.conv -> ours: encoder.layer{s+1}.{b}.downsample

        Our encoder has layers=(3,4,9), timm's ResNet50 has stages=[3,4,6,3].
        We load stages 0-1 fully, stage 2 partially (6/9 blocks), skip stage 3.
        Handles grayscale input by averaging RGB stem conv weights.
        """
        try:
            import timm
        except ImportError:
            print(
                "[TransUNet] WARNING: timm is not installed. "
                "Pretrained ResNetV2 weights will NOT be loaded. "
                "Install with: pip install timm"
            )
            return

        print("[TransUNet] Loading pretrained ResNetV2 (BiT) encoder weights from timm (ImageNet-21k)...")

        # Create pretrained timm BiT model to extract weights
        timm_model = timm.create_model('resnetv2_50x1_bit.goog_in21k', pretrained=True)
        timm_state = timm_model.state_dict()

        mapped = {}

        # ---- Stem ----
        stem_weight = timm_state['stem.conv.weight']  # [64, 3, 7, 7]
        if self.in_channels != 3:
            # Average RGB weights to grayscale
            stem_weight = stem_weight.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
            if self.in_channels > 1:
                stem_weight = stem_weight.repeat(1, self.in_channels, 1, 1)
        mapped['encoder.conv1.weight'] = stem_weight
        # Note: timm BiT has no stem norm; our encoder.gn1 stays randomly initialized

        # ---- Stages ----
        # timm stages: [3, 4, 6, 3] blocks -> our layers: [3, 4, 9] blocks
        # Load stages 0-1 fully, stage 2 partially (first 6 of 9 blocks)
        timm_block_counts = [3, 4, 6]  # How many blocks we can load per our stage
        our_layer_names = ['layer1', 'layer2', 'layer3']

        for s, (layer_name, n_load) in enumerate(zip(our_layer_names, timm_block_counts)):
            for b in range(n_load):
                src_prefix = f'stages.{s}.blocks.{b}'
                dst_prefix = f'encoder.{layer_name}.{b}'

                # norm1/2/3 (GroupNorm) -> gn1/2/3
                for n in [1, 2, 3]:
                    src_key_w = f'{src_prefix}.norm{n}.weight'
                    src_key_b = f'{src_prefix}.norm{n}.bias'
                    if src_key_w in timm_state:
                        mapped[f'{dst_prefix}.gn{n}.weight'] = timm_state[src_key_w]
                        mapped[f'{dst_prefix}.gn{n}.bias'] = timm_state[src_key_b]

                # conv1/2/3 (StdConv2d)
                for n in [1, 2, 3]:
                    src_key = f'{src_prefix}.conv{n}.weight'
                    if src_key in timm_state:
                        mapped[f'{dst_prefix}.conv{n}.weight'] = timm_state[src_key]

                # downsample projection (only in block 0 of each stage)
                ds_key = f'{src_prefix}.downsample.conv.weight'
                if ds_key in timm_state:
                    mapped[f'{dst_prefix}.downsample.weight'] = timm_state[ds_key]

        # ---- Load mapped weights ----
        load_result = self.load_state_dict(mapped, strict=False)

        loaded_count = len(mapped)
        missing = [k for k in load_result.missing_keys if 'encoder.' in k]
        print(f"[TransUNet] Loaded {loaded_count} weight tensors from pretrained ResNetV2 (BiT).")
        if missing:
            print(f"[TransUNet]   Encoder keys not loaded (randomly init): {len(missing)} keys")
            # Show which layers weren't loaded
            unloaded_layers = set()
            for k in missing:
                parts = k.split('.')
                if len(parts) >= 3:
                    unloaded_layers.add('.'.join(parts[:3]))
            if unloaded_layers:
                print(f"[TransUNet]   Unloaded encoder sub-modules: {sorted(unloaded_layers)}")

        del timm_model, timm_state

    def _load_pretrained_vit(self):
        """
        Load pretrained ViT-B/16 weights from timm (ImageNet-21k).

        Maps timm ViT block weights to our VisionTransformer structure.
        Handles the conversion from timm's separate qkv to PyTorch
        nn.MultiheadAttention's in_proj_weight/in_proj_bias format.

        Skips: patch_embed, pos_embed, cls_token (different structure in our code).
        """
        try:
            import timm
        except ImportError:
            print(
                "[TransUNet] WARNING: timm is not installed. "
                "Pretrained ViT-B/16 weights will NOT be loaded. "
                "Install with: pip install timm"
            )
            return

        print("[TransUNet] Loading pretrained ViT-B/16 weights from timm (ImageNet-21k)...")

        # Create pretrained timm ViT model to extract weights
        timm_model = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True)
        timm_state = timm_model.state_dict()

        our_state = self.transformer.state_dict()
        mapped = {}

        num_layers = len(self.transformer.layers)

        for i in range(num_layers):
            prefix_timm = f'blocks.{i}.'
            prefix_ours = f'layers.{i}.'

            # LayerNorm 1 (pre-attention)
            mapped[f'{prefix_ours}norm1.weight'] = timm_state[f'{prefix_timm}norm1.weight']
            mapped[f'{prefix_ours}norm1.bias'] = timm_state[f'{prefix_timm}norm1.bias']

            # Multi-head attention
            # timm uses a fused qkv linear: blocks.{i}.attn.qkv.weight/bias (shape [3*dim, dim])
            # PyTorch nn.MultiheadAttention uses in_proj_weight/in_proj_bias (same shape [3*dim, dim])
            qkv_weight = timm_state[f'{prefix_timm}attn.qkv.weight']
            qkv_bias = timm_state[f'{prefix_timm}attn.qkv.bias']
            mapped[f'{prefix_ours}attn.in_proj_weight'] = qkv_weight
            mapped[f'{prefix_ours}attn.in_proj_bias'] = qkv_bias

            # Output projection
            mapped[f'{prefix_ours}attn.out_proj.weight'] = timm_state[f'{prefix_timm}attn.proj.weight']
            mapped[f'{prefix_ours}attn.out_proj.bias'] = timm_state[f'{prefix_timm}attn.proj.bias']

            # LayerNorm 2 (pre-MLP)
            mapped[f'{prefix_ours}norm2.weight'] = timm_state[f'{prefix_timm}norm2.weight']
            mapped[f'{prefix_ours}norm2.bias'] = timm_state[f'{prefix_timm}norm2.bias']

            # MLP: our Sequential is [Linear, GELU, Dropout, Linear, Dropout]
            # indices: 0=fc1, 3=fc2
            mapped[f'{prefix_ours}mlp.0.weight'] = timm_state[f'{prefix_timm}mlp.fc1.weight']
            mapped[f'{prefix_ours}mlp.0.bias'] = timm_state[f'{prefix_timm}mlp.fc1.bias']
            mapped[f'{prefix_ours}mlp.3.weight'] = timm_state[f'{prefix_timm}mlp.fc2.weight']
            mapped[f'{prefix_ours}mlp.3.bias'] = timm_state[f'{prefix_timm}mlp.fc2.bias']

        # Final LayerNorm
        mapped['norm.weight'] = timm_state['norm.weight']
        mapped['norm.bias'] = timm_state['norm.bias']

        # Load mapped weights (strict=False to skip keys we did not map, e.g. proj, pos_encoding)
        load_result = self.transformer.load_state_dict(mapped, strict=False)

        loaded_count = len(mapped)
        missing = load_result.missing_keys
        unexpected = load_result.unexpected_keys

        print(f"[TransUNet] Loaded {loaded_count} weight tensors from pretrained ViT-B/16.")
        if missing:
            print(f"[TransUNet]   Missing keys (expected, not in pretrained): {missing}")
        if unexpected:
            print(f"[TransUNet]   Unexpected keys: {unexpected}")

        # Clean up timm model to free memory
        del timm_model, timm_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, num_classes, H, W)
        """
        B, C, H_in, W_in = x.shape

        # ---------------------------------------------------------------
        # Encoder: ResNetV2 returns (x0, x1, x2, x3)
        #   x0: /2,  64 ch  (skip for decoder1)
        #   x1: /4,  256 ch (skip for decoder2)
        #   x2: /8,  512 ch (skip for decoder3)
        #   x3: /16, 1024 ch (input to transformer)
        # ---------------------------------------------------------------
        x0, x1, x2, x3 = self.encoder(x)

        # Flatten for transformer
        B, C_feat, H_feat, W_feat = x3.shape
        x_flat = rearrange(x3, 'b c h w -> b (h w) c')

        # Transformer
        x_trans = self.transformer(x_flat)

        # ---------------------------------------------------------------
        # Fix 3: Reshape to spatial BEFORE Conv1x1 projection
        # ---------------------------------------------------------------
        x_trans = rearrange(x_trans, 'b (h w) c -> b c h w', h=H_feat, w=W_feat)
        x_trans = self.trans_proj(x_trans)  # Conv1x1: 768 -> 512

        # ---------------------------------------------------------------
        # Decoder with skip connections
        # ---------------------------------------------------------------
        d3 = self.decoder3(x_trans, x2)  # /8,  skip=x2 (512ch)
        d2 = self.decoder2(d3, x1)       # /4,  skip=x1 (256ch)
        d1 = self.decoder1(d2, x0)       # /2,  skip=x0 (64ch)
        d0 = self.decoder0(d1, None)     # /1,  no skip

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
    """Base TransUNet with ViT-B/16 (12 layers, 768-dim, 12 heads)."""
    return TransUNet(
        in_channels, num_classes, img_size,
        pretrained=pretrained,
        vit_layers=12, vit_heads=12, vit_dim=768
    )


def transunet_small(
    in_channels: int = 1,
    num_classes: int = 4,
    img_size: int = 224,
    pretrained: bool = False
) -> TransUNet:
    """Small TransUNet with 6 ViT layers (no pretrained weights for non-standard dims)."""
    return TransUNet(
        in_channels, num_classes, img_size,
        pretrained=pretrained,
        vit_layers=6, vit_heads=8, vit_dim=512
    )


def transunet_large(
    in_channels: int = 1,
    num_classes: int = 4,
    img_size: int = 224,
    pretrained: bool = False
) -> TransUNet:
    """Large TransUNet with 24 ViT layers (no pretrained weights for non-standard dims)."""
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
