"""
Mamba Block Implementations for Medical Image Segmentation.

This module contains three variants of Mamba blocks:
1. MambaBlock - Original Mamba (S6) with selective state spaces
2. Mamba2Block - Mamba-2 with State Space Duality (SSD) 
3. VMMambaBlock - Visual Mamba with 2D cross-scan for images

Each block is designed to be a drop-in replacement for transformer
or convolutional blocks in segmentation architectures.
"""

import math
from typing import Optional, Tuple, Union
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Try to import mamba_ssm, fall back to pure PyTorch implementation
try:
    from mamba_ssm import Mamba, Mamba2
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not installed. Using pure PyTorch implementation (slower).")


# =============================================================================
# Helper Functions
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D feature maps (B, C, H, W)."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


# =============================================================================
# Original Mamba Block (S6 - Selective State Space)
# =============================================================================

class MambaBlock(nn.Module):
    """
    Original Mamba Block with Selective State Space Model (S6).
    
    This is the core Mamba block from "Mamba: Linear-Time Sequence Modeling 
    with Selective State Spaces" (Gu & Dao, 2023).
    
    Architecture:
        Input -> Norm -> Linear (expand) -> Conv1d -> SiLU -> SSM -> Linear (project) -> Output
                      |-> Linear -> SiLU (gate) --|
                                                  |-> Multiply
    
    Args:
        dim: Input/output dimension
        d_state: SSM state dimension (N in paper)
        d_conv: Local convolution width
        expand: Expansion factor for inner dimension
        dt_rank: Rank of delta projection ("auto" = dim // 16)
        dt_min: Minimum delta value
        dt_max: Maximum delta value
        dt_init: Delta initialization method
        dt_scale: Delta scale factor
        bias: Whether to use bias in linear layers
        conv_bias: Whether to use bias in conv layer
        use_fast_path: Use optimized CUDA kernels if available
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        use_fast_path: bool = True,
    ):
        super().__init__()
        
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(dim * expand)
        self.dt_rank = math.ceil(dim / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path and MAMBA_AVAILABLE
        
        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=bias)
        
        # Depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias
        )
        
        # SSM parameters
        # x_proj: project x to (delta, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # dt_proj: project delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # S4D real initialization for A
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            'n -> d n',
            d=self.d_inner
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, L, D) or (B, D, H, W)
            
        Returns:
            Output tensor of same shape as input
        """
        # Handle 2D input (images)
        is_2d = x.ndim == 4
        if is_2d:
            B, C, H, W = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
        
        batch_size, seq_len, _ = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)
        
        # Convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, 'b d l -> b l d')
        
        # Activation
        x = F.silu(x)
        
        # SSM
        y = self.ssm(x)
        
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        if is_2d:
            output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
        
        return output
    
    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective State Space Model computation.
        
        Args:
            x: Input tensor (B, L, D)
            
        Returns:
            Output tensor (B, L, D)
        """
        batch_size, seq_len, _ = x.shape
        
        # Get A (negative for stability)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        D = self.D.float()
        
        # Compute delta, B, C from x
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        delta, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Project delta
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        
        # Selective scan
        y = self.selective_scan(x, delta, A, B, C, D)
        
        return y
    
    def selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor
    ) -> torch.Tensor:
        """
        Selective scan algorithm (pure PyTorch implementation).
        
        This is the core S6 algorithm that makes Mamba selective.
        Uses discretization: A_bar = exp(delta * A), B_bar = delta * B
        
        Args:
            x: Input (B, L, D)
            delta: Time step (B, L, D)
            A: State matrix (D, N)
            B: Input matrix (B, L, N)
            C: Output matrix (B, L, N)
            D: Skip connection (D,)
            
        Returns:
            Output (B, L, D)
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Discretize A and B
        # A_bar = exp(delta * A)
        delta_A = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, N)
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, N)
        
        # Scan
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for i in range(seq_len):
            h = delta_A[:, i] * h + delta_B[:, i] * x[:, i].unsqueeze(-1)
            y = (h * C[:, i].unsqueeze(1)).sum(dim=-1)  # (B, D)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (B, L, D)
        y = y + x * D
        
        return y


class MambaLayer(nn.Module):
    """
    Full Mamba layer with normalization and residual connection.
    
    Architecture:
        Input -> Norm -> MambaBlock -> + -> Output
          |__________________________|
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        drop_path: float = 0.0,
        norm_type: str = "layer",
        **kwargs
    ):
        super().__init__()
        
        if norm_type == "layer":
            self.norm = nn.LayerNorm(dim)
        elif norm_type == "rms":
            self.norm = RMSNorm(dim)
        else:
            self.norm = nn.Identity()
        
        self.mamba = MambaBlock(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs
        )
        
        self.drop_path = nn.Identity() if drop_path == 0 else DropPath(drop_path)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle 2D input
        is_2d = x.ndim == 4
        if is_2d:
            B, C, H, W = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Residual + Mamba
        x = x + self.drop_path(self.mamba(self.norm(x)))
        
        if is_2d:
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x


# =============================================================================
# Mamba-2 Block (State Space Duality)
# =============================================================================

class Mamba2Block(nn.Module):
    """
    Mamba-2 Block with State Space Duality (SSD).
    
    From "Transformers are SSMs: Generalized Models and Efficient Algorithms
    Through Structured State Space Duality" (Dao & Gu, 2024).
    
    Key differences from Mamba-1:
    - Uses matrix-valued state spaces (scalar -> matrix A)
    - More efficient parallel scan with SSD framework
    - Supports multi-head structure similar to attention
    
    Args:
        dim: Input/output dimension
        d_state: SSM state dimension (per head)
        d_conv: Local convolution width
        expand: Expansion factor
        n_heads: Number of SSM heads
        head_dim: Dimension per head (auto if None)
        chunk_size: Size of chunks for parallel scan
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 8,
        head_dim: Optional[int] = None,
        chunk_size: int = 256,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(dim * expand)
        self.n_heads = n_heads
        self.head_dim = head_dim or (self.d_inner // n_heads)
        self.chunk_size = chunk_size
        
        assert self.d_inner % n_heads == 0, "d_inner must be divisible by n_heads"
        
        # Input projection: x -> (z, x, B, C, dt)
        d_in_proj = self.d_inner + self.d_inner + n_heads + d_state + d_state
        self.in_proj = nn.Linear(dim, d_in_proj, bias=bias)
        
        # Depthwise convolution on x
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias
        )
        
        # A parameter (log form for stability)
        self.A_log = nn.Parameter(torch.zeros(n_heads))
        
        # D (skip connection per head)
        self.D = nn.Parameter(torch.ones(n_heads))
        
        # dt bias
        self.dt_bias = nn.Parameter(torch.zeros(n_heads))
        
        # Output norm and projection
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Mamba-2.
        
        Args:
            x: Input (B, L, D) or (B, C, H, W)
            
        Returns:
            Output of same shape
        """
        is_2d = x.ndim == 4
        if is_2d:
            B, C, H, W = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
        
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        zxBCdt = self.in_proj(x)
        
        # Split projections
        z, x, B, C, dt = zxBCdt.split(
            [self.d_inner, self.d_inner, self.n_heads, self.d_state, self.d_state],
            dim=-1
        )
        
        # Convolution on x
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)
        
        # Reshape for multi-head
        x = rearrange(x, 'b l (h d) -> b l h d', h=self.n_heads)
        
        # Get A (negative for stability)
        A = -torch.exp(self.A_log.float())  # (n_heads,)
        
        # Process dt
        dt = F.softplus(dt + self.dt_bias)  # (B, L, n_heads)
        
        # SSD scan
        y = self.ssd_scan(x, dt, A, B, C)
        
        # Reshape back
        y = rearrange(y, 'b l h d -> b l (h d)')
        
        # Gate with z
        y = y * F.silu(z)
        
        # Output
        y = self.norm(y)
        output = self.out_proj(y)
        
        if is_2d:
            output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
        
        return output
    
    def ssd_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        State Space Duality scan.
        
        Args:
            x: Input (B, L, H, D)
            dt: Time step (B, L, H)
            A: State matrix (H,)
            B: Input projection (B, L, N)
            C: Output projection (B, L, N)
            
        Returns:
            Output (B, L, H, D)
        """
        batch_size, seq_len, n_heads, head_dim = x.shape
        
        # Expand A, B, C for computation
        # A: (H,) -> (B, L, H, 1)
        A = A.view(1, 1, n_heads, 1).expand(batch_size, seq_len, -1, -1)
        
        # Discretize: A_bar = exp(dt * A)
        dt = dt.unsqueeze(-1)  # (B, L, H, 1)
        A_bar = torch.exp(dt * A)  # (B, L, H, 1)
        
        # B and C: (B, L, N) - broadcast over heads
        B = B.unsqueeze(2).expand(-1, -1, n_heads, -1)  # (B, L, H, N)
        C = C.unsqueeze(2).expand(-1, -1, n_heads, -1)  # (B, L, H, N)
        
        # Simple sequential scan (can be parallelized with chunk-wise processing)
        d_state = B.shape[-1]
        h = torch.zeros(batch_size, n_heads, head_dim, d_state, device=x.device, dtype=x.dtype)
        
        ys = []
        for i in range(seq_len):
            # h = A_bar * h + B * x
            h = A_bar[:, i].unsqueeze(-1) * h + B[:, i].unsqueeze(2) * x[:, i].unsqueeze(-1)
            # y = (h * C).sum(-1)
            y = (h * C[:, i].unsqueeze(2)).sum(dim=-1)  # (B, H, D)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (B, L, H, D)
        
        # Add skip connection
        y = y + x * self.D.view(1, 1, n_heads, 1)
        
        return y


class Mamba2Layer(nn.Module):
    """
    Full Mamba-2 layer with normalization and residual.
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 8,
        drop_path: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.mamba2 = Mamba2Block(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_heads=n_heads,
            **kwargs
        )
        self.drop_path = nn.Identity() if drop_path == 0 else DropPath(drop_path)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_2d = x.ndim == 4
        if is_2d:
            B, C, H, W = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
        
        x = x + self.drop_path(self.mamba2(self.norm(x)))
        
        if is_2d:
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x


# =============================================================================
# Visual Mamba Block (VM-Mamba / VMamba with 2D Cross-Scan)
# =============================================================================

class SS2D(nn.Module):
    """
    2D Selective Scan module for Visual Mamba.
    
    Implements cross-scan strategy: scan image patches in 4 directions
    (left-right, right-left, top-bottom, bottom-top) to capture 2D spatial
    relationships in a 1D SSM framework.
    
    From "VMamba: Visual State Space Model" (Liu et al., 2024)
    
    Args:
        dim: Feature dimension
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion ratio
        dt_rank: Rank of dt projection
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        dt_rank: Union[int, str] = "auto",
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(dim * expand)
        self.dt_rank = math.ceil(dim / 16) if dt_rank == "auto" else dt_rank
        
        # Number of scan directions
        self.K = 4  # 4 directions for cross-scan
        
        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=bias)
        
        # 2D convolution (instead of 1D for spatial features)
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_inner,
            bias=conv_bias
        )
        
        # Separate projections for each scan direction
        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
            for _ in range(self.K)
        ])
        
        self.dt_projs = nn.ModuleList([
            nn.Linear(self.dt_rank, self.d_inner, bias=True)
            for _ in range(self.K)
        ])
        
        # Initialize dt_projs bias
        for dt_proj in self.dt_projs:
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            ).clamp(min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
        
        # A and D for each direction
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            'n -> k d n',
            k=self.K,
            d=self.d_inner
        )
        self.A_logs = nn.Parameter(torch.log(A))
        self.Ds = nn.Parameter(torch.ones(self.K, self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with 2D cross-scan.
        
        Args:
            x: Input (B, C, H, W)
            
        Returns:
            Output (B, C, H, W)
        """
        B, C, H, W = x.shape
        L = H * W
        
        # Input projection
        x = rearrange(x, 'b c h w -> b (h w) c')
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner)
        
        # Reshape for conv2d
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.conv2d(x)
        x = F.silu(x)
        
        # Cross-scan in 4 directions
        xs = self.cross_scan(x)  # List of 4 tensors, each (B, L, d_inner)
        
        # Apply SSM to each direction
        ys = []
        for k in range(self.K):
            # Get parameters for this direction
            A = -torch.exp(self.A_logs[k].float())
            D = self.Ds[k].float()
            
            # Project x to get delta, B, C
            x_k = xs[k]
            x_dbl = self.x_proj[k](x_k)
            delta, B_k, C_k = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
            delta = F.softplus(self.dt_projs[k](delta))
            
            # SSM scan
            y_k = self.selective_scan(x_k, delta, A, B_k, C_k, D)
            ys.append(y_k)
        
        # Cross-merge: reverse the scans and sum
        y = self.cross_merge(ys, H, W)  # (B, L, d_inner)
        
        # Gate
        y = y * F.silu(z)
        
        # Output projection
        y = self.out_proj(y)
        y = rearrange(y, 'b (h w) c -> b c h w', h=H, w=W)
        
        return y
    
    def cross_scan(self, x: torch.Tensor) -> list:
        """
        Scan image in 4 directions.
        
        Args:
            x: Input (B, C, H, W)
            
        Returns:
            List of 4 tensors, each (B, L, C)
        """
        B, C, H, W = x.shape
        
        # Direction 1: Left to Right (row-major)
        x1 = rearrange(x, 'b c h w -> b (h w) c')
        
        # Direction 2: Right to Left (reversed row-major)
        x2 = rearrange(x.flip(-1), 'b c h w -> b (h w) c')
        
        # Direction 3: Top to Bottom (column-major)
        x3 = rearrange(x.permute(0, 1, 3, 2), 'b c w h -> b (w h) c')
        
        # Direction 4: Bottom to Top (reversed column-major)
        x4 = rearrange(x.permute(0, 1, 3, 2).flip(-1), 'b c w h -> b (w h) c')
        
        return [x1, x2, x3, x4]
    
    def cross_merge(self, ys: list, H: int, W: int) -> torch.Tensor:
        """
        Merge scanned outputs back to original order and sum.
        
        Args:
            ys: List of 4 tensors, each (B, L, C)
            H, W: Spatial dimensions
            
        Returns:
            Merged output (B, L, C)
        """
        B, L, C = ys[0].shape
        
        # Reverse direction 2
        y2 = rearrange(ys[1], 'b (h w) c -> b c h w', h=H, w=W)
        y2 = y2.flip(-1)
        y2 = rearrange(y2, 'b c h w -> b (h w) c')
        
        # Reverse direction 3 (transpose back)
        y3 = rearrange(ys[2], 'b (w h) c -> b c w h', h=H, w=W)
        y3 = y3.permute(0, 1, 3, 2)  # Back to (B, C, H, W)
        y3 = rearrange(y3, 'b c h w -> b (h w) c')
        
        # Reverse direction 4 (transpose + flip back)
        y4 = rearrange(ys[3], 'b (w h) c -> b c w h', h=H, w=W)
        y4 = y4.flip(-1).permute(0, 1, 3, 2)
        y4 = rearrange(y4, 'b c h w -> b (h w) c')
        
        # Sum all directions
        y = ys[0] + y2 + y3 + y4
        
        return y
    
    def selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor
    ) -> torch.Tensor:
        """
        Selective scan (same as MambaBlock).
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        delta_A = torch.exp(delta.unsqueeze(-1) * A)
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)
        
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for i in range(seq_len):
            h = delta_A[:, i] * h + delta_B[:, i] * x[:, i].unsqueeze(-1)
            y = (h * C[:, i].unsqueeze(1)).sum(dim=-1)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        y = y + x * D
        
        return y


class VSSBlock(nn.Module):
    """
    Visual State Space Block (from VMamba).
    
    Contains SS2D with proper normalization and residual.
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
        **kwargs
    ):
        super().__init__()
        
        self.norm1 = LayerNorm2d(dim)
        self.ss2d = SS2D(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs
        )
        self.drop_path = nn.Identity() if drop_path == 0 else DropPath(drop_path)
        
        # Optional MLP
        if mlp_ratio > 0:
            self.norm2 = LayerNorm2d(dim)
            mlp_hidden = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Conv2d(dim, mlp_hidden, 1),
                nn.GELU(),
                nn.Conv2d(mlp_hidden, dim, 1),
            )
        else:
            self.mlp = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, C, H, W)
        """
        x = x + self.drop_path(self.ss2d(self.norm1(x)))
        
        if self.mlp is not None:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class VMMambaBlock(nn.Module):
    """
    Visual Mamba Block - Alias for VSSBlock.
    
    A complete Visual Mamba block with cross-scan SS2D,
    layer normalization, residual connections, and optional MLP.
    
    This is the main block to use for image/video tasks.
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        drop_path: float = 0.0,
        mlp_ratio: float = 0.0,  # No MLP by default
        **kwargs
    ):
        super().__init__()
        
        self.block = VSSBlock(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            **kwargs
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, C, H, W)
            
        Returns:
            Output (B, C, H, W)
        """
        return self.block(x)


# =============================================================================
# Utility Classes
# =============================================================================

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        
        return output


# =============================================================================
# Factory Function
# =============================================================================

def create_mamba_block(
    variant: str,
    dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create Mamba blocks.
    
    Args:
        variant: 'mamba', 'mamba2', or 'vmamba'
        dim: Feature dimension
        **kwargs: Additional arguments for the block
        
    Returns:
        Mamba block module
    """
    variants = {
        'mamba': MambaLayer,
        'mamba1': MambaLayer,
        'mamba2': Mamba2Layer,
        'vmamba': VMMambaBlock,
        'vss': VSSBlock,
    }
    
    if variant.lower() not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")
    
    return variants[variant.lower()](dim=dim, **kwargs)


if __name__ == "__main__":
    # Test all blocks
    print("Testing Mamba blocks...")
    
    B, C, H, W = 2, 96, 32, 32
    x = torch.randn(B, C, H, W)
    
    # Test MambaLayer
    print("\n1. Testing MambaLayer (original Mamba)...")
    mamba = MambaLayer(dim=C, d_state=16, d_conv=4, expand=2)
    out = mamba(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test Mamba2Layer
    print("\n2. Testing Mamba2Layer...")
    mamba2 = Mamba2Layer(dim=C, d_state=64, n_heads=4)
    out = mamba2(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test VMMambaBlock
    print("\n3. Testing VMMambaBlock (Visual Mamba with cross-scan)...")
    vmamba = VMMambaBlock(dim=C, d_state=16, d_conv=3)
    out = vmamba(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test factory function
    print("\n4. Testing factory function...")
    for variant in ['mamba', 'mamba2', 'vmamba']:
        block = create_mamba_block(variant, dim=C)
        out = block(x)
        print(f"   {variant}: {x.shape} -> {out.shape}")
    
    print("\nAll tests passed!")
