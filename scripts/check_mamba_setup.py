#!/usr/bin/env python3
"""
Check if mamba-ssm is properly installed and being used.
"""

import os
# Disable Triton autotuning to avoid map::at errors in Mamba2
os.environ.setdefault('TRITON_DISABLE_AUTOTUNE', '1')

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("MAMBA-SSM INSTALLATION CHECK")
print("="*70)

# Check 1: Can we import mamba_ssm?
print("\n1. Checking mamba_ssm installation...")
try:
    import mamba_ssm
    print("   ✓ mamba_ssm is installed")
    print(f"   Version: {mamba_ssm.__version__ if hasattr(mamba_ssm, '__version__') else 'unknown'}")
    print(f"   Location: {mamba_ssm.__file__}")
    MAMBA_AVAILABLE = True
except ImportError as e:
    print(f"   ✗ mamba_ssm NOT installed: {e}")
    MAMBA_AVAILABLE = False

# Check 2: Can we import the fast kernels?
if MAMBA_AVAILABLE:
    print("\n2. Checking CUDA kernels...")
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        print("   ✓ selective_scan_fn available (CUDA kernels for Mamba & VMamba)")
    except ImportError as e:
        print(f"   ✗ selective_scan_fn NOT available: {e}")
    
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        print("   ✓ mamba_chunk_scan_combined available (Triton kernels for Mamba2)")
    except ImportError as e:
        print(f"   ✗ mamba_chunk_scan_combined NOT available: {e}")
        print("     (Mamba2Block will use PyTorch fallback - this is normal for older mamba-ssm versions)")
    
    try:
        from mamba_ssm import Mamba, Mamba2
        print("   ✓ Mamba and Mamba2 classes available")
    except ImportError as e:
        print(f"   ✗ Mamba classes NOT available: {e}")

# Check 3: Import our MambaBlock
print("\n3. Checking our Mamba implementations...")
try:
    from models.modules import MambaBlock, Mamba2Block, VMMambaBlock
    print("   ✓ MambaBlock, Mamba2Block, VMMambaBlock imported successfully")
    
    # Create test blocks
    mamba1 = MambaBlock(dim=64, d_state=16)
    mamba2 = Mamba2Block(dim=64, d_state=16)
    vmamba = VMMambaBlock(dim=64, d_state=16)
    
    print(f"   MambaBlock fast path: {mamba1.use_fast_path}")
    print(f"   Mamba2Block fast path: {mamba2.use_fast_path}")
    print(f"   VMMambaBlock fast path: {vmamba.block.ss2d.use_fast_path}")
    
    all_fast = mamba1.use_fast_path and mamba2.use_fast_path and vmamba.block.ss2d.use_fast_path
    
    if not all_fast:
        print("   ⚠ WARNING: Some implementations have fast path DISABLED!")
        print("   This will cause ~100-1000x slowdown!")
    
except Exception as e:
    print(f"   ✗ Error importing Mamba blocks: {e}")
    import traceback
    traceback.print_exc()

# Check 4: Test actual performance
if MAMBA_AVAILABLE:
    print("\n4. Performance test (all variants)...")
    try:
        from models.modules import MambaBlock, Mamba2Block, VMMambaBlock
        import time
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Test MambaBlock
        print("\n   Testing MambaBlock:")
        block = MambaBlock(dim=64, d_state=16).to(device)
        x_test = torch.randn(1, 64, 32, 32).to(device)
        
        # Warmup
        with torch.no_grad():
            _ = block(x_test)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = block(x_test)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"     32x32 image: {elapsed*1000:.2f}ms")
        print(f"     Estimated 256x256: {elapsed * 64:.2f}s")
        
        # Test Mamba2Block (auto-computed heads for optimal headdim)
        # Use dim=128 for better Mamba2 compatibility (d_inner=256, headdim=64, n_heads=4)
        print("\n   Testing Mamba2Block (auto n_heads, dim=128):")
        x_test_m2 = torch.randn(1, 128, 32, 32).to(device)
        mamba2 = Mamba2Block(dim=128, d_state=64).to(device)  # n_heads auto-computed
        print(f"     Config: dim=128, d_inner={mamba2.d_inner}, heads={mamba2.n_heads}, headdim={mamba2.head_dim}")
        print(f"     Using native: {getattr(mamba2, '_use_native', False)}")
        
        with torch.no_grad():
            _ = mamba2(x_test_m2)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = mamba2(x_test_m2)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed_m2 = time.time() - start
        
        print(f"     32x32 image: {elapsed_m2*1000:.2f}ms")
        print(f"     Estimated 256x256: {elapsed_m2 * 64:.2f}s")
        
        # Test VMMambaBlock (4-direction scan)
        print("\n   Testing VMMambaBlock (4-direction scan):")
        vmblock = VMMambaBlock(dim=64, d_state=16).to(device)
        
        with torch.no_grad():
            _ = vmblock(x_test)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = vmblock(x_test)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed_vm = time.time() - start
        
        print(f"     32x32 image: {elapsed_vm*1000:.2f}ms")
        print(f"     Estimated 256x256: {elapsed_vm * 64:.2f}s")
        
        # Warning if too slow
        print("\n   Performance Analysis:")
        if elapsed * 64 > 10:
            print("     ⚠ MambaBlock: TOO SLOW - using Python implementation!")
        else:
            print(f"     ✓ MambaBlock: Good ({elapsed * 64:.1f}s for 256x256)")
            
        if elapsed_m2 * 64 > 30:
            print("     ⚠ Mamba2Block: TOO SLOW - using Python implementation!")
        elif elapsed_m2 * 64 > 15:
            print(f"     ⚠ Mamba2Block: Acceptable but slow ({elapsed_m2 * 64:.1f}s for 256x256)")
            print("       This is expected with 8 heads - consider using fewer heads or MambaBlock")
        else:
            print(f"     ✓ Mamba2Block: Good ({elapsed_m2 * 64:.1f}s for 256x256)")
            
        if elapsed_vm * 64 > 40:
            print("     ⚠ VMMambaBlock: TOO SLOW - using Python implementation!")
        else:
            print(f"     ✓ VMMambaBlock: Good ({elapsed_vm * 64:.1f}s for 256x256)")
            
    except Exception as e:
        print(f"   Error during performance test: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
if not MAMBA_AVAILABLE:
    print("\n❌ mamba-ssm is NOT installed!")
    print("\nTo install:")
    print("   pip install mamba-ssm")
    print("\nOr from source:")
    print("   pip install git+https://github.com/state-spaces/mamba.git")
else:
    from models.modules import MambaBlock
    test_block = MambaBlock(dim=64)
    if test_block.use_fast_path:
        print("\n✅ Everything looks good! Fast CUDA kernels should be used.")
        print("   All three Mamba variants (Mamba, Mamba2, VMamba) will use fast kernels.")
    else:
        print("\n⚠️  mamba-ssm is installed but fast path is disabled!")
        print("   Check if there are compilation issues.")
