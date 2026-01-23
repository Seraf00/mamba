#!/usr/bin/env python3
"""
Check if mamba-ssm is properly installed and being used.
"""

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
        print("   ✓ selective_scan_fn available (CUDA kernels)")
    except ImportError as e:
        print(f"   ✗ CUDA kernels NOT available: {e}")
    
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
        from models.modules import MambaBlock, VMMambaBlock
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
        
        # Test VMMambaBlock (4x slower due to 4 scans)
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
        if elapsed * 64 > 10 or elapsed_vm * 64 > 40:
            print("\n   ⚠ CRITICAL: Performance is too slow!")
            print("   The slow Python implementation is likely being used.")
            
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
