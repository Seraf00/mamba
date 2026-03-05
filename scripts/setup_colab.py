#!/usr/bin/env python3
"""
Environment setup script for Google Colab and Kaggle.

Handles installation of mamba_ssm with CUDA support and all other dependencies.

Usage:
    # In Colab/Kaggle notebook:
    !python scripts/setup_colab.py

    # Or import and call:
    from scripts.setup_colab import setup_environment
    setup_environment()
"""

import subprocess
import sys
import os


def run_cmd(cmd, check=True):
    """Run a shell command and print output."""
    print(f"  Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")
    if result.returncode != 0 and check:
        print(f"  ERROR: {result.stderr.strip()}")
    return result.returncode == 0


def detect_environment():
    """Detect if running on Colab or Kaggle."""
    if os.path.exists('/content'):
        return 'colab'
    elif os.path.exists('/kaggle'):
        return 'kaggle'
    return 'local'


def check_cuda():
    """Check CUDA availability and version."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  CUDA {cuda_version} available on {gpu_name}")
            return cuda_version
        else:
            print("  WARNING: No CUDA GPU detected. Mamba will use slow Python fallback.")
            return None
    except ImportError:
        print("  PyTorch not installed yet.")
        return None


def install_base_dependencies():
    """Install base Python dependencies."""
    print("\n[1/4] Installing base dependencies...")
    deps = [
        'torch', 'torchvision',
        'nibabel', 'albumentations',
        'einops', 'tqdm',
        'tensorboard', 'scipy',
        'scikit-image', 'pillow',
    ]
    for dep in deps:
        try:
            __import__(dep.replace('-', '_'))
        except ImportError:
            run_cmd(f'{sys.executable} -m pip install {dep} -q')


def install_mamba():
    """Install mamba_ssm with CUDA support."""
    print("\n[2/4] Installing mamba_ssm...")

    # Check if already installed
    try:
        import mamba_ssm
        print(f"  mamba_ssm already installed (version: {mamba_ssm.__version__})")
        return True
    except ImportError:
        pass

    # Install causal-conv1d first (required dependency)
    print("  Installing causal-conv1d...")
    run_cmd(f'{sys.executable} -m pip install causal-conv1d>=1.1.0 -q', check=False)

    # Try installing mamba-ssm
    print("  Installing mamba-ssm...")
    success = run_cmd(
        f'{sys.executable} -m pip install mamba-ssm --no-build-isolation -q',
        check=False
    )

    if not success:
        print("  Direct install failed. Trying from source...")
        success = run_cmd(
            f'{sys.executable} -m pip install mamba-ssm -q',
            check=False
        )

    if not success:
        print("  WARNING: mamba_ssm installation failed.")
        print("  Models will use slower Python fallback implementations.")
        print("  MambaBlock (original) will still work but Mamba2/VMamba may be slow.")
        return False

    # Verify installation
    try:
        import mamba_ssm
        print(f"  mamba_ssm installed successfully (version: {mamba_ssm.__version__})")
        return True
    except ImportError:
        print("  WARNING: mamba_ssm import failed despite successful install.")
        return False


def verify_setup():
    """Verify the complete setup."""
    print("\n[3/4] Verifying setup...")

    checks = {
        'PyTorch': 'torch',
        'nibabel': 'nibabel',
        'albumentations': 'albumentations',
        'einops': 'einops',
        'scikit-image': 'skimage',
    }

    all_ok = True
    for name, module in checks.items():
        try:
            __import__(module)
            print(f"  {name}: OK")
        except ImportError:
            print(f"  {name}: MISSING")
            all_ok = False

    # Check mamba_ssm (optional)
    try:
        from mamba_ssm import Mamba
        print("  mamba_ssm (Mamba): OK (CUDA-optimized)")
    except ImportError:
        print("  mamba_ssm: NOT AVAILABLE (using Python fallback)")

    try:
        from mamba_ssm import Mamba2
        print("  mamba_ssm (Mamba2): OK (CUDA-optimized)")
    except ImportError:
        print("  mamba_ssm (Mamba2): NOT AVAILABLE (using Python fallback)")

    return all_ok


def test_model_creation():
    """Quick test to verify model creation works."""
    print("\n[4/4] Testing model creation...")
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models import get_model

        model = get_model('unet_v1', in_channels=1, num_classes=4)
        print(f"  UNet V1: OK ({sum(p.numel() for p in model.parameters()):,} params)")
        del model

        model = get_model('mamba_unet_v1', in_channels=1, num_classes=4, mamba_type='mamba')
        print(f"  Mamba UNet V1: OK ({sum(p.numel() for p in model.parameters()):,} params)")
        del model

        print("  Model creation: SUCCESS")
        return True
    except Exception as e:
        print(f"  Model creation failed: {e}")
        return False


def setup_environment():
    """Full environment setup for Colab/Kaggle."""
    env = detect_environment()
    print(f"Detected environment: {env}")
    print("=" * 50)

    cuda_version = check_cuda()
    install_base_dependencies()
    install_mamba()
    all_ok = verify_setup()
    test_model_creation()

    print("\n" + "=" * 50)
    if all_ok:
        print("Setup complete! Ready to train.")
    else:
        print("Setup completed with warnings. Check messages above.")


if __name__ == '__main__':
    setup_environment()
