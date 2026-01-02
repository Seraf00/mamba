#!/usr/bin/env python3
"""
Verification script to check if the project setup is correct.

Usage:
    python scripts/verify_setup.py
    python scripts/verify_setup.py --check-data
    python scripts/verify_setup.py --full
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import Dict, List, Tuple


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_result(name: str, success: bool, message: str = ""):
    """Print a result line."""
    status = "✓" if success else "✗"
    color_start = "\033[92m" if success else "\033[91m"
    color_end = "\033[0m"
    print(f"  {color_start}{status}{color_end} {name}", end="")
    if message:
        print(f" - {message}")
    else:
        print()


def check_python_version() -> bool:
    """Check Python version."""
    import platform
    version = platform.python_version()
    major, minor, _ = version.split(".")
    success = int(major) >= 3 and int(minor) >= 9
    print_result("Python version", success, f"{version} {'(OK)' if success else '(need 3.9+)'}")
    return success


def check_core_imports() -> Dict[str, bool]:
    """Check core package imports."""
    results = {}
    
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("nibabel", "NiBabel"),
        ("albumentations", "Albumentations"),
        ("matplotlib", "Matplotlib"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
    ]
    
    for module, name in packages:
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "unknown")
            results[name] = True
            print_result(name, True, f"v{version}")
        except ImportError as e:
            results[name] = False
            print_result(name, False, str(e))
    
    return results


def check_mamba_packages() -> Dict[str, bool]:
    """Check Mamba-specific packages."""
    results = {}
    
    # Check mamba-ssm
    try:
        import mamba_ssm
        version = getattr(mamba_ssm, "__version__", "unknown")
        results["mamba-ssm"] = True
        print_result("mamba-ssm", True, f"v{version}")
    except ImportError:
        results["mamba-ssm"] = False
        print_result("mamba-ssm", False, "Not installed (required for Mamba models)")
    
    # Check causal-conv1d
    try:
        import causal_conv1d
        version = getattr(causal_conv1d, "__version__", "unknown")
        results["causal-conv1d"] = True
        print_result("causal-conv1d", True, f"v{version}")
    except ImportError:
        results["causal-conv1d"] = False
        print_result("causal-conv1d", False, "Not installed (required for Mamba)")
    
    return results


def check_cuda() -> bool:
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print_result("CUDA", True, f"{device_name} (CUDA {cuda_version})")
            
            # Check memory
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_result("GPU Memory", True, f"{memory_gb:.1f} GB")
            return True
        else:
            print_result("CUDA", False, "Not available (CPU only mode)")
            return False
    except Exception as e:
        print_result("CUDA", False, str(e))
        return False


def check_project_structure() -> bool:
    """Check project directory structure."""
    required_dirs = [
        "data",
        "data/splits",
        "models",
        "training",
        "utils",
        "scripts",
        "notebooks",
        "configs",
        "checkpoints",
    ]
    
    required_files = [
        "requirements.txt",
        "LICENSE",
        "data/splits/train.txt",
        "data/splits/val.txt",
        "data/splits/test.txt",
        "scripts/train.py",
        "scripts/evaluate.py",
    ]
    
    root = Path(__file__).parent.parent
    all_good = True
    
    for dir_path in required_dirs:
        exists = (root / dir_path).is_dir()
        print_result(f"Directory: {dir_path}", exists)
        all_good = all_good and exists
    
    for file_path in required_files:
        exists = (root / file_path).is_file()
        print_result(f"File: {file_path}", exists)
        all_good = all_good and exists
    
    return all_good


def check_project_imports() -> bool:
    """Check project module imports."""
    all_good = True
    
    modules = [
        ("data", "CAMUSDataset"),
        ("models", "MambaUNet"),
        ("training", "Trainer"),
        ("utils", "set_seed"),
    ]
    
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            obj = getattr(module, class_name, None)
            if obj is not None:
                print_result(f"{module_name}.{class_name}", True)
            else:
                print_result(f"{module_name}.{class_name}", False, "Class not found")
                all_good = False
        except ImportError as e:
            print_result(f"{module_name}.{class_name}", False, str(e))
            all_good = False
    
    return all_good


def check_data_directory(data_dir: str = "./data/CAMUS") -> bool:
    """Check if CAMUS data is present."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print_result("CAMUS data directory", False, f"Not found at {data_dir}")
        print("        → Download from: https://www.creatis.insa-lyon.fr/Challenge/camus/")
        return False
    
    # Check for patient folders
    patient_folders = list(data_path.glob("patient*"))
    if not patient_folders:
        # Try training subfolder
        patient_folders = list(data_path.glob("training/patient*"))
    
    if not patient_folders:
        print_result("CAMUS data directory", False, "No patient folders found")
        return False
    
    print_result("CAMUS data directory", True, f"{len(patient_folders)} patients found")
    
    # Check sample patient
    sample_patient = patient_folders[0]
    expected_files = [
        "*_2CH_ED.nii",
        "*_2CH_ES.nii",
        "*_4CH_ED.nii",
        "*_4CH_ES.nii",
    ]
    
    files_found = 0
    for pattern in expected_files:
        matches = list(sample_patient.glob(pattern))
        if matches:
            files_found += 1
    
    if files_found == len(expected_files):
        print_result("Sample patient files", True, f"All required NIfTI files present")
        return True
    else:
        print_result("Sample patient files", False, f"Only {files_found}/{len(expected_files)} files found")
        return False


def check_model_instantiation() -> bool:
    """Check if models can be instantiated."""
    try:
        import torch
        from models import MambaUNet
        
        # Try to create model
        model = MambaUNet(in_channels=1, num_classes=4)
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print_result("MambaUNet instantiation", True, f"{params/1e6:.2f}M parameters")
        
        # Try forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        x = torch.randn(1, 1, 256, 256).to(device)
        
        with torch.no_grad():
            y = model(x)
        
        print_result("MambaUNet forward pass", True, f"Output shape: {y.shape}")
        return True
        
    except Exception as e:
        print_result("Model instantiation", False, str(e))
        return False


def check_dataset_loading(data_dir: str = "./data/CAMUS") -> bool:
    """Check if dataset can be loaded."""
    try:
        from data import CAMUSDataset
        
        dataset = CAMUSDataset(root_dir=data_dir, split="train")
        print_result("CAMUSDataset loading", True, f"{len(dataset)} samples")
        
        # Try to get a sample
        sample = dataset[0]
        img_shape = sample["image"].shape if hasattr(sample["image"], "shape") else "unknown"
        print_result("Sample loading", True, f"Image shape: {img_shape}")
        
        return True
        
    except Exception as e:
        print_result("Dataset loading", False, str(e))
        return False


def run_quick_training_test() -> bool:
    """Run a quick training test (1 batch, 1 epoch)."""
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from models import MambaUNet
        
        print("  Running quick training test...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create dummy data
        x = torch.randn(2, 1, 256, 256)
        y = torch.randint(0, 4, (2, 256, 256))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=2)
        
        # Create model
        model = MambaUNet(in_channels=1, num_classes=4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        # One training step
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).long()
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            print_result("Quick training test", True, f"Loss: {loss.item():.4f}")
            return True
        
    except Exception as e:
        print_result("Quick training test", False, str(e))
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify project setup")
    parser.add_argument("--check-data", action="store_true", help="Check CAMUS data")
    parser.add_argument("--data-dir", type=str, default="./data/CAMUS", help="CAMUS data directory")
    parser.add_argument("--full", action="store_true", help="Run all checks including training test")
    args = parser.parse_args()
    
    results = {"passed": 0, "failed": 0}
    
    def update_results(success: bool):
        if success:
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # Python version
    print_header("Python Environment")
    update_results(check_python_version())
    
    # Core packages
    print_header("Core Packages")
    for success in check_core_imports().values():
        update_results(success)
    
    # Mamba packages
    print_header("Mamba Packages")
    for success in check_mamba_packages().values():
        update_results(success)
    
    # CUDA
    print_header("GPU/CUDA")
    update_results(check_cuda())
    
    # Project structure
    print_header("Project Structure")
    update_results(check_project_structure())
    
    # Project imports
    print_header("Project Module Imports")
    update_results(check_project_imports())
    
    # Data check
    if args.check_data or args.full:
        print_header("CAMUS Dataset")
        update_results(check_data_directory(args.data_dir))
        update_results(check_dataset_loading(args.data_dir))
    
    # Model check
    print_header("Model Verification")
    update_results(check_model_instantiation())
    
    # Training test
    if args.full:
        print_header("Training Test")
        update_results(run_quick_training_test())
    
    # Summary
    print_header("Summary")
    total = results["passed"] + results["failed"]
    print(f"\n  Passed: {results['passed']}/{total}")
    print(f"  Failed: {results['failed']}/{total}")
    
    if results["failed"] == 0:
        print("\n  🎉 All checks passed! You're ready to go.")
    else:
        print("\n  ⚠️  Some checks failed. Please fix the issues above.")
    
    print()
    return results["failed"] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
