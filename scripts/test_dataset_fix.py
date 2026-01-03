"""
Quick test script to verify the dataset loading fix works.
Run this in Kaggle before starting training to ensure everything is working.
"""

import sys
from pathlib import Path

# Ensure the mamba package is in the path
sys.path.insert(0, '/kaggle/working/mamba/mamba')

import warnings
warnings.simplefilter('always')

from data.camus_dataset import CAMUSDataset

print("=" * 70)
print("Testing CAMUS Dataset Loading (with fixes)")
print("=" * 70)

# Test configuration
DATA_DIR = "/kaggle/working/camus_extracted/database_nifti"

try:
    print("\n1. Testing basic dataset loading (ED/ES frames only)...")
    dataset_basic = CAMUSDataset(
        root_dir=DATA_DIR,
        split='train',
        include_sequences=False
    )
    print(f"   ✓ Loaded {len(dataset_basic)} ED/ES samples")
    print(f"   ✓ From {len(dataset_basic.patients)} patients")
    
    print("\n2. Testing dataset with sequences...")
    dataset_full = CAMUSDataset(
        root_dir=DATA_DIR,
        split='train',
        include_sequences=True
    )
    print(f"   ✓ Loaded {len(dataset_full)} total samples")
    print(f"   ✓ Sequence frames: {len(dataset_full) - len(dataset_basic)}")
    
    print("\n3. Testing data loading (sample from dataset)...")
    sample = dataset_full[0]
    print(f"   ✓ Image shape: {sample['image'].shape}")
    print(f"   ✓ Mask shape: {sample['mask'].shape}")
    print(f"   ✓ Image dtype: {sample['image'].dtype}")
    print(f"   ✓ Mask dtype: {sample['mask'].dtype}")
    
    print("\n" + "=" * 70)
    print("SUCCESS! Dataset loading is working correctly.")
    print("You can now proceed with training.")
    print("=" * 70)
    
except Exception as e:
    print("\n" + "=" * 70)
    print("ERROR! Dataset loading failed:")
    print("=" * 70)
    print(f"\n{e}\n")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 70)
    sys.exit(1)
