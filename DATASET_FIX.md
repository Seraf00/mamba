# Dataset Loading Fix Summary

## Problem
The training was failing with a `FileNotFoundError` when trying to load segmentation files for patient0482:
```
FileNotFoundError: Segmentation not found: /kaggle/working/camus_extracted/database_nifti/patient0482/patient0482_2CH_ED_gt.nii
```

## Root Causes

### 1. Bug in `load_segmentation` and `load_image` methods
The methods had a critical bug where they would:
1. Loop through extensions (`.nii.gz`, `.nii`) to find an existing file
2. If found, break from the loop
3. **But then immediately override** the `filepath` variable with a hardcoded `.nii` extension
4. Check if the hardcoded path exists (which might not!)

**Fixed Code:**
```python
def load_segmentation(self, view: str, phase: str) -> np.ndarray:
    # Try both .nii and .nii.gz extensions
    filepath = None
    for ext in ['.nii.gz', '.nii']:
        filename = f'{self.patient_id}_{view}_{phase}_gt{ext}'
        candidate_path = self.patient_dir / filename
        if candidate_path.exists():
            filepath = candidate_path
            break
    
    if filepath is None:
        raise FileNotFoundError(...)
    
    seg_array = load_nifti(filepath)
    # ... rest of method
```

### 2. Missing File Detection
The dataset had some patients with missing files (like patient0482), but the code didn't properly handle this during dataset initialization. It would only discover the missing files during training, causing crashes.

**Enhanced `_build_sample_index`:**
- Now explicitly checks for missing files during initialization
- Logs warnings for each missing image/segmentation
- Counts and reports total skipped samples
- Training will skip problematic patients gracefully

## Changes Made

### 1. Fixed `data/camus_dataset.py`

**`CAMUSPatient.load_image` (lines ~107-123):**
- Fixed file path resolution to use the found path instead of overriding it
- Properly handles both `.nii` and `.nii.gz` extensions

**`CAMUSPatient.load_segmentation` (lines ~138-154):**
- Fixed file path resolution to use the found path instead of overriding it
- Properly handles both `.nii` and `.nii.gz` extensions

**`CAMUSDataset._build_sample_index` (lines ~363-440):**
- Added `skipped_count` tracking
- Added explicit warnings for missing images and segmentations
- Added explicit warnings for missing sequences
- Reports total skipped samples at the end
- Now only adds samples that have both image and segmentation files

### 2. Created `scripts/validate_dataset.py`

A new utility script to validate your dataset before training:

**Usage:**
```bash
# Validate entire dataset
python scripts/validate_dataset.py --data_dir /path/to/camus

# Validate specific split
python scripts/validate_dataset.py --data_dir /path/to/camus --split train

# Include sequence validation
python scripts/validate_dataset.py --data_dir /path/to/camus --include_sequences
```

**Features:**
- Reports total samples and patients
- Lists all missing images and segmentations
- Shows statistics about the dataset
- Returns error code if problems found (useful for CI/CD)

## How to Use

### On Kaggle

1. **Upload the fixed code** to your Kaggle notebook
2. **Validate your dataset first** (optional but recommended):
   ```bash
   !python scripts/validate_dataset.py \
       --data_dir /kaggle/working/camus_extracted/database_nifti \
       --split train \
       --include_sequences
   ```

3. **Run training** - it will now skip missing patients automatically:
   ```bash
   !python scripts/train_all_models.py \
       --data_dir /kaggle/working/camus_extracted/database_nifti \
       --epochs 100 \
       --batch_size 8 \
       --mamba_variants mamba mamba2 vmamba \
       --include_sequences \
       --output_dir ./results/benchmark_full
   ```

### Expected Behavior

When you run training now:
- During dataset initialization, you'll see warnings like:
  ```
  UserWarning: Missing segmentation: patient0482/2CH/ED
  UserWarning: Skipped 15 samples due to missing or corrupted files
  ```
- Training will proceed with valid samples only
- No crashes during training due to missing files

## Verification

To verify the fix works locally:
```bash
# Test the dataset loading
python -c "
from data.camus_dataset import CAMUSDataset
import warnings
warnings.simplefilter('always')

ds = CAMUSDataset(
    root_dir='data/CAMUS',
    split='train',
    include_sequences=True
)
print(f'Loaded {len(ds)} samples from {len(ds.patients)} patients')
"
```

## Notes

- The TensorBoard warnings you saw are unrelated to this issue - they're dependency compatibility warnings
- The `GradScaler` deprecation warning is also unrelated - can be fixed separately if needed
- Patient0482 will be skipped during training if its files are missing, which is expected behavior
