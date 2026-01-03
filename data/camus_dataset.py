"""
CAMUS Dataset for 2D Echocardiography Segmentation.

The CAMUS dataset contains 2D echocardiography images with segmentations of:
- Left Ventricle Endocardium (LV_endo)
- Left Ventricle Epicardium (LV_epi)  
- Left Atrium (LA)

For both End-Diastolic (ED) and End-Systolic (ES) phases.
Includes half sequences (cardiac cycle) with ground truth for all frames.

File format: NIfTI (.nii)
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import nibabel as nib
import torch
from torch.utils.data import Dataset
import warnings


def load_nifti(filepath: Path) -> np.ndarray:
    """Load a NIfTI file and return as numpy array."""
    img = nib.load(str(filepath))
    return img.get_fdata()


class CAMUSPatient:
    """
    Represents a single patient in the CAMUS dataset.
    
    Each patient has:
    - 2CH (2-chamber) and 4CH (4-chamber) views
    - ED (end-diastolic) and ES (end-systolic) frames for each view
    - Ground truth segmentations
    - Clinical information (EF, volumes, image quality)
    """
    
    def __init__(self, patient_dir: Path):
        self.patient_dir = Path(patient_dir)
        self.patient_id = self.patient_dir.name
        self._info = None
        
    @property
    def info(self) -> Dict:
        """Load patient info from Info_xCH.cfg files."""
        if self._info is None:
            self._info = self._load_info()
        return self._info
    
    def _load_info(self) -> Dict:
        """Parse patient information files."""
        info = {
            'patient_id': self.patient_id,
            '2CH': {},
            '4CH': {}
        }
        
        for view in ['2CH', '4CH']:
            info_file = self.patient_dir / f'Info_{view}.cfg'
            if info_file.exists():
                with open(info_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            # Convert numeric values
                            try:
                                if '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            except ValueError:
                                pass
                            info[view][key] = value
        
        return info
    
    def get_ef(self, view: str = '2CH') -> Optional[float]:
        """Get ejection fraction for specified view."""
        return self.info.get(view, {}).get('EF', None)
    
    def get_image_quality(self, view: str = '2CH') -> Optional[str]:
        """Get image quality grade (Good, Medium, Poor)."""
        quality = self.info.get(view, {}).get('ImageQuality', None)
        if quality is None:
            return None
        # Handle both string and numeric formats from cfg files
        if isinstance(quality, str):
            return quality  # Already a string like "Good", "Medium", "Poor"
        # Numeric format (some versions of CAMUS use 0, 1, 2)
        quality_map = {0: 'Poor', 1: 'Medium', 2: 'Good'}
        return quality_map.get(quality, None)
    
    def get_lv_volumes(self, view: str = '2CH') -> Tuple[Optional[float], Optional[float]]:
        """Get LV volumes at ED and ES."""
        ed_vol = self.info.get(view, {}).get('LVedv', None)
        es_vol = self.info.get(view, {}).get('LVesv', None)
        return ed_vol, es_vol
    
    def load_image(self, view: str, phase: str) -> np.ndarray:
        """
        Load image for specified view and phase.
        
        Args:
            view: '2CH' or '4CH'
            phase: 'ED' or 'ES'
            
        Returns:
            Image as numpy array (H, W)
        """
        # Try both .nii and .nii.gz extensions
        filepath = None
        for ext in ['.nii.gz', '.nii']:
            filename = f'{self.patient_id}_{view}_{phase}{ext}'
            candidate_path = self.patient_dir / filename
            if candidate_path.exists():
                filepath = candidate_path
                break
        
        if filepath is None:
            raise FileNotFoundError(f"Image not found: {self.patient_dir}/{self.patient_id}_{view}_{phase}.[nii|nii.gz]")
        
        image_array = load_nifti(filepath)
        
        # Handle different array shapes
        if image_array.ndim == 3:
            # Take first slice if 3D with singleton dimension
            if image_array.shape[2] == 1:
                image_array = image_array[:, :, 0]
            else:
                image_array = image_array[:, :, 0]  # Take first frame
            
        return image_array.astype(np.float32)
    
    def load_segmentation(self, view: str, phase: str) -> np.ndarray:
        """
        Load ground truth segmentation for specified view and phase.
        
        Labels:
            0: Background
            1: LV endocardium
            2: LV epicardium (myocardium)
            3: Left atrium
            
        Args:
            view: '2CH' or '4CH'
            phase: 'ED' or 'ES'
            
        Returns:
            Segmentation mask as numpy array (H, W)
        """
        # Try both .nii and .nii.gz extensions
        filepath = None
        for ext in ['.nii.gz', '.nii']:
            filename = f'{self.patient_id}_{view}_{phase}_gt{ext}'
            candidate_path = self.patient_dir / filename
            if candidate_path.exists():
                filepath = candidate_path
                break
        
        if filepath is None:
            raise FileNotFoundError(f"Segmentation not found: {self.patient_dir}/{self.patient_id}_{view}_{phase}_gt.[nii|nii.gz]")
        
        seg_array = load_nifti(filepath)
        
        # Handle different array shapes
        if seg_array.ndim == 3:
            if seg_array.shape[2] == 1:
                seg_array = seg_array[:, :, 0]
            else:
                seg_array = seg_array[:, :, 0]
            
        return seg_array.astype(np.int64)
    
    def load_half_sequence(self, view: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load half sequence (cardiac cycle) with ground truth.
        
        Args:
            view: '2CH' or '4CH'
            
        Returns:
            Tuple of (images, segmentations) as numpy arrays (T, H, W)
        """
        # Try both .nii and .nii.gz extensions for images
        img_filepath = None
        for ext in ['.nii.gz', '.nii']:
            filepath = self.patient_dir / f'{self.patient_id}_{view}_half_sequence{ext}'
            if filepath.exists():
                img_filepath = filepath
                break
        if img_filepath is None:
            raise FileNotFoundError(f"Half sequence not found: {self.patient_dir}/{self.patient_id}_{view}_half_sequence.[nii|nii.gz]")
        
        # Try both .nii and .nii.gz extensions for ground truth
        gt_filepath = None
        for ext in ['.nii.gz', '.nii']:
            filepath = self.patient_dir / f'{self.patient_id}_{view}_half_sequence_gt{ext}'
            if filepath.exists():
                gt_filepath = filepath
                break
        if gt_filepath is None:
            raise FileNotFoundError(f"Half sequence GT not found: {self.patient_dir}/{self.patient_id}_{view}_half_sequence_gt.[nii|nii.gz]")
        
        images = load_nifti(img_filepath)
        segmentations = load_nifti(gt_filepath)
        
        # NIfTI stores as (H, W, T), transpose to (T, H, W)
        if images.ndim == 3:
            images = np.transpose(images, (2, 0, 1))
            segmentations = np.transpose(segmentations, (2, 0, 1))
        
        return images.astype(np.float32), segmentations.astype(np.int64)
    
    def load_sequence(self, view: str) -> np.ndarray:
        """
        Load half sequence images only (for backward compatibility).
        
        Args:
            view: '2CH' or '4CH'
            
        Returns:
            Sequence as numpy array (T, H, W)
        """
        images, _ = self.load_half_sequence(view)
        return images


class CAMUSDataset(Dataset):
    """
    PyTorch Dataset for CAMUS echocardiography data.
    
    Supports multiple views (2CH, 4CH) and phases (ED, ES).
    Can include half sequences (all frames with ground truth).
    Can be used for segmentation training and EF computation.
    """
    
    # Class labels
    LABELS = {
        0: 'background',
        1: 'LV_endo',
        2: 'LV_epi',
        3: 'LA'
    }
    
    NUM_CLASSES = 4
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        views: List[str] = ['2CH', '4CH'],
        phases: List[str] = ['ED', 'ES'],
        transform=None,
        include_info: bool = False,
        patient_ids: Optional[List[str]] = None,
        quality_filter: Optional[List[str]] = None,
        include_sequences: bool = False
    ):
        """
        Initialize CAMUS dataset.
        
        Args:
            root_dir: Path to CAMUS dataset root
            split: 'train', 'val', or 'test'
            views: List of views to include ('2CH', '4CH')
            phases: List of phases to include ('ED', 'ES')
            transform: Albumentations transform
            include_info: Whether to include patient info in output
            patient_ids: Specific patient IDs to include (optional)
            quality_filter: Only include images with these quality grades
            include_sequences: Include all frames from half sequences (significantly more data)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.views = views
        self.phases = phases
        self.transform = transform
        self.include_info = include_info
        self.quality_filter = quality_filter
        self.include_sequences = include_sequences
        
        # Discover patients
        self.patients = self._discover_patients(patient_ids)
        
        # Build sample index
        self.samples = self._build_sample_index()
        
        if len(self.samples) == 0:
            warnings.warn(f"No samples found in {root_dir} for split={split}")
    
    def _load_split_file(self) -> Optional[List[str]]:
        """Load patient IDs from official split file."""
        # Look for split files in data/splits/ directory
        splits_dir = Path(__file__).parent / 'splits'
        
        split_map = {
            'train': 'train.txt',
            'val': 'val.txt',
            'validation': 'val.txt',
            'test': 'test.txt',
            'testing': 'test.txt'
        }
        
        split_file = splits_dir / split_map.get(self.split, f'{self.split}.txt')
        
        if split_file.exists():
            with open(split_file, 'r') as f:
                patient_ids = [line.strip() for line in f if line.strip()]
            return patient_ids
        
        return None
    
    def _discover_patients(self, patient_ids: Optional[List[str]] = None) -> List[CAMUSPatient]:
        """Discover all patients in the dataset directory."""
        patients = []
        
        # Load official split if no patient_ids provided
        if patient_ids is None:
            patient_ids = self._load_split_file()
        
        # Convert to set for faster lookup
        patient_id_set = set(patient_ids) if patient_ids else None
        
        # Search for patient directories
        # Try different directory structures
        search_dirs = [
            self.root_dir,
            self.root_dir / 'training',
            self.root_dir / 'testing',
            self.root_dir / 'database_nifti',
        ]
        
        found_patients = set()
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            for patient_dir in sorted(search_dir.iterdir()):
                if patient_dir.is_dir() and patient_dir.name.startswith('patient'):
                    patient_name = patient_dir.name
                    
                    # Skip if already found
                    if patient_name in found_patients:
                        continue
                    
                    # Check if patient is in split
                    if patient_id_set is None or patient_name in patient_id_set:
                        patients.append(CAMUSPatient(patient_dir))
                        found_patients.add(patient_name)
        
        return patients
        
        return patients
    
    def _build_sample_index(self) -> List[Dict]:
        """Build index of all samples (patient, view, phase combinations)."""
        samples = []
        skipped_count = 0
        
        def file_exists(patient_dir, pattern):
            """Check if file exists with .nii or .nii.gz extension."""
            for ext in ['.nii.gz', '.nii']:
                if (patient_dir / f'{pattern}{ext}').exists():
                    return True
            return False
        
        for patient in self.patients:
            for view in self.views:
                # Check quality filter
                if self.quality_filter is not None:
                    quality = patient.get_image_quality(view)
                    if quality not in self.quality_filter:
                        continue
                
                # Add ED/ES frames
                for phase in self.phases:
                    # Verify files exist (check both .nii and .nii.gz)
                    try:
                        img_pattern = f'{patient.patient_id}_{view}_{phase}'
                        seg_pattern = f'{patient.patient_id}_{view}_{phase}_gt'
                        
                        # Check if both files exist before adding sample
                        if file_exists(patient.patient_dir, img_pattern) and file_exists(patient.patient_dir, seg_pattern):
                            samples.append({
                                'patient': patient,
                                'view': view,
                                'phase': phase,
                                'patient_id': patient.patient_id,
                                'is_sequence': False,
                                'frame_idx': None
                            })
                        else:
                            # Log when files are missing
                            if not file_exists(patient.patient_dir, img_pattern):
                                warnings.warn(f"Missing image: {patient.patient_id}/{view}/{phase}")
                            if not file_exists(patient.patient_dir, seg_pattern):
                                warnings.warn(f"Missing segmentation: {patient.patient_id}/{view}/{phase}")
                            skipped_count += 1
                    except Exception as e:
                        warnings.warn(f"Error checking {patient.patient_id}/{view}/{phase}: {e}")
                        skipped_count += 1
                
                # Add half sequence frames if requested
                if self.include_sequences:
                    try:
                        seq_pattern = f'{patient.patient_id}_{view}_half_sequence'
                        gt_pattern = f'{patient.patient_id}_{view}_half_sequence_gt'
                        
                        if file_exists(patient.patient_dir, seq_pattern) and file_exists(patient.patient_dir, gt_pattern):
                            # Load to get number of frames
                            images, _ = patient.load_half_sequence(view)
                            n_frames = images.shape[0]
                            
                            for frame_idx in range(n_frames):
                                samples.append({
                                    'patient': patient,
                                    'view': view,
                                    'phase': 'sequence',
                                    'patient_id': patient.patient_id,
                                    'is_sequence': True,
                                    'frame_idx': frame_idx
                                })
                        else:
                            if not file_exists(patient.patient_dir, seq_pattern):
                                warnings.warn(f"Missing half sequence: {patient.patient_id}/{view}")
                            if not file_exists(patient.patient_dir, gt_pattern):
                                warnings.warn(f"Missing half sequence GT: {patient.patient_id}/{view}")
                    except Exception as e:
                        warnings.warn(f"Error loading sequence {patient.patient_id}/{view}: {e}")
                        skipped_count += 1
        
        if skipped_count > 0:
            warnings.warn(f"Skipped {skipped_count} samples due to missing or corrupted files")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]
        patient = sample_info['patient']
        view = sample_info['view']
        
        if sample_info['is_sequence']:
            # Load from half sequence
            frame_idx = sample_info['frame_idx']
            images, masks = patient.load_half_sequence(view)
            image = images[frame_idx]
            mask = masks[frame_idx]
            phase = f'seq_{frame_idx}'  # Mark as sequence frame
        else:
            # Load ED/ES frame
            phase = sample_info['phase']
            image = patient.load_image(view, phase)
            mask = patient.load_segmentation(view, phase)
        
        # Normalize image to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        
        # Add channel dimension if needed
        if image.ndim == 2:
            image = image.unsqueeze(0)
        
        # Return tuple (image, mask) for training compatibility
        # Use include_info=True to get full dictionary with metadata
        if self.include_info:
            output = {
                'image': image,
                'mask': mask,
                'patient_id': sample_info['patient_id'],
                'view': view,
                'phase': phase,
                'ef': patient.get_ef(view) or -1.0,
                'quality': patient.get_image_quality(view) or 'Unknown',
            }
            ed_vol, es_vol = patient.get_lv_volumes(view)
            output['lv_ed_volume'] = ed_vol or -1.0
            output['lv_es_volume'] = es_vol or -1.0
            return output
        
        # Default: return tuple for standard training loops
        return image, mask
    
    def get_patient_pairs(self, patient_id: str, view: str) -> Tuple[Dict, Dict]:
        """
        Get ED and ES samples for a patient-view combination.
        Useful for EF computation.
        """
        ed_sample = None
        es_sample = None
        
        for idx, sample in enumerate(self.samples):
            if sample['patient_id'] == patient_id and sample['view'] == view:
                if sample['phase'] == 'ED':
                    ed_sample = self[idx]
                elif sample['phase'] == 'ES':
                    es_sample = self[idx]
        
        return ed_sample, es_sample


class CAMUSEFDataset(Dataset):
    """
    Dataset specifically for Ejection Fraction computation.
    Returns ED-ES pairs for each patient-view combination.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        views: List[str] = ['2CH', '4CH'],
        transform=None,
        patient_ids: Optional[List[str]] = None
    ):
        self.base_dataset = CAMUSDataset(
            root_dir=root_dir,
            split=split,
            views=views,
            phases=['ED', 'ES'],
            transform=transform,
            include_info=True,
            patient_ids=patient_ids
        )
        
        # Build pairs index
        self.pairs = self._build_pairs_index()
    
    def _build_pairs_index(self) -> List[Dict]:
        """Build index of ED-ES pairs."""
        pairs = []
        seen = set()
        
        for sample in self.base_dataset.samples:
            key = (sample['patient_id'], sample['view'])
            if key not in seen:
                seen.add(key)
                pairs.append({
                    'patient_id': sample['patient_id'],
                    'view': sample['view'],
                    'patient': sample['patient']
                })
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        pair_info = self.pairs[idx]
        patient = pair_info['patient']
        view = pair_info['view']
        
        ed_sample, es_sample = self.base_dataset.get_patient_pairs(
            pair_info['patient_id'], view
        )
        
        return {
            'ed_image': ed_sample['image'],
            'ed_mask': ed_sample['mask'],
            'es_image': es_sample['image'],
            'es_mask': es_sample['mask'],
            'ef_gt': patient.get_ef(view) or -1.0,
            'patient_id': pair_info['patient_id'],
            'view': view
        }
