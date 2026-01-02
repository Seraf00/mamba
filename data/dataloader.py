"""
DataLoader utilities for CAMUS dataset.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.utils.data import DataLoader, Subset

from .camus_dataset import CAMUSDataset, CAMUSEFDataset
from .transforms import get_train_transforms, get_val_transforms, get_test_transforms


def get_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    img_size: Tuple[int, int] = (256, 256),
    train_split: float = 0.7,
    val_split: float = 0.15,
    views: List[str] = ['2CH', '4CH'],
    phases: List[str] = ['ED', 'ES'],
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        root_dir: Path to CAMUS dataset
        batch_size: Batch size
        img_size: Image size (H, W)
        train_split: Fraction for training
        val_split: Fraction for validation
        views: List of views to include
        phases: List of phases to include
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    np.random.seed(seed)
    
    # Create full dataset to get patient list
    full_dataset = CAMUSDataset(
        root_dir=root_dir,
        split='train',
        views=views,
        phases=phases,
        transform=None
    )
    
    # Get unique patient IDs
    patient_ids = list(set([s['patient_id'] for s in full_dataset.samples]))
    patient_ids.sort()
    
    # Split patients (not samples) to avoid data leakage
    train_patients, test_patients = train_test_split(
        patient_ids,
        test_size=1 - train_split,
        random_state=seed
    )
    
    val_size = val_split / (1 - train_split)  # Adjust for remaining data
    val_patients, test_patients = train_test_split(
        test_patients,
        test_size=1 - val_size,
        random_state=seed
    )
    
    # Create transforms
    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)
    
    # Create datasets
    train_dataset = CAMUSDataset(
        root_dir=root_dir,
        split='train',
        views=views,
        phases=phases,
        transform=train_transform,
        patient_ids=train_patients,
        include_info=True
    )
    
    val_dataset = CAMUSDataset(
        root_dir=root_dir,
        split='train',  # Val comes from training split
        views=views,
        phases=phases,
        transform=val_transform,
        patient_ids=val_patients,
        include_info=True
    )
    
    test_dataset = CAMUSDataset(
        root_dir=root_dir,
        split='train',  # Test also from same pool initially
        views=views,
        phases=phases,
        transform=val_transform,
        patient_ids=test_patients,
        include_info=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_patients)} patients, {len(train_dataset)} samples")
    print(f"  Val:   {len(val_patients)} patients, {len(val_dataset)} samples")
    print(f"  Test:  {len(test_patients)} patients, {len(test_dataset)} samples")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def get_cross_val_dataloaders(
    root_dir: str,
    n_folds: int = 5,
    batch_size: int = 8,
    img_size: Tuple[int, int] = (256, 256),
    views: List[str] = ['2CH', '4CH'],
    phases: List[str] = ['ED', 'ES'],
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    stratify_by: str = 'quality'
) -> List[Dict[str, DataLoader]]:
    """
    Create cross-validation dataloaders.
    
    Args:
        root_dir: Path to CAMUS dataset
        n_folds: Number of folds
        batch_size: Batch size
        img_size: Image size (H, W)
        views: List of views to include
        phases: List of phases to include
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed
        stratify_by: Stratification criterion ('quality' or None)
        
    Returns:
        List of dictionaries, each with 'train' and 'val' dataloaders
    """
    np.random.seed(seed)
    
    # Create full dataset
    full_dataset = CAMUSDataset(
        root_dir=root_dir,
        split='train',
        views=views,
        phases=phases,
        transform=None,
        include_info=True
    )
    
    # Get unique patients with their quality for stratification
    patient_info = {}
    for sample in full_dataset.samples:
        pid = sample['patient_id']
        if pid not in patient_info:
            patient = sample['patient']
            quality = patient.get_image_quality(views[0]) or 'Medium'
            patient_info[pid] = quality
    
    patient_ids = list(patient_info.keys())
    patient_ids.sort()
    
    # Stratification labels
    if stratify_by == 'quality':
        quality_map = {'Good': 0, 'Medium': 1, 'Poor': 2, 'Unknown': 1}
        stratify_labels = [quality_map.get(patient_info[p], 1) for p in patient_ids]
    else:
        stratify_labels = None
    
    # Create folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    folds = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, stratify_labels)):
        train_patients = [patient_ids[i] for i in train_idx]
        val_patients = [patient_ids[i] for i in val_idx]
        
        # Create transforms
        train_transform = get_train_transforms(img_size)
        val_transform = get_val_transforms(img_size)
        
        # Create datasets
        train_dataset = CAMUSDataset(
            root_dir=root_dir,
            split='train',
            views=views,
            phases=phases,
            transform=train_transform,
            patient_ids=train_patients,
            include_info=True
        )
        
        val_dataset = CAMUSDataset(
            root_dir=root_dir,
            split='train',
            views=views,
            phases=phases,
            transform=val_transform,
            patient_ids=val_patients,
            include_info=True
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        folds.append({
            'train': train_loader,
            'val': val_loader,
            'fold': fold_idx,
            'train_patients': train_patients,
            'val_patients': val_patients
        })
        
        print(f"Fold {fold_idx + 1}/{n_folds}:")
        print(f"  Train: {len(train_patients)} patients, {len(train_dataset)} samples")
        print(f"  Val:   {len(val_patients)} patients, {len(val_dataset)} samples")
    
    return folds


def get_ef_dataloaders(
    root_dir: str,
    batch_size: int = 4,
    img_size: Tuple[int, int] = (256, 256),
    train_split: float = 0.7,
    val_split: float = 0.15,
    views: List[str] = ['2CH', '4CH'],
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for Ejection Fraction computation.
    Returns ED-ES pairs.
    
    Args:
        root_dir: Path to CAMUS dataset
        batch_size: Batch size
        img_size: Image size (H, W)
        train_split: Fraction for training
        val_split: Fraction for validation
        views: List of views to include
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    np.random.seed(seed)
    
    # Create full dataset to get patient list
    full_dataset = CAMUSEFDataset(
        root_dir=root_dir,
        split='train',
        views=views,
        transform=None
    )
    
    # Get unique patient IDs
    patient_ids = list(set([p['patient_id'] for p in full_dataset.pairs]))
    patient_ids.sort()
    
    # Split patients
    train_patients, test_patients = train_test_split(
        patient_ids,
        test_size=1 - train_split,
        random_state=seed
    )
    
    val_size = val_split / (1 - train_split)
    val_patients, test_patients = train_test_split(
        test_patients,
        test_size=1 - val_size,
        random_state=seed
    )
    
    # Create transforms
    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)
    
    # Create datasets
    train_dataset = CAMUSEFDataset(
        root_dir=root_dir,
        split='train',
        views=views,
        transform=train_transform,
        patient_ids=train_patients
    )
    
    val_dataset = CAMUSEFDataset(
        root_dir=root_dir,
        split='train',
        views=views,
        transform=val_transform,
        patient_ids=val_patients
    )
    
    test_dataset = CAMUSEFDataset(
        root_dir=root_dir,
        split='train',
        views=views,
        transform=val_transform,
        patient_ids=test_patients
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def collate_fn_with_info(batch: List[Dict]) -> Dict:
    """
    Custom collate function that handles variable-length info.
    """
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    
    output = {
        'image': images,
        'mask': masks,
        'patient_id': [item['patient_id'] for item in batch],
        'view': [item['view'] for item in batch],
        'phase': [item['phase'] for item in batch]
    }
    
    # Add optional info if present
    if 'ef' in batch[0]:
        output['ef'] = torch.tensor([item['ef'] for item in batch])
    if 'quality' in batch[0]:
        output['quality'] = [item['quality'] for item in batch]
    
    return output
