"""
Data augmentation transforms for CAMUS dataset.
Uses Albumentations library for efficient augmentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, Optional, Tuple


def get_train_transforms(
    img_size: Tuple[int, int] = (256, 256),
    mean: float = 0.5,
    std: float = 0.5
) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Includes:
    - Geometric transforms (flip, rotate, scale, shift)
    - Intensity transforms (brightness, contrast)
    - Elastic deformation (common in medical imaging)
    - Grid distortion
    
    Args:
        img_size: Target image size (H, W)
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        # Resize
        A.Resize(height=img_size[0], width=img_size[1]),
        
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,  # cv2.BORDER_CONSTANT
            value=0,
            mask_value=0,
            p=0.5
        ),
        
        # Elastic deformation - important for ultrasound
        A.ElasticTransform(
            alpha=120,
            sigma=6,
            p=0.3
        ),
        
        # Grid distortion
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            p=0.3
        ),
        
        # Intensity transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        
        # Gaussian noise
        A.GaussNoise(
            var_limit=(0.001, 0.01),
            p=0.3
        ),
        
        # Gaussian blur
        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.2
        ),
        
        # Speckle noise (common in ultrasound)
        A.MultiplicativeNoise(
            multiplier=(0.9, 1.1),
            per_channel=False,
            p=0.3
        ),
        
        # Normalize
        A.Normalize(mean=mean, std=std),
        
        # Convert to tensor
        ToTensorV2()
    ])


def get_transforms(
    split: str = 'train',
    img_size: Tuple[int, int] = (256, 256),
    mean: float = 0.5,
    std: float = 0.5
) -> A.Compose:
    """
    Get transforms for specified split.
    
    Args:
        split: 'train', 'val', or 'test'
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Albumentations transform pipeline
    """
    if split == 'train':
        return get_train_transforms(img_size, mean, std)
    elif split in ('val', 'validation'):
        return get_val_transforms(img_size, mean, std)
    elif split == 'test':
        return get_test_transforms(img_size, mean, std)
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'")

def get_val_transforms(
    img_size: Tuple[int, int] = (256, 256),
    mean: float = 0.5,
    std: float = 0.5
) -> A.Compose:
    """
    Get validation/inference transform pipeline.
    
    Only includes resize and normalization.
    
    Args:
        img_size: Target image size (H, W)
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def get_test_transforms(
    img_size: Tuple[int, int] = (256, 256),
    mean: float = 0.5,
    std: float = 0.5
) -> A.Compose:
    """
    Get test transform pipeline (same as validation).
    
    Args:
        img_size: Target image size (H, W)
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Albumentations Compose transform
    """
    return get_val_transforms(img_size, mean, std)


def get_tta_transforms(
    img_size: Tuple[int, int] = (256, 256),
    mean: float = 0.5,
    std: float = 0.5
) -> list:
    """
    Get Test-Time Augmentation transforms.
    
    Returns a list of transforms to apply during inference.
    Predictions should be aggregated (e.g., averaged).
    
    Args:
        img_size: Target image size (H, W)
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        List of Albumentations Compose transforms
    """
    base = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    hflip = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    rotate_5 = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Rotate(limit=(5, 5), p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    rotate_neg5 = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Rotate(limit=(-5, -5), p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    scale_up = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.RandomScale(scale_limit=(0.05, 0.05), p=1.0),
        A.CenterCrop(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return [base, hflip, rotate_5, rotate_neg5, scale_up]


class IntensityWindowTransform:
    """
    Custom transform for intensity windowing.
    Useful for enhancing specific tissue contrasts.
    """
    
    def __init__(self, window_center: float = 0.5, window_width: float = 1.0):
        self.window_center = window_center
        self.window_width = window_width
    
    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        min_val = self.window_center - self.window_width / 2
        max_val = self.window_center + self.window_width / 2
        
        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val + 1e-8)
        
        return image


class UltrasoundSpeckleNoise:
    """
    Custom transform for realistic ultrasound speckle noise.
    Models multiplicative noise characteristic of ultrasound.
    """
    
    def __init__(self, intensity: float = 0.1, p: float = 0.5):
        self.intensity = intensity
        self.p = p
    
    def __call__(self, image: np.ndarray, **kwargs) -> Dict:
        if np.random.random() > self.p:
            return {'image': image}
        
        # Rayleigh-distributed speckle noise
        noise = np.random.rayleigh(scale=self.intensity, size=image.shape)
        noisy_image = image * (1 + noise - noise.mean())
        noisy_image = np.clip(noisy_image, 0, 1)
        
        return {'image': noisy_image.astype(np.float32)}


def get_strong_augmentation(
    img_size: Tuple[int, int] = (256, 256),
    mean: float = 0.5,
    std: float = 0.5
) -> A.Compose:
    """
    Get strong augmentation pipeline for challenging training.
    
    Args:
        img_size: Target image size (H, W)
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        # Resize
        A.Resize(height=img_size[0], width=img_size[1]),
        
        # Strong geometric transforms
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=30,
            border_mode=0,
            p=0.7
        ),
        
        # Elastic deformation
        A.OneOf([
            A.ElasticTransform(alpha=150, sigma=8, p=1.0),
            A.GridDistortion(num_steps=6, distort_limit=0.4, p=1.0),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=1.0),
        ], p=0.5),
        
        # Strong intensity transforms
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.5),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(0.005, 0.02), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.3),
        
        # Cutout/CoarseDropout
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.3
        ),
        
        # Normalize
        A.Normalize(mean=mean, std=std),
        
        # Convert to tensor
        ToTensorV2()
    ])
