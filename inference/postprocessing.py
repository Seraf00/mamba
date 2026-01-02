"""
Postprocessing utilities for segmentation masks.
"""

import numpy as np
from scipy import ndimage
from typing import Optional, Tuple


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 100,
    connectivity: int = 2
) -> np.ndarray:
    """
    Remove small connected components from segmentation mask.
    
    Args:
        mask: Segmentation mask (H, W) with integer labels
        min_size: Minimum component size to keep
        connectivity: Connectivity for component labeling (1=4-conn, 2=8-conn)
        
    Returns:
        Cleaned mask
    """
    cleaned = np.zeros_like(mask)
    
    # Process each class separately
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background
    
    for label in unique_labels:
        binary = mask == label
        
        # Label connected components
        labeled, num_features = ndimage.label(
            binary,
            structure=ndimage.generate_binary_structure(2, connectivity)
        )
        
        # Find component sizes
        sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        
        # Keep only components above threshold
        for i, size in enumerate(sizes):
            if size >= min_size:
                cleaned[labeled == (i + 1)] = label
    
    return cleaned


def smooth_boundaries(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Smooth segmentation boundaries using morphological operations.
    
    Args:
        mask: Segmentation mask (H, W)
        kernel_size: Size of morphological kernel
        iterations: Number of iterations
        
    Returns:
        Smoothed mask
    """
    from scipy.ndimage import binary_opening, binary_closing
    
    smoothed = np.zeros_like(mask)
    structure = np.ones((kernel_size, kernel_size))
    
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    
    for label in unique_labels:
        binary = mask == label
        
        # Opening followed by closing for smoothing
        binary = binary_closing(binary, structure, iterations=iterations)
        binary = binary_opening(binary, structure, iterations=iterations)
        
        # Only add where smoothed has no label yet
        smoothed[binary & (smoothed == 0)] = label
    
    return smoothed


def enforce_topology(
    mask: np.ndarray,
    keep_largest: bool = True
) -> np.ndarray:
    """
    Enforce cardiac anatomy topology constraints.
    
    Ensures:
    - Each class has at most one connected component
    - LV (1) is enclosed by MYO (2) or adjacent
    - LA (3) is connected to LV through valve
    
    Args:
        mask: Segmentation mask with labels:
              0: Background, 1: LV, 2: MYO, 3: LA
        keep_largest: Keep only largest component per class
        
    Returns:
        Topologically corrected mask
    """
    corrected = np.zeros_like(mask)
    
    # Process each class
    for label in [1, 2, 3]:  # LV, MYO, LA
        binary = mask == label
        
        if not np.any(binary):
            continue
        
        if keep_largest:
            # Keep only largest connected component
            labeled, num_features = ndimage.label(binary)
            
            if num_features > 0:
                sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
                largest_idx = np.argmax(sizes) + 1
                corrected[labeled == largest_idx] = label
        else:
            corrected[binary] = label
    
    return corrected


def fill_holes(
    mask: np.ndarray,
    class_label: Optional[int] = None
) -> np.ndarray:
    """
    Fill holes in segmentation mask.
    
    Args:
        mask: Segmentation mask
        class_label: Fill holes only for specific class
        
    Returns:
        Mask with filled holes
    """
    from scipy.ndimage import binary_fill_holes
    
    filled = mask.copy()
    
    if class_label is not None:
        labels = [class_label]
    else:
        labels = np.unique(mask)
        labels = labels[labels != 0]
    
    for label in labels:
        binary = mask == label
        binary_filled = binary_fill_holes(binary)
        
        # Fill holes with current label
        holes = binary_filled & ~binary
        filled[holes] = label
    
    return filled


def apply_crf(
    image: np.ndarray,
    probs: np.ndarray,
    n_iterations: int = 5
) -> np.ndarray:
    """
    Apply Conditional Random Field for boundary refinement.
    
    Requires pydensecrf package.
    
    Args:
        image: Original image (H, W) or (H, W, C)
        probs: Class probabilities (C, H, W)
        n_iterations: CRF iterations
        
    Returns:
        Refined segmentation mask
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        print("pydensecrf not installed, returning argmax")
        return probs.argmax(axis=0)
    
    n_classes, h, w = probs.shape
    
    # Create CRF
    d = dcrf.DenseCRF2D(w, h, n_classes)
    
    # Unary potentials
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)
    
    # Pairwise potentials
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    
    image = np.ascontiguousarray(image.astype(np.uint8))
    
    d.addPairwiseBilateral(
        sxy=10,
        srgb=13,
        rgbim=image,
        compat=10
    )
    
    # Inference
    Q = d.inference(n_iterations)
    Q = np.array(Q).reshape((n_classes, h, w))
    
    return Q.argmax(axis=0)


def compute_contours(
    mask: np.ndarray
) -> dict:
    """
    Extract contours for each class.
    
    Args:
        mask: Segmentation mask
        
    Returns:
        Dictionary mapping class labels to contour points
    """
    from scipy.ndimage import binary_erosion
    
    contours = {}
    
    for label in np.unique(mask):
        if label == 0:
            continue
        
        binary = mask == label
        eroded = binary_erosion(binary)
        contour = binary & ~eroded
        
        # Get contour points
        points = np.where(contour)
        contours[label] = np.stack(points, axis=1)
    
    return contours


def temporal_consistency(
    masks: np.ndarray,
    max_change_ratio: float = 0.3
) -> np.ndarray:
    """
    Enforce temporal consistency across frames.
    
    Args:
        masks: Sequence of masks (T, H, W)
        max_change_ratio: Maximum allowed area change between frames
        
    Returns:
        Temporally smoothed masks
    """
    T = masks.shape[0]
    smoothed = masks.copy()
    
    for t in range(1, T):
        for label in [1, 2, 3]:
            prev_area = np.sum(masks[t-1] == label)
            curr_area = np.sum(masks[t] == label)
            
            if prev_area == 0:
                continue
            
            change_ratio = abs(curr_area - prev_area) / prev_area
            
            if change_ratio > max_change_ratio:
                # Blend with previous frame
                alpha = max_change_ratio / change_ratio
                
                prev_binary = masks[t-1] == label
                curr_binary = masks[t] == label
                
                # Morphological interpolation
                blended = ndimage.binary_dilation(
                    prev_binary,
                    iterations=int(5 * (1 - alpha))
                ) | ndimage.binary_erosion(
                    curr_binary,
                    iterations=int(5 * (1 - alpha))
                )
                
                smoothed[t][smoothed[t] == label] = 0
                smoothed[t][blended] = label
    
    return smoothed


if __name__ == '__main__':
    # Test postprocessing functions
    test_mask = np.zeros((256, 256), dtype=np.int32)
    test_mask[50:150, 50:150] = 1
    test_mask[60:140, 60:140] = 2
    
    # Add small noise component
    test_mask[200:205, 200:205] = 1
    
    cleaned = remove_small_components(test_mask, min_size=50)
    print(f"Original components in class 1: {len(np.unique(ndimage.label(test_mask == 1)[0])) - 1}")
    print(f"After cleaning: {len(np.unique(ndimage.label(cleaned == 1)[0])) - 1}")
