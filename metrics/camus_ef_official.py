"""
Official CAMUS Ejection Fraction calculation using Simpson's biplane method of disks.

This implementation is based on the official CAMUS challenge script:
https://www.creatis.insa-lyon.fr/Challenge/camus/

Reference:
    Leclerc S, Smistad E, Pedrosa J, Østvik A, Cervenansky F, Espinosa F, Espeland T, 
    Rye Berg EA, Jodoin PM, Grenier T, Lartizien C, D'hooge J, Lovstakken L, Bernard O. 
    "Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography" 
    IEEE Trans Med Imaging, 2019:38:2198-2210, DOI: 10.1109/TMI.2019.2900516

WARNING:
    The way in which Simpson's biplane method is implemented can have a significant 
    influence on the final values calculated. This implementation follows the official
    CAMUS script for reproducibility.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List, Union
from dataclasses import dataclass

import numpy as np
import PIL
from PIL.Image import Resampling

try:
    from skimage.measure import find_contours
except ImportError:
    find_contours = None
    
import torch

logger = logging.getLogger(__name__)


@dataclass
class CAMUSEFResult:
    """CAMUS Ejection Fraction result."""
    ef_percent: float
    edv_ml: float
    esv_ml: float
    sv_ml: float
    ef_ground_truth: Optional[float] = None
    ef_error: Optional[float] = None
    patient_id: Optional[str] = None


def resize_image(
    image: np.ndarray, 
    size: Tuple[int, int], 
    resample: Resampling = Resampling.NEAREST
) -> np.ndarray:
    """
    Resizes the image to the specified dimensions.

    Args:
        image: (H, W), Input image to resize. Must be in a format supported by PIL.
        size: Width (W') and height (H') dimensions of the resized image to output.
        resample: Resampling filter to use.

    Returns:
        (H', W'), Input image resized to the specified dimensions.
    """
    resized_image = np.array(PIL.Image.fromarray(image).resize(size, resample=resample))
    return resized_image


def resize_image_to_isotropic(
    image: np.ndarray, 
    spacing: Tuple[float, float], 
    resample: Resampling = Resampling.NEAREST
) -> Tuple[np.ndarray, float]:
    """
    Resizes the image to attain isotropic spacing, by resampling the dimension 
    with the biggest voxel size.

    Args:
        image: (H, W), Input image to resize. Must be in a format supported by PIL.
        spacing: Size of the image's pixels along each (height, width) dimension.
        resample: Resampling filter to use.

    Returns:
        (H', W'), Input image resized so that the spacing is isotropic, 
        and the isotropic value of the new spacing.
    """
    scaling = np.array(spacing) / min(spacing)
    new_height, new_width = (np.array(image.shape) * scaling).round().astype(int)
    return resize_image(image, (new_width, new_height), resample=resample), min(spacing)


def _distance_line_to_points(
    line_point_0: np.ndarray, 
    line_point_1: np.ndarray, 
    points: np.ndarray
) -> np.ndarray:
    """Calculate distance from a line to multiple points."""
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    return np.absolute(
        np.cross(line_point_1 - line_point_0, line_point_0 - points)
    ) / np.linalg.norm(line_point_1 - line_point_0)


def _get_angle_of_lines_to_point(
    reference_point: np.ndarray, 
    moving_points: np.ndarray
) -> np.ndarray:
    """Get angles of lines from reference point to moving points."""
    diff = moving_points - reference_point
    return abs(np.degrees(np.arctan2(diff[:, 0], diff[:, 1])))


def _find_distance_to_edge(
    segmentation: np.ndarray, 
    point_on_mid_line: np.ndarray, 
    normal_direction: np.ndarray
) -> float:
    """Find distance from a point on the midline to the edge of the segmentation."""
    distance = 8  # start a bit in to avoid line stopping early at base
    while True:
        current_position = point_on_mid_line + distance * normal_direction

        y, x = np.round(current_position).astype(int)
        if segmentation.shape[0] <= y or y < 0 or segmentation.shape[1] <= x or x < 0:
            # out of bounds
            return distance

        elif segmentation[y, x] == 0:
            # Edge found
            return distance

        distance += 0.5


def _compute_diameters(
    segmentation: np.ndarray, 
    voxelspacing: Tuple[float, float],
    n_disks: int = 20
) -> Tuple[np.ndarray, float]:
    """
    Compute diameters at n_disks positions along the LV long axis.
    
    This is the official CAMUS method that:
    1. Makes the image isotropic
    2. Finds the contour
    3. Identifies the AV plane (base) and apex
    4. Computes diameters perpendicular to the long axis

    Args:
        segmentation: Binary segmentation of the LV.
        voxelspacing: Size of the segmentations' voxels along each (height, width) dimension (in mm).
        n_disks: Number of disks for Simpson's method.

    Returns:
        Array of diameters and the step size.
    """
    if find_contours is None:
        raise ImportError("scikit-image is required for official CAMUS EF computation. "
                          "Install with: pip install scikit-image")
    
    # Make image isotropic, have same spacing in both directions
    segmentation, isotropic_spacing = resize_image_to_isotropic(segmentation, voxelspacing)

    # Find contour
    contours = find_contours(segmentation, 0.5)
    if len(contours) == 0:
        return np.zeros(n_disks), 0.0
    
    contour = contours[0]

    # Find AV plane (base of the heart)
    # For each pair of contour points:
    # - Check if angle is acceptable (acute)
    # - Check that intermediate points are close to the line
    # - Select the longest stretch
    best_length = 0
    best_i = 0
    best_j = 0
    
    for point_idx in range(2, len(contour)):
        previous_points = contour[:point_idx]
        angles_to_previous_points = _get_angle_of_lines_to_point(contour[point_idx], previous_points)

        for acute_angle_idx in np.nonzero(angles_to_previous_points <= 45)[0]:
            intermediate_points = contour[acute_angle_idx + 1 : point_idx]
            if len(intermediate_points) == 0:
                continue
                
            distance_to_intermediate_points = _distance_line_to_points(
                contour[point_idx], contour[acute_angle_idx], intermediate_points
            )
            if np.all(distance_to_intermediate_points <= 8):
                distance = np.linalg.norm(contour[point_idx] - contour[acute_angle_idx])
                if best_length < distance:
                    best_length = distance
                    best_i = point_idx
                    best_j = acute_angle_idx

    if best_length == 0:
        # Fallback: use simple bounding box method
        return _compute_diameters_simple(segmentation, isotropic_spacing, n_disks)

    # Find midpoint of AV plane
    mid_point = int(best_j + round((best_i - best_j) / 2))
    
    # Apex is the point furthest from midpoint
    mid_line_length = 0
    apex = 0
    for i in range(len(contour)):
        length = np.linalg.norm(contour[mid_point] - contour[i])
        if mid_line_length < length:
            mid_line_length = length
            apex = i

    # Direction from midpoint to apex
    direction = contour[apex] - contour[mid_point]
    
    # Normal direction (perpendicular to long axis)
    normal_direction = np.array([-direction[1], direction[0]])
    normal_direction = normal_direction / np.linalg.norm(normal_direction)  # Normalize
    
    # Compute diameters at each disk position
    diameters = []
    for fraction in np.linspace(0, 1, n_disks, endpoint=False):
        point_on_mid_line = contour[mid_point] + direction * fraction

        distance1 = _find_distance_to_edge(segmentation, point_on_mid_line, normal_direction)
        distance2 = _find_distance_to_edge(segmentation, point_on_mid_line, -normal_direction)
        diameters.append((distance1 + distance2) * isotropic_spacing)

    step_size = (mid_line_length * isotropic_spacing) / n_disks
    return np.array(diameters), step_size


def _compute_diameters_simple(
    segmentation: np.ndarray,
    spacing: float,
    n_disks: int = 20
) -> Tuple[np.ndarray, float]:
    """
    Simple fallback method for computing diameters using bounding box.
    Used when contour-based method fails.
    """
    rows = segmentation.any(axis=1)
    if not rows.any():
        return np.zeros(n_disks), 0.0
    
    row_indices = np.where(rows)[0]
    start_row = row_indices.min()
    end_row = row_indices.max()
    
    disk_height_px = (end_row - start_row) / n_disks
    
    diameters = []
    for i in range(n_disks):
        row_idx = int(start_row + (i + 0.5) * disk_height_px)
        row_idx = min(row_idx, segmentation.shape[0] - 1)
        
        row_mask = segmentation[row_idx, :]
        if row_mask.any():
            cols = np.where(row_mask)[0]
            diameter_px = cols.max() - cols.min() + 1
            diameter_mm = diameter_px * spacing
        else:
            diameter_mm = 0.0
        
        diameters.append(diameter_mm)
    
    step_size = (end_row - start_row) * spacing / n_disks
    return np.array(diameters), step_size


def _compute_left_ventricle_volume_by_instant(
    a2c_diameters: np.ndarray, 
    a4c_diameters: np.ndarray, 
    step_size: float
) -> float:
    """
    Compute left ventricle volume using Biplane Simpson's method.

    Args:
        a2c_diameters: Diameters from the 2-chamber apical view (in mm).
        a4c_diameters: Diameters from the 4-chamber apical view (in mm).
        step_size: Height of each disk (in mm).

    Returns:
        Left ventricle volume (in millilitres).
    """
    # All measures are in millimeters, convert to meters by dividing by 1000
    a2c_diameters = a2c_diameters / 1000
    a4c_diameters = a4c_diameters / 1000
    step_size = step_size / 1000

    # Estimate left ventricle volume from orthogonal disks
    # V = π/4 * h * Σ(a_i * b_i)
    lv_volume = np.sum(a2c_diameters * a4c_diameters) * step_size * np.pi / 4

    # Volume is now in cubic meters, convert to milliliters (1 m³ = 1,000,000 mL)
    return round(lv_volume * 1e6)


def compute_left_ventricle_volumes(
    a2c_ed: np.ndarray,
    a2c_es: np.ndarray,
    a2c_voxelspacing: Tuple[float, float],
    a4c_ed: np.ndarray,
    a4c_es: np.ndarray,
    a4c_voxelspacing: Tuple[float, float],
    n_disks: int = 20
) -> Tuple[float, float]:
    """
    Computes the ED and ES volumes of the left ventricle from 2 orthogonal 2D views.
    
    This is the official CAMUS method using Simpson's biplane method of disks.

    Args:
        a2c_ed: (H,W), Binary segmentation map of the LV from ED instant of A2C view.
        a2c_es: (H,W), Binary segmentation map of the LV from ES instant of A2C view.
        a2c_voxelspacing: Size (in mm) of A2C voxels along each (height, width) dimension.
        a4c_ed: (H,W), Binary segmentation map of the LV from ED instant of A4C view.
        a4c_es: (H,W), Binary segmentation map of the LV from ES instant of A4C view.
        a4c_voxelspacing: Size (in mm) of A4C voxels along each (height, width) dimension.
        n_disks: Number of disks for Simpson's method (default: 20).

    Returns:
        Left ventricle ED and ES volumes in milliliters.
    """
    # Validate inputs
    for mask_name, mask in [("a2c_ed", a2c_ed), ("a2c_es", a2c_es), 
                             ("a4c_ed", a4c_ed), ("a4c_es", a4c_es)]:
        if mask.max() > 1:
            logger.warning(
                f"`compute_left_ventricle_volumes` expects binary segmentation masks of the LV. "
                f"However, `{mask_name}` contains labels > 1. If intentional, ignore this warning. "
                f"Most likely you forgot to extract the binary LV mask from a multi-class segmentation."
            )

    # Compute diameters for each view and instant
    a2c_ed_diameters, a2c_ed_step_size = _compute_diameters(a2c_ed, a2c_voxelspacing, n_disks)
    a2c_es_diameters, a2c_es_step_size = _compute_diameters(a2c_es, a2c_voxelspacing, n_disks)
    a4c_ed_diameters, a4c_ed_step_size = _compute_diameters(a4c_ed, a4c_voxelspacing, n_disks)
    a4c_es_diameters, a4c_es_step_size = _compute_diameters(a4c_es, a4c_voxelspacing, n_disks)
    
    # Use the maximum step size (most conservative estimate)
    step_size = max(a2c_ed_step_size, a2c_es_step_size, a4c_ed_step_size, a4c_es_step_size)

    # Compute volumes using biplane method
    ed_volume = _compute_left_ventricle_volume_by_instant(a2c_ed_diameters, a4c_ed_diameters, step_size)
    es_volume = _compute_left_ventricle_volume_by_instant(a2c_es_diameters, a4c_es_diameters, step_size)
    
    return ed_volume, es_volume


def compute_ejection_fraction(
    a2c_ed: np.ndarray,
    a2c_es: np.ndarray,
    a2c_voxelspacing: Tuple[float, float],
    a4c_ed: np.ndarray,
    a4c_es: np.ndarray,
    a4c_voxelspacing: Tuple[float, float],
    lv_label: int = 1,
    n_disks: int = 20
) -> CAMUSEFResult:
    """
    Compute Ejection Fraction using official CAMUS Simpson's biplane method.

    Args:
        a2c_ed: (H,W), Segmentation from ED instant of A2C view.
        a2c_es: (H,W), Segmentation from ES instant of A2C view.
        a2c_voxelspacing: Voxel spacing (height, width) in mm for A2C.
        a4c_ed: (H,W), Segmentation from ED instant of A4C view.
        a4c_es: (H,W), Segmentation from ES instant of A4C view.
        a4c_voxelspacing: Voxel spacing (height, width) in mm for A4C.
        lv_label: Label value for LV endocardium (default: 1).
        n_disks: Number of disks for Simpson's method (default: 20).

    Returns:
        CAMUSEFResult with EF and volumes.
    """
    # Extract binary LV masks
    a2c_ed_lv = (a2c_ed == lv_label).astype(np.uint8)
    a2c_es_lv = (a2c_es == lv_label).astype(np.uint8)
    a4c_ed_lv = (a4c_ed == lv_label).astype(np.uint8)
    a4c_es_lv = (a4c_es == lv_label).astype(np.uint8)

    # Compute volumes
    edv, esv = compute_left_ventricle_volumes(
        a2c_ed_lv, a2c_es_lv, a2c_voxelspacing,
        a4c_ed_lv, a4c_es_lv, a4c_voxelspacing,
        n_disks=n_disks
    )

    # Compute EF
    if edv > 0:
        ef = round(100 * (edv - esv) / edv)
    else:
        ef = 0.0

    sv = edv - esv

    return CAMUSEFResult(
        ef_percent=ef,
        edv_ml=edv,
        esv_ml=esv,
        sv_ml=sv
    )


class CAMUSEFCalculator:
    """
    Calculator for CAMUS Ejection Fraction using official Simpson's biplane method.
    
    This class provides a convenient interface for computing EF from model predictions
    on the CAMUS dataset.
    
    Args:
        lv_label: Label value for LV endocardium in segmentation masks (default: 1).
        n_disks: Number of disks for Simpson's method (default: 20).
    """
    
    def __init__(self, lv_label: int = 1, n_disks: int = 20):
        self.lv_label = lv_label
        self.n_disks = n_disks
        self.results: List[CAMUSEFResult] = []
    
    def reset(self):
        """Reset accumulated results."""
        self.results = []
    
    def compute_ef(
        self,
        a2c_ed: Union[np.ndarray, torch.Tensor],
        a2c_es: Union[np.ndarray, torch.Tensor],
        a2c_spacing: Tuple[float, float],
        a4c_ed: Union[np.ndarray, torch.Tensor],
        a4c_es: Union[np.ndarray, torch.Tensor],
        a4c_spacing: Tuple[float, float],
        ef_ground_truth: Optional[float] = None,
        patient_id: Optional[str] = None
    ) -> CAMUSEFResult:
        """
        Compute EF for a single patient.
        
        Args:
            a2c_ed: A2C view ED segmentation.
            a2c_es: A2C view ES segmentation.
            a2c_spacing: A2C voxel spacing (height, width) in mm.
            a4c_ed: A4C view ED segmentation.
            a4c_es: A4C view ES segmentation.
            a4c_spacing: A4C voxel spacing (height, width) in mm.
            ef_ground_truth: Ground truth EF from cfg file (optional).
            patient_id: Patient identifier (optional).
            
        Returns:
            CAMUSEFResult with computed EF and volumes.
        """
        # Convert tensors to numpy
        if isinstance(a2c_ed, torch.Tensor):
            a2c_ed = a2c_ed.cpu().numpy()
        if isinstance(a2c_es, torch.Tensor):
            a2c_es = a2c_es.cpu().numpy()
        if isinstance(a4c_ed, torch.Tensor):
            a4c_ed = a4c_ed.cpu().numpy()
        if isinstance(a4c_es, torch.Tensor):
            a4c_es = a4c_es.cpu().numpy()
        
        # Squeeze if needed
        if a2c_ed.ndim == 3:
            a2c_ed = a2c_ed.squeeze(0)
        if a2c_es.ndim == 3:
            a2c_es = a2c_es.squeeze(0)
        if a4c_ed.ndim == 3:
            a4c_ed = a4c_ed.squeeze(0)
        if a4c_es.ndim == 3:
            a4c_es = a4c_es.squeeze(0)
        
        result = compute_ejection_fraction(
            a2c_ed, a2c_es, a2c_spacing,
            a4c_ed, a4c_es, a4c_spacing,
            lv_label=self.lv_label,
            n_disks=self.n_disks
        )
        
        result.patient_id = patient_id
        result.ef_ground_truth = ef_ground_truth
        
        if ef_ground_truth is not None:
            result.ef_error = result.ef_percent - ef_ground_truth
        
        self.results.append(result)
        return result
    
    def compute_statistics(self) -> Dict[str, float]:
        """
        Compute statistics over all accumulated results.
        
        Returns:
            Dictionary with MAE, correlation, bias, and other statistics.
        """
        if not self.results:
            return {}
        
        ef_predicted = np.array([r.ef_percent for r in self.results])
        edv_predicted = np.array([r.edv_ml for r in self.results])
        esv_predicted = np.array([r.esv_ml for r in self.results])
        
        stats = {
            'n_patients': len(self.results),
            'ef_mean': float(np.mean(ef_predicted)),
            'ef_std': float(np.std(ef_predicted)),
            'edv_mean': float(np.mean(edv_predicted)),
            'esv_mean': float(np.mean(esv_predicted)),
        }
        
        # If ground truth available, compute error metrics
        results_with_gt = [r for r in self.results if r.ef_ground_truth is not None]
        if results_with_gt:
            ef_pred = np.array([r.ef_percent for r in results_with_gt])
            ef_gt = np.array([r.ef_ground_truth for r in results_with_gt])
            ef_errors = ef_pred - ef_gt
            
            stats.update({
                'ef_mae': float(np.mean(np.abs(ef_errors))),
                'ef_rmse': float(np.sqrt(np.mean(ef_errors ** 2))),
                'ef_bias': float(np.mean(ef_errors)),
                'ef_std_error': float(np.std(ef_errors)),
            })
            
            # Pearson correlation
            if len(ef_pred) > 2:
                correlation = np.corrcoef(ef_pred, ef_gt)[0, 1]
                stats['ef_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
            
            # Bland-Altman limits of agreement
            mean_ef = (ef_pred + ef_gt) / 2
            diff_ef = ef_pred - ef_gt
            bias = np.mean(diff_ef)
            std_diff = np.std(diff_ef)
            stats['bland_altman_bias'] = float(bias)
            stats['bland_altman_loa_lower'] = float(bias - 1.96 * std_diff)
            stats['bland_altman_loa_upper'] = float(bias + 1.96 * std_diff)
        
        return stats
    
    def get_results_dataframe(self):
        """
        Get results as a pandas DataFrame (if pandas is available).
        """
        try:
            import pandas as pd
            
            data = []
            for r in self.results:
                data.append({
                    'patient_id': r.patient_id,
                    'ef_predicted': r.ef_percent,
                    'ef_ground_truth': r.ef_ground_truth,
                    'ef_error': r.ef_error,
                    'edv_ml': r.edv_ml,
                    'esv_ml': r.esv_ml,
                    'sv_ml': r.sv_ml,
                })
            
            return pd.DataFrame(data)
        except ImportError:
            logger.warning("pandas not available. Install with: pip install pandas")
            return None


if __name__ == '__main__':
    # Test with synthetic data
    print("Testing CAMUS Official EF Calculator...")
    
    # Create synthetic LV masks
    a2c_ed = np.zeros((256, 256), dtype=np.uint8)
    a2c_es = np.zeros((256, 256), dtype=np.uint8)
    a4c_ed = np.zeros((256, 256), dtype=np.uint8)
    a4c_es = np.zeros((256, 256), dtype=np.uint8)
    
    # Simulate LV (elliptical shape)
    from skimage.draw import ellipse
    
    # ED (larger)
    rr, cc = ellipse(128, 128, 60, 40)
    a2c_ed[rr, cc] = 1
    rr, cc = ellipse(128, 128, 60, 45)
    a4c_ed[rr, cc] = 1
    
    # ES (smaller)
    rr, cc = ellipse(128, 128, 45, 30)
    a2c_es[rr, cc] = 1
    rr, cc = ellipse(128, 128, 45, 35)
    a4c_es[rr, cc] = 1
    
    # Assume 1mm spacing
    spacing = (1.0, 1.0)
    
    calculator = CAMUSEFCalculator()
    result = calculator.compute_ef(
        a2c_ed, a2c_es, spacing,
        a4c_ed, a4c_es, spacing,
        ef_ground_truth=55.0,
        patient_id='test_patient'
    )
    
    print(f"Patient: {result.patient_id}")
    print(f"EF: {result.ef_percent:.1f}%")
    print(f"EDV: {result.edv_ml:.1f} mL")
    print(f"ESV: {result.esv_ml:.1f} mL")
    print(f"SV: {result.sv_ml:.1f} mL")
    print(f"GT EF: {result.ef_ground_truth}")
    print(f"Error: {result.ef_error:.1f}%")
    
    print("\nStatistics:")
    stats = calculator.compute_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
