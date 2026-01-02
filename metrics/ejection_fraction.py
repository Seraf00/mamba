"""
Ejection Fraction calculation from cardiac segmentation.

Implements volume estimation methods and EF computation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class VolumeResult:
    """Volume calculation result."""
    volume_ml: float
    area_mm2: float
    length_mm: float
    method: str


@dataclass
class EFResult:
    """Ejection fraction result."""
    ef_percent: float
    edv_ml: float
    esv_ml: float
    sv_ml: float
    category: str
    view: str


class VolumeCalculator:
    """
    Calculate LV volume from 2D segmentation.
    
    Args:
        pixel_spacing: Pixel spacing in mm (height, width)
        method: Volume estimation method ('simpson', 'area_length', 'disk_sum')
    """
    
    def __init__(
        self,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
        method: str = 'simpson'
    ):
        self.pixel_spacing = pixel_spacing
        self.method = method
    
    def calculate(
        self,
        segmentation: torch.Tensor,
        lv_class: int = 1
    ) -> VolumeResult:
        """
        Calculate LV volume from segmentation.
        
        Args:
            segmentation: Segmentation mask (H, W) or (B, H, W)
            lv_class: Class index for LV endocardium
            
        Returns:
            VolumeResult with volume, area, and length
        """
        if segmentation.dim() == 3:
            segmentation = segmentation[0]
        
        mask = (segmentation == lv_class).float().cpu().numpy()
        
        # Calculate area
        area_px = mask.sum()
        area_mm2 = area_px * self.pixel_spacing[0] * self.pixel_spacing[1]
        
        # Calculate length (major axis)
        length_mm = self._calculate_length(mask)
        
        # Calculate volume based on method
        if self.method == 'simpson':
            volume_ml = self._simpson_method(mask)
        elif self.method == 'area_length':
            volume_ml = self._area_length_method(area_mm2, length_mm)
        elif self.method == 'disk_sum':
            volume_ml = self._disk_sum_method(mask)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return VolumeResult(
            volume_ml=volume_ml,
            area_mm2=area_mm2,
            length_mm=length_mm,
            method=self.method
        )
    
    def _calculate_length(self, mask: np.ndarray) -> float:
        """Calculate LV long axis length."""
        rows = mask.any(axis=1)
        if not rows.any():
            return 0.0
        
        row_indices = np.where(rows)[0]
        length_px = row_indices.max() - row_indices.min() + 1
        return length_px * self.pixel_spacing[0]
    
    def _simpson_method(
        self,
        mask: np.ndarray,
        n_disks: int = 20
    ) -> float:
        """
        Simpson's method of disks.
        
        Divides LV into n_disks equally spaced disks and sums volumes.
        V = Σ (π * (d_i/2)² * h)
        """
        rows = mask.any(axis=1)
        if not rows.any():
            return 0.0
        
        row_indices = np.where(rows)[0]
        start_row = row_indices.min()
        end_row = row_indices.max()
        
        disk_height_px = (end_row - start_row) / n_disks
        disk_height_mm = disk_height_px * self.pixel_spacing[0]
        
        volume = 0.0
        
        for i in range(n_disks):
            row_start = int(start_row + i * disk_height_px)
            row_end = int(start_row + (i + 1) * disk_height_px)
            
            # Get disk width (diameter)
            disk_mask = mask[row_start:row_end, :]
            if disk_mask.any():
                cols = disk_mask.any(axis=0)
                col_indices = np.where(cols)[0]
                diameter_px = col_indices.max() - col_indices.min() + 1
                diameter_mm = diameter_px * self.pixel_spacing[1]
                
                # Volume of disk
                disk_volume = np.pi * (diameter_mm / 2) ** 2 * disk_height_mm
                volume += disk_volume
        
        # Convert mm³ to mL
        return volume / 1000
    
    def _area_length_method(
        self,
        area_mm2: float,
        length_mm: float
    ) -> float:
        """
        Area-length method.
        
        V = (8 * A²) / (3 * π * L)
        """
        if length_mm == 0:
            return 0.0
        
        volume_mm3 = (8 * area_mm2 ** 2) / (3 * np.pi * length_mm)
        return volume_mm3 / 1000  # mm³ to mL
    
    def _disk_sum_method(self, mask: np.ndarray) -> float:
        """
        Simple disk summation (each row is a disk).
        """
        volume = 0.0
        height_mm = self.pixel_spacing[0]
        
        for row in range(mask.shape[0]):
            row_mask = mask[row, :]
            if row_mask.any():
                cols = np.where(row_mask)[0]
                diameter_px = cols.max() - cols.min() + 1
                diameter_mm = diameter_px * self.pixel_spacing[1]
                
                disk_volume = np.pi * (diameter_mm / 2) ** 2 * height_mm
                volume += disk_volume
        
        return volume / 1000


class BiplaneSimpson:
    """
    Biplane Simpson's method using 2CH and 4CH views.
    
    More accurate volume estimation using two orthogonal views.
    V = π/4 * h * Σ (a_i * b_i)
    where a_i and b_i are diameters from 2CH and 4CH views.
    """
    
    def __init__(
        self,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
        n_disks: int = 20
    ):
        self.pixel_spacing = pixel_spacing
        self.n_disks = n_disks
    
    def calculate(
        self,
        seg_2ch: torch.Tensor,
        seg_4ch: torch.Tensor,
        lv_class: int = 1
    ) -> VolumeResult:
        """
        Calculate volume using biplane method.
        
        Args:
            seg_2ch: 2-chamber view segmentation
            seg_4ch: 4-chamber view segmentation
            lv_class: LV endocardium class
            
        Returns:
            Volume result
        """
        mask_2ch = (seg_2ch == lv_class).float().cpu().numpy()
        mask_4ch = (seg_4ch == lv_class).float().cpu().numpy()
        
        if mask_2ch.ndim == 3:
            mask_2ch = mask_2ch[0]
        if mask_4ch.ndim == 3:
            mask_4ch = mask_4ch[0]
        
        # Get LV lengths
        length_2ch = self._get_length(mask_2ch)
        length_4ch = self._get_length(mask_4ch)
        
        # Use average length
        length_mm = (length_2ch + length_4ch) / 2
        
        if length_mm == 0:
            return VolumeResult(0, 0, 0, 'biplane_simpson')
        
        disk_height = length_mm / self.n_disks
        
        # Get diameters from each view
        diameters_2ch = self._get_diameters(mask_2ch, self.n_disks)
        diameters_4ch = self._get_diameters(mask_4ch, self.n_disks)
        
        # Calculate volume
        volume = 0.0
        for a, b in zip(diameters_2ch, diameters_4ch):
            disk_volume = (np.pi / 4) * a * b * disk_height
            volume += disk_volume
        
        volume_ml = volume / 1000
        
        # Calculate total area
        area_2ch = mask_2ch.sum() * self.pixel_spacing[0] * self.pixel_spacing[1]
        area_4ch = mask_4ch.sum() * self.pixel_spacing[0] * self.pixel_spacing[1]
        avg_area = (area_2ch + area_4ch) / 2
        
        return VolumeResult(
            volume_ml=volume_ml,
            area_mm2=avg_area,
            length_mm=length_mm,
            method='biplane_simpson'
        )
    
    def _get_length(self, mask: np.ndarray) -> float:
        """Get LV length in mm."""
        rows = mask.any(axis=1)
        if not rows.any():
            return 0.0
        row_indices = np.where(rows)[0]
        length_px = row_indices.max() - row_indices.min() + 1
        return length_px * self.pixel_spacing[0]
    
    def _get_diameters(
        self,
        mask: np.ndarray,
        n_disks: int
    ) -> List[float]:
        """Get diameters at n_disk positions."""
        rows = mask.any(axis=1)
        if not rows.any():
            return [0.0] * n_disks
        
        row_indices = np.where(rows)[0]
        start_row = row_indices.min()
        end_row = row_indices.max()
        
        disk_height = (end_row - start_row) / n_disks
        diameters = []
        
        for i in range(n_disks):
            row_idx = int(start_row + (i + 0.5) * disk_height)
            row_idx = min(row_idx, mask.shape[0] - 1)
            
            row_mask = mask[row_idx, :]
            if row_mask.any():
                cols = np.where(row_mask)[0]
                diameter_px = cols.max() - cols.min() + 1
                diameter_mm = diameter_px * self.pixel_spacing[1]
            else:
                diameter_mm = 0.0
            
            diameters.append(diameter_mm)
        
        return diameters


class EjectionFractionCalculator:
    """
    Calculate ejection fraction from cardiac segmentation.
    
    EF = (EDV - ESV) / EDV * 100
    
    Args:
        pixel_spacing: Pixel spacing in mm
        method: Volume calculation method
    """
    
    # EF classification thresholds
    EF_THRESHOLDS = {
        'normal': 55,
        'mildly_reduced': 40,
        'moderately_reduced': 30,
        'severely_reduced': 0
    }
    
    def __init__(
        self,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
        method: str = 'simpson'
    ):
        self.volume_calculator = VolumeCalculator(pixel_spacing, method)
        self.biplane = BiplaneSimpson(pixel_spacing)
    
    def calculate(
        self,
        seg_ed: torch.Tensor,
        seg_es: torch.Tensor,
        lv_class: int = 1,
        view: str = '4CH'
    ) -> EFResult:
        """
        Calculate EF from ED and ES segmentations.
        
        Args:
            seg_ed: End-diastolic segmentation
            seg_es: End-systolic segmentation
            lv_class: LV endocardium class index
            view: Echocardiography view
            
        Returns:
            EFResult with EF and volumes
        """
        # Calculate volumes
        vol_ed = self.volume_calculator.calculate(seg_ed, lv_class)
        vol_es = self.volume_calculator.calculate(seg_es, lv_class)
        
        edv = vol_ed.volume_ml
        esv = vol_es.volume_ml
        sv = edv - esv
        
        # Calculate EF
        ef = (sv / edv * 100) if edv > 0 else 0.0
        
        # Classify
        category = self._classify_ef(ef)
        
        return EFResult(
            ef_percent=ef,
            edv_ml=edv,
            esv_ml=esv,
            sv_ml=sv,
            category=category,
            view=view
        )
    
    def calculate_biplane(
        self,
        seg_ed_2ch: torch.Tensor,
        seg_ed_4ch: torch.Tensor,
        seg_es_2ch: torch.Tensor,
        seg_es_4ch: torch.Tensor,
        lv_class: int = 1
    ) -> EFResult:
        """
        Calculate EF using biplane Simpson's method.
        
        Uses both 2CH and 4CH views for more accurate estimation.
        """
        vol_ed = self.biplane.calculate(seg_ed_2ch, seg_ed_4ch, lv_class)
        vol_es = self.biplane.calculate(seg_es_2ch, seg_es_4ch, lv_class)
        
        edv = vol_ed.volume_ml
        esv = vol_es.volume_ml
        sv = edv - esv
        
        ef = (sv / edv * 100) if edv > 0 else 0.0
        category = self._classify_ef(ef)
        
        return EFResult(
            ef_percent=ef,
            edv_ml=edv,
            esv_ml=esv,
            sv_ml=sv,
            category=category,
            view='biplane'
        )
    
    def _classify_ef(self, ef: float) -> str:
        """Classify EF into clinical categories."""
        if ef >= self.EF_THRESHOLDS['normal']:
            return 'Normal'
        elif ef >= self.EF_THRESHOLDS['mildly_reduced']:
            return 'Mildly Reduced'
        elif ef >= self.EF_THRESHOLDS['moderately_reduced']:
            return 'Moderately Reduced'
        else:
            return 'Severely Reduced'
    
    def get_reference_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get normal reference ranges."""
        return {
            'ef_normal': (55, 70),
            'edv_male': (62, 150),  # mL
            'edv_female': (46, 106),  # mL
            'esv_male': (21, 61),  # mL
            'esv_female': (14, 42),  # mL
        }


class BatchEFCalculator:
    """
    Calculate EF for a batch of patients.
    """
    
    def __init__(self, pixel_spacing: Tuple[float, float] = (1.0, 1.0)):
        self.ef_calculator = EjectionFractionCalculator(pixel_spacing)
    
    def calculate_batch(
        self,
        seg_ed_batch: torch.Tensor,
        seg_es_batch: torch.Tensor,
        lv_class: int = 1
    ) -> List[EFResult]:
        """Calculate EF for each sample in batch."""
        results = []
        
        batch_size = seg_ed_batch.shape[0]
        
        for i in range(batch_size):
            result = self.ef_calculator.calculate(
                seg_ed_batch[i],
                seg_es_batch[i],
                lv_class
            )
            results.append(result)
        
        return results
    
    def compute_statistics(
        self,
        results: List[EFResult]
    ) -> Dict[str, float]:
        """Compute statistics over batch."""
        ef_values = [r.ef_percent for r in results]
        edv_values = [r.edv_ml for r in results]
        esv_values = [r.esv_ml for r in results]
        
        return {
            'ef_mean': np.mean(ef_values),
            'ef_std': np.std(ef_values),
            'ef_min': np.min(ef_values),
            'ef_max': np.max(ef_values),
            'edv_mean': np.mean(edv_values),
            'esv_mean': np.mean(esv_values),
            'normal_count': sum(1 for r in results if r.category == 'Normal'),
            'reduced_count': sum(1 for r in results if r.category != 'Normal'),
        }


if __name__ == '__main__':
    # Test EF calculation
    seg_ed = torch.zeros(256, 256, dtype=torch.long)
    seg_es = torch.zeros(256, 256, dtype=torch.long)
    
    # Create synthetic LV
    seg_ed[50:200, 80:180] = 1  # Larger at ED
    seg_es[80:180, 100:160] = 1  # Smaller at ES
    
    calculator = EjectionFractionCalculator()
    result = calculator.calculate(seg_ed, seg_es)
    
    print(f"EF: {result.ef_percent:.1f}%")
    print(f"EDV: {result.edv_ml:.1f} mL")
    print(f"ESV: {result.esv_ml:.1f} mL")
    print(f"Category: {result.category}")
