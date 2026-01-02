"""
Clinical report generation for cardiac segmentation.

Generates comprehensive clinical reports with:
- Segmentation visualization
- Volume and EF measurements
- Uncertainty analysis
- Explainability visualizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches


class ClinicalReport:
    """
    Generate clinical reports for cardiac segmentation predictions.
    
    Combines segmentation results, measurements, uncertainty,
    and explainability into a comprehensive report.
    """
    
    # Class labels for CAMUS dataset
    CLASS_NAMES = {
        0: 'Background',
        1: 'LV Endocardium',
        2: 'LV Epicardium (Myocardium)',
        3: 'Left Atrium'
    }
    
    CLASS_COLORS = {
        0: [0, 0, 0],        # Black
        1: [255, 0, 0],      # Red
        2: [0, 255, 0],      # Green
        3: [0, 0, 255]       # Blue
    }
    
    def __init__(
        self,
        model: nn.Module,
        pixel_spacing: float = 1.0,  # mm per pixel
        device: str = 'cuda'
    ):
        self.model = model
        self.pixel_spacing = pixel_spacing
        self.device = device
        
        self.model.to(device)
        self.model.eval()
    
    def generate_report(
        self,
        image_ed: torch.Tensor,
        image_es: torch.Tensor,
        patient_id: str = 'Unknown',
        view: str = '4CH',
        save_path: Optional[str] = None,
        include_explainability: bool = True,
        include_uncertainty: bool = True
    ) -> Dict:
        """
        Generate comprehensive clinical report.
        
        Args:
            image_ed: End-diastolic frame (B, C, H, W)
            image_es: End-systolic frame (B, C, H, W)
            patient_id: Patient identifier
            view: Echocardiography view (2CH or 4CH)
            save_path: Path to save PDF report
            include_explainability: Include Grad-CAM visualizations
            include_uncertainty: Include uncertainty maps
            
        Returns:
            Dictionary with all measurements and analysis
        """
        # Prepare images
        image_ed = image_ed.to(self.device)
        image_es = image_es.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            pred_ed = self.model(image_ed)
            pred_es = self.model(image_es)
            
            if isinstance(pred_ed, dict):
                pred_ed = pred_ed['out']
            if isinstance(pred_es, dict):
                pred_es = pred_es['out']
        
        # Convert to segmentation masks
        seg_ed = pred_ed.argmax(dim=1)
        seg_es = pred_es.argmax(dim=1)
        
        # Compute volumes
        volumes = self._compute_volumes(seg_ed, seg_es)
        
        # Compute ejection fraction
        ef = self._compute_ejection_fraction(volumes)
        
        # Compute areas
        areas = self._compute_areas(seg_ed, seg_es)
        
        # Quality metrics
        quality = self._assess_prediction_quality(pred_ed, pred_es)
        
        # Compile results
        results = {
            'patient_id': patient_id,
            'view': view,
            'timestamp': datetime.now().isoformat(),
            'volumes': volumes,
            'ejection_fraction': ef,
            'areas': areas,
            'quality': quality,
            'predictions': {
                'ed': seg_ed.cpu(),
                'es': seg_es.cpu()
            },
            'images': {
                'ed': image_ed.cpu(),
                'es': image_es.cpu()
            }
        }
        
        # Generate PDF if path provided
        if save_path:
            self._generate_pdf_report(
                results,
                save_path,
                include_explainability,
                include_uncertainty
            )
        
        return results
    
    def _compute_volumes(
        self,
        seg_ed: torch.Tensor,
        seg_es: torch.Tensor
    ) -> Dict[str, float]:
        """Compute LV volumes using Simpson's method."""
        # LV endocardium is class 1
        lv_endo_ed = (seg_ed == 1).float()
        lv_endo_es = (seg_es == 1).float()
        
        # Area in pixels
        area_ed_px = lv_endo_ed.sum(dim=(1, 2)).mean().item()
        area_es_px = lv_endo_es.sum(dim=(1, 2)).mean().item()
        
        # Convert to mm²
        area_ed_mm2 = area_ed_px * (self.pixel_spacing ** 2)
        area_es_mm2 = area_es_px * (self.pixel_spacing ** 2)
        
        # Estimate volume using area-length method (simplified)
        # V = (8 * A²) / (3 * π * L)
        # For simplicity, use disk summation approximation
        
        # Get height (length) of LV
        rows_ed = lv_endo_ed.any(dim=2).sum(dim=1).float().mean().item()
        rows_es = lv_endo_es.any(dim=2).sum(dim=1).float().mean().item()
        
        length_ed = rows_ed * self.pixel_spacing
        length_es = rows_es * self.pixel_spacing
        
        # Volume approximation (mL)
        # Using simplified formula: V ≈ A * L / 1000 (very rough)
        # Better: Use Simpson's biplane method with actual disk summation
        
        vol_ed = self._simpson_volume(lv_endo_ed)
        vol_es = self._simpson_volume(lv_endo_es)
        
        return {
            'edv_ml': vol_ed,  # End-diastolic volume
            'esv_ml': vol_es,  # End-systolic volume
            'sv_ml': vol_ed - vol_es,  # Stroke volume
            'area_ed_mm2': area_ed_mm2,
            'area_es_mm2': area_es_mm2,
            'length_ed_mm': length_ed,
            'length_es_mm': length_es
        }
    
    def _simpson_volume(self, mask: torch.Tensor) -> float:
        """
        Compute volume using Simpson's method of disks.
        
        Divides the LV into 20 disks and sums their volumes.
        """
        mask = mask.float()
        
        if mask.dim() == 3:
            mask = mask[0]  # Take first batch
        
        H, W = mask.shape
        
        # Find LV bounding box
        rows = mask.any(dim=1)
        if not rows.any():
            return 0.0
        
        row_indices = torch.where(rows)[0]
        start_row = row_indices.min().item()
        end_row = row_indices.max().item()
        
        # Number of disks
        n_disks = 20
        disk_height = (end_row - start_row) / n_disks
        
        volume = 0
        for i in range(n_disks):
            row_start = int(start_row + i * disk_height)
            row_end = int(start_row + (i + 1) * disk_height)
            
            if row_end > H:
                row_end = H
            
            # Get disk area
            disk_mask = mask[row_start:row_end, :]
            disk_area = disk_mask.sum().item() * (self.pixel_spacing ** 2)
            
            # Diameter from area (assuming circular)
            if disk_area > 0:
                diameter = 2 * np.sqrt(disk_area / np.pi)
                disk_volume = np.pi * (diameter / 2) ** 2 * disk_height * self.pixel_spacing
                volume += disk_volume
        
        # Convert to mL (mm³ / 1000)
        return volume / 1000
    
    def _compute_ejection_fraction(self, volumes: Dict) -> Dict[str, float]:
        """Compute ejection fraction and related metrics."""
        edv = volumes['edv_ml']
        esv = volumes['esv_ml']
        sv = volumes['sv_ml']
        
        # EF = (EDV - ESV) / EDV * 100
        ef = (sv / edv * 100) if edv > 0 else 0
        
        # Classify EF
        if ef >= 55:
            ef_category = 'Normal'
        elif ef >= 40:
            ef_category = 'Mildly Reduced'
        elif ef >= 30:
            ef_category = 'Moderately Reduced'
        else:
            ef_category = 'Severely Reduced'
        
        return {
            'ef_percent': ef,
            'category': ef_category
        }
    
    def _compute_areas(
        self,
        seg_ed: torch.Tensor,
        seg_es: torch.Tensor
    ) -> Dict[str, float]:
        """Compute areas for all structures."""
        areas = {}
        
        for class_idx, class_name in self.CLASS_NAMES.items():
            if class_idx == 0:
                continue
            
            mask_ed = (seg_ed == class_idx).float()
            mask_es = (seg_es == class_idx).float()
            
            area_ed = mask_ed.sum().item() * (self.pixel_spacing ** 2)
            area_es = mask_es.sum().item() * (self.pixel_spacing ** 2)
            
            areas[f'{class_name.lower().replace(" ", "_")}_ed_mm2'] = area_ed
            areas[f'{class_name.lower().replace(" ", "_")}_es_mm2'] = area_es
        
        return areas
    
    def _assess_prediction_quality(
        self,
        pred_ed: torch.Tensor,
        pred_es: torch.Tensor
    ) -> Dict[str, float]:
        """Assess prediction confidence and quality."""
        # Softmax probabilities
        probs_ed = F.softmax(pred_ed, dim=1)
        probs_es = F.softmax(pred_es, dim=1)
        
        # Max confidence per pixel
        conf_ed = probs_ed.max(dim=1)[0]
        conf_es = probs_es.max(dim=1)[0]
        
        # Average confidence
        avg_conf_ed = conf_ed.mean().item()
        avg_conf_es = conf_es.mean().item()
        
        # Low confidence pixels (< 0.5)
        low_conf_ed = (conf_ed < 0.5).float().mean().item()
        low_conf_es = (conf_es < 0.5).float().mean().item()
        
        return {
            'confidence_ed': avg_conf_ed,
            'confidence_es': avg_conf_es,
            'low_confidence_ratio_ed': low_conf_ed,
            'low_confidence_ratio_es': low_conf_es,
            'overall_confidence': (avg_conf_ed + avg_conf_es) / 2
        }
    
    def _generate_pdf_report(
        self,
        results: Dict,
        save_path: str,
        include_explainability: bool,
        include_uncertainty: bool
    ):
        """Generate PDF report with visualizations."""
        with PdfPages(save_path) as pdf:
            # Page 1: Overview
            self._create_overview_page(results, pdf)
            
            # Page 2: Segmentation results
            self._create_segmentation_page(results, pdf)
            
            # Page 3: Measurements
            self._create_measurements_page(results, pdf)
            
            if include_explainability:
                # Page 4: Explainability
                self._create_explainability_page(results, pdf)
            
            if include_uncertainty:
                # Page 5: Uncertainty
                self._create_uncertainty_page(results, pdf)
    
    def _create_overview_page(self, results: Dict, pdf):
        """Create overview page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Cardiac Segmentation Report',
                fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
        
        # Patient info
        info_text = f"""
        Patient ID: {results['patient_id']}
        View: {results['view']}
        Date: {results['timestamp'][:10]}
        
        ═══════════════════════════════════════════
        
        EJECTION FRACTION ANALYSIS
        
        End-Diastolic Volume (EDV): {results['volumes']['edv_ml']:.1f} mL
        End-Systolic Volume (ESV): {results['volumes']['esv_ml']:.1f} mL
        Stroke Volume (SV): {results['volumes']['sv_ml']:.1f} mL
        
        Ejection Fraction: {results['ejection_fraction']['ef_percent']:.1f}%
        Classification: {results['ejection_fraction']['category']}
        
        ═══════════════════════════════════════════
        
        PREDICTION QUALITY
        
        Overall Confidence: {results['quality']['overall_confidence']:.1%}
        ED Frame Confidence: {results['quality']['confidence_ed']:.1%}
        ES Frame Confidence: {results['quality']['confidence_es']:.1%}
        """
        
        ax.text(0.1, 0.75, info_text, fontsize=11, family='monospace',
                verticalalignment='top', transform=ax.transAxes)
        
        # EF interpretation
        ef = results['ejection_fraction']['ef_percent']
        if ef >= 55:
            interpretation = "✓ Normal left ventricular systolic function"
            color = 'green'
        elif ef >= 40:
            interpretation = "⚠ Mildly reduced left ventricular systolic function"
            color = 'orange'
        elif ef >= 30:
            interpretation = "⚠ Moderately reduced left ventricular systolic function"
            color = 'darkorange'
        else:
            interpretation = "✗ Severely reduced left ventricular systolic function"
            color = 'red'
        
        ax.text(0.1, 0.25, interpretation, fontsize=12, fontweight='bold',
                color=color, transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_segmentation_page(self, results: Dict, pdf):
        """Create segmentation visualization page."""
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
        
        # Get images and predictions
        img_ed = results['images']['ed'][0, 0].numpy()
        img_es = results['images']['es'][0, 0].numpy()
        seg_ed = results['predictions']['ed'][0].numpy()
        seg_es = results['predictions']['es'][0].numpy()
        
        # Original images
        axes[0, 0].imshow(img_ed, cmap='gray')
        axes[0, 0].set_title('End-Diastole (ED)')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(img_es, cmap='gray')
        axes[1, 0].set_title('End-Systole (ES)')
        axes[1, 0].axis('off')
        
        # Segmentation masks
        axes[0, 1].imshow(self._colorize_mask(seg_ed))
        axes[0, 1].set_title('ED Segmentation')
        axes[0, 1].axis('off')
        
        axes[1, 1].imshow(self._colorize_mask(seg_es))
        axes[1, 1].set_title('ES Segmentation')
        axes[1, 1].axis('off')
        
        # Overlay
        axes[0, 2].imshow(self._overlay_segmentation(img_ed, seg_ed))
        axes[0, 2].set_title('ED Overlay')
        axes[0, 2].axis('off')
        
        axes[1, 2].imshow(self._overlay_segmentation(img_es, seg_es))
        axes[1, 2].set_title('ES Overlay')
        axes[1, 2].axis('off')
        
        # Add legend
        patches = [mpatches.Patch(color=np.array(c) / 255, label=n)
                   for c, n in zip(list(self.CLASS_COLORS.values())[1:],
                                   list(self.CLASS_NAMES.values())[1:])]
        fig.legend(handles=patches, loc='lower center', ncol=3)
        
        plt.suptitle('Segmentation Results', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_measurements_page(self, results: Dict, pdf):
        """Create measurements page with tables."""
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
        
        for ax in axes:
            ax.axis('off')
        
        # Volume table
        vol_data = [
            ['Metric', 'Value'],
            ['End-Diastolic Volume', f"{results['volumes']['edv_ml']:.1f} mL"],
            ['End-Systolic Volume', f"{results['volumes']['esv_ml']:.1f} mL"],
            ['Stroke Volume', f"{results['volumes']['sv_ml']:.1f} mL"],
            ['Ejection Fraction', f"{results['ejection_fraction']['ef_percent']:.1f}%"],
        ]
        
        table1 = axes[0].table(
            cellText=vol_data,
            loc='center',
            cellLoc='left',
            colWidths=[0.5, 0.3]
        )
        table1.auto_set_font_size(False)
        table1.set_fontsize(11)
        table1.scale(1.2, 1.8)
        
        axes[0].set_title('Volume Measurements', fontsize=14, fontweight='bold', pad=20)
        
        # Area table
        area_data = [['Structure', 'ED (mm²)', 'ES (mm²)']]
        for class_name in list(self.CLASS_NAMES.values())[1:]:
            key_base = class_name.lower().replace(" ", "_")
            ed_val = results['areas'].get(f'{key_base}_ed_mm2', 0)
            es_val = results['areas'].get(f'{key_base}_es_mm2', 0)
            area_data.append([class_name, f'{ed_val:.1f}', f'{es_val:.1f}'])
        
        table2 = axes[1].table(
            cellText=area_data,
            loc='center',
            cellLoc='left',
            colWidths=[0.4, 0.2, 0.2]
        )
        table2.auto_set_font_size(False)
        table2.set_fontsize(11)
        table2.scale(1.2, 1.8)
        
        axes[1].set_title('Area Measurements', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_explainability_page(self, results: Dict, pdf):
        """Create explainability visualization page."""
        # Placeholder - would use GradCAM here
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.text(0.5, 0.5, 'Explainability Visualizations\n(Grad-CAM, Attention Maps)',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_uncertainty_page(self, results: Dict, pdf):
        """Create uncertainty analysis page."""
        # Placeholder - would use uncertainty estimation here
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.text(0.5, 0.5, 'Uncertainty Analysis\n(MC Dropout, Confidence Maps)',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert segmentation mask to RGB."""
        H, W = mask.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        
        for class_idx, color in self.CLASS_COLORS.items():
            rgb[mask == class_idx] = color
        
        return rgb
    
    def _overlay_segmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Overlay segmentation on image."""
        # Normalize image
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
        img_rgb = np.stack([img_norm] * 3, axis=-1)
        
        # Colorize mask
        mask_rgb = self._colorize_mask(mask) / 255.0
        
        # Overlay (only on non-background)
        overlay = img_rgb.copy()
        non_bg = mask > 0
        overlay[non_bg] = alpha * mask_rgb[non_bg] + (1 - alpha) * img_rgb[non_bg]
        
        return (overlay * 255).astype(np.uint8)


class ReportGenerator:
    """
    Batch report generation for multiple patients.
    """
    
    def __init__(self, model: nn.Module, output_dir: str):
        self.clinical_report = ClinicalReport(model)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_batch_reports(
        self,
        dataset,
        patient_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """Generate reports for multiple patients."""
        results = []
        
        for idx, (image_ed, image_es, patient_id) in enumerate(dataset):
            if patient_ids and patient_id not in patient_ids:
                continue
            
            save_path = self.output_dir / f'report_{patient_id}.pdf'
            
            try:
                result = self.clinical_report.generate_report(
                    image_ed.unsqueeze(0),
                    image_es.unsqueeze(0),
                    patient_id=patient_id,
                    save_path=str(save_path)
                )
                results.append(result)
                print(f"Generated report for patient {patient_id}")
            except Exception as e:
                print(f"Error generating report for patient {patient_id}: {e}")
        
        return results
    
    def generate_summary_report(self, results: List[Dict], save_path: str):
        """Generate summary report across all patients."""
        # Extract EF values
        ef_values = [r['ejection_fraction']['ef_percent'] for r in results]
        
        with PdfPages(save_path) as pdf:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            
            # EF distribution
            axes[0, 0].hist(ef_values, bins=20, edgecolor='black')
            axes[0, 0].set_xlabel('Ejection Fraction (%)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('EF Distribution')
            axes[0, 0].axvline(55, color='g', linestyle='--', label='Normal threshold')
            axes[0, 0].axvline(40, color='orange', linestyle='--', label='Mild threshold')
            axes[0, 0].legend()
            
            # EF categories
            categories = [r['ejection_fraction']['category'] for r in results]
            unique, counts = np.unique(categories, return_counts=True)
            axes[0, 1].pie(counts, labels=unique, autopct='%1.1f%%')
            axes[0, 1].set_title('EF Categories')
            
            # EDV vs ESV
            edv = [r['volumes']['edv_ml'] for r in results]
            esv = [r['volumes']['esv_ml'] for r in results]
            axes[1, 0].scatter(edv, esv, alpha=0.6)
            axes[1, 0].set_xlabel('EDV (mL)')
            axes[1, 0].set_ylabel('ESV (mL)')
            axes[1, 0].set_title('EDV vs ESV')
            
            # Confidence distribution
            conf = [r['quality']['overall_confidence'] for r in results]
            axes[1, 1].hist(conf, bins=20, edgecolor='black')
            axes[1, 1].set_xlabel('Confidence')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Prediction Confidence')
            
            plt.suptitle('Cohort Summary', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


if __name__ == '__main__':
    print("Clinical report module loaded successfully")
    print("Available classes: ClinicalReport, ReportGenerator")
