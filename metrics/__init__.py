"""
Metrics module for cardiac segmentation evaluation.

Includes segmentation metrics, clinical metrics (EF), and efficiency benchmarks.
"""

from .segmentation_metrics import (
    DiceScore,
    IoUScore,
    HausdorffDistance,
    SurfaceDistance,
    SegmentationMetrics
)
from .ejection_fraction import (
    EjectionFractionCalculator,
    VolumeCalculator,
    BiplaneSimpson
)
from .camus_ef_official import (
    CAMUSEFCalculator,
    CAMUSEFResult,
    compute_ejection_fraction,
    compute_left_ventricle_volumes
)
from .efficiency_metrics import (
    EfficiencyBenchmark,
    PortabilityAnalysis,
    count_parameters,
    measure_inference_time,
    measure_memory_usage
)

__all__ = [
    # Segmentation
    'DiceScore',
    'IoUScore',
    'HausdorffDistance',
    'SurfaceDistance',
    'SegmentationMetrics',
    # Clinical
    'EjectionFractionCalculator',
    'VolumeCalculator',
    'BiplaneSimpson',
    # CAMUS Official EF (Simpson's biplane)
    'CAMUSEFCalculator',
    'CAMUSEFResult',
    'compute_ejection_fraction',
    'compute_left_ventricle_volumes',
    # Efficiency
    'EfficiencyBenchmark',
    'PortabilityAnalysis',
    'count_parameters',
    'measure_inference_time',
    'measure_memory_usage'
]
