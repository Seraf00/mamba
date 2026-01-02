"""
Training module for cardiac segmentation models.

Includes trainers, loss functions, schedulers, and callbacks.
"""

from .trainer import Trainer, TrainingConfig
from .losses import (
    DiceLoss,
    FocalLoss,
    CombinedLoss,
    BoundaryLoss,
    TverskyLoss
)
from .scheduler import (
    WarmupCosineScheduler,
    PolyScheduler,
    get_scheduler
)
from .callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoardLogger,
    CSVLogger
)

__all__ = [
    # Trainer
    'Trainer',
    'TrainingConfig',
    # Losses
    'DiceLoss',
    'FocalLoss',
    'CombinedLoss',
    'BoundaryLoss',
    'TverskyLoss',
    # Schedulers
    'WarmupCosineScheduler',
    'PolyScheduler',
    'get_scheduler',
    # Callbacks
    'EarlyStopping',
    'ModelCheckpoint',
    'TensorBoardLogger',
    'CSVLogger'
]
