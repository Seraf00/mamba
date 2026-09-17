"""
Boundary-weighted mask loss for Ultralytics segmentation.

Motivation
----------
Across the sweep, Dice saturates (0.9127-0.9176 over a 10x parameter range and
three YOLO generations, most pairwise tests non-significant) while HD95
separates every model. The polygon encoding ceiling is 0.308 mm and we sit at
3.74 mm, so there is ~12x of headroom in boundary localisation that is not
representational.

Nothing in the stock objective targets it. The mask loss is plain per-pixel
BCE, which is indifferent to *where* an error lies: a false positive one pixel
outside the contour and one twenty pixels out cost exactly the same. HD95, the
metric with the headroom, cares about nothing else.

This re-weights the existing BCE by proximity to the ground-truth contour:

    w = 1 + lambda * band(gt)

where `band` is a morphological gradient of the GT mask (dilate minus erode),
computed with max-pooling so it is GPU-native, allocation-light and needs no
scipy round-trip. Pixels on the contour cost (1 + lambda) times as much as
interior pixels; the interior term is retained so the model cannot trade away
region accuracy to chase the boundary.

This is deliberately the cheapest intervention that targets HD95 directly: no
architecture change, no extra parameters, no change to inference or latency.
`band_px` controls the width of the emphasised band at PROTOTYPE resolution
(1/4 of input by default), so 1 means a 3x3 neighbourhood of the contour.

Install before training:

    from boundary_loss import install_boundary_loss
    install_boundary_loss(weight=5.0, band_px=1)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_ORIGINAL = None
_WEIGHT = 0.0
_BAND = 1


def boundary_band(gt: torch.Tensor, band_px: int = 1) -> torch.Tensor:
    """
    Morphological gradient of a binary mask: 1 on a band straddling the
    contour, 0 elsewhere. Shape-preserving, no gradients required.

    Args:
        gt: (N, H, W) binary ground-truth masks.
        band_px: half-width of the band, in prototype pixels.
    """
    k = 2 * band_px + 1
    x = gt.unsqueeze(1).float()
    dil = F.max_pool2d(x, kernel_size=k, stride=1, padding=band_px)
    ero = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=band_px)
    return (dil - ero).squeeze(1)


def _boundary_single_mask_loss(gt_mask, pred, proto, xyxy, area):
    """Drop-in replacement for v8SegmentationLoss.single_mask_loss."""
    from ultralytics.utils.ops import crop_mask

    pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
    loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")

    if _WEIGHT > 0:
        with torch.no_grad():
            w = 1.0 + _WEIGHT * boundary_band(gt_mask, _BAND)
        loss = loss * w
        # Renormalise so the boundary term changes the loss SHAPE, not its
        # scale -- otherwise lambda silently rescales the mask loss relative to
        # the box/cls terms and the comparison against the baseline run is
        # confounded by an effective learning-rate change.
        loss = loss * (w.numel() / w.sum().clamp(min=1.0))

    return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()


def install_boundary_loss(weight: float = 5.0, band_px: int = 1) -> None:
    """Patch the Ultralytics segmentation loss in place. Idempotent."""
    global _ORIGINAL, _WEIGHT, _BAND
    from ultralytics.utils.loss import v8SegmentationLoss

    _WEIGHT, _BAND = float(weight), int(band_px)
    if _ORIGINAL is None:
        _ORIGINAL = v8SegmentationLoss.single_mask_loss
    v8SegmentationLoss.single_mask_loss = staticmethod(_boundary_single_mask_loss)
    print(f"[boundary-loss] installed: weight={_WEIGHT}, band_px={_BAND}")


def uninstall_boundary_loss() -> None:
    """Restore the stock loss."""
    global _ORIGINAL
    from ultralytics.utils.loss import v8SegmentationLoss

    if _ORIGINAL is not None:
        v8SegmentationLoss.single_mask_loss = _ORIGINAL
        _ORIGINAL = None
