"""
Memory-safe HD95 / ASSD, numerically identical to
`metrics.segmentation_metrics.HausdorffDistance` and `SurfaceDistance`.

The project implementations build the full |A| x |B| pairwise distance matrix
with `scipy.spatial.distance.cdist`. At CAMUS native resolution a contour can
carry >1000 boundary pixels, and on a machine near its Windows commit limit
that allocation fails. These versions stream the same computation in row
chunks: never more than (chunk x |B|) floats are live, and the returned
min-distance vectors -- hence HD95 and ASSD -- are bit-for-bit the same.

Boundary extraction matches the project definition exactly:
    boundary = mask & ~binary_erosion(mask)
with boundary coordinates scaled by the pixel spacing before distances.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion


def get_boundary(mask: np.ndarray) -> np.ndarray:
    """Boundary pixel coordinates as an (N, 2) array, project convention."""
    eroded = binary_erosion(mask)
    boundary = mask & ~eroded
    return np.array(np.where(boundary)).T


def _min_distances(a: np.ndarray, b: np.ndarray, chunk: int = 256):
    """
    Return (min over b for each a, min over a for each b) Euclidean distances,
    computed in row chunks so the full pairwise matrix is never allocated.
    """
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    d_ab = np.empty(len(a), dtype=np.float64)
    d_ba = np.full(len(b), np.inf, dtype=np.float64)

    b_sq = (b ** 2).sum(axis=1)
    for i in range(0, len(a), chunk):
        blk = a[i:i + chunk]
        # squared Euclidean via the expansion, then clipped for numerical safety
        d2 = (blk ** 2).sum(axis=1)[:, None] + b_sq[None, :] - 2.0 * (blk @ b.T)
        np.maximum(d2, 0.0, out=d2)
        d = np.sqrt(d2, out=d2)
        d_ab[i:i + chunk] = d.min(axis=1)
        np.minimum(d_ba, d.min(axis=0), out=d_ba)
    return d_ab, d_ba


def hd95(pred: np.ndarray, target: np.ndarray, spacing=(1.0, 1.0)) -> float:
    """95th-percentile Hausdorff distance in spacing units (mm)."""
    pb = get_boundary(pred)
    tb = get_boundary(target)
    if len(pb) == 0 or len(tb) == 0:
        return float("nan")
    s = np.asarray(spacing, dtype=np.float64)
    d_pt, d_tp = _min_distances(pb * s, tb * s)
    return float(max(np.percentile(d_pt, 95), np.percentile(d_tp, 95)))


def assd(pred: np.ndarray, target: np.ndarray, spacing=(1.0, 1.0)) -> float:
    """
    ASSD exactly as defined by `metrics.SurfaceDistance._compute_asd`, which is
    what produced the published baseline numbers.

    Note this is boundary-to-REGION, not boundary-to-boundary: the distance
    transform is taken of the opposite class's filled mask, so a boundary pixel
    lying inside the other structure contributes 0. Replicated verbatim so that
    YOLO rows and baseline rows in the comparison table are commensurable;
    `assd_surface` below is the textbook boundary-to-boundary definition.
    """
    from scipy.ndimage import distance_transform_edt

    if not pred.any() or not target.any():
        return float("nan")
    pred_dist = distance_transform_edt(~pred, sampling=spacing)
    target_dist = distance_transform_edt(~target, sampling=spacing)
    pb = pred & ~binary_erosion(pred)
    tb = target & ~binary_erosion(target)
    p2t = target_dist[pb]
    t2p = pred_dist[tb]
    if len(p2t) == 0 or len(t2p) == 0:
        return float("nan")
    return float((p2t.mean() + t2p.mean()) / 2.0)


def assd_surface(pred: np.ndarray, target: np.ndarray, spacing=(1.0, 1.0)) -> float:
    """Textbook average symmetric SURFACE distance (boundary to boundary), mm."""
    pb = get_boundary(pred)
    tb = get_boundary(target)
    if len(pb) == 0 or len(tb) == 0:
        return float("nan")
    s = np.asarray(spacing, dtype=np.float64)
    d_pt, d_tp = _min_distances(pb * s, tb * s)
    return float((d_pt.sum() + d_tp.sum()) / (len(d_pt) + len(d_tp)))


def dice_iou(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    inter = np.logical_and(pred, target).sum()
    ps, ts = pred.sum(), target.sum()
    dice = (2.0 * inter / (ps + ts)) if (ps + ts) > 0 else 1.0
    union = np.logical_or(pred, target).sum()
    iou = (inter / union) if union > 0 else 1.0
    return float(dice), float(iou)


def score_labelmap(pred: np.ndarray, target: np.ndarray, spacing=(1.0, 1.0),
                   classes=(1, 2, 3)) -> dict:
    """Per-class Dice/IoU/HD95/ASSD for two CAMUS 4-label maps."""
    out: dict[str, float] = {}
    for c in classes:
        p, t = pred == c, target == c
        d, i = dice_iou(p, t)
        out[f"dice_{c}"], out[f"iou_{c}"] = d, i
        out[f"hd95_{c}"] = hd95(p, t, spacing) if (p.any() and t.any()) else float("nan")
        out[f"assd_{c}"] = assd(p, t, spacing) if (p.any() and t.any()) else float("nan")
        out[f"masd_{c}"] = assd_surface(p, t, spacing) if (p.any() and t.any()) else float("nan")
    return out
