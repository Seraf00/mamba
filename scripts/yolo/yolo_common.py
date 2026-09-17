"""
Shared helpers for the YOLO-on-CAMUS experiments.

The central piece is `polys_to_labelmap`, which turns a set of YOLO polygons
(or predicted binary masks) back into a CAMUS 4-label map at native image
resolution, so that predictions can be scored with exactly the same
`metrics.SegmentationMetrics` used for the U-Net/Transformer/Mamba baselines.

CAMUS label convention (the target space):
    0 background, 1 LV endocardium, 2 LV myocardium, 3 left atrium
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CAMUS_CLASS_NAMES = {1: "LV_endocardium", 2: "LV_epicardium", 3: "Left_atrium"}


def read_yolo_label(path: Path) -> list[tuple[int, np.ndarray]]:
    """Parse a YOLO segmentation label file into [(cls, normalised_xy), ...]."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = int(parts[0])
        pts = np.asarray([float(v) for v in parts[1:]], dtype=np.float64).reshape(-1, 2)
        out.append((cls, pts))
    return out


def rasterize(pts_xy: np.ndarray, h: int, w: int) -> np.ndarray:
    """Fill a polygon (absolute pixel xy) into a binary mask of shape (h, w)."""
    m = np.zeros((h, w), dtype=np.uint8)
    poly = np.round(pts_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [poly], 1)
    return m.astype(bool)


def binaries_to_labelmap(
    endo: np.ndarray | None,
    mid: np.ndarray | None,
    la: np.ndarray | None,
    encoding: str,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Compose three predicted binary masks into a CAMUS 4-label map.

    `mid` is the model's second class: the FILLED epicardial region when
    encoding == "filled", or the myocardial ring otherwise. Painting order
    guarantees the myocardium is exactly (epicardium minus endocardium).

    A detector may emit nothing at all for a frame, which is a legitimate
    prediction (and one a semantic segmenter cannot make). With `shape` given
    that yields an all-background map, so the frame is scored as a total miss
    rather than crashing the run.
    """
    ref = next((x for x in (endo, mid, la) if x is not None), None)
    if ref is None:
        if shape is None:
            raise ValueError("no masks given and no shape to fall back on")
        return np.zeros(shape, dtype=np.int64)
    out = np.zeros(ref.shape, dtype=np.int64)

    if la is not None:
        out[la] = 3
    if mid is not None:
        out[mid] = 2                      # epicardium (filled) or myocardium ring
    if endo is not None:
        if encoding == "filled":
            out[endo] = 1                 # carve the blood pool out of the filled epi
        else:
            out[endo & (out != 2)] = 1
            out[endo & (out == 2)] = 1    # endo wins over an overlapping ring
    return out


def polys_to_labelmap(
    polys: list[tuple[int, np.ndarray]],
    h: int,
    w: int,
    encoding: str,
    normalised: bool = True,
) -> np.ndarray:
    """Rasterize YOLO polygons into a CAMUS 4-label map of shape (h, w)."""
    per_class: dict[int, np.ndarray] = {}
    for cls, pts in polys:
        p = pts.astype(np.float64).copy()
        if normalised:
            p[:, 0] *= w
            p[:, 1] *= h
        m = rasterize(p, h, w)
        per_class[cls] = m if cls not in per_class else (per_class[cls] | m)
    return binaries_to_labelmap(
        per_class.get(0), per_class.get(1), per_class.get(2), encoding
    )


def load_meta(dataset_dir: Path) -> dict:
    return json.loads((Path(dataset_dir) / "meta.json").read_text())


def native_gt(camus_root: Path, patient_id: str, view: str, phase: str) -> np.ndarray:
    """Load the untouched native-resolution CAMUS ground truth label map."""
    import nibabel as nib

    p = Path(camus_root) / patient_id / f"{patient_id}_{view}_{phase}_gt.nii"
    return np.asarray(nib.load(str(p)).dataobj).astype(np.int64)
