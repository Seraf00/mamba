"""
Baseline-compatible scoring for CAMUS predictions.

Reproduces the exact protocol of `scripts/evaluate_all_models.py`, which is what
produced the published U-Net / Transformer / Mamba numbers:

  * Dice and IoU are computed on the 256x256 resized grid, with the project's
    smoothing constant (2*inter + eps) / (union + eps).
  * HD95 and ASSD are computed at NATIVE resolution using each patient's true
    NIfTI pixel spacing, in millimetres.

Any deviation would make the YOLO rows incomparable with the existing table, so
both conventions are kept here rather than "improved". Native-resolution Dice is
additionally reported under `*_native` keys as a robustness check.
"""
from __future__ import annotations

import numpy as np

from boundary_metrics import assd, assd_surface, hd95

CLASS_KEY = {1: "lv_endocardium", 2: "lv_epicardium", 3: "left_atrium"}
EPS = 1e-6


def resize_label(lbl: np.ndarray, size: int = 256) -> np.ndarray:
    import cv2

    return cv2.resize(lbl.astype(np.uint8), (size, size),
                      interpolation=cv2.INTER_NEAREST).astype(np.int64)


def dice_iou_smooth(pred: np.ndarray, target: np.ndarray, c: int) -> tuple[float, float]:
    """Project convention: smoothed Dice/IoU on a single sample and class."""
    p = (pred == c).astype(np.float64)
    t = (target == c).astype(np.float64)
    inter = float((p * t).sum())
    dice = (2.0 * inter + EPS) / (float(p.sum() + t.sum()) + EPS)
    union = float(p.sum() + t.sum() - inter)
    iou = (inter + EPS) / (union + EPS)
    return dice, iou


class CamusScorer:
    """Accumulates per-sample metrics and emits a baseline-compatible summary."""

    def __init__(self, grid: int = 256):
        self.grid = grid
        self.records: list[dict] = []

    def add(self, pred_native: np.ndarray, gt_native: np.ndarray,
            spacing: tuple[float, float], meta: dict | None = None) -> dict:
        meta = meta or {}
        rec: dict = dict(meta)

        p256 = resize_label(pred_native, self.grid)
        g256 = resize_label(gt_native, self.grid)

        for c in (1, 2, 3):
            k = CLASS_KEY[c]
            d, i = dice_iou_smooth(p256, g256, c)
            rec[f"dice_{k}"], rec[f"iou_{k}"] = d, i

            dn, _ = dice_iou_smooth(pred_native, gt_native, c)
            rec[f"dice_{k}_native"] = dn

            p, t = pred_native == c, gt_native == c
            if p.any() and t.any():
                rec[f"hd95_{k}"] = hd95(p, t, spacing)
                rec[f"assd_{k}"] = assd(p, t, spacing)
                rec[f"masd_{k}"] = assd_surface(p, t, spacing)
            else:
                rec[f"hd95_{k}"] = float("nan")
                rec[f"assd_{k}"] = float("nan")
                rec[f"masd_{k}"] = float("nan")
            rec[f"detected_{k}"] = bool(p.any())

        for pre in ("dice", "iou", "hd95", "assd", "masd"):
            vals = [rec[f"{pre}_{CLASS_KEY[c]}"] for c in (1, 2, 3)]
            vals = [v for v in vals if np.isfinite(v)]
            rec[f"{pre}_mean"] = float(np.mean(vals)) if vals else float("nan")

        self.records.append(rec)
        return rec

    # -- aggregation ------------------------------------------------------
    def _mean(self, key: str, rows: list[dict]) -> float:
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def summary(self) -> dict:
        rows = self.records
        out: dict = {}
        if not rows:
            return out

        keys = [f"{p}_{CLASS_KEY[c]}" for p in ("dice", "iou", "hd95", "assd", "masd")
                for c in (1, 2, 3)]
        keys += [f"dice_{CLASS_KEY[c]}_native" for c in (1, 2, 3)]
        keys += [f"{p}_mean" for p in ("dice", "iou", "hd95", "assd", "masd")]

        for k in keys:
            out[k] = self._mean(k, rows)

        for phase in ("ED", "ES"):
            sub = [r for r in rows if r.get("phase") == phase]
            if sub:
                for k in keys:
                    out[f"{k}_{phase.lower()}"] = self._mean(k, sub)

        for q in ("Good", "Medium", "Poor"):
            sub = [r for r in rows if r.get("quality") == q]
            if sub:
                out[f"dice_mean_quality_{q.lower()}"] = self._mean("dice_mean", sub)
                out[f"hd95_mean_quality_{q.lower()}"] = self._mean("hd95_mean", sub)
                out[f"n_quality_{q.lower()}"] = len(sub)

        dm = np.array([r["dice_mean"] for r in rows if np.isfinite(r["dice_mean"])])
        out["dice_std"] = float(dm.std())
        out["dice_ci_lower"] = float(np.percentile(dm, 2.5))
        out["dice_ci_upper"] = float(np.percentile(dm, 97.5))
        out["n_samples"] = len(rows)
        for c in (1, 2, 3):
            k = CLASS_KEY[c]
            out[f"n_missing_{k}"] = int(sum(1 for r in rows if not r.get(f"detected_{k}", True)))
        out["boundary_resolution"] = "native"
        return out
