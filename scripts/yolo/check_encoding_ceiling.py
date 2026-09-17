#!/usr/bin/env python3
"""
Measure the POLYGON ENCODING CEILING on the CAMUS test split.

A YOLO segmentation model can never beat the accuracy of its own label
representation. This script takes the ground-truth label maps, converts them to
polygons exactly as `prepare_yolo_dataset.py` does, rasterises them back, and
scores the result against the untouched native ground truth. The output is an
upper bound on Dice/HD95/ASSD for every (encoding, vertex-budget) pair.

Usage:
    python scripts/yolo/check_encoding_ceiling.py
    python scripts/yolo/check_encoding_ceiling.py --n-points 32 64 96 128 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "yolo"))

from prepare_yolo_dataset import mask_to_polygons, read_split  # noqa: E402
from scoring import CamusScorer  # noqa: E402
from yolo_common import binaries_to_labelmap, rasterize  # noqa: E402


def polys_to_native(polys, h, w, encoding):
    per_class = {}
    for cls, pts in polys:
        m = rasterize(np.asarray(pts, dtype=np.float64), h, w)
        per_class[cls] = m if cls not in per_class else (per_class[cls] | m)
    return binaries_to_labelmap(
        per_class.get(0), per_class.get(1), per_class.get(2), encoding
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/CAMUS")
    ap.add_argument("--splits-dir", default="./data/splits")
    ap.add_argument("--split", default="test")
    ap.add_argument("--encodings", nargs="+", default=["filled", "seam"])
    ap.add_argument("--n-points", nargs="+", type=int, default=[32, 64, 96, 128, 0])
    ap.add_argument("--out", default="./results/yolo/encoding_ceiling.json")
    args = ap.parse_args()

    import nibabel as nib

    pids = read_split(Path(args.splits_dir) / f"{args.split}.txt")
    samples = []
    for pid in pids:
        pdir = Path(args.data_dir) / pid
        for view in ("2CH", "4CH"):
            for phase in ("ED", "ES"):
                gp = pdir / f"{pid}_{view}_{phase}_gt.nii"
                if gp.exists():
                    samples.append((pid, view, phase, gp))

    results = {}
    for enc in args.encodings:
        for npts in args.n_points:
            key = f"{enc}_n{npts}"
            scorer = CamusScorer()
            target_dice: list[float] = []
            for pid, view, phase, gp in tqdm(samples, desc=f"{key:18s}", ncols=80, leave=False):
                nii = nib.load(str(gp))
                gt = np.asarray(nii.dataobj).astype(np.int64)
                z = nii.header.get_zooms()
                sp = ((float(z[0]), float(z[1]))
                      if len(z) >= 2 and z[0] > 0 and z[1] > 0 else (1.0, 1.0))
                h, w = gt.shape[:2]
                polys = mask_to_polygons(gt, enc, npts)
                rec = polys_to_native(polys, h, w, enc)
                scorer.add(rec, gt, sp, meta={"patient_id": pid, "view": view, "phase": phase})

                # Fidelity of the RAW class-1 target the mask head must learn,
                # before the label map is composed. This is where `filled`
                # (a disc) and `seam` (a true annulus) actually differ: the
                # composed map hides it, because the endocardium polygon
                # re-carves the hole either way.
                tgt = (gt == 1) | (gt == 2) if enc == "filled" else (gt == 2)
                raw = np.zeros((h, w), dtype=bool)
                for cls, pts in polys:
                    if cls == 1:
                        raw |= rasterize(np.asarray(pts, dtype=np.float64), h, w)
                inter = np.logical_and(raw, tgt).sum()
                denom = raw.sum() + tgt.sum()
                target_dice.append((2.0 * inter / denom) if denom else 1.0)

            r = scorer.summary()
            r["target_dice_class1"] = float(np.mean(target_dice))
            results[key] = r
            print(
                f"{key:18s} Dice={r.get('dice_mean', float('nan')):.4f}  "
                f"HD95={r.get('hd95_mean', float('nan')):.3f} mm  "
                f"ASSD={r.get('assd_mean', float('nan')):.3f} mm  "
                f"cls1-target Dice={r['target_dice_class1']:.4f}",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
