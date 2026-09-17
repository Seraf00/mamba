#!/usr/bin/env python3
"""
Ejection fraction from YOLO predictions, via the official CAMUS method.

This is where the cardiac cycle phase actually earns its keep. EF is defined
across the pair of phases,
    EF = (EDV - ESV) / EDV,
so it can only be computed by pairing a patient's end-diastolic and
end-systolic predictions -- in both apical views. Segmentation metrics are
per-frame and blind to that structure; EF is the clinical endpoint the
segmentation exists to serve, and it punishes exactly the errors that Dice
forgives (a systematic volume bias barely moves Dice but shifts EF directly).

Volumes use Simpson's biplane method of disks from
`metrics.camus_ef_official`, the same code path used for the Paper 1 EF table,
at native resolution with each view's true NIfTI spacing.

Run this AFTER the sweep (it needs the GPU only briefly):

    python scripts/yolo/eval_ef.py --weights yolo_runs/E9_seed0/weights/best.pt \
        --name E9_seed0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "yolo"))

from metrics import CAMUSEFCalculator  # noqa: E402
from yolo_common import binaries_to_labelmap, load_meta, native_gt  # noqa: E402


def predict_labelmap(model, img_path: Path, shape, encoding: str, pred_kw) -> np.ndarray:
    h, w = shape
    res = model.predict(source=str(img_path), **pred_kw)[0]
    best: dict[int, np.ndarray] = {}
    if res.masks is not None and len(res.masks) > 0:
        md = res.masks.data.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        for c in (0, 1, 2):
            idx = np.where(cls == c)[0]
            if len(idx):
                m = md[idx[int(np.argmax(cf[idx]))]] > 0.5
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                best[c] = m
    return binaries_to_labelmap(best.get(0), best.get(1), best.get(2),
                                encoding, shape=(h, w))


def largest_cc(m: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected component of a binary LV mask."""
    from scipy import ndimage

    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return (lab == (int(np.argmax(sizes)) + 1)).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", default="./yolo_data/camus_edes")
    ap.add_argument("--camus-root", default="./data/CAMUS")
    ap.add_argument("--encoding", default="filled")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default="./results/yolo")
    ap.add_argument("--gt-oracle", action="store_true",
                    help="score ground-truth masks instead, to separate EF "
                         "method error from segmentation error")
    args = ap.parse_args()

    ds = Path(args.dataset)
    meta = load_meta(ds)
    img_dir = ds / "images" / args.split
    stems = [s for s in sorted(p.stem for p in img_dir.glob("*.png"))
             if s in meta and not meta[s]["is_sequence"]]

    model = None
    pred_kw = {}
    if not args.gt_oracle:
        from ultralytics import YOLO

        model = YOLO(args.weights)
        pred_kw = dict(imgsz=args.imgsz, conf=args.conf, retina_masks=True,
                       device=args.device, verbose=False)
        model.predict(source=str(img_dir / f"{stems[0]}.png"), **pred_kw)

    # patient -> (view, phase) -> (binary LV mask, spacing)
    per_patient: dict[str, dict] = defaultdict(dict)
    gt_ef: dict[str, float] = {}

    for stem in tqdm(stems, desc=f"EF {args.name}", ncols=80):
        info = meta[stem]
        gt = native_gt(Path(args.camus_root), info["patient_id"], info["view"], info["phase"])
        lab = gt if args.gt_oracle else predict_labelmap(
            model, img_dir / f"{stem}.png", gt.shape[:2], args.encoding, pred_kw)
        per_patient[info["patient_id"]][(info["view"], info["phase"])] = (
            (lab == 1).astype(np.uint8), tuple(info["spacing"]))
        if np.isfinite(info["ef"]):
            gt_ef[info["patient_id"]] = float(info["ef"])

    calc = CAMUSEFCalculator(lv_label=1, n_disks=20)
    rows, skipped = [], []
    for pid, d in sorted(per_patient.items()):
        need = [("2CH", "ED"), ("2CH", "ES"), ("4CH", "ED"), ("4CH", "ES")]
        if not all(k in d for k in need):
            skipped.append((pid, "missing view/phase"))
            continue
        if any(d[k][0].sum() == 0 for k in need):
            # a detector may emit nothing; EF is undefined without an LV
            skipped.append((pid, "empty LV prediction"))
            continue
        (a2e, sp2), (a2s, _), (a4e, sp4), (a4s, _) = (d[k] for k in need)
        # See eval_baseline_ef.largest_cc: the official Simpson implementation
        # takes find_contours(...)[0], so a stray blob can be measured instead
        # of the ventricle. Without this filter the EF column is noise.
        a2e, a2s, a4e, a4s = (largest_cc(x) for x in (a2e, a2s, a4e, a4s))
        try:
            r = calc.compute_ef(a2e, a2s, sp2, a4e, a4s, sp4,
                                ef_ground_truth=gt_ef.get(pid), patient_id=pid)
            rows.append({"patient_id": pid, "ef_pred": r.ef_percent,
                         "ef_gt": gt_ef.get(pid), "edv_ml": r.edv_ml,
                         "esv_ml": r.esv_ml})
        except Exception as e:
            skipped.append((pid, f"{type(e).__name__}: {str(e)[:60]}"))

    stats = calc.compute_statistics()
    stats["n_skipped"] = len(skipped)
    stats["skipped"] = skipped
    stats["gt_oracle"] = args.gt_oracle

    out = Path(args.out_root) / args.name
    out.mkdir(parents=True, exist_ok=True)
    tag = "ef_oracle" if args.gt_oracle else "ef"
    (out / f"{tag}.json").write_text(json.dumps({"stats": stats, "per_patient": rows}, indent=1))

    print(f"\n{args.name}  ({'GT oracle' if args.gt_oracle else 'predicted'})")
    print(f"  patients   {stats.get('n_patients', 0)}  (skipped {len(skipped)})")
    if "ef_mae" in stats:
        print(f"  EF MAE     {stats['ef_mae']:.2f} %")
        print(f"  EF bias    {stats['ef_bias']:+.2f} %")
        print(f"  EF corr    {stats.get('ef_correlation', float('nan')):.3f}")
        print(f"  B-A LoA    [{stats['bland_altman_loa_lower']:+.1f}, "
              f"{stats['bland_altman_loa_upper']:+.1f}] %")
    if skipped:
        print(f"  skipped    {skipped[:3]}{' ...' if len(skipped) > 3 else ''}")
    print(f"  saved      {out / (tag + '.json')}")


if __name__ == "__main__":
    main()
