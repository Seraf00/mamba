#!/usr/bin/env python3
"""
Evaluate a trained YOLO segmentation model on CAMUS at NATIVE resolution.

Predicted instance masks are mapped back to the CAMUS 4-label space and scored
with `scoring.CamusScorer`, which reproduces the exact protocol behind the
published U-Net / Transformer / Mamba baselines (Dice & IoU on the 256x256
grid, HD95 & ASSD at native resolution in mm). The YOLO rows therefore drop
straight into the existing comparison tables.

Per-sample metrics are written out so paired significance tests against the
baselines can be run later.

Usage:
    python scripts/yolo/eval_yolo.py --weights yolo_runs/<run>/weights/best.pt \
        --dataset yolo_data/camus_edes --name v11n_base
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "yolo"))

from scoring import CamusScorer  # noqa: E402
from yolo_common import binaries_to_labelmap, load_meta, native_gt  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", default="./yolo_data/camus_edes")
    ap.add_argument("--camus-root", default="./data/CAMUS")
    ap.add_argument("--encoding", default="filled")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--tta", action="store_true", help="test-time augmentation")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default="./results/yolo")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N images")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    from ultralytics import YOLO

    ds = Path(args.dataset)
    meta = load_meta(ds)
    img_dir = ds / "images" / args.split
    stems = sorted(p.stem for p in img_dir.glob("*.png"))
    stems = [s for s in stems if s in meta and not meta[s]["is_sequence"]]
    if args.limit:
        stems = stems[: args.limit]

    model = YOLO(args.weights)

    # `half` is deprecated upstream and warns on every call, so only pass it
    # when actually requested.
    pred_kw = dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                   retina_masks=True, augment=args.tta,
                   device=args.device, verbose=False)
    if args.half:
        pred_kw["half"] = True

    # warm-up so the first-call CUDA/cuDNN setup is not charged to latency
    for _ in range(3):
        model.predict(source=str(img_dir / f"{stems[0]}.png"), **pred_kw)

    scorer = CamusScorer()
    wall_ms: list[float] = []
    infer_ms: list[float] = []

    for stem in tqdm(stems, desc=f"eval {args.name}", ncols=80):
        info = meta[stem]
        gt = native_gt(Path(args.camus_root), info["patient_id"], info["view"], info["phase"])
        h, w = gt.shape[:2]
        sp = tuple(info["spacing"])

        t0 = time.perf_counter()
        res = model.predict(source=str(img_dir / f"{stem}.png"), **pred_kw)[0]
        wall_ms.append((time.perf_counter() - t0) * 1000.0)
        sp_d = getattr(res, "speed", {}) or {}
        infer_ms.append(float(sp_d.get("inference", np.nan)))

        # highest-confidence instance per class
        best: dict[int, np.ndarray] = {}
        if res.masks is not None and len(res.masks) > 0:
            md = res.masks.data.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cfd = res.boxes.conf.cpu().numpy()
            for c in (0, 1, 2):
                idx = np.where(cls == c)[0]
                if len(idx):
                    best[c] = md[idx[int(np.argmax(cfd[idx]))]] > 0.5

        def fit(m):
            if m is None:
                return None
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            return m

        pred = binaries_to_labelmap(
            fit(best.get(0)), fit(best.get(1)), fit(best.get(2)), args.encoding,
            shape=(h, w),
        )
        if pred.shape != gt.shape:
            pred = np.zeros_like(gt)

        scorer.add(pred, gt, sp, meta={
            "stem": stem, "patient_id": info["patient_id"], "view": info["view"],
            "phase": info["phase"], "quality": info["quality"], "ef": info["ef"],
        })

    summary = scorer.summary()
    summary["latency_ms_mean"] = float(np.mean(wall_ms))
    summary["latency_ms_std"] = float(np.std(wall_ms))
    summary["inference_ms_mean"] = float(np.nanmean(infer_ms))
    summary["fps"] = float(1000.0 / np.mean(wall_ms))
    summary["config"] = vars(args)

    # NB: YOLO.info() returns None in ultralytics 8.4.x, so read the module
    # directly -- otherwise params/GFLOPs silently vanish from every result.
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params

        summary["params_M"] = get_num_params(model.model) / 1e6
        summary["gflops"] = float(get_flops(model.model, args.imgsz))
    except Exception as e:
        print(f"  [warn] could not read params/FLOPs: {e}")

    out = Path(args.out_root) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(summary, indent=1))
    (out / "per_sample.json").write_text(json.dumps(scorer.records, indent=1))

    print(f"\n{args.name}")
    print(f"  Dice  {summary.get('dice_mean', float('nan')):.4f}")
    print(f"  HD95  {summary.get('hd95_mean', float('nan')):.3f} mm")
    print(f"  ASSD  {summary.get('assd_mean', float('nan')):.3f} mm")
    print(f"  lat   {summary['latency_ms_mean']:.2f} ms  ({summary['fps']:.1f} FPS)")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
