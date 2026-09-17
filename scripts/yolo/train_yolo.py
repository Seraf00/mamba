#!/usr/bin/env python3
"""
Train one YOLO segmentation model on the CAMUS-derived dataset.

Thin, explicit wrapper over Ultralytics so every knob that the paper ablates is
a named flag and every run writes a reproducible config next to its weights.

Echo-specific defaults differ from the Ultralytics COCO defaults:
  * mosaic/mixup/copy-paste OFF -- each CAMUS frame holds exactly one instance
    of each structure in a fixed anatomical layout; pasting sectors together
    creates images that cannot occur and destroys the geometric prior.
  * fliplr OFF -- a horizontal flip swaps the anatomical left/right convention
    of an apical view.
  * hsv_h/hsv_s OFF -- the images are grayscale, so hue/saturation jitter is a
    no-op; only value jitter is meaningful.
Pass --coco-aug to restore the stock recipe for the augmentation ablation.

Usage:
    python scripts/yolo/train_yolo.py --model yolo11n-seg.pt --name v11n_base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11n-seg.pt",
                    help="pretrained checkpoint, or a .yaml for from-scratch")
    ap.add_argument("--data", default="./yolo_data/camus_edes/data.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="./yolo_runs")

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=float, default=8,
                    help="int batch, or a fraction in (0,1) for AutoBatch")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--cache", default="False", choices=["False", "ram", "disk"])

    ap.add_argument("--optimizer", default="auto")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--cos-lr", action="store_true")
    ap.add_argument("--box", type=float, default=7.5)
    ap.add_argument("--cls", type=float, default=0.5)
    ap.add_argument("--dfl", type=float, default=1.5)

    ap.add_argument("--overlap-mask", default="True", choices=["True", "False"])
    ap.add_argument("--mask-ratio", type=int, default=4,
                    help="mask head downsample; 1 = full-resolution prototypes")

    ap.add_argument("--coco-aug", action="store_true",
                    help="use stock Ultralytics augmentation instead of echo defaults")
    ap.add_argument("--degrees", type=float, default=10.0)
    ap.add_argument("--translate", type=float, default=0.1)
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--shear", type=float, default=0.0)
    ap.add_argument("--fliplr", type=float, default=0.0)
    ap.add_argument("--flipud", type=float, default=0.0)
    ap.add_argument("--mosaic", type=float, default=0.0)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--copy-paste", type=float, default=0.0)
    ap.add_argument("--hsv-h", type=float, default=0.0)
    ap.add_argument("--hsv-s", type=float, default=0.0)
    ap.add_argument("--hsv-v", type=float, default=0.3)
    ap.add_argument("--erasing", type=float, default=0.0)

    ap.add_argument("--boundary-weight", type=float, default=0.0,
                    help="lambda for the boundary-weighted mask loss (0 = stock loss)")
    ap.add_argument("--boundary-band", type=int, default=1,
                    help="half-width of the emphasised contour band, in prototype pixels")
    ap.add_argument("--amp", default="True", choices=["True", "False"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    batch = int(args.batch) if args.batch >= 1 else float(args.batch)
    cache = False if args.cache == "False" else args.cache

    cfg = dict(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        cache=cache,
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        overlap_mask=(args.overlap_mask == "True"),
        mask_ratio=args.mask_ratio,
        amp=(args.amp == "True"),
        plots=True,
        val=True,
        deterministic=True,
    )

    if not args.coco_aug:
        cfg.update(
            degrees=args.degrees, translate=args.translate, scale=args.scale,
            shear=args.shear, fliplr=args.fliplr, flipud=args.flipud,
            mosaic=args.mosaic, mixup=args.mixup, copy_paste=args.copy_paste,
            hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
            erasing=args.erasing,
        )

    print(json.dumps({k: str(v) for k, v in cfg.items()}, indent=1))
    if args.dry_run:
        return

    # Boundary-weighted mask loss: patches the Ultralytics segmentation loss in
    # place, so it must be installed before the trainer builds its criterion.
    # At weight 0 the patched loss is bit-identical to the stock one, so the
    # ablation is not confounded by the patch itself.
    if args.boundary_weight > 0:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from boundary_loss import install_boundary_loss

        install_boundary_loss(weight=args.boundary_weight, band_px=args.boundary_band)

    model = YOLO(args.model)
    model.train(resume=args.resume, **cfg)

    out = Path(args.project) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(
        json.dumps({"model": args.model, **{k: str(v) for k, v in cfg.items()}}, indent=1)
    )
    print(f"\nDone: {out}")


if __name__ == "__main__":
    main()
