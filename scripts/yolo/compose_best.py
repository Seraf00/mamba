#!/usr/bin/env python3
"""
Compose the best-performing configuration from the completed sweep, and emit it
as an experiment group (E11) plus a boundary-loss arm (E12).

The plan's E9 group is labelled "best configuration" but actually re-runs the
DEFAULT configuration with three seeds -- the substitution its comment describes
was never implemented. This closes that gap by reading what the sweep actually
measured and composing the winner.

Selection rule
--------------
Per ablation group, pick the run with the lowest HD95 rather than the highest
Dice. Across this sweep Dice was saturated (0.9127-0.9176 spanning a 10x
parameter range and three generations, most pairwise tests non-significant)
while HD95 separated every model, so HD95 is the discriminating statistic.
Dice and miss counts are reported alongside for sanity.

A setting is only adopted if it beats the group's reference run by more than
`--min-gain` mm, so noise-level differences do not get baked in.

Usage:
    python scripts/yolo/compose_best.py                 # write E11/E12 groups
    python scripts/yolo/compose_best.py --dry-run       # just show the picks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]

# group -> (run name -> the config override that run represents)
GROUP_SETTINGS: dict[str, dict[str, dict]] = {
    "E1_encoding": {
        "E1_filled": {"data": "./yolo_data/camus_edes/data.yaml", "encoding": "filled"},
        "E1_seam": {"data": "./yolo_data/camus_edes_seam/data.yaml", "encoding": "seam"},
    },
    "E2_scale": {
        "E2_v26n": {"model": "yolo26n-seg.pt"},
        "E2_v26s": {"model": "yolo26s-seg.pt", "batch": 8},
        "E2_v26m": {"model": "yolo26m-seg.pt", "batch": 4},
        "E2_v11n": {"model": "yolo11n-seg.pt"},
        "E2_v8n": {"model": "yolov8n-seg.pt"},
    },
    "E3_imgsz": {
        "E3_sz512": {"imgsz": 512},
        "E3_sz640": {"imgsz": 640},
        "E3_sz800": {"imgsz": 800, "batch": 6},
        "E3_sz960": {"imgsz": 960, "batch": 4},
    },
    "E4_maskratio": {
        "E4_mr4": {"mask_ratio": 4},
        "E4_mr2": {"mask_ratio": 2, "batch": 6},
        "E4_mr1": {"mask_ratio": 1, "batch": 4},
    },
    "E5_aug": {
        "E5_echo": {},
        "E5_coco": {"coco_aug": True},
        "E5_none": {"degrees": 0.0, "translate": 0.0, "scale": 0.0, "hsv_v": 0.0},
        "E5_echo_fliplr": {"fliplr": 0.5},
    },
    "E6_pretrain": {
        "E6_pretrained": {},
        "E6_scratch": {"model": "yolo26n-seg.yaml"},
    },
    "E7_data": {
        "E7_edes": {"data": "./yolo_data/camus_edes/data.yaml", "epochs": 100},
        "E7_seq": {"data": "./yolo_data/camus_seq/data.yaml", "epochs": 40},
    },
    "E8_vertices": {
        "E8_n96": {"data": "./yolo_data/camus_edes/data.yaml"},
        "E8_n32": {"data": "./yolo_data/camus_edes_n32/data.yaml"},
    },
}

# the run in each group that represents the current default
REFERENCE = {
    "E1_encoding": "E1_filled", "E2_scale": "E2_v26n", "E3_imgsz": "E3_sz640",
    "E4_maskratio": "E4_mr4", "E5_aug": "E5_echo", "E6_pretrain": "E6_pretrained",
    "E7_data": "E7_edes", "E8_vertices": "E8_n96",
}

CLASSES = ["lv_endocardium", "lv_epicardium", "left_atrium"]


def load(res: Path, name: str):
    f = res / name / "evaluation.json"
    return json.loads(f.read_text()) if f.exists() else None


def miss_count(res: Path, name: str) -> int:
    f = res / name / "per_sample.json"
    if not f.exists():
        return -1
    recs = json.loads(f.read_text())
    return sum(1 for r in recs
               if not all(r.get(f"detected_{c}", True) for c in CLASSES))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "yolo"))
    ap.add_argument("--plan", default=str(Path(__file__).resolve().parent / "experiments.yaml"))
    ap.add_argument("--min-gain", type=float, default=None,
                    help="mm of HD95 a setting must win by to be adopted; "
                         "default = measured seed noise from the E9 repeats")
    ap.add_argument("--boundary-weights", nargs="+", type=float, default=[2.0, 5.0, 10.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    res = Path(args.results)

    # Threshold from measured seed noise, not a guess. Repeats of one identical
    # configuration (E9 + its dedup twin E1_filled) differ by a non-trivial
    # margin -- enough that a paired Wilcoxon on 200 frames calls pure seed
    # variation significant. Adopting settings that win by less than that would
    # compose a configuration out of coin flips.
    if args.min_gain is None:
        seeds = [load(res, n) for n in ("E1_filled", "E9_seed1", "E9_seed2")]
        h = [s["hd95_mean"] for s in seeds if s]
        args.min_gain = (max(h) - min(h)) if len(h) >= 2 else 0.20
        print(f"seed-noise threshold from {len(h)} repeats of one config: "
              f"{args.min_gain:.3f} mm HD95 "
              f"({'measured' if len(h) >= 2 else 'fallback'})\n")

    chosen: dict = {}
    print(f"{'group':16s} {'pick':16s} {'HD95':>7} {'ref HD95':>9} {'gain':>7} {'Dice':>7} {'miss':>5}")
    for gid, runs in GROUP_SETTINGS.items():
        avail = {n: load(res, n) for n in runs}
        avail = {n: v for n, v in avail.items() if v}
        if not avail:
            print(f"{gid:16s} (no results yet -- keeping default)")
            continue
        ref_name = REFERENCE[gid]
        ref = avail.get(ref_name)
        best = min(avail.items(), key=lambda kv: kv[1].get("hd95_mean", np.inf))
        bname, bval = best
        gain = (ref.get("hd95_mean", np.inf) - bval["hd95_mean"]) if ref else np.inf

        if ref is not None and gain <= args.min_gain:
            bname, bval, gain = ref_name, ref, 0.0
        chosen[gid] = bname
        print(f"{gid:16s} {bname:16s} {bval['hd95_mean']:7.3f} "
              f"{(ref or {}).get('hd95_mean', float('nan')):9.3f} {gain:+7.3f} "
              f"{bval['dice_mean']:7.4f} {miss_count(res, bname):5d}")

    # merge the winning overrides
    cfg: dict = {}
    for gid, name in chosen.items():
        cfg.update(GROUP_SETTINGS[gid][name])
    # batch is a per-group artefact of memory limits; keep the most restrictive
    batches = [GROUP_SETTINGS[g][n].get("batch") for g, n in chosen.items()
               if GROUP_SETTINGS[g][n].get("batch")]
    if batches:
        cfg["batch"] = min(batches)

    print("\ncomposed configuration:")
    for k, v in sorted(cfg.items()):
        print(f"  {k:16s} {v}")

    if args.dry_run:
        return

    plan = yaml.safe_load(Path(args.plan).read_text())
    plan["groups"] = [g for g in plan["groups"] if g["id"] not in ("E11_best", "E12_boundary")]

    plan["groups"].append({
        "id": "E11_best",
        "question": "Does combining the per-group winners beat each individually?",
        "runs": [{"name": f"E11_best_seed{s}", "seed": s, **cfg} for s in args.seeds],
    })
    plan["groups"].append({
        "id": "E12_boundary",
        "question": "Does a boundary-weighted mask loss reduce HD95?",
        "runs": [{"name": "E12_bw0", **cfg}] +
                [{"name": f"E12_bw{str(w).replace('.', '')}", "boundary_weight": w, **cfg}
                 for w in args.boundary_weights],
    })

    Path(args.plan).write_text(yaml.safe_dump(plan, sort_keys=False, width=100))
    print(f"\nWrote E11_best ({len(args.seeds)} seeds) and "
          f"E12_boundary ({len(args.boundary_weights) + 1} runs) to {args.plan}")
    print("Run with: python scripts/yolo/run_experiments.py --groups E11_best E12_boundary")


if __name__ == "__main__":
    main()
