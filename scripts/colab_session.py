#!/usr/bin/env python3
"""
colab_session.py -- the single GPU session that finishes BOTH papers.

Run this ONCE on Colab (L4), where the trained checkpoints and the CAMUS
dataset live. In one pass over the official test split it produces every
GPU-dependent artifact the two papers still need:

  1. Per-patient EF arrays (ef_pred / ef_true) injected into each
     evaluation_results.json  ->  unlocks the Bland-Altman figure
     (make_figures.py fig_bland_altman) for Paper 1.
  2. Quality-stratified Dice by CAMUS Good / Medium / Poor grade, written as
     quality_stratified.json + a ready T5_quality.tex  ->  makes the
     "robustness across image-quality grades" contribution real in both papers.
  3. Qualitative prediction-vs-GT overlay grids (grades x models)  ->  replaces
     the \framebox placeholder figures in both papers.

Nothing here re-trains anything: it is inference-only over checkpoints that
already exist, so it fits comfortably in ~0.5-1 GPU-h on an L4.

Typical Colab usage (adjust paths to your Drive layout)::

    # Paper 1 base models
    python scripts/colab_session.py \
        --checkpoint_dir /content/results/base_models \
        --data_dir       /content/data/CAMUS \
        --out_dir        /content/session_out/p1 \
        --grid_models transunet nnunet unet_resnet unet_v1 deeplab_v3

    # Paper 2 Mamba variants (point at each group dir you want overlays for)
    python scripts/colab_session.py \
        --checkpoint_dir /content/results/vmamba_models \
        --data_dir       /content/data/CAMUS \
        --out_dir        /content/session_out/p2 \
        --grid_models mamba_unet_resnet_vmamba mamba_nnunet_vmamba \
                      mamba_unet_v1_vmamba pure_mamba_unet_vmamba

Afterwards (locally or on Colab) re-run the table/figure generators and
recompile::

    python scripts/fill_tables.py --results_root results \
        --benchmark_csv results/benchmark_efficiency.csv \
        --paper1_tables ../Paper1/paper/tables --strict
    python scripts/make_figures.py --results_root results \
        --benchmark_csv results/benchmark_efficiency.csv \
        --paper1_figs ../Paper1/paper/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import CAMUSDataset, get_transforms
from metrics import CAMUSEFCalculator

# Reuse the exact checkpoint discovery + construction logic the evaluator uses,
# so a model built here is byte-for-byte the one that produced the JSON numbers.
from scripts.evaluate_all_models import (
    find_model_checkpoints,
    _construct_and_load,
    get_img_size,
)

GRADES = ["Good", "Medium", "Poor"]
LV_LABEL = 1  # LV-endocardium class index (0 bg, 1 endo, 2 epi, 3 LA)


# ---------------------------------------------------------------------------
# Collate that preserves everything we need (spacing, phase, quality, ef)
# ---------------------------------------------------------------------------

def _collate(batch: List[Dict]) -> Dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "patient_id": [b["patient_id"] for b in batch],
        "view": [b["view"] for b in batch],
        "phase": [b["phase"] for b in batch],
        "pixel_spacing": [tuple(b["pixel_spacing"]) for b in batch],
        "quality": [b.get("quality", "Unknown") for b in batch],
    }


def _per_sample_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean foreground Dice (classes 1..3) for one frame -- matches
    evaluate_all_models.compute_per_sample_dice exactly."""
    dices = []
    for c in range(1, 4):
        pc = (pred == c).astype(float)
        tc = (target == c).astype(float)
        union = pc.sum() + tc.sum()
        dices.append(2 * (pc * tc).sum() / union if union > 0 else 1.0)
    return float(np.mean(dices))


# ---------------------------------------------------------------------------
# Pass 1: quality-stratified Dice (per frame) over the flat test set
# ---------------------------------------------------------------------------

def quality_stratified_dice(model, dataset, device, img_size, batch_size=8) -> Dict[str, Dict]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, collate_fn=_collate)
    per_grade: Dict[str, List[float]] = {g: [] for g in GRADES}
    other: List[float] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="  quality Dice", leave=False):
            images = batch["image"].to(device)
            out = model(images)
            if isinstance(out, dict):
                out = out["out"]
            preds = out.argmax(dim=1).cpu().numpy()
            masks = batch["mask"].cpu().numpy()
            for i, grade in enumerate(batch["quality"]):
                d = _per_sample_dice(preds[i], masks[i])
                (per_grade[grade] if grade in per_grade else other).append(d)
    summary = {}
    for g in GRADES:
        v = per_grade[g]
        if v:
            summary[g] = {"dice_mean": float(np.mean(v)),
                          "dice_std": float(np.std(v)), "n": len(v)}
    return summary


# ---------------------------------------------------------------------------
# Pass 2: per-patient biplane EF, capturing the ef_pred / ef_true arrays
# ---------------------------------------------------------------------------

def biplane_ef_arrays(model, data_dir, device, img_size) -> Dict[str, object]:
    """Mirror evaluate_all_models.evaluate_ef_biplane but also return the
    per-patient predicted/true EF arrays for the Bland-Altman plot."""
    transform = get_transforms(split="val", img_size=(img_size, img_size))
    dataset = CAMUSDataset(root_dir=data_dir, split="test",
                           views=["2CH", "4CH"], phases=["ED", "ES"],
                           transform=transform, include_info=True)
    by_patient: Dict[str, Dict] = {}
    for idx in range(len(dataset)):
        s = dataset[idx]
        by_patient.setdefault(s["patient_id"], {})[f"{s['view']}_{s['phase']}"] = {
            "image": s["image"], "pixel_spacing": s["pixel_spacing"], "ef_gt": s["ef"],
        }

    calc = CAMUSEFCalculator(lv_label=LV_LABEL)
    need = ["2CH_ED", "2CH_ES", "4CH_ED", "4CH_ES"]
    model.eval()
    with torch.no_grad():
        for pid, s in tqdm(by_patient.items(), desc="  biplane EF", leave=False):
            if not all(k in s for k in need):
                continue
            preds = {}
            for k in need:
                img = s[k]["image"].unsqueeze(0).to(device)
                out = model(img)
                if isinstance(out, dict):
                    out = out["out"]
                preds[k] = out.argmax(dim=1).squeeze(0).cpu().numpy()
            ef_gt = s["4CH_ED"]["ef_gt"]
            if ef_gt <= 0:
                ef_gt = s["2CH_ED"]["ef_gt"]
            calc.compute_ef(
                a2c_ed=preds["2CH_ED"], a2c_es=preds["2CH_ES"],
                a2c_spacing=s["2CH_ED"]["pixel_spacing"],
                a4c_ed=preds["4CH_ED"], a4c_es=preds["4CH_ES"],
                a4c_spacing=s["4CH_ED"]["pixel_spacing"],
                ef_ground_truth=ef_gt if ef_gt > 0 else None, patient_id=pid,
            )
    stats = calc.compute_statistics()
    pred, true = [], []
    for r in calc.results:
        if getattr(r, "ef_ground_truth", None) is not None:
            pred.append(float(r.ef_percent))
            true.append(float(r.ef_ground_truth))
    stats["ef_pred"] = pred
    stats["ef_true"] = true
    return stats


# ---------------------------------------------------------------------------
# Pass 3: qualitative overlay grid (grades x models)
# ---------------------------------------------------------------------------

_CMAP = np.array([[0, 0, 0], [231, 76, 60], [46, 204, 113], [52, 152, 219]]) / 255.0


def _overlay(ax, image: np.ndarray, mask: np.ndarray, title: str):
    img = image.squeeze()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    ax.imshow(img, cmap="gray")
    rgba = np.zeros((*mask.shape, 4))
    for c in range(1, 4):
        rgba[mask == c, :3] = _CMAP[c]
        rgba[mask == c, 3] = 0.45
    ax.imshow(rgba)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def qualitative_grid(models_info, dataset, device, grid_models, out_png):
    """One representative test patient per grade (4CH ED frame) x each model."""
    # Pick a 4CH ED sample index for each grade.
    picks: Dict[str, int] = {}
    for idx in range(len(dataset)):
        s = dataset[idx]
        if s["view"] == "4CH" and s["phase"] == "ED" and s.get("quality") in GRADES \
                and s["quality"] not in picks:
            picks[s["quality"]] = idx
        if len(picks) == len(GRADES):
            break
    grades = [g for g in GRADES if g in picks]
    if not grades:
        print("  [qualitative] no graded 4CH/ED frames found; skipping grid")
        return

    rows = 1 + len(grid_models)  # GT row + one row per model
    fig, axes = plt.subplots(rows, len(grades),
                             figsize=(3.0 * len(grades), 3.0 * rows), squeeze=False)
    # Row 0: ground truth
    for j, g in enumerate(grades):
        s = dataset[picks[g]]
        _overlay(axes[0][j], s["image"].numpy(), s["mask"].numpy(),
                 f"{g} -- ground truth")
    # Remaining rows: each model's prediction
    name_to_info = {m["name"]: m for m in models_info}
    name_to_info.update({m["display_name"]: m for m in models_info})
    for r, mname in enumerate(grid_models, start=1):
        info = name_to_info.get(mname)
        if info is None:
            for j in range(len(grades)):
                axes[r][j].axis("off")
            print(f"  [qualitative] model '{mname}' not among checkpoints; skipped row")
            continue
        model = _construct_and_load(info, device).to(device).eval()
        img_size = get_img_size(info["name"])
        with torch.no_grad():
            for j, g in enumerate(grades):
                s = dataset[picks[g]]
                img = s["image"].unsqueeze(0).to(device)
                out = model(img)
                if isinstance(out, dict):
                    out = out["out"]
                pred = out.argmax(dim=1).squeeze(0).cpu().numpy()
                _overlay(axes[r][j], s["image"].numpy(), pred, info["display_name"])
        del model
        torch.cuda.empty_cache()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


# ---------------------------------------------------------------------------
# Quality-stratified LaTeX table (Paper 1 T5)
# ---------------------------------------------------------------------------

def write_quality_table(quality: Dict[str, Dict], out_tex: Path,
                        display: Dict[str, str]):
    order = sorted(quality.items(),
                   key=lambda kv: -(kv[1].get("Good", {}).get("dice_mean", 0)))
    body = []
    for name, gr in order:
        disp = display.get(name, name.replace("_", r"\_"))
        cells = []
        for g in GRADES:
            v = gr.get(g)
            cells.append(f"{v['dice_mean']:.4f}" if v else "{---}")
        body.append(f"{disp:<18} & " + " & ".join(cells) + r" \\")
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(
        "%% Auto-generated by colab_session.py -- quality-stratified Dice\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Mean Dice on the CAMUS test split stratified by the "
        "expert-assigned image-quality grade (Good / Medium / Poor).}\n"
        "\\label{tab:quality}\n\\small\n\\setlength{\\tabcolsep}{6pt}\n"
        "\\begin{tabular}{l c c c}\n\\toprule\n"
        "Architecture & {Good} & {Medium} & {Poor} \\\\\n\\midrule\n"
        + "\n".join(body) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")
    print(f"  wrote {out_tex}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_dir", type=Path, required=True,
                    help="Dir with model subfolders (e.g. results/base_models)")
    ap.add_argument("--data_dir", type=str, default="./data/CAMUS")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--eval_json", type=Path, default=None,
                    help="evaluation_results.json to inject EF arrays into "
                         "(default: <checkpoint_dir>/evaluation/evaluation_results.json)")
    ap.add_argument("--grid_models", nargs="*", default=None,
                    help="Model names/display names for the qualitative grid")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--skip_ef", action="store_true")
    ap.add_argument("--skip_quality", action="store_true")
    ap.add_argument("--skip_grid", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models_info = find_model_checkpoints(args.checkpoint_dir)
    if not models_info:
        print(f"No checkpoints under {args.checkpoint_dir}")
        return
    print(f"Found {len(models_info)} checkpoints; device = {device}")

    eval_json = args.eval_json or (args.checkpoint_dir / "evaluation" /
                                   "evaluation_results.json")
    eval_data = json.loads(eval_json.read_text()) if eval_json.exists() else {"results": {}}
    display_map = {m["name"]: m["display_name"] for m in models_info}

    quality_all: Dict[str, Dict] = {}
    for info in models_info:
        name, disp = info["name"], info["display_name"]
        print(f"\n=== {disp} ===")
        img_size = get_img_size(name)
        transform = get_transforms(split="val", img_size=(img_size, img_size))
        flat = CAMUSDataset(root_dir=args.data_dir, split="test",
                            transform=transform, include_info=True)

        if not args.skip_quality:
            model = _construct_and_load(info, device).to(device).eval()
            quality_all[disp] = quality_stratified_dice(model, flat, device,
                                                        img_size, args.batch_size)
            print(f"  quality Dice: "
                  + ", ".join(f"{g} {quality_all[disp].get(g, {}).get('dice_mean', float('nan')):.4f}"
                              for g in GRADES))
            del model
            torch.cuda.empty_cache()

        if not args.skip_ef:
            ef = biplane_ef_arrays(_construct_and_load(info, device).to(device).eval(),
                                   args.data_dir, device, img_size)
            # merge arrays into the existing ef_metrics of the JSON
            res = eval_data.setdefault("results", {}).setdefault(disp, {})
            em = res.setdefault("ef_metrics", {})
            em["ef_pred"] = ef.get("ef_pred", [])
            em["ef_true"] = ef.get("ef_true", [])
            print(f"  persisted {len(em['ef_pred'])} per-patient EF pairs")
            torch.cuda.empty_cache()

    # Write outputs
    if not args.skip_quality and quality_all:
        (args.out_dir / "quality_stratified.json").write_text(
            json.dumps(quality_all, indent=2))
        write_quality_table(quality_all, args.out_dir / "T5_quality.tex", display_map)

    if not args.skip_ef:
        eval_json.parent.mkdir(parents=True, exist_ok=True)
        eval_json.write_text(json.dumps(eval_data, indent=2))
        print(f"\nInjected EF arrays into {eval_json}")

    if not args.skip_grid:
        grid_models = args.grid_models or [m["name"] for m in models_info[:5]]
        img_size = get_img_size(grid_models[0] if grid_models else "unet_v1")
        transform = get_transforms(split="val", img_size=(img_size, img_size))
        flat = CAMUSDataset(root_dir=args.data_dir, split="test",
                            transform=transform, include_info=True)
        qualitative_grid(models_info, flat, device, grid_models,
                         args.out_dir / "fig_qualitative.png")

    print("\n[colab_session] Done. Next: re-run fill_tables.py + make_figures.py "
          "and recompile.")


if __name__ == "__main__":
    main()
