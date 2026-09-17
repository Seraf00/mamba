#!/usr/bin/env python3
"""
Generate the figures for the ISBI paper from completed YOLO runs.

    fig_pareto.pdf      accuracy vs latency, YOLO against the baselines
    fig_ablation.pdf    resolution and mask-prototype sweeps vs HD95
    fig_ceiling.pdf     polygon encoding ceiling vs vertex budget
    fig_qualitative.pdf predicted contours on representative test frames

Figures whose inputs are missing are skipped with a note rather than failing,
so this can be run at any point during the sweep.

Usage:
    python scripts/yolo/make_figures.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "yolo"))

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.4,
})

BASE_LABEL = {
    "transunet": "TransUNet", "nnunet": "nnU-Net", "unet_v1": "UNet-V1",
    "unet_resnet": "UNet-ResNet", "unet_v2": "UNet-V2", "swin_unet": "Swin-UNet",
    "deeplab_v3": "DeepLabV3+", "fpn": "FPN-UNet",
    "dense_context_unet": "DenseCtx-UNet",
}


def load_runs(res: Path) -> dict:
    out = {}
    for d in sorted(res.iterdir()):
        f = d / "evaluation.json"
        if d.is_dir() and f.exists():
            out[d.name] = json.loads(f.read_text())
    return out


def load_baselines(p: Path) -> dict:
    return json.loads(p.read_text())["results"] if p.exists() else {}


# ------------------------------------------------------------------ pareto --
def fig_pareto(runs, baselines, eff_csv: Path, out: Path) -> bool:
    if not runs or not baselines:
        return False
    lat = {}
    if eff_csv.exists():
        import csv as _csv

        with eff_csv.open(newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                lat[row["Model"]] = float(row["Inference Time (ms)"])
    alias = {"unet_v1": "unet", "deeplab_v3": "deeplab_v3"}

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    bx, by, bl = [], [], []
    for k, r in baselines.items():
        t = lat.get(alias.get(k, k), r.get("inference_time_ms"))
        if t and r.get("dice_mean"):
            bx.append(t)
            by.append(r["dice_mean"])
            bl.append(BASE_LABEL.get(k, k))
    ax.scatter(bx, by, s=18, c="#4C72B0", marker="o", label="Semantic seg. baselines",
               zorder=3, edgecolors="white", linewidths=0.4)
    for x, y, l in zip(bx, by, bl):
        ax.annotate(l, (x, y), fontsize=5, xytext=(2, 2), textcoords="offset points",
                    color="#33507d")

    yx, yy, yl = [], [], []
    for n, r in runs.items():
        if r.get("latency_ms_mean") and r.get("dice_mean"):
            yx.append(r["latency_ms_mean"])
            yy.append(r["dice_mean"])
            yl.append(n)
    if yx:
        ax.scatter(yx, yy, s=22, c="#C44E52", marker="^", label="YOLO-seg (this work)",
                   zorder=4, edgecolors="white", linewidths=0.4)

    ax.set_xlabel("Latency per frame (ms)")
    ax.set_ylabel("Mean Dice")
    ax.set_xscale("log")
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    fig.savefig(out)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- ablation --
def fig_ablation(runs, out: Path) -> bool:
    sz = [("E3_sz512", 512), ("E3_sz640", 640), ("E3_sz800", 800), ("E3_sz960", 960)]
    mr = [("E4_mr1", "1/1"), ("E4_mr2", "1/2"), ("E4_mr4", "1/4")]
    sz = [(n, v) for n, v in sz if n in runs]
    mr = [(n, v) for n, v in mr if n in runs]
    if not sz and not mr:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
    if sz:
        x = [v for _, v in sz]
        d = [runs[n]["dice_mean"] for n, _ in sz]
        h = [runs[n]["hd95_mean"] for n, _ in sz]
        a = axes[0]
        a.plot(x, d, "o-", color="#C44E52", lw=1.2, ms=4, label="Dice")
        a.set_xlabel("Input resolution (px)")
        a.set_ylabel("Mean Dice", color="#C44E52")
        a2 = a.twinx()
        a2.plot(x, h, "s--", color="#4C72B0", lw=1.2, ms=4, label="HD95")
        a2.set_ylabel("HD95 (mm)", color="#4C72B0")
        a2.grid(False)
        a.set_title("Input resolution")
    else:
        axes[0].set_visible(False)

    if mr:
        lbl = [v for _, v in mr]
        d = [runs[n]["dice_mean"] for n, _ in mr]
        h = [runs[n]["hd95_mean"] for n, _ in mr]
        a = axes[1]
        xi = np.arange(len(lbl))
        a.bar(xi - 0.18, d, 0.36, color="#C44E52", label="Dice")
        a.set_xticks(xi)
        a.set_xticklabels(lbl)
        a.set_xlabel("Mask prototype resolution")
        a.set_ylabel("Mean Dice", color="#C44E52")
        a.set_ylim(min(d) - 0.02, max(d) + 0.01)
        a2 = a.twinx()
        a2.bar(xi + 0.18, h, 0.36, color="#4C72B0", label="HD95")
        a2.set_ylabel("HD95 (mm)", color="#4C72B0")
        a2.grid(False)
        a.set_title("Mask head resolution")
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


# ----------------------------------------------------------------- ceiling --
def fig_ceiling(path: Path, out: Path) -> bool:
    if not path.exists():
        return False
    d = json.loads(path.read_text())
    encs: dict[str, list[tuple[int, float, float]]] = {}
    for k, v in d.items():
        enc, n = k.rsplit("_n", 1)
        n = int(n)
        if n == 0:
            continue
        encs.setdefault(enc, []).append((n, v["dice_mean"], v["target_dice_class1"]))
    if not encs:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
    colors = {"filled": "#C44E52", "seam": "#4C72B0"}
    names = {"filled": "Filled contour", "seam": "Seam-cut annulus"}
    for enc, vals in encs.items():
        vals.sort()
        n = [v[0] for v in vals]
        axes[0].plot(n, [v[1] for v in vals], "o-", ms=4, lw=1.2,
                     color=colors.get(enc), label=names.get(enc, enc))
        axes[1].plot(n, [v[2] for v in vals], "s-", ms=4, lw=1.2,
                     color=colors.get(enc), label=names.get(enc, enc))
    axes[0].set_xlabel("Vertices per contour")
    axes[0].set_ylabel("Composed label-map Dice")
    axes[0].set_title("Reconstructed CAMUS label map")
    axes[1].set_xlabel("Vertices per contour")
    axes[1].set_ylabel("Class-2 target Dice")
    axes[1].set_title("Myocardial target as the mask head sees it")
    for a in axes:
        a.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


# ------------------------------------------------------------- qualitative --
def fig_qualitative(weights: Path, dataset: Path, encoding: str, imgsz: int,
                    out: Path, n: int = 4) -> bool:
    if not weights.exists():
        return False
    import cv2
    from ultralytics import YOLO

    from yolo_common import binaries_to_labelmap, load_meta, native_gt

    meta = load_meta(dataset)
    img_dir = dataset / "images" / "test"
    stems = sorted(p.stem for p in img_dir.glob("*.png"))
    stems = [s for s in stems if s in meta]
    if not stems:
        return False
    pick = [stems[i] for i in np.linspace(0, len(stems) - 1, n).astype(int)]

    model = YOLO(str(weights))
    fig, axes = plt.subplots(1, len(pick), figsize=(1.75 * len(pick), 2.1))
    if len(pick) == 1:
        axes = [axes]

    for ax, stem in zip(axes, pick):
        info = meta[stem]
        img = cv2.imread(str(img_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        gt = native_gt(ROOT / "data" / "CAMUS", info["patient_id"], info["view"], info["phase"])
        h, w = gt.shape[:2]
        res = model.predict(source=str(img_dir / f"{stem}.png"), imgsz=imgsz,
                            retina_masks=True, verbose=False)[0]
        best = {}
        if res.masks is not None and len(res.masks):
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
        pred = binaries_to_labelmap(best.get(0), best.get(1), best.get(2),
                                    encoding, shape=(h, w))

        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        ax.imshow(img, cmap="gray")
        for c, col in zip((1, 2, 3), ("#55A868", "#C44E52", "#4C72B0")):
            for src, style in ((gt, "--"), (pred, "-")):
                cs, _ = cv2.findContours((src == c).astype(np.uint8),
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cc in cs:
                    if cv2.contourArea(cc) < 20:
                        continue
                    cc = cc.reshape(-1, 2)
                    ax.plot(cc[:, 0], cc[:, 1], style, color=col, lw=0.7)
        ax.set_title(f"{info['view']} {info['phase']}", fontsize=6)
        ax.axis("off")

    fig.suptitle("solid: prediction    dashed: ground truth", fontsize=6, y=0.04)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def placeholder_fig(path: Path, name: str) -> None:
    """Keep the paper compiling before the sweep has produced this figure."""
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    ax.text(0.5, 0.5, f"{name}\npending GPU sweep", ha="center", va="center",
            fontsize=9, color="red", transform=ax.transAxes)
    ax.set_axis_off()
    ax.grid(False)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "yolo"))
    ap.add_argument("--baselines",
                    default=str(ROOT / "results" / "base_models" / "evaluation" / "evaluation_results.json"))
    ap.add_argument("--efficiency", default=str(ROOT / "results" / "benchmark_efficiency.csv"))
    # E9_seed0 is usually deduplicated away and has no weights of its own, so
    # fall back to the best completed run that actually has a checkpoint.
    _w = ROOT / "yolo_runs" / "E9_seed0" / "weights" / "best.pt"
    if not _w.exists():
        _c = []
        for _d in (ROOT / "results" / "yolo").glob("*"):
            _p = ROOT / "yolo_runs" / _d.name / "weights" / "best.pt"
            _e = _d / "evaluation.json"
            if _p.exists() and _e.exists():
                try:
                    _c.append((json.loads(_e.read_text()).get("dice_mean", -1), _p))
                except json.JSONDecodeError:
                    pass
        if _c:
            _w = max(_c)[1]
    ap.add_argument("--weights", default=str(_w))
    ap.add_argument("--dataset", default=str(ROOT / "yolo_data" / "camus_edes"))
    ap.add_argument("--encoding", default="filled")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=str(ROOT / "paper2" / "figures"))
    args = ap.parse_args()

    res = Path(args.results)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(res)
    baselines = load_baselines(Path(args.baselines))

    jobs = [
        ("fig_ceiling.pdf", lambda p: fig_ceiling(res / "encoding_ceiling.json", p)),
        ("fig_pareto.pdf", lambda p: fig_pareto(runs, baselines, Path(args.efficiency), p)),
        ("fig_ablation.pdf", lambda p: fig_ablation(runs, p)),
        ("fig_qualitative.pdf", lambda p: fig_qualitative(
            Path(args.weights), Path(args.dataset), args.encoding, args.imgsz, p)),
    ]
    for fname, fn in jobs:
        try:
            ok = fn(outdir / fname)
        except Exception as e:  # a missing input should not kill the rest
            print(f"  {fname}: failed ({e})")
            ok = False
        if not ok:
            placeholder_fig(outdir / fname, fname)
            print(f"  {fname}: placeholder (inputs missing)")
        else:
            print(f"  {fname}: written")
    print(f"\nFigures in {outdir}")


if __name__ == "__main__":
    main()
