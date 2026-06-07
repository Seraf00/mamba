#!/usr/bin/env python3
"""
make_figures.py -- generate publication figures from the evaluation JSONs
and the efficiency-benchmark CSV.

Produces (PDF + PNG) into the chosen output directories:
  Paper 1:
    fig_pareto.pdf            accuracy (Dice) vs parameters, Pareto front
    fig_bland_altman.pdf      4-panel Bland-Altman for best EF models
    fig_cd_diagram.pdf        critical-difference diagram (per-sample Dice)
  Paper 2:
    fig_memory.pdf            peak train memory per config, OOMs hatched
    fig_radar.pdf             per-class Dice radar for the 3 SSM variants
    fig_dice_delta_heatmap.pdf  base->Mamba Dice delta, architecture x variant
    fig_cd_diagram_mamba.pdf  critical-difference over top-15 configs

Each figure degrades gracefully: if the data it needs is absent (e.g.
per-sample Dice arrays, per-patient EF), it prints a clear note and skips
that figure rather than crashing.

Usage (Colab, data already on disk):
    python scripts/make_figures.py \\
        --results_root /content/results \\
        --benchmark_csv /content/results/benchmark_efficiency.csv \\
        --paper1_figs /content/drive/MyDrive/Papers/Paper1/paper/figures \\
        --paper2_figs /content/drive/MyDrive/Papers/Paper2/paper/figures

Usage (local):
    python D:/Papers/Paper1/scripts/make_figures.py \\
        --results_root D:/Papers/Paper1/results \\
        --benchmark_csv D:/Papers/Paper1/results/benchmark_efficiency.csv \\
        --paper1_figs D:/Papers/Paper1/paper/figures \\
        --paper2_figs D:/Papers/Paper2/paper/figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Shared loaders (mirror fill_tables.py)
# ---------------------------------------------------------------------------

def load_all_results(results_root: Path) -> Dict[str, Dict]:
    merged: Dict[str, Dict] = {}
    for j in sorted(results_root.rglob("evaluation_results.json"),
                    key=lambda p: p.stat().st_mtime):
        with open(j) as f:
            data = json.load(f)
        for name, res in data.get("results", {}).items():
            res["_group"] = j.parent.parent.name
            merged[name] = res
    return merged


def load_benchmark_csv(path: Optional[Path]) -> Dict[str, Dict]:
    if not path or not Path(path).exists():
        return {}
    out: Dict[str, Dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("Model") or row.get("model")
            if not name:
                continue
            def _f(*keys):
                for k in keys:
                    v = row.get(k)
                    if v not in (None, "", "N/A"):
                        try:
                            return float(v)
                        except ValueError:
                            pass
                return None
            params = _f("Parameters", "params")
            out[name] = {
                "params_M": (params / 1e6) if params and params > 1e4 else params,
                "flops_G": (lambda x: x / 1e9 if x else None)(_f("FLOPs", "flops")),
                "latency_ms": _f("Inference Time (ms)", "latency_ms"),
                "memory_MB": _f("Memory (MB)", "memory_mb"),
            }
    return out


def _variant(name: str) -> str:
    n = name.lower()
    if n.endswith("_vmamba"): return "vmamba"
    if n.endswith("_mamba2"): return "mamba2"
    if n.endswith("_mamba"):  return "mamba"
    return "base"


def _savefig(fig, out_dirs: List[Path], stem: str):
    for d in out_dirs:
        if d is None:
            continue
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(d / f"{stem}.png", dpi=200, bbox_inches="tight")
        print(f"  wrote {d / stem}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: accuracy vs parameters Pareto (Paper 1 base; Paper 2 all)
# ---------------------------------------------------------------------------

def fig_pareto(results: Dict[str, Dict], out_dirs: List[Path],
               keys: List[str], stem: str, title: str):
    pts = []
    for k in keys:
        r = results.get(k)
        if not r or r.get("dice_mean", 0) < 0.5:
            continue
        p = r.get("params_M")
        d = r.get("dice_mean")
        if p and d:
            pts.append((p, d, k))
    if len(pts) < 3:
        print(f"[skip] {stem}: not enough points")
        return

    pts.sort()
    xs = [p for p, _, _ in pts]
    ys = [d for _, d, _ in pts]

    # Pareto front: max Dice at <= params
    front = []
    best = -1
    for p, d, k in pts:
        if d > best:
            front.append((p, d))
            best = d

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(xs, ys, s=42, c="#3b6ea5", zorder=3, edgecolor="white", linewidth=0.6)
    fx = [p for p, _ in front]
    fy = [d for _, d in front]
    ax.plot(fx, fy, "--", color="#c0392b", lw=1.4, zorder=2, label="Pareto front")
    for p, d, k in pts:
        lab = k.replace("mamba_", "").replace("_", " ")
        ax.annotate(lab, (p, d), fontsize=6.2, xytext=(3, 3),
                    textcoords="offset points", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (M, log scale)")
    ax.set_ylabel("Mean Dice (test)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    _savefig(fig, out_dirs, stem)


# ---------------------------------------------------------------------------
# Figure 2: memory bar chart with OOM markers (Paper 2)
# ---------------------------------------------------------------------------

# Known Mamba-2 SSD OOM configs (no peak-memory data because they never ran)
MAMBA2_OOM = {
    "mamba_unet_v1_mamba2", "mamba_unet_v2_mamba2", "mamba_swin_unet_mamba2",
    "mamba_transunet_mamba2", "mamba_fpn_mamba2",
}
L4_SRAM_KB = 101  # per-block shared memory ceiling


def fig_memory(bench: Dict[str, Dict], results: Dict[str, Dict],
               out_dirs: List[Path]):
    # Use inference peak memory from the benchmark; group/colour by variant.
    rows = []
    for name, b in bench.items():
        mem = b.get("memory_MB")
        if mem is None:
            continue
        rows.append((name, mem, _variant(name)))
    # add the OOM configs as zero-height hatched bars
    for name in MAMBA2_OOM:
        if name not in bench:
            rows.append((name, 0.0, "mamba2"))
    if len(rows) < 5:
        print("[skip] fig_memory: benchmark CSV missing or too small")
        return

    rows.sort(key=lambda x: (x[2], x[1]))
    colors = {"base": "#7f8c8d", "mamba": "#2980b9",
              "mamba2": "#27ae60", "vmamba": "#8e44ad"}
    fig, ax = plt.subplots(figsize=(11, 4.6))
    xs = range(len(rows))
    for i, (name, mem, var) in enumerate(rows):
        is_oom = name in MAMBA2_OOM
        ax.bar(i, mem if not is_oom else max(r[1] for r in rows) * 0.04,
               color=colors[var],
               hatch="////" if is_oom else None,
               edgecolor="#c0392b" if is_oom else "none",
               linewidth=1.0 if is_oom else 0,
               alpha=0.55 if is_oom else 0.9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r[0].replace("mamba_", "").replace("_", " ") for r in rows],
                       rotation=90, fontsize=6)
    ax.set_ylabel("Peak inference memory (MB)")
    ax.set_title("Per-configuration GPU memory (NVIDIA L4). "
                 "Hatched red = Mamba-2 SSD kernel OOM at train time.")
    # legend
    from matplotlib.patches import Patch
    handles = [Patch(color=colors[v], label=v) for v in
               ["base", "mamba", "mamba2", "vmamba"]]
    handles.append(Patch(facecolor="white", edgecolor="#c0392b", hatch="////",
                         label="SRAM OOM (train)"))
    ax.legend(handles=handles, fontsize=7, ncol=5, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
    _savefig(fig, out_dirs, "fig_memory")


# ---------------------------------------------------------------------------
# Figure 3: per-class Dice radar for the 3 SSM variants (Paper 2)
# ---------------------------------------------------------------------------

def fig_radar(results: Dict[str, Dict], out_dirs: List[Path]):
    # Best model per variant by dice_mean
    best = {}
    for name, r in results.items():
        v = _variant(name)
        if v == "base" or r.get("dice_mean", 0) < 0.5:
            continue
        if v not in best or r["dice_mean"] > best[v][1].get("dice_mean", 0):
            best[v] = (name, r)
    if len(best) < 2:
        print("[skip] fig_radar: insufficient variant data")
        return

    axes_labels = ["LV-endo", "LV-epi", "LA", "mean Dice",
                   "EF $r$", "1$-$HD95/3"]
    def vec(r):
        ef = r.get("ef_metrics") or {}
        hd = r.get("hd95_mean", 3.0)
        return [
            r.get("dice_lv_endocardium", 0),
            r.get("dice_lv_epicardium", 0),
            r.get("dice_left_atrium", 0),
            r.get("dice_mean", 0),
            ef.get("ef_correlation", 0),
            max(0.0, 1 - hd / 3.0),
        ]

    N = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.4, 5.4), subplot_kw=dict(polar=True))
    colors = {"mamba": "#2980b9", "mamba2": "#27ae60", "vmamba": "#8e44ad"}
    for v in ["mamba", "mamba2", "vmamba"]:
        if v not in best:
            continue
        name, r = best[v]
        vals = vec(r)
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[v], lw=1.8,
                label=f"{v}: {name.replace('mamba_','').replace('_',' ')}")
        ax.fill(angles, vals, color=colors[v], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Best model per SSM variant", fontsize=10, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=7)
    _savefig(fig, out_dirs, "fig_radar")


# ---------------------------------------------------------------------------
# Figure 4: base -> Mamba Dice delta heatmap (Paper 2)
# ---------------------------------------------------------------------------

# architecture -> (base_key, {variant: mamba_key})
ARCH_MAP = {
    "UNet-V1":     ("unet_v1", {"mamba": "mamba_unet_v1_mamba",
                                 "mamba2": "mamba_unet_v1_mamba2",
                                 "vmamba": "mamba_unet_v1_vmamba"}),
    "UNet-V2":     ("unet_v2", {"mamba": "mamba_unet_v2_mamba",
                                 "mamba2": "mamba_unet_v2_mamba2",
                                 "vmamba": "mamba_unet_v2_vmamba"}),
    "UNet-ResNet": ("unet_resnet", {"mamba": "mamba_unet_resnet_mamba",
                                     "mamba2": "mamba_unet_resnet_mamba2",
                                     "vmamba": "mamba_unet_resnet_vmamba"}),
    "DeepLabV3+":  ("deeplab_v3", {"mamba": "mamba_deeplab_mamba",
                                    "mamba2": "mamba_deeplab_mamba2",
                                    "vmamba": "mamba_deeplab_vmamba"}),
    "nnU-Net":     ("nnunet", {"mamba": "mamba_nnunet_mamba",
                                "mamba2": "mamba_nnunet_mamba2",
                                "vmamba": "mamba_nnunet_vmamba"}),
    "DenseCtx":    ("dense_context_unet", {"mamba": "mamba_dense_context_unet_mamba",
                                            "mamba2": "mamba_dense_context_unet_mamba2",
                                            "vmamba": "mamba_dense_context_unet_vmamba"}),
    "FPN-UNet":    ("fpn", {"mamba": "mamba_fpn_mamba",
                             "mamba2": "mamba_fpn_mamba2",
                             "vmamba": "mamba_fpn_vmamba"}),
    "Swin-UNet":   ("swin_unet", {"mamba": "mamba_swin_unet_mamba",
                                   "mamba2": "mamba_swin_unet_mamba2",
                                   "vmamba": "mamba_swin_unet_vmamba"}),
    "TransUNet":   ("transunet", {"mamba": "mamba_transunet_mamba",
                                   "mamba2": "mamba_transunet_mamba2",
                                   "vmamba": "mamba_transunet_vmamba"}),
}


def fig_dice_delta_heatmap(results: Dict[str, Dict], out_dirs: List[Path]):
    variants = ["mamba", "mamba2", "vmamba"]
    archs = list(ARCH_MAP.keys())
    M = np.full((len(archs), len(variants)), np.nan)
    for i, a in enumerate(archs):
        base_key, vmap = ARCH_MAP[a]
        base = results.get(base_key, {}).get("dice_mean")
        for j, v in enumerate(variants):
            mk = vmap.get(v)
            mr = results.get(mk, {})
            md = mr.get("dice_mean")
            if base is not None and md is not None:
                M[i, j] = md - base

    if np.all(np.isnan(M)):
        print("[skip] fig_dice_delta_heatmap: no paired data")
        return

    fig, ax = plt.subplots(figsize=(5.0, 5.6))
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(["Mamba/S6", "Mamba-2", "VMamba"], fontsize=9)
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels(archs, fontsize=9)
    for i in range(len(archs)):
        for j in range(len(variants)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(M[i, j]) > vmax * 0.6 else "black")
            else:
                ax.text(j, i, "--", ha="center", va="center",
                        fontsize=8, color="#888")
    ax.set_title("Dice change vs base CNN\n(blue = Mamba helps, red = hurts)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$\\Delta$ Dice")
    _savefig(fig, out_dirs, "fig_dice_delta_heatmap")


# ---------------------------------------------------------------------------
# Figure 5: critical-difference diagram (per-sample Dice)
# ---------------------------------------------------------------------------

def _nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    # q_alpha values for the two-tailed Nemenyi test at alpha=0.05
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
           8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
           14: 3.354, 15: 3.391, 16: 3.426}
    q = q05.get(k, 3.426)
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def fig_cd_diagram(results: Dict[str, Dict], out_dirs: List[Path],
                   keys: List[str], stem: str, top_n: int = 15):
    # Gather models with per_sample_dice
    have = [(k, np.asarray(results[k]["per_sample_dice"]))
            for k in keys
            if results.get(k, {}).get("per_sample_dice")]
    # keep only those sharing the same N
    if not have:
        print(f"[skip] {stem}: no per_sample_dice arrays in JSON "
              f"(add them in evaluate_all_models.py)")
        return
    n_common = min(len(v) for _, v in have)
    have = [(k, v[:n_common]) for k, v in have]
    # rank by mean Dice, keep top_n
    have.sort(key=lambda kv: -kv[1].mean())
    have = have[:top_n]
    names = [k for k, _ in have]
    mat = np.vstack([v for _, v in have])  # (k, n_samples)
    k, n = mat.shape
    if k < 3:
        print(f"[skip] {stem}: need >=3 models with per-sample Dice")
        return

    # average ranks (higher Dice = rank 1)
    # rank per sample across models
    ranks = np.zeros_like(mat)
    for s in range(n):
        order = (-mat[:, s]).argsort()
        r = np.empty(k)
        r[order] = np.arange(1, k + 1)
        ranks[:, s] = r
    avg_rank = ranks.mean(axis=1)
    cd = _nemenyi_cd(k, n)

    # Plot CD diagram
    order = avg_rank.argsort()
    names = [names[i] for i in order]
    avg_rank = avg_rank[order]

    fig, ax = plt.subplots(figsize=(8, 0.4 * k + 1.5))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, k + 1)
    ax.invert_yaxis()
    for i, (nm, rk) in enumerate(zip(names, avg_rank)):
        y = i + 1
        ax.plot([lo - 0.5, rk], [y, y], color="#bbb", lw=0.7, zorder=1)
        ax.scatter([rk], [y], s=30, color="#3b6ea5", zorder=2)
        ax.text(lo - 0.6, y, nm.replace("mamba_", "").replace("_", " "),
                ha="right", va="center", fontsize=7)
        ax.text(rk + 0.05, y, f"{rk:.2f}", va="center", fontsize=6.5,
                color="#555")
    # CD bar
    ax.plot([lo, lo + cd], [0.4, 0.4], color="#c0392b", lw=2.5)
    ax.text(lo + cd / 2, 0.2, f"CD = {cd:.2f}", ha="center", fontsize=8,
            color="#c0392b")
    ax.set_xlabel("Average rank (lower = better)")
    ax.set_yticks([])
    ax.set_title("Critical-difference diagram (per-sample Dice, Nemenyi $\\alpha$=0.05)",
                 fontsize=10)
    ax.grid(True, axis="x", alpha=0.2)
    _savefig(fig, out_dirs, stem)


# ---------------------------------------------------------------------------
# Figure 6: Bland-Altman (Paper 1) -- needs per-patient EF arrays
# ---------------------------------------------------------------------------

def fig_bland_altman(results: Dict[str, Dict], out_dirs: List[Path],
                     model_keys: List[str]):
    panels = []
    for k in model_keys:
        r = results.get(k, {})
        ef = r.get("ef_metrics") or {}
        pred = ef.get("ef_pred") or ef.get("ef_predicted")
        gt = ef.get("ef_true") or ef.get("ef_ground_truth")
        if pred and gt and len(pred) == len(gt):
            panels.append((k, np.asarray(pred), np.asarray(gt)))
    if not panels:
        print("[skip] fig_bland_altman: per-patient EF arrays not in JSON.\n"
              "       To enable, have evaluate_all_models.py store\n"
              "       results[model]['ef_metrics']['ef_pred'] and ['ef_true'].")
        return

    n = len(panels)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(9, 3.6 * rows), squeeze=False)
    for idx, (k, pred, gt) in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        mean = (pred + gt) / 2
        diff = pred - gt
        bias = diff.mean()
        sd = diff.std()
        ax.scatter(mean, diff, s=18, alpha=0.6, color="#3b6ea5")
        ax.axhline(bias, color="#c0392b", lw=1.3, label=f"bias {bias:+.2f}")
        ax.axhline(bias + 1.96 * sd, color="#c0392b", ls="--", lw=1.0)
        ax.axhline(bias - 1.96 * sd, color="#c0392b", ls="--", lw=1.0)
        ax.set_title(k.replace("mamba_", "").replace("_", " "), fontsize=9)
        ax.set_xlabel("Mean EF (%)")
        ax.set_ylabel("Predicted $-$ GT EF (%)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    _savefig(fig, out_dirs, "fig_bland_altman")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

P1_BASE = ["unet_v1", "unet_v2", "unet_resnet", "deeplab_v3", "nnunet",
           "dense_context_unet", "fpn", "swin_unet", "transunet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=Path, required=True)
    ap.add_argument("--benchmark_csv", type=Path, default=None)
    ap.add_argument("--paper1_figs", type=Path, default=None)
    ap.add_argument("--paper2_figs", type=Path, default=None)
    args = ap.parse_args()

    results = load_all_results(args.results_root)
    bench = load_benchmark_csv(args.benchmark_csv)
    print(f"[make_figures] {len(results)} models, {len(bench)} benchmark rows")

    p1 = [args.paper1_figs] if args.paper1_figs else []
    p2 = [args.paper2_figs] if args.paper2_figs else []
    all_keys = list(results.keys())

    print("\n== Paper 1 figures ==")
    fig_pareto(results, p1, P1_BASE, "fig_pareto",
               "Accuracy vs. capacity (9 base architectures)")
    fig_bland_altman(results, p1,
                     ["nnunet", "unet_v1", "unet_resnet", "transunet"])
    fig_cd_diagram(results, p1, P1_BASE, "fig_cd_diagram", top_n=9)

    print("\n== Paper 2 figures ==")
    fig_memory(bench, results, p2)
    fig_radar(results, p2)
    fig_dice_delta_heatmap(results, p2)
    fig_pareto(results, p2, all_keys, "fig_pareto_all",
               "Accuracy vs. capacity (all configurations)")
    fig_cd_diagram(results, p2, all_keys, "fig_cd_diagram_mamba", top_n=15)

    print("\n[make_figures] done.")


if __name__ == "__main__":
    main()
