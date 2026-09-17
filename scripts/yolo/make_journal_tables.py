#!/usr/bin/env python3
"""
Tables for the journal-length paper (paper3/).

The ISBI version had to omit most of what was measured. This emits the fuller
set into paper3/tables:

    Y1_main.tex      as in the short paper (copied through aggregate_results)
    Y2_ablation.tex  the FULL ablation grid, all factors, not the paper2 subset
    Y3_ceiling.tex   encoding ceiling, all vertex budgets
    Y5_latency.tex   every model at 256 and 640 px under one protocol
    Y6_ef.tex        ejection fraction, all baselines plus the oracle floor

Kept separate from aggregate_results.py so the short paper's pipeline is not
disturbed.

Usage:
    python scripts/yolo/make_journal_tables.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "yolo"))

import aggregate_results as A  # noqa: E402

BASE_NAME = {
    "transunet": "TransUNet", "nnunet": "nnU-Net", "unet_v1": "UNet-V1",
    "unet_resnet": "UNet-ResNet", "unet_v2": "UNet-V2",
    "swin_unet": "Swin-UNet", "deeplab_v3": "DeepLabV3+", "fpn": "FPN-UNet",
    "dense_context_unet": "DenseCtx-UNet",
}


def fmt(v, nd=2, dash="--"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    return f"{v:.{nd}f}"


def table_latency(path: Path, out: Path) -> None:
    if not path.exists():
        out.write_text("% latency_fair.json not found\n")
        return
    d = json.loads(path.read_text())
    L = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Latency under a single protocol: batch 1, CUDA-synchronised, "
        "median of 50 iterations after 10 warm-up, on one RTX~4060 Laptop GPU. "
        "Every model is timed at both resolutions so that input size is "
        "separated from architecture; no accuracy is claimed at the off-design "
        "size. Swin-UNet is shape-constrained and is timed at 224/448.}",
        "\\label{tab:latency}", "\\small",
        "\\begin{tabular}{lrrr}", "\\toprule",
        "Model & @256\\,px & @640\\,px & ratio \\\\",
        "\\midrule",
    ]
    rows = []
    for k, v in d.get("baselines", {}).items():
        lo = v.get("raw_fwd_256") or v.get("raw_fwd_224")
        hi = v.get("raw_fwd_640") or v.get("raw_fwd_448")
        if not (lo and hi):
            continue
        rows.append((BASE_NAME.get(k, k), lo["median_ms"], hi["median_ms"]))
    rows.sort(key=lambda r: r[2])

    y = d.get("yolo", {})
    if y.get("raw_fwd_256") and y.get("raw_fwd_640"):
        rows.insert(0, ("\\textbf{YOLO26n-seg}",
                        y["raw_fwd_256"]["median_ms"], y["raw_fwd_640"]["median_ms"]))
    for name, lo, hi in rows:
        L.append(f"{name} & {fmt(lo)} & {fmt(hi)} & {fmt(hi / lo, 1)}$\\times$ \\\\")

    extra = []
    for key, lab in (("e2e_640", "end-to-end @640"),
                     ("raw_fwd_fp16_640", "FP16 @640"),
                     ("onnx_640", "ONNX Runtime @640")):
        if key in y:
            extra.append(f"\\quad {lab} & & {fmt(y[key]['median_ms'])} & \\\\")
    if extra:
        L += ["\\midrule",
              "\\multicolumn{4}{l}{\\textit{YOLO26n-seg, deployment paths}} \\\\"] + extra
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    out.write_text("\n".join(L))


def table_ef(base: Path, oracle: Path, yolo: Path, out: Path) -> None:
    if not base.exists():
        out.write_text("% baseline_ef_native.json not found\n")
        return
    d = json.loads(base.read_text())
    # Keys beginning with "_" are provenance metadata, not models: the EF
    # artefacts carry a "_determinism" block recording the seed and precision
    # settings that produced them. Rendering it as a row emits a bare
    # underscore into LaTeX and fails the build.
    rows = [(BASE_NAME.get(k, k), v.get("ef_mae"), v.get("ef_bias"),
             v.get("ef_correlation"), v.get("edv_mean"))
            for k, v in d.items()
            if not k.startswith("_") and isinstance(v, dict)
            and v.get("ef_mae") is not None]
    if yolo.exists():
        s = json.loads(yolo.read_text())["stats"]
        rows.append(("\\textbf{YOLO26n-seg}", s.get("ef_mae"), s.get("ef_bias"),
                     s.get("ef_correlation"), s.get("edv_mean")))
    rows.sort(key=lambda r: r[1] if r[1] is not None else 9e9)

    L = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Ejection fraction against the clinically recorded value, "
        "computed with the official CAMUS biplane Simpson implementation at "
        "native resolution with a largest-component filter (Sec.~\\ref{sec:ef}). "
        "The oracle row applies the identical pipeline to the ground-truth masks "
        "and bounds what any segmentation can achieve.}",
        "\\label{tab:ef}", "\\small",
        "\\begin{tabular}{lrrrr}", "\\toprule",
        "Model & MAE (\\%) & bias (\\%) & $r$ & EDV (ml) \\\\", "\\midrule",
    ]
    for n, mae, bias, r, edv in rows:
        L.append(f"{n} & {fmt(mae)} & {fmt(bias)} & {fmt(r, 3)} & {fmt(edv, 1)} \\\\")
    if oracle.exists():
        s = json.loads(oracle.read_text())["stats"]
        L += ["\\midrule",
              f"\\textit{{oracle (ground-truth masks)}} & {fmt(s.get('ef_mae'))} & "
              f"{fmt(s.get('ef_bias'))} & {fmt(s.get('ef_correlation'), 3)} & "
              f"{fmt(s.get('edv_mean'), 1)} \\\\"]
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    out.write_text("\n".join(L))


SEED_RUNS = ["E9_seed0", "E9_seed1", "E9_seed2"]
MISSING_COLS = ["n_missing_lv_endocardium", "n_missing_lv_epicardium",
                "n_missing_left_atrium"]


def table_seed_floor(csv_path: Path, out: Path) -> None:
    """The noise floor: N trainings of one configuration differing only in seed.

    Previously typed by hand into paper3/sections/04_results.tex. It reports
    both the range and the standard deviation on purpose. The range is the
    conservative attribution threshold the section actually uses, but a range
    grows with sample size, so it cannot be compared against the spread of a
    larger set of configurations. Any such comparison must use the SD.
    """
    import csv as _csv
    import statistics as _st

    if not csv_path.exists():
        out.write_text("% all_runs.csv not found\n")
        return
    rows = {r["run"]: r for r in _csv.DictReader(open(csv_path))}
    seeds = [rows[k] for k in SEED_RUNS if k in rows]
    if len(seeds) < 2:
        out.write_text("% fewer than two seed runs in all_runs.csv\n")
        return

    dice = [float(r["dice_mean"]) for r in seeds]
    hd95 = [float(r["hd95_mean"]) for r in seeds]
    miss = [sum(int(r[c] or 0) for c in MISSING_COLS) for r in seeds]

    L = [
        "%=========================================================================",
        "% Paper 3 / Y7 - Seed noise floor (auto-generated)",
        "%   <- results/yolo/all_runs.csv, runs " + ", ".join(SEED_RUNS),
        "%=========================================================================",
        "\\begin{table}[t]", "\\centering",
        "\\caption{%d trainings of one identical configuration differing only in "
        "random seed. The range bounds the smallest difference that can be "
        "credibly attributed to a design choice and is used as that threshold "
        "throughout. The standard deviation is given because a range grows with "
        "the number of runs, so only the SD may be compared against the spread "
        "of a larger set of configurations.}" % len(seeds),
        "\\label{tab:seeds}",
        "\\begin{tabular}{lccc}", "\\toprule",
        "& Dice & HD95 (mm) & missed frames \\\\", "\\midrule",
    ]
    for i, r in enumerate(seeds):
        L.append(f"seed {i} & {dice[i]:.4f} & {hd95[i]:.3f} & {miss[i]} \\\\")
    L.append("\\midrule")
    L.append(f"range  & {max(dice) - min(dice):.4f} & "
             f"\\textbf{{{max(hd95) - min(hd95):.3f}}} & "
             f"{max(miss) - min(miss)} \\\\")
    L.append(f"SD     & {_st.stdev(dice):.4f} & {_st.stdev(hd95):.3f} & "
             f"{_st.stdev(miss):.1f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper3" / "tables"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = ROOT / "results" / "yolo"

    # Y1 / Y3 / Y2 via the shared generator, but with EVERY group in the grid.
    A.PAPER_GROUPS = [
        "E1_encoding", "E8_vertices", "E10_ceiling_test", "E2_scale", "E3_imgsz",
        "E4_maskratio", "E5_aug", "E6_pretrain", "E7_data", "E12_boundary",
        "E13_optim", "E9_final",
    ]
    sys.argv = ["aggregate_results", "--out", str(out)]
    A.main()

    table_latency(res / "latency_fair.json", out / "Y5_latency.tex")
    table_ef(res / "baseline_ef_native.json",
             res / "_EF_ORACLE" / "ef_oracle.json",
             res / "E1_filled" / "ef.json",
             out / "Y6_ef.tex")
    table_seed_floor(ROOT / "results" / "yolo" / "all_runs.csv",
                     out / "Y7_seedfloor.tex")
    print(f"Wrote {out / 'Y5_latency.tex'}")
    print(f"Wrote {out / 'Y6_ef.tex'}")
    print(f"Wrote {out / 'Y7_seedfloor.tex'}")


if __name__ == "__main__":
    main()
