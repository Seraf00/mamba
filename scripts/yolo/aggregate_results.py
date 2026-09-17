#!/usr/bin/env python3
"""
Collect every YOLO run, compare against the semantic-segmentation baselines,
and emit the LaTeX tables for the ISBI paper.

Outputs (paper2/tables/):
    Y1_main.tex        YOLO vs the nine baselines: Dice, HD95, params, FPS
    Y2_ablation.tex    the E1-E8 ablation grid
    Y3_ceiling.tex     polygon encoding ceiling
    Y4_stats.tex       paired Wilcoxon of the best YOLO vs each baseline
    summary.csv        every run, flat

Per-sample pairing with the baselines is index-based. The baseline evaluator
stored `per_sample_dice` in dataset order (patients sorted, then 2CH/4CH, then
ED/ES) without identifiers; this script rebuilds that order and VERIFIES it by
checking that the ED and ES subset means reproduce each baseline's reported
`dice_mean_ed`/`dice_mean_es`. If the check fails the stats table is skipped
rather than silently mispaired.

Usage:
    python scripts/yolo/aggregate_results.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

BASELINE_LABEL = {
    "transunet": ("TransUNet", "Hybrid"),
    "nnunet": ("nnU-Net", "CNN"),
    "unet_v1": ("UNet-V1", "CNN"),
    "unet_resnet": ("UNet-ResNet", "CNN"),
    "unet_v2": ("UNet-V2", "CNN"),
    "swin_unet": ("Swin-UNet", "Transformer"),
    "deeplab_v3": ("DeepLabV3+", "CNN"),
    "fpn": ("FPN-UNet", "CNN"),
    "dense_context_unet": ("DenseContextU-Net", "CNN"),
}

GROUP_TITLE = {
    "E1_encoding": "Annulus encoding",
    "E2_scale": "Model scale / generation",
    "E3_imgsz": "Input resolution",
    "E4_maskratio": "Mask prototype resolution",
    "E5_aug": "Augmentation recipe",
    "E6_pretrain": "COCO pretraining",
    "E7_data": "Training data",
    "E8_vertices": "Polygon vertex budget",
    "E9_final": "Final configuration",
    "E10_ceiling_test": "Encoding at 32 vertices",
    "E11_best": "Composed best configuration",
    "E12_boundary": "Boundary-weighted loss",
    "E13_optim": "Loss gains and optimiser",
    "inference": "Inference-time settings",
}

RUN_LABEL = {
    "E1_filled": "Filled contour (ours)", "E1_seam": "Seam-cut annulus",
    "E2_v26n": "YOLO26n-seg", "E2_v26s": "YOLO26s-seg", "E2_v26m": "YOLO26m-seg",
    "E2_v11n": "YOLO11n-seg", "E2_v8n": "YOLOv8n-seg",
    "E3_sz512": "512 px", "E3_sz640": "640 px", "E3_sz800": "800 px",
    "E3_sz960": "960 px",
    "E4_mr4": "1/4 resolution", "E4_mr2": "1/2 resolution", "E4_mr1": "full resolution",
    "E5_echo": "Echo-aware", "E5_coco": "Stock COCO", "E5_none": "None",
    "E5_echo_fliplr": "Echo-aware + h-flip",
    "E6_pretrained": "COCO-pretrained", "E6_scratch": "From scratch",
    "E7_edes": "ED/ES only (1.6k)", "E7_seq": "+ cycle frames (~20k)",
    "E8_n96": "96 vertices", "E8_n32": "32 vertices",
    "E9_seed0": "seed 0", "E9_seed1": "seed 1", "E9_seed2": "seed 2",
    "I1_conf10": "conf 0.10", "I1_conf25": "conf 0.25", "I1_conf50": "conf 0.50",
    "I2_tta": "+ TTA", "I3_half": "FP16",
    "E10_filled_n32": "Filled, 32 vtx", "E10_seam_n32": "Seam, 32 vtx",
    "E11_best_seed0": "seed 0", "E11_best_seed1": "seed 1",
    "E11_best_seed2": "seed 2",
    "E12_bw0": "$\lambda=0$ (stock)", "E12_bw20": "$\lambda=2$",
    "E12_bw50": "$\lambda=5$", "E12_bw100": "$\lambda=10$",
    "E13_box_half": "box $\times$0.5", "E13_box_double": "box $\times$2",
    "E13_cls_double": "cls $\times$2", "E13_dfl_double": "dfl $\times$2",
    "E13_adamw": "AdamW", "E13_sgd": "SGD",
}


# Groups shown in the paper's ablation table. E9/E11 are seed repeats of the
# default (their spread is quoted in the text as the noise floor, so listing
# them as ablation rows would be redundant), and the inference sweeps are
# discussed in prose. Everything still lands in results/yolo/summary.csv.
PAPER_GROUPS = [
    "E1_encoding", "E8_vertices", "E10_ceiling_test", "E2_scale", "E3_imgsz",
    "E4_maskratio", "E5_aug", "E6_pretrain", "E7_data", "E12_boundary",
]


def tex_escape(t: str) -> str:
    """Escape characters that break LaTeX when a label falls back to a raw id."""
    if "$" in t or "\\" in t:          # already contains deliberate markup
        return t
    for a, b in (("_", "\_"), ("&", "\&"), ("%", "\%"), ("#", "\#")):
        t = t.replace(a, b)
    return t


# ---------------------------------------------------------------- loading ---
def load_yolo_runs(res_dir: Path) -> dict[str, dict]:
    runs = {}
    for d in sorted(res_dir.iterdir()):
        f = d / "evaluation.json"
        if d.is_dir() and f.exists():
            runs[d.name] = json.loads(f.read_text())
    return runs


def load_baselines(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())["results"]


def test_stems(splits_dir: Path) -> list[str]:
    ids = [ln.strip() for ln in (splits_dir / "test.txt").read_text().splitlines() if ln.strip()]
    return [f"{p}_{v}_{ph}" for p in ids for v in ("2CH", "4CH") for ph in ("ED", "ES")]


def verify_pairing(baselines: dict, stems: list[str]) -> bool:
    """Confirm the reconstructed sample order matches the stored per-sample list."""
    ed = [i for i, s in enumerate(stems) if s.endswith("_ED")]
    es = [i for i, s in enumerate(stems) if s.endswith("_ES")]
    for name, r in baselines.items():
        ps = np.asarray(r.get("per_sample_dice", []))
        if len(ps) != len(stems):
            return False
        if not np.isclose(ps[ed].mean(), r["dice_mean_ed"], atol=1e-3):
            return False
        if not np.isclose(ps[es].mean(), r["dice_mean_es"], atol=1e-3):
            return False
    return True


# ----------------------------------------------------------------- tables ---
def placeholder(caption: str, label: str) -> str:
    """A table stub so the paper still compiles before the sweep has run."""
    return (
        "% auto-generated placeholder -- rerun aggregate_results.py after the sweep\n"
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption} -- pending the GPU sweep.}}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{c}\n\\toprule\n"
        "\\textcolor{red}{\\textbf{[results pending]}} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def fmt(v, nd=4, dash="--"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    return f"{v:.{nd}f}"


# baseline key -> row name in results/benchmark_efficiency.csv
EFF_ALIAS = {"unet_v1": "unet"}


def load_efficiency(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["Model"]] = float(row["Inference Time (ms)"])
            except (KeyError, ValueError):
                pass
    return out


# Seeds of the final configuration. Significance is computed on the mean over
# these rather than on any single run: the estimand is the difference between
# configurations, averaged over the seed distribution, not between two
# particular fitted models.
SEED_RUNS = ["E9_seed0", "E9_seed1", "E9_seed2"]


def _to_patients(frame_vals: np.ndarray) -> np.ndarray:
    """Collapse 200 frames to 50 patient means.

    The frames are 50 patients x 2 views x 2 phases, ordered patient-major in
    blocks of four. Treating them as independent inflates n fourfold and makes
    differences smaller than the seed floor testable -- which is exactly the
    practice this paper argues against, so it cannot be the practice this paper
    uses.
    """
    a = np.asarray(frame_vals, dtype=float)
    return a.reshape(-1, 4).mean(axis=1) if a.size % 4 == 0 else a


def paired_dice(res_dir: Path, best: str, stems: list[str]) -> np.ndarray | None:
    """Patient-level mean Dice, averaged over the available seed runs.

    Falls back to `best` alone if the seed runs are not on disk, so the
    function still works on a partial results tree.
    """
    runs = [r for r in SEED_RUNS if (Path(res_dir) / r / "per_sample.json").exists()]
    if not runs:
        runs = [best]
    per_run = []
    for run in runs:
        ps = Path(res_dir) / run / "per_sample.json"
        if not ps.exists():
            return None
        recs = {r["stem"]: r for r in json.loads(ps.read_text())}
        if not all(s in recs for s in stems):
            return None
        per_run.append(_to_patients(np.asarray([recs[s]["dice_mean"] for s in stems])))
    return np.mean(per_run, axis=0)


def compute_sig(res_dir: Path, baselines: dict, best: str,
                stems: list[str]) -> dict[str, float]:
    """Holm-corrected paired Wilcoxon p-values, best YOLO vs each baseline."""
    from scipy.stats import wilcoxon

    yolo = paired_dice(res_dir, best, stems)
    if yolo is None or not verify_pairing(baselines, stems):
        return {}

    raw: list[tuple[str, float]] = []
    for key in BASELINE_LABEL:
        if key not in baselines:
            continue
        b = _to_patients(np.asarray(baselines[key]["per_sample_dice"]))
        try:
            _, p = wilcoxon(yolo, b)
        except ValueError:          # identical vectors -> no difference to test
            p = 1.0
        raw.append((key, float(p)))

    order = sorted(range(len(raw)), key=lambda i: raw[i][1])
    out, running, m = {}, 0.0, len(raw)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * raw[i][1]))
        out[raw[i][0]] = running
    return out


def table_main(runs: dict, baselines: dict, best: str, out: Path,
               eff: dict[str, float] | None = None,
               sig: dict[str, float] | None = None) -> None:
    """`sig` maps baseline key -> Holm-corrected p vs the best YOLO run."""
    eff = eff or {}
    sig = sig or {}
    rows = []
    for key, (label, para) in BASELINE_LABEL.items():
        if key in baselines:
            r = baselines[key]
            mark = ""
            if key in sig:
                mark = "$^{*}$" if sig[key] < 0.05 else "$^{\\dagger}$"
            rows.append((label + mark, para, r.get("dice_mean"), r.get("hd95_mean"),
                         r.get("params_M"), False))

    # One row for the detector, averaged over seeds. Reporting the best of three
    # would be selecting on the outcome, in a paper whose own noise-floor section
    # argues against exactly that.
    seeds = [runs[n] for n in sorted(runs) if n.startswith("E9_")]
    if seeds:
        ds = [s["dice_mean"] for s in seeds if s.get("dice_mean") is not None]
        hs = [s["hd95_mean"] for s in seeds if s.get("hd95_mean") is not None]
        rows.append((f"YOLO26-seg (ours, mean of {len(seeds)} seeds)", "Detector",
                     float(np.mean(ds)) if ds else None,
                     float(np.mean(hs)) if hs else None,
                     seeds[0].get("params_M"), True))
    rows.sort(key=lambda x: -(x[2] if x[2] is not None else -1))

    best_d = max((r[2] for r in rows if r[2] is not None), default=None)
    best_h = min((r[3] for r in rows if r[3] is not None and np.isfinite(r[3])), default=None)

    L = [
        "%" + "=" * 76,
        "% Y1 -- YOLO vs semantic-segmentation baselines (auto-generated)",
        "%" + "=" * 76,
        "\\begin{table*}[t]", "\\centering",
        "\\caption{CAMUS official test split (50 patients). Dice on the "
        "$256\\times256$ grid, HD95 at native resolution in mm. The detector row "
        "is the mean over three seeds, not the best of them. Best per column in "
        "\\textbf{bold}. Markers: Holm-corrected paired Wilcoxon over patients "
        "against that mean, $^{*}$ $p<0.05$, $^{\\dagger}$ not significant.}",
        "\\label{tab:main}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l l c c c}", "\\toprule",
        "Model & Paradigm & Dice & HD95 (mm) & Params (M) \\\\",
        "\\midrule",
    ]
    for label, para, d, h, p, is_yolo in rows:
        ds = fmt(d)
        hs = fmt(h, 2)
        if best_d is not None and d is not None and np.isclose(d, best_d):
            ds = f"\\textbf{{{ds}}}"
        if best_h is not None and h is not None and np.isclose(h, best_h):
            hs = f"\\textbf{{{hs}}}"
        L.append(f"{label:26s} & {para:11s} & {ds} & {hs} & {fmt(p, 1)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    out.write_text("\n".join(L))


def table_ablation(runs: dict, plan_groups: dict, out: Path) -> None:
    """
    Ablation grid laid out as two side-by-side panels.

    A single-column list of every run comes to roughly 35 rows, which eats most
    of a column in a 4-page paper. Dealing the blocks into two panels of three
    columns each halves the height for identical content.
    """
    blocks: list[list[str]] = []
    for gid in [g for g in PAPER_GROUPS if g in plan_groups]:
        names = plan_groups[gid]
        present = [n for n in names if n in runs]
        if not present:
            continue
        lines = ["\\textit{" + GROUP_TITLE.get(gid, gid) + "} & & \\\\"]
        vals = [runs[n].get("dice_mean") for n in present]
        bd = max((v for v in vals if v is not None), default=None)
        for n in present:
            r = runs[n]
            d = r.get("dice_mean")
            ds = fmt(d)
            if bd is not None and d is not None and np.isclose(d, bd):
                ds = "\\textbf{" + ds + "}"
            lines.append("\\quad " + RUN_LABEL.get(n, n) + " & " + ds + " & "
                         + fmt(r.get("hd95_mean"), 2) + " \\\\")
        blocks.append(lines)

    if not blocks:
        out.write_text(placeholder("Ablations", "tab:ablation"))
        return

    # split so the two panels come out as close to equal height as possible
    total = sum(len(b) for b in blocks)
    left, acc = [], 0
    for b in blocks:
        if not left or acc + len(b) <= (total + 1) // 2:
            left.append(b)
            acc += len(b)
        else:
            break
    right = blocks[len(left):]

    lcol = [ln for b in left for ln in b]
    rcol = [ln for b in right for ln in b]
    n = max(len(lcol), len(rcol), 1)
    blank = " & & \\\\"
    lcol += [blank] * (n - len(lcol))
    rcol += [blank] * (n - len(rcol))

    L = [
        "%" + "=" * 76,
        "% Y2 -- ablation grid (auto-generated)",
        "%" + "=" * 76,
        "\\begin{table}[t]", "\\centering",
        "\\caption{Ablations on the CAMUS test split. Each block varies one "
        "factor, all others held at the default configuration (YOLO26n-seg, "
        "640\\,px, filled-contour encoding, echo-aware augmentation, "
        "COCO-pretrained). D: mean Dice; H: HD95 in mm. Best per block in "
        "\\textbf{bold}.}",
        "\\label{tab:ablation}", "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{@{}lcc@{\\hspace{5pt}}lcc@{}}", "\\toprule",
        "Setting & D & H & Setting & D & H \\\\", "\\midrule",
    ]
    for a, b in zip(lcol, rcol):
        L.append(a[:-2].rstrip() + " & " + b.lstrip())
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    out.write_text("\n".join(L))


def table_ceiling(path: Path, out: Path, keep: set[str] | None = None) -> None:
    if not path.exists():
        out.write_text(placeholder("Encoding ceiling", "tab:ceiling"))
        return
    d = json.loads(path.read_text())
    L = [
        "%" + "=" * 76,
        "% Y3 -- polygon encoding ceiling (auto-generated)",
        "%" + "=" * 76,
        "\\begin{table}[t]", "\\centering",
        "\\caption{Polygon encoding ceiling: ground-truth masks converted to "
        "YOLO polygons and rasterised back, scored against the untouched native "
        "ground truth. These are upper bounds no detector trained on that "
        "encoding can exceed. $n$ is the vertex budget per contour "
        "($n{=}0$ keeps every boundary pixel).}",
        "\\label{tab:ceiling}", "\\small",
        "\\begin{tabular}{l c c c c}", "\\toprule",
        "Encoding & $n$ & Dice & HD95 (mm) & Myo.\\ target \\\\", "\\midrule",
    ]
    enc_label = {"filled": "Filled contour (ours)", "seam": "Seam-cut annulus"}
    last = None
    for key, v in d.items():
        enc, n = key.rsplit("_n", 1)
        if keep is not None and n not in keep:
            continue
        if last is not None and enc != last:
            L.append("\\midrule")
        last = enc
        nlab = "all" if n == "0" else n
        L.append(f"{enc_label.get(enc, enc):24s} & {nlab} & {fmt(v.get('dice_mean'))} & "
                 f"{fmt(v.get('hd95_mean'), 3)} & "
                 f"{fmt(v.get('target_dice_class1'))} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    out.write_text("\n".join(L))


def table_stats(res_dir: Path, baselines: dict, best: str, stems: list[str], out: Path) -> None:
    from scipy.stats import wilcoxon

    if not baselines:
        out.write_text("% per-sample data unavailable\n")
        return
    # Seed mean at the patient level -- see paired_dice for why neither the
    # single best run nor the 200 frames is the right basis.
    yolo = paired_dice(res_dir, best, stems)
    if yolo is None:
        out.write_text("% YOLO per-sample records incomplete\n")
        return
    n_seeds = sum(1 for r in SEED_RUNS
                  if (Path(res_dir) / r / "per_sample.json").exists()) or 1

    L = [
        "%" + "=" * 76,
        "% Y4 -- paired Wilcoxon signed-rank tests (auto-generated)",
        "%" + "=" * 76,
        "\\begin{table}[t]", "\\centering",
        "\\caption{Paired Wilcoxon on mean Dice, final configuration averaged "
        f"over {n_seeds} seeds, tested over the $50$ patients not the $200$ "
        "frames. $\\Delta$ is mean Dice (YOLO $-$ baseline); $p$ "
        "Holm-corrected.}",
        "\\label{tab:stats}", "\\small",
        "\\begin{tabular}{l c c c}", "\\toprule",
        "Baseline & $\\Delta$ Dice & $W$ & $p$ \\\\", "\\midrule",
    ]
    raw = []
    for key, (label, _) in BASELINE_LABEL.items():
        if key not in baselines:
            continue
        b = _to_patients(np.asarray(baselines[key]["per_sample_dice"]))
        diff = yolo - b
        try:
            w, p = wilcoxon(yolo, b)
        except ValueError:
            w, p = float("nan"), float("nan")
        raw.append((label, float(diff.mean()), float(w), float(p)))

    # Holm correction
    order = sorted(range(len(raw)), key=lambda i: raw[i][3])
    adj = [0.0] * len(raw)
    m = len(raw)
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * raw[i][3])
        prev = max(prev, val)
        adj[i] = prev

    for (label, dd, w, _), p in zip(raw, adj):
        star = "$^{*}$" if p < 0.05 else ""
        ps = "$<$0.001" if p < 1e-3 else f"{p:.3f}"
        L.append(f"{label:22s} & {dd:+.4f} & {w:.0f} & {ps}{star} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    out.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "yolo"))
    ap.add_argument("--baselines",
                    default=str(ROOT / "results" / "base_models" / "evaluation" / "evaluation_results.json"))
    ap.add_argument("--splits", default=str(ROOT / "data" / "splits"))
    ap.add_argument("--plan", default=str(Path(__file__).resolve().parent / "experiments.yaml"))
    ap.add_argument("--out", default=str(ROOT / "paper2" / "tables"))
    args = ap.parse_args()

    import yaml

    res_dir = Path(args.results)
    runs = load_yolo_runs(res_dir)
    baselines = load_baselines(Path(args.baselines))
    stems = test_stems(Path(args.splits))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"YOLO runs found : {len(runs)}")
    print(f"baselines found : {len(baselines)}")

    plan = yaml.safe_load(Path(args.plan).read_text())
    plan_groups = {g["id"]: [r["name"] for r in g["runs"]] for g in plan["groups"]}
    plan_groups["inference"] = [s["name"] for s in plan.get("inference_sweeps", [])]

    # flat CSV of everything
    with (outdir.parent / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ["run", "dice_mean", "dice_lv_endocardium", "dice_lv_epicardium",
                "dice_left_atrium", "hd95_mean", "assd_mean", "latency_ms_mean",
                "fps", "params_M", "gflops"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for name, r in sorted(runs.items()):
            w.writerow({"run": name, **r})

    if not runs:
        print("\nNo YOLO runs yet -- ceiling table only; the rest are placeholders.")
        table_ceiling(res_dir / "encoding_ceiling.json", outdir / "Y3_ceiling.tex",
                      keep={"32", "96"})
        for f, cap, lab in (("Y1_main.tex", "Main comparison", "tab:main"),
                            ("Y2_ablation.tex", "Ablations", "tab:ablation"),
                            ("Y4_stats.tex", "Significance tests", "tab:stats")):
            (outdir / f).write_text(placeholder(cap, lab))
        for f in sorted(outdir.glob("Y*.tex")):
            print(f"Wrote {f}")
        return

    finals = [n for n in runs if n.startswith("E9_")] or list(runs)
    best = max(finals, key=lambda n: runs[n].get("dice_mean", -1))
    print(f"best run        : {best} (Dice={runs[best].get('dice_mean', float('nan')):.4f})")

    eff = load_efficiency(ROOT / "results" / "benchmark_efficiency.csv")
    sig = compute_sig(res_dir, baselines, best, stems) if baselines else {}
    table_main(runs, baselines, best, outdir / "Y1_main.tex", eff, sig)
    table_ablation(runs, plan_groups, outdir / "Y2_ablation.tex")
    table_ceiling(res_dir / "encoding_ceiling.json", outdir / "Y3_ceiling.tex",
                  keep={"32", "96"})

    if baselines and verify_pairing(baselines, stems):
        table_stats(res_dir, baselines, best, stems, outdir / "Y4_stats.tex")
        print("per-sample pairing verified against reported ED/ES means")
    else:
        (outdir / "Y4_stats.tex").write_text(
            "% pairing check FAILED -- per-sample order could not be verified\n")
        print("!! pairing check failed; Y4_stats.tex not generated")

    for f in sorted(outdir.glob("Y*.tex")):
        print(f"Wrote {f}")


if __name__ == "__main__":
    main()
