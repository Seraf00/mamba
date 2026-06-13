#!/usr/bin/env python3
"""
fill_tables.py — rewrite every paper-table .tex from the evaluation JSONs
and the efficiency-benchmark CSV.

Usage on Colab (after evaluation + benchmark have run):
    python scripts/fill_tables.py \\
        --results_root /content/results \\
        --paper1_tables /content/Paper1/paper/tables \\
        --paper2_tables /content/Paper2/paper/tables

Usage locally (Drive-synced):
    python D:/Papers/Paper1/scripts/fill_tables.py \\
        --results_root D:/Papers/Paper1/results \\
        --paper1_tables D:/Papers/Paper1/paper/tables \\
        --paper2_tables D:/Papers/Paper2/paper/tables

Each ``evaluation_results.json`` looks like::

    {
      "evaluation_date": "...",
      "split": "test",
      "results": {
        "transunet": {
          "dice_mean": 0.9122,
          "dice_lv_endocardium": 0.9367,
          "dice_lv_epicardium": 0.8787,
          "dice_left_atrium": 0.9213,
          "iou_mean": 0.8422,
          "hd95_mean": 1.82,
          "assd_mean": 0.36,
          "hd95_mean_ed": 1.85,
          "hd95_mean_es": 1.79,
          "dice_mean_ed": ...,
          "dice_mean_es": ...,
          "ef_metrics": {"ef_mae": 7.52, "ef_correlation": 0.810,
                         "bland_altman_bias": -3.52},
          "params_M": 102.1,
          "per_sample_dice": [...]
        },
        ...
      },
      "statistical_comparison": {
        "model_a_vs_model_b": {"p_corrected": ..., "mean_diff": ...,
                               "significant": true/false},
        ...
      }
    }

The script does NOT touch tables T0, T1 (architectures, prior work) — those
are static. It rewrites every other table whose numbers come from the
evaluation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def find_eval_jsons(results_root: Path) -> Dict[str, Path]:
    """Find every evaluation_results.json under results_root, keyed by
    its parent group name (base_models, mamba_models, mamba2_models,
    vmamba_models, param_matched, ...)."""
    out: Dict[str, Path] = {}
    for j in results_root.rglob("evaluation_results.json"):
        group = j.parent.parent.name  # group/evaluation/evaluation_results.json
        out[group] = j
    return out


def load_all_results(results_root: Path) -> Dict[str, Dict]:
    """Return ``{model_display_name: per_model_dict}`` merged across groups.

    When the same display name appears in multiple groups (e.g. ``nnunet``
    re-trained inside param_matched), the *most recent* file wins. The
    function prints a one-line conflict notice so you can decide whether to
    rename.
    """
    merged: Dict[str, Dict] = {}
    conflicts: List[str] = []
    for j in sorted(find_eval_jsons(results_root).values(),
                    key=lambda p: p.stat().st_mtime):
        with open(j) as f:
            data = json.load(f)
        for name, res in data.get("results", {}).items():
            if name in merged and merged[name] != res:
                conflicts.append(f"  {name}  ({merged[name].get('_group','?')} -> {j.parent.parent.name})")
            res["_group"] = j.parent.parent.name
            merged[name] = res
    if conflicts:
        print("[fill_tables] Duplicate model names across groups, kept latest:")
        for c in conflicts[:20]:
            print(c)
    return merged


def load_benchmark_csv(path: Path) -> Dict[str, Dict]:
    """Read efficiency benchmark CSV produced by ``scripts/benchmark.py``.

    Returns ``{model_name: {'params_M', 'flops_G', 'latency_ms', 'memory_MB'}}``
    """
    if not path or not path.exists():
        return {}
    out: Dict[str, Dict] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Model") or row.get("model")
            if not name:
                continue
            params = float(row.get("Parameters", row.get("params", 0)) or 0) / 1e6
            flops = row.get("FLOPs", row.get("flops"))
            flops_g = (float(flops) / 1e9) if flops and flops != "N/A" else None
            try:
                t_ms = float(row.get("Inference Time (ms)", row.get("latency_ms", 0)))
            except (TypeError, ValueError):
                t_ms = None
            try:
                mem_mb = float(row.get("Memory (MB)", row.get("memory_mb", 0)))
            except (TypeError, ValueError):
                mem_mb = None
            out[name] = {"params_M": params, "flops_G": flops_g,
                         "latency_ms": t_ms, "memory_MB": mem_mb}
    return out


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v, spec: str, default: str = "{---}") -> str:
    try:
        if v is None:
            return default
        return format(float(v), spec)
    except (TypeError, ValueError):
        return default


def _bold(s: str) -> str:
    return r"\textbf{" + s + "}"


def _ul(s: str) -> str:
    return r"\underline{" + s + "}"


def _bold_min(values: List[float], idx: int) -> bool:
    finite = [v for v in values if v is not None]
    return finite and values[idx] == min(finite)


def _bold_max(values: List[float], idx: int) -> bool:
    finite = [v for v in values if v is not None]
    return finite and values[idx] == max(finite)


def _texttt(name: str) -> str:
    return r"\texttt{" + name.replace("_", r"\_") + "}"


# benchmark.py registers UNet-V1 under the canonical alias 'unet' (and the
# Mamba variants as 'mamba_unet_*' rather than 'mamba_unet_v1_*'), so the
# efficiency CSV keys do not match the evaluation keys for that one model.
_BENCH_ALIAS = {
    "unet_v1": "unet",
    "mamba_unet_v1_mamba": "mamba_unet_mamba",
    "mamba_unet_v1_mamba2": "mamba_unet_mamba2",
    "mamba_unet_v1_vmamba": "mamba_unet_vmamba",
}


def _bench(bench: Dict[str, Dict], key: str) -> Dict:
    """Look up a model in the benchmark CSV, trying known name aliases."""
    if key in bench:
        return bench[key]
    if key in _BENCH_ALIAS and _BENCH_ALIAS[key] in bench:
        return bench[_BENCH_ALIAS[key]]
    return {}


# ---------------------------------------------------------------------------
# Display name mapping
# ---------------------------------------------------------------------------

# Internal key -> human-friendly display name for Paper 1 base table.
P1_BASE_DISPLAY = {
    "unet_v1":            "UNet-V1",
    "unet_v2":            "UNet-V2",
    "unet_resnet":        "UNet-ResNet",
    "deeplab_v3":         "DeepLabV3+",
    "nnunet":             "nnU-Net",
    "dense_context_unet": "DenseContextU-Net",
    "fpn":                "FPN-UNet",
    "swin_unet":          "Swin-UNet",
    "transunet":          "TransUNet",
}

P1_PARADIGM = {
    "unet_v1": "CNN", "unet_v2": "CNN", "unet_resnet": "CNN",
    "deeplab_v3": "CNN", "nnunet": "CNN", "dense_context_unet": "CNN",
    "fpn": "CNN",
    "transunet": "Hybrid", "swin_unet": "Transformer",
}


def _get(r: Dict, *keys, default=None):
    """Dict get with multiple fallback keys."""
    for k in keys:
        if k in r and r[k] is not None:
            return r[k]
        # nested via .
        cur = r
        for part in k.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                cur = None
                break
        if cur is not None:
            return cur
    return default


# ---------------------------------------------------------------------------
# Paper 1 — base-model tables
# ---------------------------------------------------------------------------

def gen_p1_t1_main_dice(results: Dict[str, Dict]) -> str:
    """Mean and per-class Dice for the 9 base architectures."""
    rows: List[Tuple[str, str, float, float, float, float, float]] = []
    for key, disp in P1_BASE_DISPLAY.items():
        r = results.get(key)
        if not r:
            continue
        rows.append((
            disp, P1_PARADIGM[key],
            float(_get(r, "dice_mean", default=0)),
            float(_get(r, "dice_lv_endocardium", default=0)),
            float(_get(r, "dice_lv_epicardium", default=0)),
            float(_get(r, "dice_left_atrium", default=0)),
            float(_get(r, "params_M", default=0)),
        ))
    rows.sort(key=lambda x: -x[2])

    # column-wise best
    dice_col = [r[2] for r in rows]
    le_col   = [r[3] for r in rows]
    lp_col   = [r[4] for r in rows]
    la_col   = [r[5] for r in rows]

    body = []
    for i, (disp, par, d, le, lp, la, p) in enumerate(rows):
        d_str  = _bold(f"{d:.4f}")  if _bold_max(dice_col, i) else (_ul(f"{d:.4f}") if i == 1 else f"{d:.4f}")
        le_str = _bold(f"{le:.4f}") if _bold_max(le_col, i)   else f"{le:.4f}"
        lp_str = _bold(f"{lp:.4f}") if _bold_max(lp_col, i)   else f"{lp:.4f}"
        la_str = _bold(f"{la:.4f}") if _bold_max(la_col, i)   else f"{la:.4f}"
        body.append(f"{disp:<18} & {par:<11} & {d_str} & {le_str} & {lp_str} & {la_str} & {p:>5.1f} \\\\")

    return _wrap_table_p1_t1("\n".join(body))


def _wrap_table_p1_t1(body: str) -> str:
    return (
        "%==============================================================================\n"
        "% Paper 1 / T1 — Main Dice (auto-generated by fill_tables.py)\n"
        "%==============================================================================\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Mean and per-class Dice on the CAMUS official test split for the\n"
        "nine base architectures. Best per column in \\textbf{bold}, second-best\n"
        "\\underline{underlined}. Per-class scores are LV-endocardium, LV-epicardium,\n"
        "and left atrium.}\n"
        "\\label{tab:main}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\begin{tabular}{l c c c c c c}\n"
        "\\toprule\n"
        "Architecture       & Paradigm & {Mean Dice} & {LV-endo} & {LV-epi} & {LA} & {Params (M)} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


def gen_p1_t2_boundary(results: Dict[str, Dict]) -> str:
    rows: List[Tuple[str, float, float]] = []
    for key, disp in P1_BASE_DISPLAY.items():
        r = results.get(key)
        if not r:
            continue
        rows.append((disp,
                     float(_get(r, "hd95_mean", default=0)),
                     float(_get(r, "assd_mean", default=0))))
    rows.sort(key=lambda x: x[1])

    hd_col = [r[1] for r in rows]
    as_col = [r[2] for r in rows]
    body = []
    for i, (disp, hd, asd) in enumerate(rows):
        hd_str  = _bold(f"{hd:.2f}")  if _bold_min(hd_col, i) else f"{hd:.2f}"
        as_str  = _bold(f"{asd:.2f}") if _bold_min(as_col, i) else f"{asd:.2f}"
        body.append(f"{disp:<18} & {hd_str} & {as_str} \\\\")

    return _wrap_table_p1_t2("\n".join(body))


def _wrap_table_p1_t2(body: str) -> str:
    return (
        "%==============================================================================\n"
        "% Paper 1 / T2 — Boundary metrics in mm (auto-generated)\n"
        "%==============================================================================\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Boundary metrics on the CAMUS test split. HD95 is the\n"
        "95\\textsuperscript{th}-percentile Hausdorff distance, ASSD is the average\n"
        "symmetric surface distance, both reported in millimetres using per-sample\n"
        "NIfTI pixel spacing. Best per column in \\textbf{bold}.}\n"
        "\\label{tab:boundary}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{6pt}\n"
        "\\begin{tabular}{l c c}\n"
        "\\toprule\n"
        "Architecture       & {HD95 (mm)} & {ASSD (mm)} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def gen_p1_t3_edes(results: Dict[str, Dict]) -> str:
    rows = []
    for key, disp in P1_BASE_DISPLAY.items():
        r = results.get(key)
        if not r:
            continue
        rows.append((disp,
                     _get(r, "dice_mean_ed"),
                     _get(r, "dice_mean_es"),
                     _get(r, "hd95_mean_ed"),
                     _get(r, "hd95_mean_es")))
    rows.sort(key=lambda x: -(x[1] or 0))
    body = []
    for disp, ded, des, hed, hes in rows:
        body.append(
            f"{disp:<18} & {_fmt(ded,'.4f')} & {_fmt(des,'.4f')} & "
            f"{_fmt(hed,'.2f')} & {_fmt(hes,'.2f')} \\\\"
        )

    return (
        "%==============================================================================\n"
        "% Paper 1 / T3 — ED/ES stratified (auto-generated)\n"
        "%==============================================================================\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{End-diastole (ED) and end-systole (ES) stratified mean Dice and\n"
        "HD95 (mm) on the CAMUS test split.}\n"
        "\\label{tab:edes}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{l c c c c}\n"
        "\\toprule\n"
        "\\multirow{2}{*}{Architecture} & \\multicolumn{2}{c}{Dice} & \\multicolumn{2}{c}{HD95 (mm)} \\\\\n"
        "                              & {ED}   & {ES}   & {ED}   & {ES}   \\\\\n"
        "\\midrule\n"
        f"{chr(10).join(body)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def gen_p1_t4_ef(results: Dict[str, Dict]) -> str:
    rows = []
    for key, disp in P1_BASE_DISPLAY.items():
        r = results.get(key)
        if not r:
            continue
        ef = r.get("ef_metrics") or {}
        rows.append((disp,
                     ef.get("ef_mae"),
                     ef.get("ef_correlation"),
                     ef.get("bland_altman_bias")))
    rows = [r for r in rows if r[1] is not None]
    rows.sort(key=lambda x: x[1])

    mae_col = [r[1] for r in rows]
    r_col   = [r[2] for r in rows]
    body = []
    for i, (disp, mae, rr, bias) in enumerate(rows):
        m_str = _bold(f"{mae:.2f}")  if _bold_min(mae_col, i) else f"{mae:.2f}"
        r_str = _bold(f"{rr:.3f}")   if _bold_max(r_col, i)   else f"{rr:.3f}"
        bias_str = f"{bias:+.2f}" if bias is not None else "{---}"
        body.append(f"{disp:<18} & {m_str} & {r_str} & {bias_str} \\\\")

    return (
        "%==============================================================================\n"
        "% Paper 1 / T4 — EF metrics (auto-generated)\n"
        "%==============================================================================\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Biplane Simpson's ejection fraction (EF) metrics on the CAMUS\n"
        "test set. Best per column in \\textbf{bold}.}\n"
        "\\label{tab:ef}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{6pt}\n"
        "\\begin{tabular}{l c c c}\n"
        "\\toprule\n"
        "Architecture       & {MAE (\\%)} & {$r$} & {Bias (\\%)} \\\\\n"
        "\\midrule\n"
        f"{chr(10).join(body)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def gen_p1_t6_efficiency(results: Dict[str, Dict], bench: Dict[str, Dict]) -> str:
    rows = []
    for key, disp in P1_BASE_DISPLAY.items():
        r = results.get(key)
        b = _bench(bench, key)
        if not r:
            continue
        rows.append((disp,
                     float(_get(r, "params_M", default=0)),
                     b.get("flops_G"),
                     b.get("latency_ms"),
                     b.get("memory_MB")))
    rows.sort(key=lambda x: x[1])
    body = []
    for disp, p, f, t, m in rows:
        body.append(
            f"{disp:<18} & {p:>5.1f} & {_fmt(f,'.1f')} & {_fmt(t,'.2f')} & {_fmt(m,'.0f')} \\\\"
        )

    return (
        "%==============================================================================\n"
        "% Paper 1 / T6 — Efficiency (auto-generated)\n"
        "%==============================================================================\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Efficiency profile on a single NVIDIA L4 GPU. Latency is the mean\n"
        "over 100 forward passes after 10 warmup at batch 1, FP16, $256\\times 256$\n"
        "(or $224\\times 224$ for Swin-UNet).}\n"
        "\\label{tab:efficiency}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{l c c c c}\n"
        "\\toprule\n"
        "Architecture       & {Params (M)} & {FLOPs (G)} & {Latency (ms)} & {Peak mem (MB)} \\\\\n"
        "\\midrule\n"
        f"{chr(10).join(body)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def gen_p1_t7_wilcoxon(results: Dict[str, Dict],
                       stats_top_level: Dict[str, Dict]) -> str:
    """9×9 Wilcoxon corrected p-value matrix on per-sample Dice."""
    # Compute per-sample Dice via scipy.stats.wilcoxon if per_sample_dice present.
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        return "% T7 requires numpy + scipy; install and re-run fill_tables.py\n"

    keys = [k for k in P1_BASE_DISPLAY if k in results
            and results[k].get("per_sample_dice")]
    if len(keys) < 2:
        return "% T7: insufficient per-sample Dice data\n"

    disp = {k: P1_BASE_DISPLAY[k] for k in keys}
    n = len(keys)
    pairs = n * (n - 1) // 2

    pvals = {}
    for i, a in enumerate(keys):
        for j in range(i + 1, n):
            b = keys[j]
            da = np.asarray(results[a]["per_sample_dice"])
            db = np.asarray(results[b]["per_sample_dice"])
            try:
                _, p = stats.wilcoxon(da, db)
            except ValueError:
                p = 1.0
            pvals[(a, b)] = min(1.0, p * pairs)  # Bonferroni

    # Render matrix
    header = " & " + " & ".join(disp[k] for k in keys) + " \\\\"
    lines = [header, "\\midrule"]
    for i, a in enumerate(keys):
        row = [disp[a]]
        for j, b in enumerate(keys):
            if i == j:
                row.append("---")
            elif j > i:
                p = pvals.get((a, b), 1.0)
                row.append("NS" if p >= 0.05 else f"{p:.0e}")
            else:
                row.append("")
        lines.append(" & ".join(row) + " \\\\")

    body = "\n".join(lines)
    cols = "l " + ("c " * n)

    return (
        "%==============================================================================\n"
        "% Paper 1 / T7 — Wilcoxon (auto-generated)\n"
        "%==============================================================================\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Pairwise Wilcoxon signed-rank test on per-sample mean Dice,\n"
        f"Bonferroni-corrected over $\\binom{{{n}}}{{2}} = {pairs}$ pairs. ``NS'' marks\n"
        "pairs not significantly different at $p < 0.05$ after correction.}\n"
        "\\label{tab:wilcoxon}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        f"\\begin{{tabular}}{{{cols}}}\n"
        "\\toprule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


# ---------------------------------------------------------------------------
# Paper 2 — Mamba tables
# ---------------------------------------------------------------------------

# Nice display names for base architectures in the Paper 2 leaderboard.
P2_BASE_DISPLAY = {
    "unet_v1": "UNet-V1", "unet_v2": "UNet-V2", "unet_resnet": "UNet-ResNet",
    "deeplab_v3": "DeepLabV3+", "nnunet": "nnU-Net",
    "dense_context_unet": "DenseContextU-Net", "fpn": "FPN-UNet",
    "swin_unet": "Swin-UNet", "transunet": "TransUNet",
}


def _classify(name: str) -> Tuple[str, str]:
    """Return (group_label, ssm_variant) for a model name.

    Parameter-matched widened baselines (``*_wide``) are tagged ``wide`` so
    the leaderboard can drop them -- they belong only in the param-matched
    table T7, not the main results leaderboard.
    """
    n = name.lower()
    if n.endswith("_wide"):
        return ("Parameter-matched widened (see T7)", "wide")
    if n.endswith("_vmamba"):
        return ("VMamba/SS2D variants (4-directional 2D cross-scan)", "vmamba")
    if n.endswith("_mamba2"):
        return ("Mamba-2/SSD variants (chunked scan, Triton)", "mamba2")
    if n.endswith("_mamba"):
        return ("Mamba/S6 variants (1D selective scan)", "mamba")
    return ("Base architectures (no SSM)", "base")


def gen_p2_t2_leaderboard(results: Dict[str, Dict]) -> str:
    """Full leaderboard for all trained configurations grouped by SSM."""
    # group by SSM variant
    groups: Dict[str, List[Tuple[str, Dict]]] = {
        "base": [], "mamba": [], "mamba2": [], "vmamba": [],
    }
    for name, r in results.items():
        _, variant = _classify(name)
        # drop param-matched _wide baselines from the leaderboard (they are
        # reported only in T7), and skip anything without a Dice number
        if variant == "wide" or r.get("dice_mean") is None:
            continue
        groups[variant].append((name, r))

    # within each group, sort by Dice desc
    for v in groups:
        groups[v].sort(key=lambda x: -x[1].get("dice_mean", 0))

    # Find global best/2nd by Dice, HD95, EF MAE, EF r across all configs
    all_models = [(n, r) for v in groups.values() for n, r in v]
    dice_vals  = [r.get("dice_mean", 0) for n, r in all_models]
    hd_vals    = [r.get("hd95_mean") for n, r in all_models]
    efm_vals   = [(r.get("ef_metrics") or {}).get("ef_mae") for n, r in all_models]
    efr_vals   = [(r.get("ef_metrics") or {}).get("ef_correlation") for n, r in all_models]

    def best2(v_list, mode="max"):
        vals = [v for v in v_list if v is not None]
        if not vals: return None, None
        if mode == "max":
            s = sorted(vals, reverse=True)
        else:
            s = sorted(vals)
        return s[0], (s[1] if len(s) > 1 else None)

    dice_b, dice_2 = best2(dice_vals, "max")
    hd_b, hd_2     = best2([v for v in hd_vals if v is not None], "min")
    efm_b, efm_2   = best2([v for v in efm_vals if v is not None], "min")
    efr_b, efr_2   = best2([v for v in efr_vals if v is not None], "max")

    def cell(v, best, second, fmt, mode="max"):
        if v is None: return "{---}"
        s = format(v, fmt)
        if mode == "max":
            if best is not None and abs(v - best) < 1e-9: return _bold(s)
            if second is not None and abs(v - second) < 1e-9: return _ul(s)
        else:
            if best is not None and abs(v - best) < 1e-9: return _bold(s)
            if second is not None and abs(v - second) < 1e-9: return _ul(s)
        return s

    body = []
    headers = [
        ("Base architectures", "base"),
        ("Mamba/S6 variants (1D selective scan)", "mamba"),
        ("Mamba-2/SSD variants (chunked scan, Triton)", "mamba2"),
        ("VMamba/SS2D variants (4-directional 2D cross-scan)", "vmamba"),
    ]
    for title, key in headers:
        body.append("\\midrule")
        body.append("\\multicolumn{6}{l}{\\emph{" + title + "}} \\\\")
        body.append("\\midrule")
        for name, r in groups[key]:
            d  = r.get("dice_mean")
            hd = r.get("hd95_mean")
            ef = r.get("ef_metrics") or {}
            mae = ef.get("ef_mae")
            efr = ef.get("ef_correlation")
            params = r.get("params_M", "")
            params_s = f"{params:>5.1f}" if isinstance(params, (int, float)) else "  ---"
            display = P2_BASE_DISPLAY.get(name, name) if key == "base" else _texttt(name)
            d_c   = cell(d, dice_b, dice_2, ".4f", "max")
            hd_c  = cell(hd, hd_b, hd_2, ".2f", "min")
            ef_c  = cell(mae, efm_b, efm_2, ".2f", "min")
            efr_c = cell(efr, efr_b, efr_2, ".3f", "max")
            body.append(f"{display} & {d_c} & {hd_c} & {ef_c} & {efr_c} & {params_s} \\\\")

    return (
        "%==============================================================================\n"
        "% Paper 2 / T2 — Main leaderboard (auto-generated)\n"
        "%==============================================================================\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{CAMUS test-set results for all successfully trained configurations.\n"
        "\\textbf{Dice} is the mean of LV-endo, LV-epi and LA. \\textbf{HD95} is in\n"
        "millimetres using per-sample NIfTI pixel spacing. Best per metric in\n"
        "\\textbf{bold}, second-best \\underline{underlined}.}\n"
        "\\label{tab:main}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{l c c c c c}\n"
        "\\toprule\n"
        "Model & {Dice} & {HD95 (mm)} & {EF MAE (\\%)} & {EF $r$} & {Params (M)} \\\\\n"
        f"{chr(10).join(body)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


def gen_p2_t5_variants(results: Dict[str, Dict]) -> str:
    """Aggregate metrics per SSM variant."""
    buckets = {"mamba": [], "mamba2": [], "vmamba": []}
    for name, r in results.items():
        _, v = _classify(name)
        if v in buckets and r.get("dice_mean") is not None:
            buckets[v].append(r)

    # Total mamba models in registry: 10 mamba / 9 mamba2 / 9 vmamba
    totals = {"mamba": 10, "mamba2": 9, "vmamba": 9}

    rows = []
    for v in ("mamba", "mamba2", "vmamba"):
        rs = [r for r in buckets[v] if r.get("dice_mean", 0) > 0.5]  # exclude collapses
        if not rs:
            continue
        trainable = f"{len(rs)}/{totals[v]}"
        mean_d  = sum(r["dice_mean"] for r in rs) / len(rs)
        mean_hd = sum(r["hd95_mean"] for r in rs if r.get("hd95_mean")) / max(1, sum(1 for r in rs if r.get("hd95_mean")))
        best_d  = max(r["dice_mean"] for r in rs)
        best_hd = min(r["hd95_mean"] for r in rs if r.get("hd95_mean"))
        mae_vals = [(r.get("ef_metrics") or {}).get("ef_mae") for r in rs]
        best_mae = min(v for v in mae_vals if v is not None)
        label = {"mamba": "Mamba/S6  (1D)",
                 "mamba2": "Mamba-2/SSD",
                 "vmamba": "VMamba/SS2D"}[v]
        rows.append((label, trainable, mean_d, mean_hd, best_d, best_hd, best_mae))

    # column-wise winners
    md  = [r[2] for r in rows]; mh = [r[3] for r in rows]
    bd  = [r[4] for r in rows]; bh = [r[5] for r in rows]
    bm  = [r[6] for r in rows]
    body = []
    for i, (lbl, tr, m_d, m_h, b_d, b_h, b_m) in enumerate(rows):
        md_s = _bold(f"{m_d:.4f}") if _bold_max(md, i) else f"{m_d:.4f}"
        mh_s = _bold(f"{m_h:.2f}") if _bold_min(mh, i) else f"{m_h:.2f}"
        bd_s = _bold(f"{b_d:.4f}") if _bold_max(bd, i) else f"{b_d:.4f}"
        bh_s = _bold(f"{b_h:.2f}") if _bold_min(bh, i) else f"{b_h:.2f}"
        bm_s = _bold(f"{b_m:.2f}") if _bold_min(bm, i) else f"{b_m:.2f}"
        body.append(f"{lbl:<22}  & {tr:<5} & {md_s} & {mh_s} & {bd_s} & {bh_s} & {bm_s} \\\\")

    return (
        "%==============================================================================\n"
        "% Paper 2 / T5 — Aggregate SSM variant comparison (auto-generated)\n"
        "%==============================================================================\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Aggregate comparison of the three SSM variants over architectures\n"
        "on which each variant was trainable.}\n"
        "\\label{tab:variants}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{l c c c c c c}\n"
        "\\toprule\n"
        "SSM variant & {Trainable} & {Mean Dice} & {Mean HD95} & {Best Dice} & {Best HD95} & {Best EF MAE} \\\\\n"
        "\\midrule\n"
        f"{chr(10).join(body)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def gen_p2_t8_efficiency(bench: Dict[str, Dict], results: Dict[str, Dict],
                         top_n: int = 12) -> str:
    """Efficiency profile for top-N models from the leaderboard plus best base."""
    # Sort all models by Dice, excluding param-matched _wide baselines
    ranked = sorted(((n, r) for n, r in results.items()
                     if r.get("dice_mean", 0) > 0.5
                     and _classify(n)[1] != "wide"),
                    key=lambda x: -x[1]["dice_mean"])[:top_n]
    body = []
    for name, r in ranked:
        b = _bench(bench, name)
        params = float(_get(r, "params_M", default=0))
        body.append(
            f"{_texttt(name)} & {params:>5.1f} & {_fmt(b.get('flops_G'),'.1f')} & "
            f"{_fmt(b.get('latency_ms'),'.2f')} & {_fmt(b.get('memory_MB'),'.0f')} \\\\"
        )

    return (
        "%==============================================================================\n"
        "% Paper 2 / T8 — Efficiency (auto-generated)\n"
        "%==============================================================================\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Efficiency profile of top configurations on a single NVIDIA L4 GPU.}\n"
        "\\label{tab:efficiency}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{l c c c c}\n"
        "\\toprule\n"
        "Configuration & {Params (M)} & {FLOPs (G)} & {Latency (ms)} & {Peak mem (MB)} \\\\\n"
        "\\midrule\n"
        f"{chr(10).join(body)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


# Pairs for the param-matched table (Paper 2 / T7)
PARAMM_PAIRS = [
    # (base_key, wide_key, mamba_key, base_label, wide_label, mamba_label)
    ("unet_v1",           "unet_v1_wide",           "mamba_unet_v1_mamba",
     "UNet-V1 (base, $b_f{=}64$)",
     "UNet-V1\\textsubscript{wide} (param-matched)",
     "mamba_unet_v1_mamba"),
    ("unet_v2",           "unet_v2_wide",           "mamba_unet_v2_mamba",
     "UNet-V2 (base, $b_f{=}64$)",
     "UNet-V2\\textsubscript{wide} (param-matched)",
     "mamba_unet_v2_mamba"),
    ("unet_resnet",       "unet_resnet_wide",       "mamba_unet_resnet_mamba2",
     "UNet-ResNet (ResNet-34)",
     "UNet-ResNet\\textsubscript{wide} (closest backbone)",
     "mamba_unet_resnet_mamba2"),
    ("deeplab_v3",        "deeplab_v3_wide",        "mamba_deeplab_mamba2",
     "DeepLabV3+ (ResNet-50)",
     "DeepLabV3+\\textsubscript{wide} (closest backbone)",
     "mamba_deeplab_mamba2"),
    ("nnunet",            "nnunet_wide",            "mamba_nnunet_mamba2",
     "nnU-Net (base, $b_f{=}32$)",
     "nnU-Net\\textsubscript{wide} (closest)",
     "mamba_nnunet_mamba2"),
    ("dense_context_unet","dense_context_unet_wide","mamba_dense_context_unet_mamba",
     "DenseContextU-Net (base)",
     "DenseContextU-Net\\textsubscript{wide} (param-matched)",
     "mamba_dense_context_unet_mamba"),
]


def gen_p2_t7_param_matched(results: Dict[str, Dict]) -> str:
    body = []
    for base_k, wide_k, mamba_k, base_label, wide_label, mamba_label in PARAMM_PAIRS:
        for k, lbl in [(base_k, base_label), (wide_k, wide_label), (mamba_k, mamba_label)]:
            r = results.get(k)
            if not r:
                body.append(f"{lbl:<60} & {{---}} & {{---}} & {{---}} & {{---}} \\\\")
                continue
            params = float(_get(r, "params_M", default=0))
            d  = _get(r, "dice_mean")
            hd = _get(r, "hd95_mean")
            mae = (r.get("ef_metrics") or {}).get("ef_mae")
            disp = lbl if k.endswith("_wide") or "(base" in lbl or "(closest" in lbl else _texttt(k)
            body.append(
                f"{disp:<60} & {params:>5.1f} & {_fmt(d,'.4f')} & {_fmt(hd,'.2f')} & {_fmt(mae,'.2f')} \\\\"
            )
        body.append("\\midrule")

    if body and body[-1] == "\\midrule":
        body = body[:-1]

    return (
        "%==============================================================================\n"
        "% Paper 2 / T7 — Param-matched (auto-generated)\n"
        "%==============================================================================\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Parameter-matched comparison: each Mamba-enhanced model is\n"
        "compared against (i) its default base CNN and (ii) a widened version of the\n"
        "base CNN. Swin-UNet and TransUNet excluded as their capacity is tied to\n"
        "fixed pretrained encoders.}\n"
        "\\label{tab:parammatch}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lr c c c}\n"
        "\\toprule\n"
        "Configuration & Params (M) & {Dice} & {HD95 (mm)} & {EF MAE (\\%)} \\\\\n"
        "\\midrule\n"
        f"{chr(10).join(body)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Rewrite paper tables from eval outputs.")
    ap.add_argument("--results_root", type=Path, required=True,
                    help="Directory containing base_models/, mamba_models/, etc.")
    ap.add_argument("--benchmark_csv", type=Path, default=None,
                    help="Path to benchmark_efficiency.csv from scripts/benchmark.py")
    ap.add_argument("--paper1_tables", type=Path, default=None,
                    help="Paper 1 tables/ directory to write")
    ap.add_argument("--paper2_tables", type=Path, default=None,
                    help="Paper 2 tables/ directory to write")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print to stdout instead of writing files")
    args = ap.parse_args()

    print(f"[fill_tables] Loading evaluation JSONs from {args.results_root}")
    results = load_all_results(args.results_root)
    print(f"[fill_tables] {len(results)} models loaded.")

    bench_path = args.benchmark_csv
    if bench_path is None:
        # Try common defaults
        for c in [args.results_root / "benchmark_efficiency.csv",
                  args.results_root.parent / "benchmark_efficiency.csv"]:
            if c.exists():
                bench_path = c; break
    bench = load_benchmark_csv(bench_path) if bench_path else {}
    print(f"[fill_tables] {len(bench)} models in benchmark CSV "
          f"({bench_path}).")

    # Build per-paper outputs
    p1_tables = {
        "T1_main_dice.tex":   gen_p1_t1_main_dice(results),
        "T2_boundary.tex":    gen_p1_t2_boundary(results),
        "T3_edes.tex":        gen_p1_t3_edes(results),
        "T4_ef.tex":          gen_p1_t4_ef(results),
        "T6_efficiency.tex":  gen_p1_t6_efficiency(results, bench),
        "T7_wilcoxon.tex":    gen_p1_t7_wilcoxon(results, {}),
    }
    p2_tables = {
        "T2_main_leaderboard.tex": gen_p2_t2_leaderboard(results),
        "T5_variants.tex":         gen_p2_t5_variants(results),
        "T7_param_matched.tex":    gen_p2_t7_param_matched(results),
        "T8_efficiency.tex":       gen_p2_t8_efficiency(bench, results),
    }

    def write_or_print(target_dir: Optional[Path], tables: Dict[str, str], tag: str):
        if target_dir is None:
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        for name, body in tables.items():
            p = target_dir / name
            if args.dry_run:
                print(f"\n=== {tag} / {name} ===\n{body}")
            else:
                p.write_text(body, encoding="utf-8")
                print(f"  wrote {p}")

    write_or_print(args.paper1_tables, p1_tables, "Paper 1")
    write_or_print(args.paper2_tables, p2_tables, "Paper 2")

    print("\n[fill_tables] Done. Remember to re-compile both PDFs.")


if __name__ == "__main__":
    main()
