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
          "ef_metrics": {"ef_mae": 7.30, "ef_correlation": 0.824,
                         "bland_altman_bias": -3.52,
                         "bland_altman_loa_lower": -17.4,
                         "bland_altman_loa_upper": 16.8},
          "dice_std": 0.07, "dice_ci_lower": 0.905, "dice_ci_upper": 0.914,
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
    """Return ``{model_name: per_model_dict}`` merged across result groups.

    IMPORTANT: the ``param_matched`` session re-trains the base models in a
    separate run purely to provide a within-session reference for the widened
    (``*_wide``) baselines. Those re-trained base models can differ from the
    canonical ``base_models`` session by several Dice points due to
    training variance, so we must NOT let them overwrite the canonical
    numbers. We therefore take from ``param_matched`` ONLY the ``*_wide``
    models; every other model comes from its canonical group
    (``base_models`` / ``mamba_models`` / ``mamba2_models`` /
    ``vmamba_models``).
    """
    merged: Dict[str, Dict] = {}
    for j in sorted(find_eval_jsons(results_root).values(),
                    key=lambda p: p.stat().st_mtime):
        group = j.parent.parent.name
        with open(j) as f:
            data = json.load(f)
        for name, res in data.get("results", {}).items():
            is_wide = name.lower().endswith("_wide")
            if group == "param_matched" and not is_wide:
                # skip param_matched re-trained base models (use canonical)
                continue
            res["_group"] = group
            merged[name] = res
    return merged


def load_group_results(results_root: Path, group: str) -> Dict[str, Dict]:
    """Return the per-model dict for a single result group (e.g.
    ``param_matched``), without merging across sessions."""
    for g, j in find_eval_jsons(results_root).items():
        if g == group:
            with open(j) as f:
                return json.load(f).get("results", {})
    return {}


def load_training_summaries(results_root: Path) -> Dict[str, Dict]:
    """Return ``{display_name: record}`` from every group's all_results.json.

    This is the only artefact that carries ``epochs_trained``, which the
    failure table needs in order to separate a *collapsed* run (trained to
    the budget, converged to a degenerate mask) from an *early-stopped* one
    (patience fired while cosine LR was still near peak) from a run that
    never launched at all.
    """
    out: Dict[str, Dict] = {}
    for j in sorted(results_root.glob("*/all_results.json")):
        group = j.parent.name
        try:
            with open(j) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for rec in data:
            name = rec.get("display_name") or rec.get("model_name")
            if not name:
                continue
            rec["_group"] = group
            # Canonical group wins; param_matched only supplies *_wide.
            if group == "param_matched" and not name.lower().endswith("_wide"):
                out.setdefault(name, rec)
            else:
                out[name] = rec
    return out


# Native-resolution EF artefacts, in load order. Later files win, which mirrors
# load_all_results: the canonical sessions supply their own models, and
# param_matched contributes only its *_wide rows (enforced in load_ef_native).
_EF_NATIVE_FILES = [
    ("base_models",   "yolo/baseline_ef_native.json"),
    ("mamba_models",  "yolo/ef_native_mamba_models.json"),
    ("mamba2_models", "yolo/ef_native_mamba2_models.json"),
    ("vmamba_models", "yolo/ef_native_vmamba_models.json"),
    ("param_matched", "yolo/ef_native_param_matched.json"),
]


def load_ef_native(results_root: Path) -> Dict[str, Dict]:
    """Return ``{model: ef_stats}`` from the native-resolution EF artefacts.

    Paper 1's evaluator computed EF from predictions on the resized 256x256
    grid while passing NATIVE pixel spacing, yielding ~10 ml ventricles. The
    ``ef_metrics`` block inside evaluation_results.json is therefore NOT a
    usable EF source and is overwritten by ``apply_native_ef`` below.

    Produced by ``scripts/yolo/eval_baseline_ef.py --checkpoint-dir <session>``.
    A session whose file is absent simply contributes nothing, and the models
    in it render as {---} rather than carrying a superseded number.
    """
    out: Dict[str, Dict] = {}
    for session, rel in _EF_NATIVE_FILES:
        p = results_root / rel
        if not p.exists():
            continue
        try:
            with open(p) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for k, v in data.items():
            if not isinstance(v, dict) or v.get("ef_mae") is None:
                continue
            # param_matched re-trained the base models; only its widened rows
            # are canonical, exactly as in load_all_results.
            if session == "param_matched" and not k.lower().endswith("_wide"):
                continue
            out[k] = v
    return out


def load_ef_oracle(results_root: Path) -> Optional[Dict]:
    """Ground-truth-mask EF, the floor any segmentation can reach."""
    p = results_root / "yolo" / "_EF_ORACLE" / "ef_oracle.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f).get("stats")
    except (OSError, json.JSONDecodeError):
        return None


def apply_native_ef(results: Dict[str, Dict],
                    ef_native: Dict[str, Dict]) -> Tuple[int, int]:
    """Overwrite every model's ``ef_metrics`` with the native measurement.

    This is deliberately destructive. The resized-grid EF block is wrong by
    construction, so leaving it reachable means any future table or prose
    anchor can silently pick it up again -- which is how it survived into
    Paper A's T4 and Paper D's leaderboard in the first place. Substituting at
    load time makes the broken values unreachable from anywhere downstream,
    and a model with no native measurement gets an empty dict so it formats as
    {---} instead of a plausible-looking wrong number.

    Returns ``(n_substituted, n_cleared)``.
    """
    sub = cleared = 0
    for name, r in results.items():
        native = ef_native.get(name)
        if native and not native.get("n_skipped"):
            r["ef_metrics"] = dict(native)
            sub += 1
        else:
            # Either no native measurement, or one computed over a partial
            # cohort. EF needs all four of a patient's frames, so a model that
            # emits an empty LV mask on any frame drops that whole patient:
            # pure_mamba_unet_mamba is scored over 14 of 50 patients, and its
            # 42.29% is not on the same footing as a 50-patient figure. Both
            # cases render as {---} rather than inviting a false comparison.
            if r.get("ef_metrics"):
                cleared += 1
            r["ef_metrics"] = {}
    return sub, cleared


def load_ssm_probe(results_root: Path) -> Dict[Tuple[str, str], Dict]:
    """Return ``{(model, variant): record}`` from the shared-memory probe.

    ``results/hardware/ssm_shared_mem_ada.jsonl`` (scripts/ssm_shared_mem_probe.py)
    is the only artefact that covers all ten architectures under all SSM
    variants, including the six where Mamba-2 never launches -- the training
    sessions record parameter counts only for runs that started. It is
    therefore the authoritative source for per-variant parameter counts and
    for the measured shared-memory requirement.
    """
    out: Dict[Tuple[str, str], Dict] = {}
    p = results_root / "hardware" / "ssm_shared_mem_ada.jsonl"
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(r["model"], r["mamba_type"])] = r
    return out


def _probe_required_bytes(rec: Dict) -> Optional[int]:
    """Pull the Triton kernel's own reported requirement out of the error."""
    import re
    m = re.search(r"Required:\s*(\d+)", rec.get("error") or "")
    return int(m.group(1)) if m else None


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
    """Mean and per-class Dice for the 9 base architectures, with bootstrap
    95% CIs on mean Dice (read from ``dice_ci_lower``/``dice_ci_upper``)."""
    rows: List[Tuple] = []
    for key, disp in P1_BASE_DISPLAY.items():
        r = results.get(key)
        if not r:
            continue
        rows.append((
            disp, P1_PARADIGM[key],
            float(_get(r, "dice_mean", default=0)),
            _get(r, "dice_ci_lower"),
            _get(r, "dice_ci_upper"),
            float(_get(r, "dice_lv_endocardium", default=0)),
            float(_get(r, "dice_lv_epicardium", default=0)),
            float(_get(r, "dice_left_atrium", default=0)),
            float(_get(r, "params_M", default=0)),
        ))
    rows.sort(key=lambda x: -x[2])

    # column-wise best
    dice_col = [r[2] for r in rows]
    le_col   = [r[5] for r in rows]
    lp_col   = [r[6] for r in rows]
    la_col   = [r[7] for r in rows]

    body = []
    for i, (disp, par, d, clo, chi, le, lp, la, p) in enumerate(rows):
        d_str  = _bold(f"{d:.4f}")  if _bold_max(dice_col, i) else (_ul(f"{d:.4f}") if i == 1 else f"{d:.4f}")
        ci_str = (f"[{float(clo):.4f}, {float(chi):.4f}]"
                  if clo is not None and chi is not None else "{---}")
        le_str = _bold(f"{le:.4f}") if _bold_max(le_col, i)   else f"{le:.4f}"
        lp_str = _bold(f"{lp:.4f}") if _bold_max(lp_col, i)   else f"{lp:.4f}"
        la_str = _bold(f"{la:.4f}") if _bold_max(la_col, i)   else f"{la:.4f}"
        body.append(f"{disp:<18} & {par:<11} & {d_str} & {ci_str} & {le_str} & {lp_str} & {la_str} & {p:>5.1f} \\\\")

    return _wrap_table_p1_t1("\n".join(body))


def _wrap_table_p1_t1(body: str) -> str:
    return (
        "%==============================================================================\n"
        "% Paper 1 / T1 — Main Dice (auto-generated by fill_tables.py)\n"
        "%==============================================================================\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Mean and per-class Dice on the CAMUS official test split for the\n"
        "nine base architectures. The bracketed interval is the bootstrap 95\\%\n"
        "confidence interval on mean Dice over the 50-patient test set. Best per\n"
        "column in \\textbf{bold}, second-best \\underline{underlined}. Per-class\n"
        "scores are LV-endocardium, LV-epicardium, and left atrium.}\n"
        "\\label{tab:main}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l c c c c c c c}\n"
        "\\toprule\n"
        "Architecture       & Paradigm & {Mean Dice} & {95\\% CI} & {LV-endo} & {LV-epi} & {LA} & {Params (M)} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}}\n"
        "\\end{table*}\n"
    )


_P1_CLASSES = [("lv_endocardium", "LV-endo"),
               ("lv_epicardium", "LV-epi"),
               ("left_atrium", "LA")]


def gen_p1_t2_boundary(results: Dict[str, Dict]) -> str:
    """Boundary metrics in mm, mean and per class.

    Per-class HD95 and ASSD were listed in the contributions and reported only
    as means. The per-class fields exist for every model in the evaluation
    JSON, so delivering them is a table-generation change, not an experiment.
    Splitting them out is also what makes the epicardial border visible as the
    hardest of the three for every architecture.
    """
    rows = []
    for key, disp in P1_BASE_DISPLAY.items():
        r = results.get(key)
        if not r:
            continue
        rows.append((disp,
                     _get(r, "hd95_mean"),
                     *[_get(r, f"hd95_{c}") for c, _ in _P1_CLASSES],
                     _get(r, "assd_mean"),
                     *[_get(r, f"assd_{c}") for c, _ in _P1_CLASSES]))
    rows.sort(key=lambda x: x[1] if x[1] is not None else 9e9)

    cols = [list(c) for c in zip(*[r[1:] for r in rows])] if rows else []
    body = []
    for i, row in enumerate(rows):
        cells = []
        for j, v in enumerate(row[1:]):
            s = _fmt(v, ".2f")
            if v is not None and _bold_min(cols[j], i):
                s = _bold(s)
            cells.append(s)
        body.append(f"{row[0]:<18} & " + " & ".join(cells) + " \\\\")

    return _wrap_table_p1_t2("\n".join(body))


def _wrap_table_p1_t2(body: str) -> str:
    return (
        "%==============================================================================\n"
        "% Paper 1 / T2 — Boundary metrics in mm (auto-generated)\n"
        "%==============================================================================\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Boundary metrics on the CAMUS test split, mean and per class.\n"
        "HD95 is the 95\\textsuperscript{th}-percentile Hausdorff distance, ASSD the\n"
        "average symmetric surface distance, both in millimetres computed at native\n"
        "resolution using per-sample NIfTI pixel spacing. The per-class columns show\n"
        "that overlap and distance disagree about which structure is hardest: the\n"
        "epicardium has the lowest Dice for all nine architectures\n"
        "(Table~\\ref{tab:main}), yet the largest HD95 for only two of them --- for\n"
        "five the worst boundary is the left atrium. A per-class overlap summary and\n"
        "a per-class distance summary are therefore not interchangeable. Best per\n"
        "column in \\textbf{bold}.}\n"
        "\\label{tab:boundary}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\begin{tabular}{l cccc cccc}\n"
        "\\toprule\n"
        "\\multirow{2}{*}{Architecture} & \\multicolumn{4}{c}{HD95 (mm)} &"
        " \\multicolumn{4}{c}{ASSD (mm)} \\\\\n"
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
        "                   & Mean & LV-endo & LV-epi & LA &"
        " Mean & LV-endo & LV-epi & LA \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


def gen_p1_t8_iou(results: Dict[str, Dict]) -> str:
    """Per-class IoU -- the second contribution the referee listed as absent.

    Reported alongside Dice rather than instead of it. The two order the
    architectures almost identically, being monotonically related overlap
    measures, and saying so plainly is more useful than printing a column that
    carries no additional ordering information. It earns its place because IoU
    is the more common metric outside medical imaging, which is what makes this
    benchmark comparable to that literature.
    """
    rows = []
    for key, disp in P1_BASE_DISPLAY.items():
        r = results.get(key)
        if not r:
            continue
        rows.append((disp, _get(r, "iou_mean"),
                     *[_get(r, f"iou_{c}") for c, _ in _P1_CLASSES]))
    rows.sort(key=lambda x: -(x[1] if x[1] is not None else 0))

    cols = [list(c) for c in zip(*[r[1:] for r in rows])] if rows else []
    body = []
    for i, row in enumerate(rows):
        cells = []
        for j, v in enumerate(row[1:]):
            s = _fmt(v, ".4f")
            if v is not None and _bold_max(cols[j], i):
                s = _bold(s)
            cells.append(s)
        body.append(f"{row[0]:<18} & " + " & ".join(cells) + " \\\\")

    return (
        "%==============================================================================\n"
        "% Paper 1 / T8 - Per-class IoU (auto-generated)\n"
        "%   <- results/base_models/evaluation/evaluation_results.json : iou_*\n"
        "%==============================================================================\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Intersection-over-union on the CAMUS test split, mean and per\n"
        "class. IoU orders the architectures almost identically to Dice, as expected\n"
        "of two monotonically related overlap measures; it is reported because IoU is\n"
        "the more common metric outside medical imaging and makes this benchmark\n"
        "comparable to that literature. Best per column in \\textbf{bold}.}\n"
        "\\label{tab:iou}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{6pt}\n"
        "\\begin{tabular}{l cccc}\n"
        "\\toprule\n"
        "Architecture       & {Mean IoU} & {LV-endo} & {LV-epi} & {LA} \\\\\n"
        "\\midrule\n"
        f"{chr(10).join(body)}\n"
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


def gen_p1_t4_ef(results: Dict[str, Dict],
                 ef_oracle: Optional[Dict] = None) -> str:
    """EF against the clinically recorded value, at native resolution.

    ``results`` must already have been through ``apply_native_ef``; the
    ``ef_metrics`` block it reads is the native measurement, not the
    resized-grid one that produced ~10 ml ventricles.

    The oracle row applies the identical pipeline to the ground-truth masks
    and bounds what any segmentation can achieve. It belongs in this table
    because without it a 6.66% MAE reads as a modest result rather than as
    1.2 points above an irreducible 5.44% floor.
    """
    rows = []
    for key, disp in P1_BASE_DISPLAY.items():
        r = results.get(key)
        if not r:
            continue
        ef = r.get("ef_metrics") or {}
        rows.append((disp,
                     ef.get("ef_mae"),
                     ef.get("ef_correlation"),
                     ef.get("bland_altman_bias"),
                     ef.get("bland_altman_loa_lower"),
                     ef.get("bland_altman_loa_upper")))
    rows = [r for r in rows if r[1] is not None]
    rows.sort(key=lambda x: x[1])

    mae_col = [r[1] for r in rows]
    r_col   = [r[2] for r in rows]
    body = []
    for i, (disp, mae, rr, bias, lo, hi) in enumerate(rows):
        m_str = _bold(f"{mae:.2f}")  if _bold_min(mae_col, i) else f"{mae:.2f}"
        r_str = _bold(f"{rr:.3f}")   if _bold_max(r_col, i)   else f"{rr:.3f}"
        bias_str = f"{bias:+.2f}" if bias is not None else "{---}"
        loa_str = (f"[{lo:+.1f}, {hi:+.1f}]"
                   if lo is not None and hi is not None else "{---}")
        body.append(f"{disp:<18} & {m_str} & {r_str} & {bias_str} & {loa_str} \\\\")

    if ef_oracle and ef_oracle.get("ef_mae") is not None:
        o_mae = _fmt(ef_oracle.get("ef_mae"), ".2f")
        o_r = _fmt(ef_oracle.get("ef_correlation"), ".3f")
        o_b = ef_oracle.get("bland_altman_bias")
        o_lo = ef_oracle.get("bland_altman_loa_lower")
        o_hi = ef_oracle.get("bland_altman_loa_upper")
        body.append("\\midrule")
        body.append(
            f"\\textit{{oracle (ground-truth masks)}} & {o_mae} & {o_r} & "
            f"{o_b:+.2f} & [{o_lo:+.1f}, {o_hi:+.1f}] \\\\"
            if o_b is not None and o_lo is not None and o_hi is not None else
            f"\\textit{{oracle (ground-truth masks)}} & {o_mae} & {o_r} & "
            "{---} & {---} \\\\")

    return (
        "%==============================================================================\n"
        "% Paper 1 / T4 - EF metrics (auto-generated)\n"
        "%   <- results/yolo/baseline_ef_native.json  (native resolution,\n"
        "%      largest-connected-component filter, official CAMUS Simpson)\n"
        "%   <- results/yolo/_EF_ORACLE/ef_oracle.json  (oracle row)\n"
        "%==============================================================================\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Biplane Simpson's ejection fraction (EF) against the clinically\n"
        "recorded value on the CAMUS test set, computed with the official CAMUS\n"
        "implementation at native resolution with a largest-component filter.\n"
        "Bias and the 95\\% limits of agreement (LoA) are the Bland--Altman\n"
        "statistics over the 50 test patients. The oracle row applies the identical\n"
        "pipeline to the ground-truth masks and bounds what any segmentation can\n"
        "achieve. Best per column in \\textbf{bold}.}\n"
        "\\label{tab:ef}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{6pt}\n"
        "\\begin{tabular}{l c c c c}\n"
        "\\toprule\n"
        "Architecture       & {MAE (\\%)} & {$r$} & {Bias (\\%)} & {95\\% LoA (\\%)} \\\\\n"
        "\\midrule\n"
        f"{chr(10).join(body)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def gen_p1_t5_quality(results_root: Path) -> Optional[str]:
    """Quality-stratified Dice table from ``quality_stratified.json`` (written
    by ``colab_session.py``). Returns ``None`` if the file is not present."""
    cand = sorted(results_root.rglob("quality_stratified.json"))
    if not cand:
        return None
    with open(cand[0]) as f:
        q = json.load(f)
    grades = ["Good", "Medium", "Poor"]
    rows: List[Tuple[str, List[Optional[float]]]] = []
    for key, disp in P1_BASE_DISPLAY.items():
        r = q.get(key)
        if not r:
            continue
        rows.append((disp, [(r.get(g) or {}).get("dice_mean") for g in grades]))
    if not rows:
        return None
    rows.sort(key=lambda x: -(sum(v for v in x[1] if v is not None) /
                              max(1, sum(1 for v in x[1] if v is not None))))
    cols = list(zip(*[r[1] for r in rows]))
    best = [max((v for v in c if v is not None), default=None) for c in cols]
    first = next(iter(q.values()))
    ns = [(first.get(g) or {}).get("n") for g in grades]
    body = []
    for disp, vals in rows:
        cells = []
        for i, v in enumerate(vals):
            if v is None:
                cells.append("{---}")
            else:
                s = f"{v:.4f}"
                cells.append(_bold(s) if best[i] is not None and abs(v - best[i]) < 1e-9 else s)
        body.append(f"{disp:<18} & " + " & ".join(cells) + r" \\")
    ncap = ", ".join(f"{g} $n{{=}}{n}$" for g, n in zip(grades, ns) if n)
    return (
        "%==============================================================================\n"
        "% Paper 1 / T5 -- quality-stratified Dice (auto-generated from\n"
        "% quality_stratified.json produced by colab_session.py)\n"
        "%==============================================================================\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Mean Dice on the CAMUS test split stratified by the\n"
        "expert-assigned image-quality grade (" + ncap + " test frames). Every\n"
        "architecture degrades monotonically from Good to Poor; the top-tier\n"
        "models remain above $0.88$ even on Poor-quality images. Best per column\n"
        "in \\textbf{bold}.}\n"
        "\\label{tab:quality}\n\\small\n\\setlength{\\tabcolsep}{6pt}\n"
        "\\begin{tabular}{l c c c}\n\\toprule\n"
        "Architecture & {Good} & {Medium} & {Poor} \\\\\n\\midrule\n"
        + "\n".join(body) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
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

    def patient_means(arr):
        """Aggregate 200 frames to 50 patient means.

        The 200 test frames are 50 patients x 2 views x 2 phases, ordered
        patient-major in blocks of four (verified against the dataset's own
        iteration order). Testing over frames treats four measurements of the
        same heart as independent, which inflates n fourfold and makes tiny
        differences significant. The patient is the inferential unit here, so
        that is what the shipped p-values use.
        """
        a = np.asarray(arr, dtype=float)
        if a.size % 4:
            return a
        return a.reshape(-1, 4).mean(axis=1)

    pvals = {}
    pvals_frame = {}
    for i, a in enumerate(keys):
        for j in range(i + 1, n):
            b = keys[j]
            fa = np.asarray(results[a]["per_sample_dice"], dtype=float)
            fb = np.asarray(results[b]["per_sample_dice"], dtype=float)
            for tgt, xa, xb in ((pvals, patient_means(fa), patient_means(fb)),
                                (pvals_frame, fa, fb)):
                try:
                    _, p = stats.wilcoxon(xa, xb)
                except ValueError:
                    p = 1.0
                tgt[(a, b)] = min(1.0, p * pairs)  # Bonferroni

    # How much the estimand matters: pairs that frame-level calls significant
    # and patient-level does not.
    n_frame_sig = sum(1 for k in pvals_frame if pvals_frame[k] < 0.05)
    n_pat_sig = sum(1 for k in pvals if pvals[k] < 0.05)
    n_lost = sum(1 for k in pvals
                 if pvals_frame[k] < 0.05 and pvals[k] >= 0.05)

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
        "\\caption{Pairwise Wilcoxon signed-rank test on mean Dice,\n"
        f"Bonferroni-corrected over $\\binom{{{n}}}{{2}} = {pairs}$ pairs. ``NS'' marks\n"
        "pairs not significantly different at $p < 0.05$ after correction.\n"
        "Tests are computed over the $50$ \\emph{patients}, not the $200$ frames:\n"
        "the frames are two views and two phases of each patient and are not\n"
        "independent, so testing over them would inflate the sample fourfold.\n"
        f"On this table the correction changes no conclusion --- {n_frame_sig} of\n"
        f"{pairs} pairs reach significance at frame level and {n_pat_sig} at\n"
        f"patient level, with {n_lost} pairs differing --- because the surviving\n"
        "differences are far larger than the extra power the wrong unit would buy.\n"
        "We report the patient-level test because it is the correct estimand, not\n"
        "because it alters the outcome.}\n"
        "\\label{tab:wilcoxon}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"\\begin{{tabular}}{{{cols}}}\n"
        "\\toprule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}}\n"
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

    # Denominator = architectures ATTEMPTED under that variant, which is the
    # registry minus the combinations documented as architecturally
    # incompatible (currently only DenseContextU-Net + VMamba, whose
    # 4-directional cross-scan meets dense skip concatenation). This was
    # previously hard-coded as 9 for Mamba-2, which understated the failure
    # rate: the shared-memory probe records all ten architectures attempted,
    # four launching and six raising OutOfResources.
    _N_ARCH = len(ARCH_ORDER) + 1          # nine base architectures + Pure-Mamba-UNet
    _INCOMPATIBLE = {"vmamba": 1}          # dense_context_unet is never attempted
    totals = {v: _N_ARCH - _INCOMPATIBLE.get(v, 0)
              for v in ("mamba", "mamba2", "vmamba")}

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
        mae_vals = [m for m in ((r.get("ef_metrics") or {}).get("ef_mae")
                                for r in rs) if m is not None]
        # None until eval_baseline_ef.py has been run over the SSM sessions;
        # rendered as {---} rather than falling back to the resized-grid value.
        best_mae = min(mae_vals) if mae_vals else None
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
        bm_s = ("{---}" if b_m is None else
                _bold(f"{b_m:.2f}") if _bold_min(bm, i) else f"{b_m:.2f}")
        body.append(f"{lbl:<22}  & {tr:<5} & {md_s} & {mh_s} & {bd_s} & {bh_s} & {bm_s} \\\\")

    return (
        "%==============================================================================\n"
        "% Paper 2 / T5 — Aggregate SSM variant comparison (auto-generated)\n"
        "%==============================================================================\n"
        "\\begin{table*}[t]\n"
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
        "\\end{table*}\n"
    )


# Static descriptive columns for the architecture tables. These are prose, not
# measurements -- the only measured column (parameters) is injected from the
# artefacts below, so no parameter count is ever typed into a .tex file.
ARCH_DESC: Dict[str, Dict[str, str]] = {
    "unet_v1":            dict(paradigm="Pure CNN", disp="UNet-V1",
                               enc="scratch / conv", dec="transposed conv", pre="---"),
    "unet_v2":            dict(paradigm="Pure CNN", disp="UNet-V2",
                               enc="scratch / conv + SE", dec="attn-gated skips", pre="---"),
    "unet_resnet":        dict(paradigm="Pure CNN", disp="UNet-ResNet",
                               enc="ResNet-34", dec="U-Net decoder", pre="ImageNet-1k"),
    "deeplab_v3":         dict(paradigm="Pure CNN", disp="DeepLabV3+",
                               enc="ResNet-50", dec="ASPP + light dec.", pre="ImageNet-1k"),
    "nnunet":             dict(paradigm="Pure CNN", disp="nnU-Net",
                               enc="auto-configured", dec="deep-supervision", pre="---"),
    "dense_context_unet": dict(paradigm="Pure CNN", disp="DenseContextU-Net",
                               enc="dense + context", dec="dense decoder", pre="---"),
    "fpn":                dict(paradigm="Pure CNN", disp="FPN-UNet",
                               enc="ResNet-50", dec="FPN + U-Net dec.", pre="ImageNet-1k"),
    "transunet":          dict(paradigm="Hybrid", disp="TransUNet",
                               enc="ResNetV2 + ViT-B/16", dec="cascaded upsmpler",
                               pre="BiT + IN-21k"),
    "swin_unet":          dict(paradigm="Pure Transformer", disp="Swin-UNet",
                               enc="Swin-Tiny", dec="Swin decoder", pre="IN-1k"),
    "pure_mamba_unet":    dict(paradigm="SSM-only", disp="Pure-Mamba-UNet",
                               enc="--- (all-SSM)", dec="--- (all-SSM)", pre="---"),
}

# Order the architecture tables present rows in.
ARCH_ORDER = ["unet_v1", "unet_v2", "unet_resnet", "deeplab_v3", "nnunet",
              "dense_context_unet", "fpn", "transunet", "swin_unet"]

# The SSM registry drops the version suffix for DeepLabV3+, so the probe and
# the Mamba sessions key it as 'mamba_deeplab' rather than 'mamba_deeplab_v3'.
_SSM_KEY = {"deeplab_v3": "mamba_deeplab", "pure_mamba_unet": "pure_mamba_unet"}


def _ssm_key(base_key: str) -> str:
    return _SSM_KEY.get(base_key, f"mamba_{base_key}")


def _variant_params(train: Dict[str, Dict], probe: Dict[Tuple[str, str], Dict],
                    base_key: str, var: str) -> Optional[float]:
    """Parameter count for one architecture under one SSM variant.

    The hardware probe is preferred because it covers every architecture
    including the ones that never trained, but it only ran the two variants
    with a hardware question attached (mamba2, vmamba). The 1D Mamba counts
    come from that session's all_results.json instead.
    """
    rec = probe.get((_ssm_key(base_key), var))
    if rec and rec.get("params_M") is not None:
        return rec["params_M"]
    t = train.get(f"{_ssm_key(base_key)}_{var}")
    if t and t.get("num_params") is not None:
        return t["num_params"] / 1e6
    return None


def _base_params(train: Dict[str, Dict], key: str) -> Optional[float]:
    rec = train.get(key)
    if not rec or rec.get("num_params") is None:
        return None
    return rec["num_params"] / 1e6


def gen_p1_t0_archs(train: Dict[str, Dict]) -> str:
    """Paper 1's architecture summary. Parameter counts come from
    ``results/base_models/all_results.json``; every other column is
    descriptive text held in ARCH_DESC."""
    body = []
    for key in ARCH_ORDER:
        d = ARCH_DESC[key]
        p = _base_params(train, key)
        body.append(
            f"{d['paradigm']:<18} & {d['disp']:<18} & {d['enc']:<20} & "
            f"{d['dec']:<17} & {d['pre']:<11}& {_fmt(p, '6.2f')} \\\\"
        )
    return (
        "%==============================================================================\n"
        "% Paper 1 / T0 - Architecture summary (auto-generated)\n"
        "%   Params (M) <- results/base_models/all_results.json : num_params\n"
        "%==============================================================================\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{The nine architectures benchmarked in this paper.\n"
        "``Pretrained'' indicates whether the encoder is initialised from\n"
        "ImageNet/BiT pretrained weights or trained from scratch.}\n"
        "\\label{tab:archs}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{llllll}\n"
        "\\toprule\n"
        "Paradigm           & Architecture       & Encoder              & "
        "Decoder           & Pretrained & Params (M) \\\\\n"
        "\\midrule\n"
        f"{chr(10).join(body)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


def gen_p2_t1_architectures(train: Dict[str, Dict],
                            probe: Dict[Tuple[str, str], Dict]) -> str:
    """Paper 2's architecture summary, with one column per SSM variant.

    The previous hand-written version carried a single ``Mamba params``
    column justified by a caption claiming Mamba-2 and VMamba differ from
    Mamba/S6 by less than 5%. Measured against the probe that is false for
    six of the nine architectures -- DenseContextU-Net differs by 53%,
    TransUNet by 20%, Swin-UNet by 18% -- so the three variants get their own
    columns and the caption states the measured worst case instead of
    asserting a tolerance.
    """
    rows = []
    devs = []
    for key in ARCH_ORDER + ["pure_mamba_unet"]:
        d = ARCH_DESC[key]
        base = _base_params(train, key)
        per = {var: _variant_params(train, probe, key, var)
               for var in ("mamba", "mamba2", "vmamba")}
        if per["mamba"]:
            for var in ("mamba2", "vmamba"):
                if per[var]:
                    devs.append(abs(per[var] - per["mamba"]) / per["mamba"] * 100)
        rows.append((d, base, per))

    body = []
    for d, base, per in rows:
        body.append(
            f"{d['paradigm']:<16} & {d['disp']:<22} & {d['enc']:<22} & "
            f"{d['pre']:<13} & {_fmt(base, '6.2f')} & {_fmt(per['mamba'], '6.2f')} "
            f"& {_fmt(per['mamba2'], '6.2f')} & {_fmt(per['vmamba'], '6.2f')} \\\\"
        )

    worst = max(devs) if devs else 0.0
    return (
        "%==============================================================================\n"
        "% Paper 2 / T1 - Architecture summary (auto-generated)\n"
        "%   Base params    <- results/base_models/all_results.json : num_params\n"
        "%   Variant params <- results/hardware/ssm_shared_mem_ada.jsonl : params_M\n"
        "%     (the probe covers all ten architectures under all three variants,\n"
        "%      including the six where Mamba-2 never launches)\n"
        "%==============================================================================\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Base and SSM-enhanced architectures evaluated in this study.\n"
        "Parameters are reported for the default base-features / backbone choice.\n"
        "The three SSM variants are listed separately because they are not\n"
        "interchangeable in size: across these architectures the Mamba-2 and\n"
        f"VMamba counts depart from Mamba/S6 by up to {worst:.0f}\\%.\n"
        "``Pretrained'' indicates whether the encoder ships with ImageNet,\n"
        "ImageNet-21k, or BiT pretrained weights. Counts are read from the\n"
        "hardware probe, so they exist for configurations that never trained.}\n"
        "\\label{tab:archs}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{llllrrrr}\n"
        "\\toprule\n"
        "\\multirow{2}{*}{Family} & \\multirow{2}{*}{Base architecture} &\n"
        "\\multirow{2}{*}{Encoder} & \\multirow{2}{*}{Pretrained} &\n"
        "Base & Mamba & Mamba-2 & VMamba \\\\\n"
        "& & & & (M) & (M) & (M) & (M) \\\\\n"
        "\\midrule\n"
        f"{chr(10).join(body)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


def gen_p2_t10_shared_mem(probe: Dict[Tuple[str, str], Dict]) -> str:
    """Measured per-block shared-memory requirement against the device ceiling.

    This replaces Paper 2's Equation 5, which evaluates to 4,194,304 B rather
    than the stated 256 KB, substitutes dimensions matching no block in the
    study, and carries no bytes-per-element term. A measured boundary is both
    correct and what the referees asked for.

    The Required column is the Triton kernel's own reported figure. Note it
    tracks the largest ``head_dim`` in the model padded up to a power of two,
    not the parameter count: TransUNet at 242 M and UNet-V1 at 68 M request
    byte-identical amounts.
    """
    rows = []
    for key in ARCH_ORDER + ["pure_mamba_unet"]:
        mk = _ssm_key(key)
        rec = probe.get((mk, "mamba2"))
        if not rec:
            continue
        head = max((b.get("head_dim") or 0) for b in rec.get("ssm_blocks") or [{}])
        req = _probe_required_bytes(rec)
        rows.append((ARCH_DESC[key]["disp"], rec.get("params_M"),
                     rec.get("max_block_dim"), head, rec.get("status"),
                     rec.get("failure_kind"), req, rec.get("max_shared_mem_bytes")))

    rows.sort(key=lambda r: (r[4] != "OK", r[3] or 0))

    body = []
    for disp, params, dim, head, status, kind, req, limit in rows:
        outcome = "trains" if status == "OK" else "\\texttt{OutOfResources}"
        req_s = f"{req:,}" if req else "{---}"
        body.append(
            f"{disp:<20} & {_fmt(params, '6.2f')} & {dim or '{---}'} & {head or '{---}'} "
            f"& {outcome} & {req_s} \\\\"
        )

    limits = {r[7] for r in rows if r[7]}
    limit_s = f"{sorted(limits)[0]:,}" if len(limits) == 1 else "device-dependent"

    return (
        "%==============================================================================\n"
        "% Paper 2 / T10 - Measured Mamba-2 shared-memory requirement (auto-generated)\n"
        "%   <- results/hardware/ssm_shared_mem_ada.jsonl\n"
        "%      (scripts/ssm_shared_mem_probe.py, RTX 4060 Laptop, Ada, cc 8.9)\n"
        "%==============================================================================\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Measured per-block shared-memory requirement of the Mamba-2 SSD\n"
        "kernel, batch 8 at 256\\,px, each configuration in its own process on one\n"
        f"RTX~4060 Laptop GPU (Ada, cc~8.9, ceiling {limit_s}\\,B). Bytes are the\n"
        "Triton kernel's own reported requirement, not a derivation. The requirement\n"
        "is set by the widest head dimension in the model, padded to a power of two,\n"
        "and is invariant to parameter count: the configurations at 68\\,M and 242\\,M\n"
        "request byte-identical amounts.}\n"
        "\\label{tab:sharedmem}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lrrrlr}\n"
        "\\toprule\n"
        "Configuration & Params (M) & Bottleneck & Max $d_{\\text{head}}$ & "
        "Outcome & Required (B) \\\\\n"
        "\\midrule\n"
        f"{chr(10).join(body)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


def gen_p2_t6_failures(
    results: Dict[str, Dict],
    train: Optional[Dict[str, Dict]] = None,
    ef_native: Optional[Dict[str, Dict]] = None,
) -> str:
    """1D Mamba/S6 divergence and its recovery under VMamba's SS2D scan.

    Previously hand-written, which is how it came to be quoting HD95 at the
    pre-fix convention (values roughly 2.2-2.5x low) while every other table
    had moved to native millimetres. Everything here now reads the same
    artefacts as the rest of the file.

    The Epochs column exists because the two collapsed 1D configurations are
    also the two shortest runs in their session, so "the 1D scan diverged"
    and "training stopped early" are not yet separable from the leaderboard
    alone. Printing the budget lets the reader see that, and answers the
    referees' request to distinguish not-attempted / failed / collapsed.
    """
    train = train or {}
    ef_native = ef_native or {}

    # (label, model key, base-model key for the EF/native lookup)
    families: List[Tuple[str, List[Tuple[str, str]]]] = [
        ("Swin-UNet family (windowed attention encoder)", [
            ("Swin-UNet (base, no SSM)", "swin_unet"),
            (_texttt("mamba_swin_unet_mamba") + " (1D)", "mamba_swin_unet_mamba"),
            (_texttt("mamba_swin_unet_vmamba") + " (SS2D)", "mamba_swin_unet_vmamba"),
        ]),
        ("Pure-Mamba-UNet family (no convolutional path)", [
            (_texttt("pure_mamba_unet_mamba") + " (1D)", "pure_mamba_unet_mamba"),
            (_texttt("pure_mamba_unet_vmamba") + " (SS2D)", "pure_mamba_unet_vmamba"),
        ]),
    ]

    any_ef_missing = False
    partial: List[Tuple[str, int]] = []
    body: List[str] = []
    for fam_label, rows in families:
        body.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{fam_label}}}}} \\\\")
        body.append("\\midrule")
        for label, key in rows:
            r = results.get(key, {})
            t = train.get(key, {})
            dice = _fmt(r.get("dice_mean"), ".4f")
            hd95 = _fmt(r.get("hd95_mean"), ".2f")
            ep = t.get("epochs_trained")
            ep_s = f"{ep:d}" if isinstance(ep, int) else "{---}"
            # Same partial-cohort rule as apply_native_ef: EF needs all four of
            # a patient's frames, so a model that emits an empty LV mask drops
            # that patient entirely and its mean is over a different cohort.
            _ef = ef_native.get(key) or {}
            mae = None if _ef.get("n_skipped") else _ef.get("ef_mae")
            n_skipped = _ef.get("n_skipped") or 0
            if mae is None:
                any_ef_missing = True
                if n_skipped:
                    partial.append((label, n_skipped))
                mae_s = "{---}"
            else:
                mae_s = _fmt(mae, ".2f")
            body.append(f"{label:<46} & {ep_s:>4} & {dice} & {hd95} & {mae_s} \\\\")
        body.append("\\midrule")
    if body and body[-1] == "\\midrule":
        body.pop()

    ef_note = ""
    if partial:
        kept = 50 - partial[0][1]
        ef_note = (
            "\nEF is computed at native resolution with the official CAMUS biplane\n"
            "implementation. It is omitted where a configuration emits an empty\n"
            "left-ventricular mask on any frame: Simpson's method needs all four of\n"
            "a patient's frames, so those patients drop out and the mean is no longer\n"
            f"over the same cohort. The collapsed 1D run is scored over {kept} of the\n"
            "50 test patients, which is not comparable to the other rows."
        )
    elif any_ef_missing:
        ef_note = (
            "\nEF is reported only where a native-resolution measurement exists in\n"
            "\\texttt{results/yolo/baseline\\_ef\\_native.json}."
        )

    return (
        "%==============================================================================\n"
        "% Paper 2 / T6 - 1D Mamba divergence and VMamba recovery (auto-generated)\n"
        "%   Dice, HD95  <- results/<group>/evaluation/evaluation_results.json\n"
        "%   Epochs      <- results/<group>/all_results.json\n"
        "%   EF MAE      <- results/yolo/baseline_ef_native.json\n"
        "%==============================================================================\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{The two configurations where 1D Mamba/S6 diverged, and their\n"
        "recovery under VMamba's 4-directional SS2D scan. The base CNN counterpart\n"
        "is shown for reference where applicable. Epochs is the number actually\n"
        "trained against a 100-epoch budget with early-stopping patience 20, so a\n"
        "short run and a degenerate result can be told apart."
        f"{ef_note}}}\n"
        "\\label{tab:failures}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{l c c c c}\n"
        "\\toprule\n"
        "Configuration & {Epochs} & {Dice} & {HD95 (mm)} & {EF MAE (\\%)} \\\\\n"
        "\\midrule\n"
        f"{chr(10).join(body)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
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


def gen_p2_t7_param_matched(results: Dict[str, Dict],
                            pm_results: Optional[Dict[str, Dict]] = None) -> str:
    # For a clean within-session comparison, read the base AND wide rows from
    # the param_matched session (pm_results); the Mamba row comes from the
    # canonical session (results). Falls back to canonical if pm is absent.
    pm = pm_results or {}
    body = []
    for base_k, wide_k, mamba_k, base_label, wide_label, mamba_label in PARAMM_PAIRS:
        sources = [
            (base_k, base_label, pm.get(base_k) or results.get(base_k)),
            (wide_k, wide_label, pm.get(wide_k) or results.get(wide_k)),
            (mamba_k, mamba_label, results.get(mamba_k)),
        ]
        for k, lbl, r in sources:
            if not r:
                body.append(f"{lbl:<60} & {{---}} & {{---}} & {{---}} & {{---}} \\\\")
                continue
            params = float(_get(r, "params_M", default=0))
            d  = _get(r, "dice_mean")
            hd = _get(r, "hd95_mean")
            mae = (r.get("ef_metrics") or {}).get("ef_mae")
            disp = lbl if k.endswith("_wide") or "(base" in lbl or "(closest" in lbl else _texttt(k)
            # A row carrying ``_retrained`` was NOT produced in the original
            # param-matched session, so its comparison against the base row in
            # the same block is cross-session. Training variance across sessions
            # is worth several Dice points (see load_all_results), which is the
            # same magnitude as the effect this table measures -- so the row is
            # marked and the caption says what the mark means.
            if r.get("_retrained"):
                disp = disp + "$^{\\dag}$"
            body.append(
                f"{disp:<60} & {params:>5.1f} & {_fmt(d,'.4f')} & {_fmt(hd,'.2f')} & {_fmt(mae,'.2f')} \\\\"
            )
        body.append("\\midrule")

    if body and body[-1] == "\\midrule":
        body = body[:-1]

    # Explain only the notations the table actually uses, so the caption never
    # describes a situation that no longer exists.
    joined_body = "\n".join(body)
    missing_note = (
        " Entries shown as --- could not be evaluated under the"
        " native-resolution boundary protocol because the corresponding"
        " checkpoint is no longer available; a pair whose widened control is"
        " missing does not support a conclusion about the value of the SSM"
        " block, and is reported only for completeness."
        if "{---}" in joined_body else ""
    )
    retrain_note = (
        " $^{\\dag}$Retrained separately from the original param-matched"
        " session under the same recipe and seed after the original"
        " checkpoint was lost; the reproduction agrees with the original"
        " session to $0.0003$ Dice, but the comparison with its base row is"
        " nonetheless cross-session."
        if "\\dag" in joined_body else ""
    )

    return (
        "%==============================================================================\n"
        "% Paper 2 / T7 — Param-matched (auto-generated)\n"
        "%==============================================================================\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Parameter-matched comparison. The base and widened rows are\n"
        "trained in the same dedicated session so their comparison is within-session;\n"
        "the Mamba row is from the main session. Swin-UNet and TransUNet are excluded\n"
        "as their capacity is tied to fixed pretrained encoders."
        f"{missing_note}{retrain_note}}}\n"
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


def _top_configs(results: Dict[str, Dict], n: int = 10) -> List[Tuple[str, Dict]]:
    """Top-n trained configs by Dice, excluding base and _wide models."""
    cand = [(name, r) for name, r in results.items()
            if r.get("dice_mean", 0) > 0.5
            and _classify(name)[1] in ("mamba", "mamba2", "vmamba")]
    cand.sort(key=lambda x: -x[1]["dice_mean"])
    return cand[:n]


def gen_p2_t3_perclass(results: Dict[str, Dict]) -> str:
    """Per-class Dice for the top-10 configs plus best base (nnU-Net)."""
    rows = _top_configs(results, 10)
    # append nnU-Net as reference if present
    if "nnunet" in results:
        rows = rows + [("nnunet", results["nnunet"])]
    le = [r.get("dice_lv_endocardium", 0) for _, r in rows]
    lp = [r.get("dice_lv_epicardium", 0) for _, r in rows]
    la = [r.get("dice_left_atrium", 0) for _, r in rows]
    body = []
    for i, (name, r) in enumerate(rows):
        disp = P2_BASE_DISPLAY.get(name, _texttt(name))
        e = _bold(f"{le[i]:.4f}") if le[i] == max(le) else f"{le[i]:.4f}"
        p = _bold(f"{lp[i]:.4f}") if lp[i] == max(lp) else f"{lp[i]:.4f}"
        a = _bold(f"{la[i]:.4f}") if la[i] == max(la) else f"{la[i]:.4f}"
        sep = "\\midrule\n" if name == "nnunet" else ""
        body.append(f"{sep}{disp} & {e} & {p} & {a} \\\\")
    return (
        "%==============================================================================\n"
        "% Paper 2 / T3 -- per-class Dice (auto-generated)\n"
        "%==============================================================================\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Per-class Dice on CAMUS test for the top-10 configurations plus\n"
        "the strongest base CNN (nnU-Net). Best per column in \\textbf{bold}.}\n"
        "\\label{tab:perclass}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{l c c c}\n\\toprule\n"
        "Model & {LV-endo} & {LV-epi} & {LA} \\\\\n\\midrule\n"
        f"{chr(10).join(body)}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def gen_p2_t4_edes(results: Dict[str, Dict]) -> str:
    """ED/ES-stratified Dice + HD95 for the top-10 configs."""
    rows = _top_configs(results, 10)
    body = []
    for name, r in rows:
        body.append(
            f"{_texttt(name)} & {_fmt(r.get('dice_mean_ed'),'.4f')} & "
            f"{_fmt(r.get('dice_mean_es'),'.4f')} & "
            f"{_fmt(r.get('hd95_mean_ed'),'.2f')} & {_fmt(r.get('hd95_mean_es'),'.2f')} \\\\"
        )
    return (
        "%==============================================================================\n"
        "% Paper 2 / T4 -- ED/ES stratified (auto-generated)\n"
        "%==============================================================================\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{ED- and ES-stratified mean Dice and HD95 (mm) for the top-10\n"
        "configurations.}\n"
        "\\label{tab:edes}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{l c c c c}\n\\toprule\n"
        "\\multirow{2}{*}{Model} & \\multicolumn{2}{c}{Dice} & \\multicolumn{2}{c}{HD95 (mm)} \\\\\n"
        "                       & {ED}   & {ES}   & {ED}   & {ES}   \\\\\n\\midrule\n"
        f"{chr(10).join(body)}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def gen_p2_t9_wilcoxon(results: Dict[str, Dict]) -> str:
    """Wilcoxon corrected-p among the top-10 configs (computed from per-sample Dice)."""
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        return "% T9 needs scipy; install and re-run fill_tables.py\n"
    rows = [(n, r) for n, r in _top_configs(results, 10)
            if r.get("per_sample_dice")]
    if len(rows) < 3:
        return "% T9: insufficient per-sample Dice\n"
    n_common = min(len(r["per_sample_dice"]) for _, r in rows)
    names = [n for n, _ in rows]
    arrs = {n: np.asarray(r["per_sample_dice"][:n_common]) for n, r in rows}
    pairs = len(names) * (len(names) - 1) // 2
    sig = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            try:
                _, p = stats.wilcoxon(arrs[a], arrs[b])
            except ValueError:
                p = 1.0
            pc = min(1.0, p * pairs)
            if pc < 0.05:
                d = float(arrs[a].mean() - arrs[b].mean())
                sig.append((a, b, pc, d))
    sig.sort(key=lambda x: x[2])
    body = []
    for a, b, pc, d in sig[:18]:
        body.append(f"{_texttt(a)} vs.\\ {_texttt(b)} & {pc:.1e} & {d:+.4f} \\\\")
    if not body:
        body = ["\\multicolumn{3}{l}{No pairs significant at $p<0.05$ after Bonferroni.} \\\\"]
    return (
        "%==============================================================================\n"
        "% Paper 2 / T9 -- Wilcoxon among top-10 (auto-generated)\n"
        "%==============================================================================\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Statistically significant pairwise differences among the top-10\n"
        "configurations (Wilcoxon signed-rank on per-sample Dice, Bonferroni-corrected\n"
        f"over $\\binom{{10}}{{2}}={pairs}$ pairs). $\\Delta$ is the mean per-sample\n"
        "Dice difference. Pairs not shown are not significant.}\n"
        "\\label{tab:wilcoxon}\n\\small\n\\setlength{\\tabcolsep}{5pt}\n"
        "\\begin{tabular}{l c c}\n\\toprule\n"
        "Pair & Corrected $p$ & $\\Delta$ Dice \\\\\n\\midrule\n"
        f"{chr(10).join(body)}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# Prose consistency lint
#
# Reviewer #1's structural fix for the 2026 EF-number regression: every
# headline number in the prose must be regenerable from the same JSON that
# feeds the tables, and the specific stale values that caused the bug must
# never reappear. Run with --check_prose (+ --strict to fail the build).
# ---------------------------------------------------------------------------

# Values that were wrong in the prose and must never come back:
#   5.72  = nnU-Net esv_mean, mis-slotted as EF MAE in the abstract
#   7.52  = the old fill_tables docstring placeholder for TransUNet EF MAE
#   0.823 = a stale nnU-Net EF correlation (canonical is 0.787)
PROSE_BLOCKLIST = ["5.72", "7.52", "0.823"]


def gen_canonical_anchors(results: Dict[str, Dict]) -> List[Tuple[str, str]]:
    """Return ``[(label, value_string)]`` that the prose MUST contain, derived
    from the same JSON that feeds the tables. Keeps headline numbers in sync so
    a prose/table disagreement is caught mechanically."""
    anchors: List[Tuple[str, str]] = []

    def add(key, field, fmt, label, sub=None):
        r = results.get(key) or {}
        v = (r.get(sub) or {}).get(field) if sub else r.get(field)
        if v is not None:
            anchors.append((label, format(float(v), fmt)))

    # EF anchors. These read the NATIVE measurement (apply_native_ef has already
    # replaced ef_metrics by the time this runs), so a regression to the
    # resized-grid values fails the build. UNet-V1 is anchored because it leads
    # the corrected table; nnU-Net is kept because the discussion compares it
    # against the 2019 CAMUS report.
    add("unet_v1", "ef_mae", ".2f", "UNet-V1 EF MAE (best)", sub="ef_metrics")
    add("nnunet", "ef_mae", ".2f", "nnU-Net EF MAE", sub="ef_metrics")
    add("transunet", "ef_mae", ".2f", "TransUNet EF MAE", sub="ef_metrics")
    add("nnunet", "ef_correlation", ".3f", "nnU-Net EF r", sub="ef_metrics")
    add("transunet", "dice_mean", ".4f", "TransUNet Dice")
    add("nnunet", "dice_mean", ".4f", "nnU-Net Dice")
    # Boundary anchors. These exist because the HD95/ASSD values were once
    # computed on the resized 256x256 grid with the native NIfTI spacing,
    # under-reporting every distance by ~2.2x; the tables were regenerated but
    # the prose kept the old sub-2 mm numbers. Anchoring HD95 makes any future
    # table/prose divergence on the boundary axis fail the build.
    add("transunet", "hd95_mean", ".2f", "TransUNet HD95")
    add("nnunet", "hd95_mean", ".2f", "nnU-Net HD95")
    return anchors


def check_prose(anchors: List[Tuple[str, str]], blocklist: List[str],
                prose_files: List[str]) -> List[str]:
    """Return a list of violation strings (empty list == clean)."""
    violations: List[str] = []
    texts: Dict[str, str] = {}
    for f in prose_files:
        try:
            texts[f] = Path(f).read_text(encoding="utf-8")
        except OSError:
            continue
    joined = "\n".join(texts.values())
    for bad in blocklist:
        for f, t in texts.items():
            for ln, line in enumerate(t.splitlines(), 1):
                if bad in line:
                    violations.append(
                        f"BLOCKLISTED '{bad}' in {f}:{ln}: {line.strip()[:90]}")
    for label, val in anchors:
        if val not in joined:
            violations.append(
                f"MISSING anchor: {label} = {val} appears in no prose file "
                f"(prose disagrees with the tables)")
    return violations


def _collect_prose_files(explicit, paper1_tables) -> List[str]:
    """Resolve the prose .tex files to lint: explicit paths/dirs if given,
    else auto-discover Paper 1's sections/ + main.tex next to the tables dir."""
    out: List[str] = []
    if explicit:
        for p in explicit:
            p = Path(p)
            if p.is_dir():
                out += [str(x) for x in sorted(p.rglob("*.tex"))]
            elif p.exists():
                out.append(str(p))
    elif paper1_tables:
        base = Path(paper1_tables).parent  # .../paper/
        secs = base / "sections"
        if secs.is_dir():
            out += [str(x) for x in sorted(secs.glob("*.tex"))]
        if (base / "main.tex").exists():
            out.append(str(base / "main.tex"))
    return out


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
    ap.add_argument("--check_prose", type=Path, nargs="*", default=None,
                    help="Prose .tex files/dirs to lint for number consistency. "
                         "If omitted, Paper 1's sections/ + main.tex are auto-discovered.")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if the prose consistency lint finds violations.")
    args = ap.parse_args()

    print(f"[fill_tables] Loading evaluation JSONs from {args.results_root}")
    results = load_all_results(args.results_root)
    print(f"[fill_tables] {len(results)} models loaded.")

    train = load_training_summaries(args.results_root)
    print(f"[fill_tables] {len(train)} training summaries "
          f"(epochs_trained) loaded.")
    probe = load_ssm_probe(args.results_root)
    print(f"[fill_tables] {len(probe)} SSM probe records loaded.")
    ef_native = load_ef_native(args.results_root)
    ef_oracle = load_ef_oracle(args.results_root)
    print(f"[fill_tables] {len(ef_native)} native-resolution EF measurements "
          f"loaded.")

    # The resized-grid EF block is wrong by construction. Replace it at load
    # time so no table or prose anchor downstream can reach it.
    _sub, _clr = apply_native_ef(results, ef_native)
    print(f"[fill_tables] EF: {_sub} models repointed at the native artefact, "
          f"{_clr} cleared (no native measurement -> renders as ---).")
    if ef_oracle:
        print(f"[fill_tables] EF oracle floor: {ef_oracle.get('ef_mae')}% MAE.")

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
        "T0_archs.tex":       gen_p1_t0_archs(train),
        "T1_main_dice.tex":   gen_p1_t1_main_dice(results),
        "T2_boundary.tex":    gen_p1_t2_boundary(results),
        "T3_edes.tex":        gen_p1_t3_edes(results),
        "T4_ef.tex":          gen_p1_t4_ef(results, ef_oracle),
        "T6_efficiency.tex":  gen_p1_t6_efficiency(results, bench),
        "T8_iou.tex":         gen_p1_t8_iou(results),
    }
    _t5 = gen_p1_t5_quality(args.results_root)
    if _t5:
        p1_tables["T5_quality.tex"] = _t5
    p1_tables = {
        **p1_tables,
        "T7_wilcoxon.tex":    gen_p1_t7_wilcoxon(results, {}),
    }
    p2_tables = {
        "T2_main_leaderboard.tex": gen_p2_t2_leaderboard(results),
        "T3_perclass.tex":         gen_p2_t3_perclass(results),
        "T4_edes.tex":             gen_p2_t4_edes(results),
        "T5_variants.tex":         gen_p2_t5_variants(results),
        "T1_architectures.tex":    gen_p2_t1_architectures(train, probe),
        "T6_failures.tex":         gen_p2_t6_failures(results, train, ef_native),
        "T10_shared_mem.tex":      gen_p2_t10_shared_mem(probe),
        "T7_param_matched.tex":    gen_p2_t7_param_matched(
                                       results,
                                       load_group_results(args.results_root, "param_matched")),
        "T8_efficiency.tex":       gen_p2_t8_efficiency(bench, results),
        "T9_wilcoxon.tex":         gen_p2_t9_wilcoxon(results),
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

    # Prose consistency lint (reviewer #1: fail if prose disagrees with tables)
    prose_files = _collect_prose_files(args.check_prose, args.paper1_tables)
    if prose_files:
        anchors = gen_canonical_anchors(results)
        violations = check_prose(anchors, PROSE_BLOCKLIST, prose_files)
        print(f"\n[fill_tables] Prose lint: {len(prose_files)} files, "
              f"{len(anchors)} anchors, {len(PROSE_BLOCKLIST)} blocklisted values.")
        if violations:
            print("[fill_tables] PROSE CONSISTENCY LINT — VIOLATIONS:")
            for v in violations:
                print("  X " + v)
            if args.strict:
                raise SystemExit("[fill_tables] Prose lint failed (--strict).")
        else:
            print("[fill_tables] Prose lint clean: every headline number matches "
                  "the tables and no stale value is present.")

    print("\n[fill_tables] Done. Remember to re-compile both PDFs.")


if __name__ == "__main__":
    main()
