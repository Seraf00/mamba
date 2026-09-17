#!/usr/bin/env python3
"""
Recompute the nine Paper 1 baselines' ejection fraction at NATIVE resolution.

Why this exists
---------------
Paper 1's evaluator computed EF from predictions on the resized 256x256 grid
while passing the NATIVE pixel spacing. That yields left-ventricular volumes of
roughly 10 ml (physiologically impossible; the correct value is ~108 ml), and
the error does NOT cancel in the EF ratio as one might assume: on ground-truth
masks the two pipelines differ by 4.8% EF on average and up to 20.9% on
individual patients -- more than the spread between models in the published EF
table.

So YOLO EF computed natively cannot be placed beside Paper 1's EF numbers. This
script re-runs every baseline through the identical native-resolution pipeline
used for the YOLO rows, making the whole EF table commensurable.

Volumes come from `metrics.camus_ef_official`, which was verified bit-identical
to the official CAMUS notebook (`script_camus_ef.ipynb`): max |EF difference|
0.0000% across all 50 test patients.

Usage:
    python scripts/yolo/eval_baseline_ef.py
    python scripts/yolo/eval_baseline_ef.py --models nnunet transunet
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data import CAMUSDataset, get_transforms  # noqa: E402
from metrics import CAMUSEFCalculator  # noqa: E402

# reuse Paper 1's own loader so construction/recovery behaviour is identical
from evaluate_all_models import (  # noqa: E402
    _construct_and_load, find_model_checkpoints, get_img_size,
)

# Input size comes from get_img_size(), which matches on substrings and so
# also catches mamba_swin_unet_* at 224 -- a plain dict keyed by the nine base
# names would silently evaluate those at 256.

DEFAULT_MODELS = [
    "unet_v1", "unet_v2", "unet_resnet", "deeplab_v3", "nnunet",
    "dense_context_unet", "swin_unet", "transunet", "fpn",
]


def pin_determinism(seed: int = 42, mode: str = "full") -> dict:
    """Make this evaluation bit-reproducible, and report what was pinned.

    Without this the EF table does not reproduce: re-running the identical
    checkpoints moved six of nine baselines and reordered the ranking.

    The dominant factor is *arithmetic precision*, not run-to-run
    nondeterminism. Ablated on FPN-UNet (see
    ``results/hardware/ef_precision_ablation.json``): unpinned 9.74 / 9.72,
    cuDNN determinism alone 9.72 (no effect), deterministic algorithms alone
    10.00, TF32 disabled alone 10.60 -- the whole shift. Turning TF32 off
    changed the EF of 26 of 50 patients, some by 14 EF points and some EDVs by
    ~20 ml, because a precision change flips borderline pixels, ``largest_cc``
    turns that into a discrete choice, and a patient's EF needs all four of
    their frames.

    TF32 off is full FP32 and is the more accurate computation, so it is what
    ships. Note this does not make results portable across machines: skimage
    version also changes the answer (0.25.2 vs 0.26.0 differ), so the
    environment must be pinned alongside these flags.

    Must be called before any CUDA work. ``CUBLAS_WORKSPACE_CONFIG`` is set
    here rather than in the shell so a bare ``python eval_baseline_ef.py`` is
    reproducible too.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ``mode`` exists so the individual controls can be ablated against each
    # other. "full" is the shipping configuration; the others are for
    # attributing a change to a specific flag.
    want_cudnn = mode in ("full", "cudnn")
    want_tf32  = mode in ("full", "tf32")
    want_algos = mode in ("full", "algos")

    torch.backends.cudnn.benchmark = not want_cudnn   # autotuning off when pinned
    torch.backends.cudnn.deterministic = want_cudnn
    torch.backends.cuda.matmul.allow_tf32 = not want_tf32
    torch.backends.cudnn.allow_tf32 = not want_tf32

    # warn_only: a few upsampling kernels have no deterministic implementation,
    # and this must not abort the run.
    torch.use_deterministic_algorithms(want_algos, warn_only=True)

    return {
        "mode": mode,
        "seed": seed,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def run_model(cfg: dict, data_dir: Path, device: torch.device):
    """Return {(patient, view, phase): (binary LV mask at native res, spacing)}.

    ``cfg`` is a model config in the shape ``find_model_checkpoints`` produces:
    registry ``name``, ``display_name``, ``checkpoint``, and -- for the SSM
    sessions -- ``mamba_type`` and any ``extra_kwargs`` the trainer used. Going
    through that structure rather than a bare name is what lets this script
    cover the Mamba, Mamba-2, VMamba and param-matched sessions, not just the
    nine base models it was written for.
    """
    size = get_img_size(cfg["name"])
    ds = CAMUSDataset(
        root_dir=str(data_dir), split="test",
        transform=get_transforms(split="val", img_size=(size, size)),
        include_info=True, include_native_mask=True,
    )
    model = _construct_and_load(cfg, device)
    model.to(device).eval()

    out: dict = {}
    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc=f'{cfg["display_name"]:24s}', ncols=80,
                      leave=False):
            s = ds[i]
            img = s["image"].unsqueeze(0).to(device).float()
            logits = model(img)
            # Some baselines return a dict (e.g. torchvision-style {'out': ...})
            # and some a tuple; Paper 1's evaluator handles both, so match it.
            if isinstance(logits, dict):
                logits = logits.get("out", next(iter(logits.values())))
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred = logits.argmax(1)                      # (1, size, size)

            nm = np.asarray(s["native_mask"])
            h, w = nm.shape[:2]
            pred_nat = F.interpolate(pred.unsqueeze(1).float(), size=(h, w),
                                     mode="nearest").squeeze(1).long().cpu().numpy()[0]
            out[(s["patient_id"], s["view"], s["phase"])] = (
                (pred_nat == 1).astype(np.uint8), tuple(s["pixel_spacing"]))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def largest_cc(m: np.ndarray) -> np.ndarray:
    """
    Keep only the biggest connected component of a binary LV mask.

    Simpson's method in `camus_ef_official` (and in the official CAMUS
    notebook) takes `find_contours(mask, 0.5)[0]` -- the FIRST contour, which
    is ordered by position, not area. Ground truth has a single clean
    component so this never bites, but model predictions carry occasional
    spurious blobs: measured here, 6% of frames had >1 component, and that was
    enough to drive EF correlation from r=0.911 down to r=0.030, because EF
    needs all four of a patient's masks and one bad frame ruins the patient.

    Filtering is therefore not cosmetic -- without it the EF column is noise.
    """
    from scipy import ndimage

    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return (lab == (int(np.argmax(sizes)) + 1)).astype(np.uint8)


def ef_for(masks: dict, gt_ef: dict, use_largest_cc: bool = True) -> dict:
    calc = CAMUSEFCalculator(lv_label=1, n_disks=20)
    skipped = []
    patients = sorted({k[0] for k in masks})
    for pid in patients:
        need = [(pid, "2CH", "ED"), (pid, "2CH", "ES"),
                (pid, "4CH", "ED"), (pid, "4CH", "ES")]
        if not all(k in masks for k in need):
            skipped.append((pid, "missing view/phase"))
            continue
        if any(masks[k][0].sum() == 0 for k in need):
            skipped.append((pid, "empty LV prediction"))
            continue
        (a2e, sp2), (a2s, _), (a4e, sp4), (a4s, _) = (masks[k] for k in need)
        if use_largest_cc:
            a2e, a2s, a4e, a4s = (largest_cc(x) for x in (a2e, a2s, a4e, a4s))
        try:
            calc.compute_ef(a2e, a2s, sp2, a4e, a4s, sp4,
                            ef_ground_truth=gt_ef.get(pid), patient_id=pid)
        except Exception as e:
            skipped.append((pid, f"{type(e).__name__}: {str(e)[:60]}"))
    st = calc.compute_statistics()
    st["n_skipped"] = len(skipped)
    st["skipped"] = skipped
    # Persist per-patient EF. Only the oracle carried this before, so when the
    # aggregate moved between runs there was no way to identify which patient
    # flipped. Additive: existing consumers read the aggregate keys unchanged.
    st["per_patient"] = [
        {"patient_id": r.patient_id, "ef_pred": r.ef_percent,
         "ef_gt": r.ef_ground_truth, "edv_ml": r.edv_ml, "esv_ml": r.esv_ml}
        for r in calc.results
    ]
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default=str(ROOT / "results" / "base_models"))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "CAMUS"))
    ap.add_argument("--models", nargs="*", default=None,
                    help="Restrict to these display names. Default: every model "
                         "with a checkpoint under --checkpoint-dir.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for the pinned deterministic evaluation.")
    ap.add_argument("--pin", default="full",
                    choices=["full", "cudnn", "tf32", "algos", "off"],
                    help="Which determinism controls to apply. 'full' ships; "
                         "the rest isolate one control for ablation.")
    ap.add_argument("--out", default=None,
                    help="Output JSON. Defaults to baseline_ef_native.json for "
                         "the base_models session, else ef_native_<session>.json.")
    ap.add_argument("--no-largest-cc", action="store_true",
                    help="Disable the largest-connected-component filter. Used "
                         "to quantify what the filter is worth; the filtered "
                         "path is what ships.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N models (smoke test).")
    args = ap.parse_args()

    determinism = pin_determinism(args.seed, args.pin)
    print(f"[determinism] mode={determinism['mode']} seed={determinism['seed']} "
          f"cudnn.benchmark={determinism['cudnn_benchmark']} "
          f"cudnn.deterministic={determinism['cudnn_deterministic']} "
          f"tf32={determinism['tf32_matmul']} ({determinism['gpu']})", flush=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    # reference EF from the CAMUS cfg files
    gt_ef: dict[str, float] = {}
    for pid in [ln.strip() for ln in (ROOT / "data" / "splits" / "test.txt").read_text().splitlines() if ln.strip()]:
        cfg = data_dir / pid / "Info_4CH.cfg"
        if cfg.exists():
            for line in cfg.read_text().splitlines():
                if line.startswith("EF:"):
                    gt_ef[pid] = float(line.split(":", 1)[1])

    ckpt_dir = Path(args.checkpoint_dir)
    session = ckpt_dir.name
    configs = find_model_checkpoints(ckpt_dir)
    if args.models:
        wanted = set(args.models)
        configs = [c for c in configs if c["display_name"] in wanted]
    if args.limit:
        configs = configs[: args.limit]
    print(f"[{session}] {len(configs)} model(s) with checkpoints")
    print()

    results = {}
    for cfg in configs:
        name = cfg["display_name"]
        try:
            masks = run_model(cfg, data_dir, device)
            results[name] = ef_for(masks, gt_ef, use_largest_cc=not args.no_largest_cc)
            r = results[name]
            print(f"{name:20s} MAE={r.get('ef_mae', float('nan')):6.2f}%  "
                  f"bias={r.get('ef_bias', float('nan')):+6.2f}%  "
                  f"r={r.get('ef_correlation', float('nan')):.3f}  "
                  f"EDV={r.get('edv_mean', float('nan')):6.1f} ml  "
                  f"(skipped {r['n_skipped']})", flush=True)
        except Exception as e:
            print(f"[fail] {name}: {type(e).__name__}: {str(e)[:120]}", flush=True)

    if args.out:
        out = Path(args.out)
    elif session == "base_models":
        out = ROOT / "results" / "yolo" / "baseline_ef_native.json"
    else:
        out = ROOT / "results" / "yolo" / f"ef_native_{session}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Record what produced these numbers alongside them. The whole point of
    # pinning is that the table reproduces, and a reader (or a future re-run)
    # has to be able to tell which determinism settings a given artefact was
    # measured under. Consumers key on ``ef_mae``, so this entry is ignored by
    # load_ef_native and by the table generators.
    results["_determinism"] = determinism
    out.write_text(json.dumps(results, indent=1))

    print(f"\nSaved {out}")
    print("\nCompare against Paper 1's resized-grid EF table (paper/tables/T4_ef.tex);")
    print("a changed ranking means that table needs regenerating.")


if __name__ == "__main__":
    main()
