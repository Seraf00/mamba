#!/usr/bin/env python3
"""
Measure the ANNOTATION CEILING on CAMUS, and place the model against it.

Why
---
Leclerc et al. (TMI 2019), who built CAMUS, report an average inter-observer
Dice of 0.899 and an average Hausdorff distance of 7.34 mm for this data. Our
model reaches 0.9151 mean Dice, i.e. it already agrees with the reference
annotator better than two cardiologists agree with each other. Above that
level, Dice is measuring one annotator's idiosyncrasies rather than anatomy,
which is exactly why seven ablation axes moved it by nothing.

So the meaningful question is not "can we beat the leaderboard" but "are we at
the annotation ceiling", and that is answerable directly: CAMUS ships a
multi-observer subset (fold 5 -- 50 patients contoured by a second and third
cardiologist O2/O3, plus a repeat pass O1b by the original annotator seven
months later). This script computes, under the SAME protocol used for every
other number in this project:

    O1a vs O1b   intra-observer agreement  (the reproducibility ceiling)
    O1a vs O2    inter-observer agreement  (the between-expert ceiling)
    model vs O1a / O2 / O1b

CAVEAT THIS SCRIPT ENFORCES: the multi-observer patients are a specific CAMUS
fold, which is NOT the same 50 patients as this project's custom test split.
Any of them that fall in our TRAINING set are contaminated -- the model saw
their reference contours. The script reports the overlap and, by default,
scores only the uncontaminated subset.

The second-observer files are not part of the standard CAMUS download used
here; obtain them from the CAMUS resource page and point --obs-dir at them.
Expected layout (either works):
    <obs-dir>/patientXXXX/patientXXXX_{2CH,4CH}_{ED,ES}_gt_O2.nii[.gz]
    <obs-dir>/patientXXXX_{2CH,4CH}_{ED,ES}_gt_O2.nii[.gz]

Usage:
    python scripts/yolo/eval_observer_variability.py --obs-dir path/to/O2
    python scripts/yolo/eval_observer_variability.py --obs-dir path/to/O2 \
        --weights yolo_runs/E1_filled/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "yolo"))

from scoring import CamusScorer  # noqa: E402
from yolo_common import binaries_to_labelmap, load_meta, native_gt  # noqa: E402

VIEWS = ("2CH", "4CH")
PHASES = ("ED", "ES")


def read_split(name: str) -> set[str]:
    f = ROOT / "data" / "splits" / f"{name}.txt"
    return {ln.strip() for ln in f.read_text().splitlines() if ln.strip()} if f.exists() else set()


def find_obs_file(obs_dir: Path, pid: str, view: str, phase: str) -> Path | None:
    """Locate a second-observer mask, tolerating the common layouts."""
    pats = [
        f"{pid}/{pid}_{view}_{phase}_gt*",
        f"{pid}_{view}_{phase}_gt*",
        f"**/{pid}_{view}_{phase}_gt*",
    ]
    for p in pats:
        for f in sorted(obs_dir.glob(p)):
            if f.suffix in (".nii", ".gz"):
                return f
    return None


def load_mask(path: Path) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.load(str(path)).dataobj).astype(np.int64)


def spacing_of(pid: str, view: str, phase: str, camus: Path) -> tuple[float, float]:
    import nibabel as nib

    for ext in (".nii", ".nii.gz"):
        f = camus / pid / f"{pid}_{view}_{phase}{ext}"
        if f.exists():
            z = nib.load(str(f)).header.get_zooms()
            if len(z) >= 2 and z[0] > 0 and z[1] > 0:
                return (float(z[0]), float(z[1]))
    return (1.0, 1.0)


def score_pair(pairs, camus: Path, label: str) -> dict:
    """Score one annotation set against another under the project protocol."""
    sc = CamusScorer()
    for pid, view, phase, a, b in pairs:
        sc.add(a, b, spacing_of(pid, view, phase, camus),
               meta={"patient_id": pid, "view": view, "phase": phase})
    r = sc.summary()
    print(f"  {label:22s} Dice={r.get('dice_mean', float('nan')):.4f}  "
          f"HD95={r.get('hd95_mean', float('nan')):.3f} mm  "
          f"ASSD={r.get('assd_mean', float('nan')):.3f} mm  (n={r.get('n_samples', 0)})")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dir", required=True,
                    help="directory of second-observer (O2) annotations")
    ap.add_argument("--obs2-dir", default=None,
                    help="optional third set, e.g. the O1b repeat pass")
    ap.add_argument("--camus-root", default=str(ROOT / "data" / "CAMUS"))
    ap.add_argument("--weights", default=None,
                    help="optional YOLO weights, to score the model on the same frames")
    ap.add_argument("--dataset", default=str(ROOT / "yolo_data" / "camus_edes"))
    ap.add_argument("--encoding", default="filled")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--include-train", action="store_true",
                    help="also score patients seen during training (CONTAMINATED)")
    ap.add_argument("--out", default=str(ROOT / "results" / "yolo" / "observer_variability.json"))
    args = ap.parse_args()

    camus = Path(args.camus_root)
    obs_dir = Path(args.obs_dir)
    if not obs_dir.exists():
        raise SystemExit(f"--obs-dir not found: {obs_dir}")

    # which patients have a second observer?
    pids = sorted({m.group(1) for m in
                   (re.search(r"(patient\d+)", str(p)) for p in obs_dir.rglob("*"))
                   if m})
    if not pids:
        raise SystemExit(f"no patientXXXX files found under {obs_dir}")

    train, val, test = read_split("train"), read_split("val"), read_split("test")
    contaminated = sorted(set(pids) & train)
    clean = [p for p in pids if p not in train]

    print(f"multi-observer patients found : {len(pids)}")
    print(f"  in our TRAIN split (contaminated): {len(contaminated)}")
    print(f"  in our val split                 : {len(set(pids) & val)}")
    print(f"  in our test split                : {len(set(pids) & test)}")
    print(f"  usable (unseen in training)      : {len(clean)}")
    if contaminated and not args.include_train:
        print("  -> scoring the usable subset only; pass --include-train to override")
    use = pids if args.include_train else clean
    if not use:
        raise SystemExit("no uncontaminated multi-observer patients; nothing to report")

    # gather aligned annotation pairs
    ref_obs, ref_obs2 = [], []
    frames = []
    for pid in use:
        for view in VIEWS:
            for phase in PHASES:
                f2 = find_obs_file(obs_dir, pid, view, phase)
                if f2 is None:
                    continue
                try:
                    o1 = native_gt(camus, pid, view, phase)
                except FileNotFoundError:
                    continue
                o2 = load_mask(f2)
                if o2.shape != o1.shape:
                    print(f"  [skip] {pid} {view} {phase}: shape {o2.shape} vs {o1.shape}")
                    continue
                ref_obs.append((pid, view, phase, o2, o1))
                frames.append((pid, view, phase, o1))
                if args.obs2_dir:
                    f3 = find_obs_file(Path(args.obs2_dir), pid, view, phase)
                    if f3 is not None:
                        o3 = load_mask(f3)
                        if o3.shape == o1.shape:
                            ref_obs2.append((pid, view, phase, o3, o1))

    if not ref_obs:
        raise SystemExit("no matching annotation pairs found")

    print(f"\naligned frames: {len(ref_obs)}\n")
    results = {"n_patients": len(use), "n_frames": len(ref_obs),
               "contaminated_excluded": [] if args.include_train else contaminated}

    print("annotation ceiling (scored with the project protocol):")
    results["inter_observer_O2_vs_O1"] = score_pair(ref_obs, camus, "O2 vs O1 (inter)")
    if ref_obs2:
        results["intra_observer_O1b_vs_O1a"] = score_pair(ref_obs2, camus, "O1b vs O1a (intra)")

    # optionally score the model on exactly these frames
    if args.weights and Path(args.weights).exists():
        import cv2
        from ultralytics import YOLO

        model = YOLO(args.weights)
        meta = load_meta(Path(args.dataset))
        img_root = Path(args.dataset) / "images"
        kw = dict(imgsz=args.imgsz, retina_masks=True, device=args.device, verbose=False)

        preds = []
        missing = 0
        for pid, view, phase, o1 in frames:
            stem = f"{pid}_{view}_{phase}"
            src = next((img_root / s / f"{stem}.png" for s in ("train", "val", "test")
                        if (img_root / s / f"{stem}.png").exists()), None)
            if src is None:
                missing += 1
                continue
            h, w = o1.shape[:2]
            res = model.predict(source=str(src), **kw)[0]
            best = {}
            if res.masks is not None and len(res.masks):
                md = res.masks.data.cpu().numpy()
                cl = res.boxes.cls.cpu().numpy().astype(int)
                cf = res.boxes.conf.cpu().numpy()
                for c in (0, 1, 2):
                    idx = np.where(cl == c)[0]
                    if len(idx):
                        m = md[idx[int(np.argmax(cf[idx]))]] > 0.5
                        if m.shape[:2] != (h, w):
                            m = cv2.resize(m.astype(np.uint8), (w, h),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                        best[c] = m
            pred = binaries_to_labelmap(best.get(0), best.get(1), best.get(2),
                                        args.encoding, shape=(h, w))
            preds.append((pid, view, phase, pred, o1))
        if missing:
            print(f"  [note] {missing} frames had no exported PNG and were skipped")
        if preds:
            print("\nmodel on the same frames:")
            results["model_vs_O1"] = score_pair(preds, camus, "model vs O1")
            o2map = {(p, v, f): a for p, v, f, a, _ in ref_obs}
            pm = [(p, v, f, pr, o2map[(p, v, f)]) for p, v, f, pr, _ in preds
                  if (p, v, f) in o2map]
            if pm:
                results["model_vs_O2"] = score_pair(pm, camus, "model vs O2")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nSaved {out}")
    print("\nRead the model row against the observer rows: matching or exceeding "
          "inter-observer agreement is the meaningful claim, not a leaderboard rank.")


if __name__ == "__main__":
    main()
