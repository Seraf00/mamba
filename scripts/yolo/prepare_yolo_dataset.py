#!/usr/bin/env python3
"""
Convert the CAMUS dataset into Ultralytics YOLO instance-segmentation format.

CAMUS ground truth uses a 4-label map:
    0 = background
    1 = LV endocardium (blood pool)
    2 = LV myocardium  (the ring between endo- and epicardial contours)
    3 = Left atrium

Label 2 is an ANNULUS: it is not simply connected, so it cannot be written as a
single YOLO polygon without either (a) filling the hole, or (b) cutting a seam
from the outer to the inner contour. We support both encodings so the choice
can be ablated:

  --encoding filled  (default)  classes are {LV_endo, LV_epi(=1 or 2), LA}.
                                The epicardial class is the FILLED contour, which
                                is simply connected. At evaluation time the
                                myocardium is recovered as epi minus endo, so the
                                mask head only ever has to produce blob shapes.
  --encoding seam               classes are {LV_endo, LV_myo(ring), LA}. The ring
                                is encoded as one polygon with a zero-width seam
                                joining the outer and inner contours, so the mask
                                head has to produce a genuine annulus.

Note there is no "outer contour only" encoding: the outer contour of the
myocardial ring IS the epicardial contour, so that option is mathematically
identical to `filled` rather than a distinct (worse) baseline.

Usage:
    python scripts/yolo/prepare_yolo_dataset.py --out camus_edes
    python scripts/yolo/prepare_yolo_dataset.py --out camus_seq --include-sequences
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Class layout written to the YOLO label files (index -> name)
ENCODING_CLASSES = {
    "filled": ["LV_endo", "LV_epi", "LA"],
    "seam": ["LV_endo", "LV_myo", "LA"],
}


def read_split(split_file: Path) -> list[str]:
    return [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]


def parse_info_cfg(path: Path) -> dict:
    info = {}
    if not path.exists():
        return info
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return info


def largest_contour(binary: np.ndarray) -> np.ndarray | None:
    """Return the largest external contour of a binary mask as (N, 2) int array."""
    cnts, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 4:
        return None
    return c.reshape(-1, 2)


def resample_contour(pts: np.ndarray, n_points: int) -> np.ndarray:
    """Uniformly resample a closed contour to exactly n_points along arc length."""
    if n_points <= 0 or len(pts) <= n_points:
        return pts
    closed = np.vstack([pts, pts[:1]]).astype(np.float64)
    seg = np.sqrt(((closed[1:] - closed[:-1]) ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return pts
    targets = np.linspace(0.0, total, n_points, endpoint=False)
    idx = np.searchsorted(cum, targets, side="right") - 1
    idx = np.clip(idx, 0, len(seg) - 1)
    denom = np.where(seg[idx] > 0, seg[idx], 1.0)
    t = ((targets - cum[idx]) / denom)[:, None]
    return closed[idx] + t * (closed[idx + 1] - closed[idx])


def annulus_seam_polygon(ring: np.ndarray, n_points: int) -> np.ndarray | None:
    """
    Encode an annulus as a single polygon by cutting a seam between its outer
    and inner contour: outer -> seam -> inner (reversed) -> seam back.
    """
    cnts, hier = cv2.findContours(
        ring.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if not cnts:
        return None
    hier = hier[0]
    outer_idx = [i for i in range(len(cnts)) if hier[i][3] == -1]
    if not outer_idx:
        return None
    o = max(outer_idx, key=lambda i: cv2.contourArea(cnts[i]))
    outer = cnts[o].reshape(-1, 2)
    holes = [i for i in range(len(cnts)) if hier[i][3] == o]
    if not holes:
        return resample_contour(outer, n_points)
    h = max(holes, key=lambda i: cv2.contourArea(cnts[i]))
    inner = cnts[h].reshape(-1, 2)

    half = max(n_points // 2, 8) if n_points > 0 else 0
    outer_r = resample_contour(outer, half).astype(np.float64)
    inner_r = resample_contour(inner, half).astype(np.float64)

    # Rotate both rings so the seam is the shortest outer->inner connection
    d = np.linalg.norm(outer_r[:, None, :] - inner_r[None, :, :], axis=2)
    oi, ii = np.unravel_index(np.argmin(d), d.shape)
    outer_r = np.roll(outer_r, -oi, axis=0)
    inner_r = np.roll(inner_r, -ii, axis=0)
    # inner traversed in the opposite winding so the hole is subtracted
    inner_r = inner_r[::-1]
    inner_r = np.roll(inner_r, 1, axis=0)

    return np.vstack([outer_r, outer_r[:1], inner_r, inner_r[:1], outer_r[:1]])


def mask_to_polygons(mask: np.ndarray, encoding: str, n_points: int) -> list[tuple[int, np.ndarray]]:
    """Return [(class_index, polygon_points_xy), ...] for one label map."""
    out: list[tuple[int, np.ndarray]] = []

    endo = mask == 1
    myo = mask == 2
    la = mask == 3

    # class 0: LV endocardium
    c = largest_contour(endo)
    if c is not None:
        out.append((0, resample_contour(c, n_points)))

    # class 1: epicardium / myocardium, depending on encoding
    if encoding == "filled":
        c = largest_contour(endo | myo)
        if c is not None:
            out.append((1, resample_contour(c, n_points)))
    elif encoding == "seam":
        p = annulus_seam_polygon(myo, n_points)
        if p is not None:
            out.append((1, p))

    # class 2: left atrium
    c = largest_contour(la)
    if c is not None:
        out.append((2, resample_contour(c, n_points)))

    return out


def write_label(path: Path, polys: list[tuple[int, np.ndarray]], h: int, w: int) -> None:
    lines = []
    for cls, pts in polys:
        p = pts.astype(np.float64).copy()
        p[:, 0] = np.clip(p[:, 0] / w, 0.0, 1.0)   # x <- column
        p[:, 1] = np.clip(p[:, 1] / h, 0.0, 1.0)   # y <- row
        if len(p) < 3:
            continue
        coords = " ".join(f"{v:.6f}" for v in p.reshape(-1))
        lines.append(f"{cls} {coords}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="./data/CAMUS")
    ap.add_argument("--splits-dir", default="./data/splits")
    ap.add_argument("--out-root", default="./yolo_data")
    ap.add_argument("--out", required=True, help="dataset name, e.g. camus_edes")
    ap.add_argument("--encoding", default="filled", choices=list(ENCODING_CLASSES))
    ap.add_argument("--n-points", type=int, default=96,
                    help="polygon vertices per contour (0 = keep every boundary pixel)")
    ap.add_argument("--include-sequences", action="store_true",
                    help="also export every annotated half-sequence frame (train split only)")
    ap.add_argument("--seq-stride", type=int, default=1,
                    help="keep every Nth sequence frame")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_root) / args.out
    class_names = ENCODING_CLASSES[args.encoding]

    for sub in ("images", "labels"):
        for sp in ("train", "val", "test"):
            (out_dir / sub / sp).mkdir(parents=True, exist_ok=True)

    meta: dict[str, dict] = {}
    counts = {"train": 0, "val": 0, "test": 0}

    for split in ("train", "val", "test"):
        pids = read_split(Path(args.splits_dir) / f"{split}.txt")
        for pid in tqdm(pids, desc=f"{split:5s}", ncols=80):
            pdir = data_dir / pid
            for view in ("2CH", "4CH"):
                info = parse_info_cfg(pdir / f"Info_{view}.cfg")
                quality = info.get("ImageQuality", "Unknown")
                try:
                    ef = float(info.get("EF", "nan"))
                except ValueError:
                    ef = float("nan")

                # ---- ED / ES frames -------------------------------------
                for phase in ("ED", "ES"):
                    ip = pdir / f"{pid}_{view}_{phase}.nii"
                    gp = pdir / f"{pid}_{view}_{phase}_gt.nii"
                    if not (ip.exists() and gp.exists()):
                        continue
                    nii = nib.load(str(ip))
                    img = np.asarray(nii.dataobj)
                    msk = np.asarray(nib.load(str(gp)).dataobj).astype(np.int64)
                    zooms = nii.header.get_zooms()
                    spacing = (float(zooms[0]), float(zooms[1])) if len(zooms) >= 2 else (1.0, 1.0)
                    if not (spacing[0] > 0 and spacing[1] > 0):
                        spacing = (1.0, 1.0)

                    stem = f"{pid}_{view}_{phase}"
                    h, w = msk.shape[:2]
                    cv2.imwrite(str(out_dir / "images" / split / f"{stem}.png"), to_uint8(img))
                    write_label(out_dir / "labels" / split / f"{stem}.txt",
                                mask_to_polygons(msk, args.encoding, args.n_points), h, w)
                    meta[stem] = {
                        "patient_id": pid, "view": view, "phase": phase, "split": split,
                        "spacing": spacing, "shape": [h, w],
                        "quality": quality, "ef": ef, "is_sequence": False,
                    }
                    counts[split] += 1

                # ---- half-sequence frames (train only) ------------------
                if args.include_sequences and split == "train":
                    sp_i = pdir / f"{pid}_{view}_half_sequence.nii"
                    sp_g = pdir / f"{pid}_{view}_half_sequence_gt.nii"
                    if not (sp_i.exists() and sp_g.exists()):
                        continue
                    seq = np.asarray(nib.load(str(sp_i)).dataobj)
                    seg = np.asarray(nib.load(str(sp_g)).dataobj).astype(np.int64)
                    n = min(seq.shape[-1], seg.shape[-1])
                    for f in range(0, n, args.seq_stride):
                        m = seg[..., f]
                        if m.max() == 0:
                            continue
                        stem = f"{pid}_{view}_seq{f:03d}"
                        h, w = m.shape[:2]
                        cv2.imwrite(str(out_dir / "images" / split / f"{stem}.png"),
                                    to_uint8(seq[..., f]))
                        write_label(out_dir / "labels" / split / f"{stem}.txt",
                                    mask_to_polygons(m, args.encoding, args.n_points), h, w)
                        counts[split] += 1

    # ---- dataset yaml + metadata -------------------------------------------
    yaml_path = out_dir / "data.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    yaml_path.write_text(
        f"# CAMUS -> YOLO segmentation ({args.encoding} encoding)\n"
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n\n"
        f"names:\n{names}\n"
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    print(f"\nWrote {out_dir}")
    print(f"  encoding : {args.encoding}  classes={class_names}")
    print(f"  images   : train={counts['train']} val={counts['val']} test={counts['test']}")
    print(f"  yaml     : {yaml_path}")


if __name__ == "__main__":
    main()
