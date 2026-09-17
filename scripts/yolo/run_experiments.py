#!/usr/bin/env python3
"""
Execute the YOLO-on-CAMUS experiment plan in `experiments.yaml`.

For every run it: builds the dataset if missing -> trains -> evaluates on the
held-out test split -> appends a row to results/yolo/all_runs.csv.

The plan is RESUMABLE. A run whose evaluation.json already exists is skipped, so
an interrupted multi-day sweep can simply be relaunched with the same command.

Usage:
    # see what would happen, run nothing
    python scripts/yolo/run_experiments.py --dry-run

    # everything, in plan order
    python scripts/yolo/run_experiments.py

    # just one group (or a few)
    python scripts/yolo/run_experiments.py --groups E1_encoding E3_imgsz

    # re-do a run that already has results
    python scripts/yolo/run_experiments.py --groups E2_scale --force
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PY = sys.executable

TRAIN_FLAG = {
    "model": "--model", "data": "--data", "epochs": "--epochs", "imgsz": "--imgsz",
    "batch": "--batch", "device": "--device", "workers": "--workers", "seed": "--seed",
    "patience": "--patience", "cache": "--cache", "optimizer": "--optimizer",
    "lr0": "--lr0", "lrf": "--lrf", "box": "--box", "cls": "--cls", "dfl": "--dfl",
    "overlap_mask": "--overlap-mask", "mask_ratio": "--mask-ratio",
    "degrees": "--degrees", "translate": "--translate", "scale": "--scale",
    "shear": "--shear", "fliplr": "--fliplr", "flipud": "--flipud",
    "mosaic": "--mosaic", "mixup": "--mixup", "copy_paste": "--copy-paste",
    "hsv_h": "--hsv-h", "hsv_s": "--hsv-s", "hsv_v": "--hsv-v",
    "erasing": "--erasing", "amp": "--amp",
    "boundary_weight": "--boundary-weight", "boundary_band": "--boundary-band",
}
TRAIN_BOOL = {"cos_lr": "--cos-lr", "coco_aug": "--coco-aug"}

# keys that configure evaluation rather than training
EVAL_ONLY = {"encoding", "conf", "iou", "tta", "half"}


def sh(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return p.returncode


def ensure_dataset(spec: dict, force: bool) -> None:
    name = spec["name"]
    out = ROOT / "yolo_data" / name
    if (out / "data.yaml").exists() and not force:
        print(f"[dataset] {name}: present")
        return
    print(f"[dataset] {name}: building")
    cmd = [PY, str(HERE / "prepare_yolo_dataset.py"), "--out", name,
           "--encoding", spec.get("encoding", "filled"),
           "--n-points", str(spec.get("n_points", 96))]
    if spec.get("include_sequences"):
        cmd += ["--include-sequences", "--seq-stride", str(spec.get("seq_stride", 1))]
    rc = sh(cmd, ROOT / "results" / "yolo" / "logs" / f"dataset_{name}.log")
    if rc != 0:
        raise SystemExit(f"dataset build failed: {name} (rc={rc})")


def train_cmd(cfg: dict) -> list[str]:
    cmd = [PY, str(HERE / "train_yolo.py"), "--name", cfg["name"]]
    for key, flag in TRAIN_FLAG.items():
        if key in cfg:
            cmd += [flag, str(cfg[key])]
    for key, flag in TRAIN_BOOL.items():
        if cfg.get(key):
            cmd += [flag]
    return cmd


def eval_cmd(cfg: dict, weights: Path) -> list[str]:
    ds = Path(cfg["data"]).parent
    cmd = [PY, str(HERE / "eval_yolo.py"),
           "--weights", str(weights),
           "--dataset", str(ds),
           "--encoding", cfg.get("encoding", "filled"),
           "--imgsz", str(cfg.get("imgsz", 640)),
           "--name", cfg["name"],
           "--device", str(cfg.get("device", "0"))]
    if "conf" in cfg:
        cmd += ["--conf", str(cfg["conf"])]
    if "iou" in cfg:
        cmd += ["--iou", str(cfg["iou"])]
    if cfg.get("tta"):
        cmd += ["--tta"]
    if cfg.get("half"):
        cmd += ["--half"]
    return cmd


def _canonical(cmd: list[str], parser, drop: set[str]) -> str:
    """
    Hash a command by parsing it with its own argparse parser, so that an
    explicitly-passed default (e.g. --mask-ratio 4) hashes identically to
    leaving it unset. Run-identifying fields in `drop` are excluded.
    """
    argv = [str(c) for c in cmd[2:]]           # strip interpreter + script path
    ns = vars(parser.parse_args(argv))
    ns = {k: v for k, v in sorted(ns.items()) if k not in drop}
    # normalise paths so ./x and absolute x agree
    for k in ("data", "dataset", "weights"):
        if k in ns and ns[k]:
            ns[k] = Path(str(ns[k])).resolve().as_posix()
    return hashlib.sha1(json.dumps(ns, default=str).encode()).hexdigest()[:12]


def train_hash(cmd: list[str]) -> str:
    import train_yolo

    return _canonical(cmd, train_yolo.build_parser(),
                      drop={"name", "project", "resume", "dry_run"})


def eval_hash(cmd: list[str]) -> str:
    import eval_yolo

    return _canonical(cmd, eval_yolo.build_parser(),
                      drop={"name", "out_root", "weights", "limit"})


def load_index(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_index(path: Path, idx: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(idx, indent=1))


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            existing = [r for r in csv.DictReader(fh) if r.get("run") != row["run"]]
    rows = existing + [row]
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarise(name: str, group: str, cfg: dict) -> dict | None:
    ev = ROOT / "results" / "yolo" / name / "evaluation.json"
    if not ev.exists():
        return None
    d = json.loads(ev.read_text())
    row = {"run": name, "group": group}
    for k in ("model", "imgsz", "batch", "epochs", "seed", "encoding",
              "mask_ratio", "data"):
        if k in cfg:
            row[k] = cfg[k]
    for k in ("dice_mean", "dice_std", "dice_lv_endocardium", "dice_lv_epicardium",
              "dice_left_atrium", "iou_mean", "hd95_mean", "hd95_lv_endocardium",
              "hd95_lv_epicardium", "hd95_left_atrium", "assd_mean", "masd_mean",
              "dice_mean_ed", "dice_mean_es", "hd95_mean_ed", "hd95_mean_es",
              "latency_ms_mean", "fps", "params_M", "gflops", "n_samples",
              "n_missing_lv_endocardium", "n_missing_lv_epicardium",
              "n_missing_left_atrium"):
        if k in d:
            row[k] = d[k]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(HERE / "experiments.yaml"))
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run completed runs")
    ap.add_argument("--skip-datasets", action="store_true")
    ap.add_argument("--csv", default=str(ROOT / "results" / "yolo" / "all_runs.csv"))
    args = ap.parse_args()

    plan = yaml.safe_load(Path(args.plan).read_text())
    defaults = plan.get("defaults", {})
    all_groups = plan.get("groups", [])
    groups = all_groups
    if args.groups:
        groups = [g for g in groups if g["id"] in args.groups]
        if not groups:
            raise SystemExit(f"no groups matched {args.groups}")

    # datasets referenced by the selected groups
    if not args.skip_datasets:
        needed = set()
        for g in groups:
            for r in g["runs"]:
                needed.add(Path(r.get("data", defaults["data"])).parent.name)
        for spec in plan.get("datasets", []):
            if spec["name"] in needed:
                if args.dry_run:
                    print(f"[dataset] {spec['name']}: would ensure")
                else:
                    ensure_dataset(spec, force=False)

    logs = ROOT / "results" / "yolo" / "logs"
    index_path = ROOT / "results" / "yolo" / "_config_index.json"
    index = load_index(index_path)

    # Rebuild the dedup index from EVERY completed run in the plan, not just
    # the groups selected on this invocation. Registering only visited runs
    # meant that `--groups E11_best` could not see E9's identical
    # configurations and retrained all three from scratch (~5 GPU-hours).
    recovered = 0
    for g in all_groups:
        for run in g["runs"]:
            c = {**defaults, **{k: v for k, v in (run or {}).items() if v is not None}}
            n = c["name"]
            if not (ROOT / "results" / "yolo" / n / "evaluation.json").exists():
                continue
            if not (ROOT / "yolo_runs" / n / "weights" / "best.pt").exists():
                continue
            th = train_hash(train_cmd({k: v for k, v in c.items() if k not in EVAL_ONLY}))
            if th not in index:
                index[th] = n
                recovered += 1
    if recovered:
        save_index(index_path, index)
        print(f"[index] recovered {recovered} config hash(es) from completed runs")

    t_start = time.time()

    for g in groups:
        print(f"\n=== {g['id']}: {g.get('question', '')} ===")
        for run in g["runs"]:
            cfg = {**defaults, **{k: v for k, v in (run or {}).items() if v is not None}}
            name = cfg["name"]
            ev = ROOT / "results" / "yolo" / name / "evaluation.json"
            tcmd = train_cmd({k: v for k, v in cfg.items() if k not in EVAL_ONLY})

            if ev.exists() and not args.force:
                d = json.loads(ev.read_text())
                print(f"[skip] {name}: done (Dice={d.get('dice_mean', float('nan')):.4f})")
                # Register the completed run in the dedup index even though we
                # are skipping it. Without this a restart loses every hash
                # learned in the previous process, and a later run with an
                # identical config retrains from scratch instead of reusing
                # these weights.
                if (ROOT / "yolo_runs" / name / "weights" / "best.pt").exists():
                    index.setdefault(train_hash(tcmd), name)
                    save_index(index_path, index)
                row = summarise(name, g["id"], cfg)
                if row:
                    append_row(Path(args.csv), row)
                continue
            weights = ROOT / "yolo_runs" / name / "weights" / "best.pt"

            # Several groups share an identical configuration (each group needs
            # its own reference row). Train each unique config once and reuse.
            th = train_hash(tcmd)
            twin = index.get(th)
            reuse_w = None
            if twin and twin != name:
                cand = ROOT / "yolo_runs" / twin / "weights" / "best.pt"
                if cand.exists() or args.dry_run:
                    reuse_w = cand

            ecmd = eval_cmd(cfg, reuse_w or weights)
            eh = eval_hash(ecmd)

            if args.dry_run:
                tag = f"  (reuses weights from {twin})" if reuse_w else ""
                print(f"[plan] {name}{tag}")
                if not reuse_w:
                    print("  $ " + " ".join(str(c) for c in tcmd))
                print("  $ " + " ".join(str(c) for c in ecmd))
                index.setdefault(th, name)
                index.setdefault(f"eval:{th}:{eh}", name)
                continue

            print(f"[run ] {name}")
            t0 = time.time()

            if reuse_w is None:
                # Auto-resume a run that was interrupted part-way. Ultralytics
                # writes weights/last.pt every epoch, so a stopped sweep costs
                # only the current epoch rather than the whole run. The hash
                # above was taken from the ORIGINAL command, so swapping the
                # model path here does not disturb dedup.
                exec_cmd = list(tcmd)
                last = ROOT / "yolo_runs" / name / "weights" / "last.pt"
                if last.exists() and "--model" in exec_cmd:
                    exec_cmd[exec_cmd.index("--model") + 1] = str(last)
                    exec_cmd.append("--resume")
                    print(f"  resuming {name} from last.pt")

                rc = sh(exec_cmd, logs / f"{name}_train.log")
                if rc != 0:
                    print(f"  !! training failed (rc={rc}); see {logs / f'{name}_train.log'}")
                    continue
                if not weights.exists():
                    print(f"  !! no weights at {weights}")
                    continue
                index[th] = name
                save_index(index_path, index)
            else:
                print(f"  reusing weights from {twin} (identical training config)")

            # identical eval of identical weights -> copy the results across
            etwin = index.get(f"eval:{th}:{eh}")
            if etwin and etwin != name:
                src = ROOT / "results" / "yolo" / etwin
                if (src / "evaluation.json").exists():
                    dst = ROOT / "results" / "yolo" / name
                    dst.mkdir(parents=True, exist_ok=True)
                    for f in ("evaluation.json", "per_sample.json"):
                        if (src / f).exists():
                            shutil.copy2(src / f, dst / f)
                    print(f"  reusing evaluation from {etwin}")
                    row = summarise(name, g["id"], cfg)
                    if row:
                        row["duplicate_of"] = etwin
                        append_row(Path(args.csv), row)
                    continue

            rc = sh(ecmd, logs / f"{name}_eval.log")
            index[f"eval:{th}:{eh}"] = name
            save_index(index_path, index)
            if rc != 0:
                print(f"  !! eval failed (rc={rc}); see {logs / f'{name}_eval.log'}")
                continue
            row = summarise(name, g["id"], cfg)
            if row:
                row["train_seconds"] = round(time.time() - t0, 1)
                append_row(Path(args.csv), row)
                print(f"  Dice={row.get('dice_mean', float('nan')):.4f}  "
                      f"HD95={row.get('hd95_mean', float('nan')):.3f} mm  "
                      f"({row['train_seconds'] / 60:.1f} min)")

    # ---- inference-only sweeps on the final weights -------------------------
    sweeps = plan.get("inference_sweeps", [])
    # The nominal final run may have been deduplicated against an identical
    # earlier config, in which case it has no weights directory of its own.
    # Fall back to whichever completed run scored best and does have weights,
    # otherwise the whole inference block silently skips.
    base = ROOT / "yolo_runs" / "E9_seed0" / "weights" / "best.pt"
    if not base.exists():
        # Only runs trained under the SAME encoding and dataset as the sweep
        # defaults are eligible: a seam-encoded model scored with
        # --encoding filled would silently produce nonsense.
        want_enc = defaults.get("encoding", "filled")
        want_data = Path(defaults["data"]).parent.resolve()
        cands = []
        for d in (ROOT / "results" / "yolo").glob("*"):
            w = ROOT / "yolo_runs" / d.name / "weights" / "best.pt"
            ev = d / "evaluation.json"
            if not (w.exists() and ev.exists()):
                continue
            try:
                j = json.loads(ev.read_text())
            except json.JSONDecodeError:
                continue
            c = j.get("config") or {}
            if c.get("encoding") != want_enc:
                continue
            if c.get("dataset") and Path(c["dataset"]).resolve() != want_data:
                continue
            cands.append((j.get("dice_mean", -1), w))
        if cands:
            base = max(cands)[1]
            print(f"[note] E9_seed0 has no weights (deduplicated); inference "
                  f"sweeps will use {base.parent.parent.name} "
                  f"(encoding={want_enc})")
    if sweeps and (base.exists() or args.dry_run):
        print("\n=== inference sweeps (no retraining) ===")
        for s in sweeps:
            cfg = {**defaults, **s}
            cfg["name"] = s["name"]
            ev = ROOT / "results" / "yolo" / s["name"] / "evaluation.json"
            if ev.exists() and not args.force:
                print(f"[skip] {s['name']}: done")
                continue
            cmd = eval_cmd(cfg, base)
            if args.dry_run:
                print("  $ " + " ".join(str(c) for c in cmd))
                continue
            if sh(cmd, logs / f"{s['name']}_eval.log") == 0:
                row = summarise(s["name"], "inference", cfg)
                if row:
                    append_row(Path(args.csv), row)
    elif sweeps:
        print(f"\n[note] inference sweeps skipped: {base} not found (train E9 first)")

    print(f"\nTotal wall time: {(time.time() - t_start) / 3600:.2f} h")
    print(f"Summary CSV: {args.csv}")


if __name__ == "__main__":
    main()
