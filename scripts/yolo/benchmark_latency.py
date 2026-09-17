#!/usr/bin/env python3
"""
One latency protocol applied to every model, so Table 1's timing column means
something.

The problem this solves
-----------------------
The published baseline timings are a bare forward pass at 256x256; the YOLO
timing was end-to-end `predict()` at 640x640 including disk read, letterboxing
and native-resolution mask upsampling. Those are not the same measurement, and
the first YOLO result looked ~10x slower partly for that reason.

Three effects were confounded -- input resolution, architecture, and framework
overhead -- so this measures each separately:

  raw_fwd    bare nn.Module forward, no pre/post-processing. Architecture only.
             Every model is measured at BOTH 256 and 640 so resolution is
             isolated from architecture. (Timing needs no retraining; accuracy
             is NOT claimed at the off-design size.)
  e2e        Ultralytics `predict()` end to end -- what you actually pay in
             that runtime.
  onnx       ONNX Runtime (CUDA provider) on the exported graph -- what you pay
             after leaving the training framework.

All timings: batch 1, CUDA-synchronised around each iteration, warmup
discarded, median reported (mean is skewed by scheduler noise on a laptop).

Usage:
    python scripts/yolo/benchmark_latency.py
    python scripts/yolo/benchmark_latency.py --skip-onnx
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

BASELINES = [
    "unet_v1", "unet_v2", "unet_resnet", "deeplab_v3", "nnunet",
    "dense_context_unet", "swin_unet", "transunet", "fpn",
]
# Swin's windowed attention is shape-constrained; 224/448 are its valid sizes.
SIZES_FOR = {"swin_unet": (224, 448)}


def timeit(fn, warmup: int, iters: int, device: torch.device) -> dict:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ts)
    return {"median_ms": float(np.median(a)), "mean_ms": float(a.mean()),
            "std_ms": float(a.std()), "p90_ms": float(np.percentile(a, 90))}


def bench_baselines(sizes, warmup, iters, device, ckpt_dir: Path) -> dict:
    from evaluate_all_models import _construct_and_load

    out = {}
    for name in BASELINES:
        ckpt = ckpt_dir / name / "best_model.pth"
        if not ckpt.exists():
            print(f"[skip] {name}: no checkpoint")
            continue
        try:
            model = _construct_and_load({"name": name, "checkpoint": ckpt}, device)
            model.to(device).eval()
        except Exception as e:
            print(f"[fail] {name}: {type(e).__name__}: {str(e)[:90]}")
            continue

        out[name] = {}
        for sz in SIZES_FOR.get(name, sizes):
            x = torch.randn(1, 1, sz, sz, device=device)
            try:
                with torch.no_grad():
                    r = timeit(lambda: model(x), warmup, iters, device)
                out[name][f"raw_fwd_{sz}"] = r
                print(f"  {name:20s} raw_fwd @{sz:<4d} {r['median_ms']:7.2f} ms")
            except Exception as e:
                print(f"  {name:20s} @{sz}: {type(e).__name__}: {str(e)[:70]}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def bench_yolo(weights: Path, sizes, warmup, iters, device, skip_onnx: bool) -> dict:
    from ultralytics import YOLO

    res: dict = {}
    y = YOLO(str(weights))
    core = y.model.to(device).eval()

    for sz in sizes:
        x = torch.randn(1, 3, sz, sz, device=device)
        with torch.no_grad():
            r = timeit(lambda: core(x), warmup, iters, device)
        res[f"raw_fwd_{sz}"] = r
        print(f"  {'YOLO':20s} raw_fwd @{sz:<4d} {r['median_ms']:7.2f} ms")

    # half precision -- CUDA only; fp16 on CPU is emulated and meaninglessly slow
    if device.type != "cuda":
        print("  (fp16 skipped: CPU)")
    else:
        try:
            half = YOLO(str(weights)).model.to(device).half().eval()
            x = torch.randn(1, 3, sizes[-1], sizes[-1], device=device, dtype=torch.half)
            with torch.no_grad():
                r = timeit(lambda: half(x), warmup, iters, device)
            res[f"raw_fwd_fp16_{sizes[-1]}"] = r
            print(f"  {'YOLO':20s} raw_fp16 @{sizes[-1]:<3d} {r['median_ms']:7.2f} ms")
            del half
        except Exception as e:
            print(f"  fp16 failed: {type(e).__name__}: {str(e)[:70]}")

    # end-to-end predict(), the number the evaluator reports
    import cv2

    tmp = ROOT / "yolo_data" / "camus_edes" / "images" / "test"
    img = next(tmp.glob("*.png"), None)
    if img is not None:
        arr = cv2.imread(str(img))
        for sz in sizes:
            r = timeit(lambda: y.predict(source=arr, imgsz=sz, retina_masks=True,
                                         device=str(device.index or 0), verbose=False),
                       max(2, warmup // 2), max(5, iters // 2), device)
            res[f"e2e_{sz}"] = r
            print(f"  {'YOLO':20s} e2e     @{sz:<4d} {r['median_ms']:7.2f} ms")

    # ONNX Runtime on the exported graph
    if not skip_onnx:
        try:
            import onnxruntime as ort

            sz = sizes[-1]
            path = YOLO(str(weights)).export(format="onnx", imgsz=sz, opset=13,
                                             simplify=False, verbose=False)
            prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                    if p in ort.get_available_providers()]
            sess = ort.InferenceSession(str(path), providers=prov)
            iname = sess.get_inputs()[0].name
            xin = np.random.rand(1, 3, sz, sz).astype(np.float32)
            r = timeit(lambda: sess.run(None, {iname: xin}), warmup, iters,
                       torch.device("cpu"))   # ORT syncs internally
            res[f"onnx_{sz}"] = r
            res["onnx_provider"] = prov[0]
            print(f"  {'YOLO':20s} onnx    @{sz:<4d} {r['median_ms']:7.2f} ms "
                  f"({prov[0]})")
        except Exception as e:
            print(f"  onnx failed: {type(e).__name__}: {str(e)[:110]}")

    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None,
                    help="YOLO weights; default = best E9 seed, else E1_filled")
    ap.add_argument("--checkpoint-dir", default=str(ROOT / "results" / "base_models"))
    ap.add_argument("--sizes", nargs="+", type=int, default=[256, 640])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-onnx", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "yolo" / "latency_fair.json"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    print(f"protocol: batch 1, {args.warmup} warmup, {args.iters} timed, "
          f"CUDA-synchronised, median reported\n")

    w = args.weights
    if w is None:
        cands = sorted(ROOT.glob("yolo_runs/E9_seed*/weights/best.pt"))
        w = str(cands[0]) if cands else str(ROOT / "yolo_runs" / "E1_filled" / "weights" / "best.pt")

    print("=== baselines (bare forward) ===")
    base = bench_baselines(args.sizes, args.warmup, args.iters, device,
                           Path(args.checkpoint_dir))
    print("\n=== YOLO ===")
    yolo = bench_yolo(Path(w), args.sizes, args.warmup, args.iters, device,
                      args.skip_onnx) if Path(w).exists() else {}
    if not yolo:
        print(f"[warn] YOLO weights not found: {w}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "weights": w,
                               "protocol": {"batch": 1, "warmup": args.warmup,
                                            "iters": args.iters,
                                            "statistic": "median"},
                               "baselines": base, "yolo": yolo}, indent=1))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
