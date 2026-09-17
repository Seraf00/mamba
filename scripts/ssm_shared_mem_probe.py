#!/usr/bin/env python3
"""
Probe one SSM configuration on the local GPU and report, as JSON on stdout:
  - whether a forward+backward step completes
  - the Triton shared-memory request when it does not
  - the SSM block dimensions that determine that request

Run one config per process so a poisoned CUDA context cannot affect the next.
"""
import argparse, json, sys, traceback

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

SWIN = {"mamba_swin_unet"}


def device_max_shared_mem():
    try:
        import triton
        return int(triton.runtime.driver.active.utils.get_device_properties(0)["max_shared_mem"])
    except Exception:
        return None


def block_dims(model):
    """Collect the dims of every Mamba-2 / VMamba block in the model."""
    out = []
    for name, m in model.named_modules():
        t = type(m).__name__
        if t == "Mamba2Block":
            out.append({"block": t, "name": name, "dim": getattr(m, "dim", None),
                        "d_inner": getattr(m, "d_inner", None),
                        "d_state": getattr(m, "d_state", None),
                        "n_heads": getattr(m, "n_heads", None),
                        "head_dim": getattr(m, "head_dim", None),
                        "chunk_size": getattr(m, "chunk_size", None)})
        elif t in ("VMMambaBlock", "SS2D"):
            out.append({"block": t, "name": name, "dim": getattr(m, "dim", None),
                        "d_state": getattr(m, "d_state", None)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mamba-type", required=True)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    rec = {"model": a.model, "mamba_type": a.mamba_type, "batch": a.batch,
           "max_shared_mem_bytes": device_max_shared_mem()}

    import torch
    from models import get_model

    sz = 224 if a.model in SWIN else 256
    rec["img_size"] = sz
    d = torch.device("cuda")
    rec["gpu"] = torch.cuda.get_device_name(0)
    p = torch.cuda.get_device_properties(0)
    rec["compute_capability"] = f"{p.major}.{p.minor}"

    try:
        model = get_model(a.model, in_channels=1, num_classes=4, mamba_type=a.mamba_type)
        rec["params_M"] = round(sum(q.numel() for q in model.parameters()) / 1e6, 2)
        rec["ssm_blocks"] = block_dims(model)
        dims = [b["dim"] for b in rec["ssm_blocks"] if b.get("dim")]
        rec["max_block_dim"] = max(dims) if dims else None
        model = model.to(d)
    except Exception as e:
        rec.update(status="BUILD_FAIL", error_type=type(e).__name__,
                   error=str(e)[:400])
        print(json.dumps(rec)); return

    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        x = torch.randn(a.batch, 1, sz, sz, device=d)
        y = torch.randint(0, 4, (a.batch, sz, sz), device=d)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                o = model(x)
                o = o["out"] if isinstance(o, dict) else o
                if isinstance(o, (list, tuple)):
                    o = o[0]
                loss = torch.nn.functional.cross_entropy(o, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        rec.update(status="OK",
                   peak_mib=round(torch.cuda.max_memory_allocated() / 2**20))
    except Exception as e:
        msg = str(e)
        et = type(e).__name__
        kind = "OTHER"
        if "OutOfResources" in et or "out of resource" in msg.lower():
            kind = "TRITON_SHARED_MEM"
        elif "out of memory" in msg.lower():
            kind = "CUDA_OOM"
        rec.update(status="FAIL", failure_kind=kind, error_type=et,
                   error=msg[:600],
                   traceback_tail=traceback.format_exc()[-400:])
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
