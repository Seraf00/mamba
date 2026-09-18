# ============================================================================
# GPU SHOOTOUT -- paste into ONE Colab cell. Run on G4, H100, A100.
# Standalone: no repo, no CAMUS. ~5 min (plus installs).
#
# Six sections, in order of how much they matter:
#
#   1 IDENTITY   what the card is, and whether this torch build has kernels
#                for its architecture at all
#   2 KERNELS    triton / causal-conv1d / mamba-ssm, and -- decisively -- each
#                real Mamba-2 configuration ACTUALLY RUN rather than predicted
#                from the 1280*P+512 formula. This section IS Paper C s4.3 on
#                this hardware.
#   3 NUMERICS   TF32, AMP, determinism -- the pinning the revision depends on
#   4 CAPACITY   can it hold DenseContextU-Net's 18.9 GiB at batch 8
#   5 SPEED      single-job throughput and MEASURED concurrency scaling
#   6 VERDICT    six lines to copy per GPU
#
# Install first if you want section 2 to mean anything:
#   !pip install -q causal-conv1d mamba-ssm --no-build-isolation
# That install is itself the test on a new architecture: if it needs a source
# build, you find out in minutes rather than six hours into R4.
# ============================================================================
import json, os, shutil, subprocess, sys, textwrap, time, traceback
import multiprocessing

BATCHES     = 30
WORKERS     = 3
CONCURRENCY = [1, 2, 4, 8]
SAVE_TO     = '/content/gpu_scorecard.json'
card = {}

def hdr(n, t):
    print('\n' + '=' * 78); print(f'{n}. {t}'); print('=' * 78)

def line(k, v, note=''):
    print(f'  {k:<26s} {v}' + (f'\n  {"":26s} {note}' if note else ''))

# ===========================================================================
hdr(1, 'IDENTITY AND ARCHITECTURE SUPPORT')
# ===========================================================================
import torch
if not torch.cuda.is_available():
    raise SystemExit('No GPU. Runtime > Change runtime type > GPU.')

p = torch.cuda.get_device_properties(0)
sm = f'sm_{p.major}{p.minor}'
line('GPU', torch.cuda.get_device_name(0))
line('architecture', f'{sm}  ({p.multi_processor_count} SMs)')
line('VRAM', f'{p.total_memory / 1024**3:.1f} GiB')
line('vCPU', multiprocessing.cpu_count())
try:
    import psutil
    line('system RAM', f'{psutil.virtual_memory().total / 1024**3:.1f} GiB')
except ImportError:
    pass
line('disk free', f'{shutil.disk_usage("/content" if os.path.isdir("/content") else ".").free / 1024**3:.0f} GiB')
line('torch', f'{torch.__version__}  cuda {torch.version.cuda}  '
              f'cudnn {torch.backends.cudnn.version()}')

# Which architectures this torch was COMPILED for. Informative, not pass/fail:
# an RTX 4060 is sm_89 and is absent from most arch lists, yet runs fine on
# sm_86 binaries via minor-version compatibility. The smoke test below is the
# real check -- a missing arch shows up as "no kernel image is available".
arch_list = torch.cuda.get_arch_list()
line('torch built for', ' '.join(arch_list))
line('this arch listed', 'yes' if sm in arch_list else
     f'NO -- relies on minor-version compat or PTX JIT (fine if the smoke test passes)')

# Does anything actually execute? This is what catches a genuine arch mismatch.
try:
    a = torch.randn(512, 512, device='cuda', requires_grad=True)
    (a @ a).sum().backward()
    c = torch.nn.Conv2d(4, 8, 3, padding=1).cuda()
    c(torch.randn(2, 4, 64, 64, device='cuda')).sum().backward()
    torch.cuda.synchronize()
    line('kernel smoke test', 'PASS  (matmul + conv2d fwd/bwd)')
    card['kernels_ok'] = True
except Exception as e:
    line('kernel smoke test', f'FAIL  {type(e).__name__}: {e}')
    card['kernels_ok'] = False

def max_shared_mem():
    try:
        import triton
        return int(triton.runtime.driver.active.utils
                   .get_device_properties(0)['max_shared_mem'])
    except Exception:
        pass
    for a_ in ('shared_memory_per_block_optin', 'sharedMemPerBlockOptin',
               'shared_memory_per_block'):
        v = getattr(p, a_, None)
        if v:
            return int(v)
    return None

shmem = max_shared_mem()
line('max shared mem/block', f'{shmem:,} B' if shmem else 'UNAVAILABLE')
line('bf16 supported', torch.cuda.is_bf16_supported())

card.update(gpu=torch.cuda.get_device_name(0), sm=sm,
            vram_gib=round(p.total_memory / 1024**3, 1),
            sm_count=p.multi_processor_count,
            vcpu=multiprocessing.cpu_count(),
            torch=torch.__version__, cuda=torch.version.cuda,
            arch_list=arch_list, arch_listed=sm in arch_list,
            max_shared_mem=shmem)

# ===========================================================================
hdr(2, 'KERNEL STACK  (blocking for R4/R5)')
# ===========================================================================
def probe(name, fn):
    try:
        v = fn()
        line(name, f'OK   {v}' if v else 'OK')
        return True, str(v or '')
    except Exception as e:
        line(name, f'FAIL  {type(e).__name__}: {str(e)[:90]}')
        return False, f'{type(e).__name__}: {e}'

def t_triton():
    import triton, triton.language as tl
    @triton.jit
    def k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(y_ptr + i, tl.load(x_ptr + i, mask=m) * 2.0, mask=m)
    x = torch.arange(256, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    k[(1,)](x, y, 256, BLOCK=256)
    torch.cuda.synchronize()
    assert torch.allclose(y, x * 2)
    return f'v{triton.__version__} (compiled and ran a kernel)'

def t_causal_conv1d():
    from causal_conv1d import causal_conv1d_fn
    x = torch.randn(2, 64, 128, device='cuda', dtype=torch.float16,
                    requires_grad=True)
    w = torch.randn(64, 4, device='cuda', dtype=torch.float16,
                    requires_grad=True)
    causal_conv1d_fn(x, w, None, activation='silu').sum().backward()
    torch.cuda.synchronize()
    return 'fwd+bwd'

def t_mamba1():
    import mamba_ssm
    from mamba_ssm import Mamba
    m = Mamba(d_model=512, d_state=16, d_conv=4, expand=2).cuda()
    x = torch.randn(2, 256, 512, device='cuda', requires_grad=True)
    m(x).sum().backward()
    torch.cuda.synchronize()
    return f'v{getattr(mamba_ssm, "__version__", "?")} fwd+bwd (selective scan)'

triton_ok, triton_note = probe('triton', t_triton)
cc1d_ok,  cc1d_note   = probe('causal-conv1d', t_causal_conv1d)
m1_ok,    m1_note     = probe('mamba-ssm (Mamba-1)', t_mamba1)

# --- the decisive test: run every real Mamba-2 configuration -----------------
# Distinct (dim, d_inner, head_dim) triples taken from the nine SSM models.
# d_state=16 because the integration modules hard-code it, overriding the
# documented default of 32 -- which is why "reduce the state dimension" is not
# an available mitigation.
MAMBA2_CONFIGS = [
    (2048, 4096, 512), (341, 682, 341), (1024, 2048, 256), (768, 1536, 192),
    (512, 1024, 128), (384, 768, 96), (320, 640, 80), (304, 608, 76),
    (256, 512, 64), (248, 496, 62), (192, 384, 48), (128, 256, 32),
]
MODEL_MAX_HEADDIM = {
    'mamba_fpn': 512, 'mamba_unet_v2': 341, 'mamba_unet_v1': 256,
    'mamba_transunet': 256, 'pure_mamba_unet': 256, 'mamba_swin_unet': 192,
    'mamba_unet_resnet': 128, 'mamba_nnunet': 80, 'mamba_deeplab': 76,
    'mamba_dense_context_unet': 62,
}

print('\n  --- Mamba-2 chunk-scan, ACTUALLY RUN per configuration ---')
print('  (this is Paper C s4.3 measured on this card, not predicted)')
m2_results, headdim_ok = {}, {}
if not m1_ok:
    print('  skipped: mamba-ssm unusable above')
else:
    from mamba_ssm import Mamba2
    import re
    # Report the kernel's OWN requirement, parsed from its error, never a
    # formula. An earlier version printed 1280*P+512 here; that line was fitted
    # to two measured points (P=256, P=512) and extrapolated downward, and the
    # extrapolation is false: head_dim 128 "needs" 164,352 B by it, yet runs on
    # a 101,376 B card. Two points always fit a two-parameter line exactly, so
    # the fit was never evidence.
    print(f'  {"head_dim":>9s} {"d_model":>8s} {"measured B":>12s}  result')
    for d_model, d_inner, head_dim in MAMBA2_CONFIGS:
        need = None
        try:
            m = Mamba2(d_model=d_model, d_state=16, d_conv=4, expand=2,
                       headdim=head_dim, chunk_size=128).cuda()
            x = torch.randn(1, 256, d_model, device='cuda', requires_grad=True)
            m(x).sum().backward()
            torch.cuda.synchronize()
            res, ok = 'RUNS', True
            del m, x
            torch.cuda.empty_cache()
        except Exception as e:
            msg = str(e).replace('\n', ' ')
            ok = False
            req = re.search(r'Required:\s*(\d+)', msg)
            need = int(req.group(1)) if req else None
            if 'shared memory' in msg.lower() or 'out of resource' in msg.lower():
                res = 'shared-memory ceiling'
            elif 'multiple of 8' in msg.lower() or 'convolution channels' in msg.lower():
                # causal-conv1d requires channel counts divisible by 8. d_inner
                # = 682 is not, so this config fails an alignment assertion
                # BEFORE it reaches the scan kernel -- a different failure from
                # the shared-memory one, and one to report separately.
                res = 'causal-conv1d alignment (channels not a multiple of 8)'
            else:
                res = f'{type(e).__name__}: {msg[:52]}'
        headdim_ok[head_dim] = ok
        m2_results[head_dim] = {'d_model': d_model, 'measured_bytes': need,
                                'runs': ok, 'detail': res}
        print(f'  {head_dim:9d} {d_model:8d} '
              f'{f"{need:,d}" if need else "-":>12s}  '
              f'{"RUNS" if ok else "fails -- " + res}')

    print('\n  --- per model (max head_dim decides) ---')
    runs = [n for n, hd in MODEL_MAX_HEADDIM.items() if headdim_ok.get(hd)]
    fails = [n for n in MODEL_MAX_HEADDIM if n not in runs]
    print(f'  {len(runs)} of 10 Mamba-2 variants run on this card')
    for n in sorted(runs):
        print(f'    RUNS   {n}')
    for n in sorted(fails):
        print(f'    fails  {n}  (head_dim {MODEL_MAX_HEADDIM[n]})')
    card['mamba2_runs'], card['mamba2_fails'] = sorted(runs), sorted(fails)

card.update(triton_ok=triton_ok, causal_conv1d_ok=cc1d_ok,
            mamba1_ok=m1_ok, mamba2_by_headdim=m2_results)

# ===========================================================================
hdr(3, 'NUMERICS AND DETERMINISM')
# ===========================================================================
# Read BOTH before pinning. Recent torch defaults matmul.allow_tf32 to False
# while cudnn.allow_tf32 stays True, so reporting either one alone misstates
# what the card would actually do untouched.
tf32_matmul_default = bool(torch.backends.cuda.matmul.allow_tf32)
tf32_cudnn_default = bool(torch.backends.cudnn.allow_tf32)
line('TF32 default',
     f'matmul={"ON" if tf32_matmul_default else "off"}, '
     f'cuDNN={"ON" if tf32_cudnn_default else "off"}',
     'the revision pins BOTH off: unpinned, TF32 moved FPN-UNet EF by 0.88 '
     'points -- more than the gaps between adjacent architectures')

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
det_algos, det_note = True, ''
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception as e:
    det_algos, det_note = False, f'{type(e).__name__}: {e}'
line('pinning applied',
     f'TF32 off={not torch.backends.cuda.matmul.allow_tf32}, '
     f'deterministic_algorithms={det_algos}', det_note)

try:
    net = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 3, padding=1),
                              torch.nn.BatchNorm2d(16), torch.nn.ReLU(),
                              torch.nn.Conv2d(16, 4, 3, padding=1)).cuda().eval()
    xx = torch.randn(2, 1, 256, 256, device='cuda')
    with torch.no_grad():
        r1, r2 = net(xx), net(xx)
    bit_identical = torch.equal(r1, r2)
    line('repeat determinism',
         'bit-identical' if bit_identical else
         f'DIFFERS by {(r1 - r2).abs().max().item():.3e}')
except Exception as e:
    bit_identical = False
    line('repeat determinism', f'FAIL {e}')

try:
    sc = torch.amp.GradScaler('cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        out = net(xx).float().sum()
    line('AMP fp16', 'OK')
    amp_ok = True
except Exception as e:
    amp_ok = False
    line('AMP fp16', f'FAIL {e}')

card.update(tf32_matmul_default=tf32_matmul_default,
            tf32_cudnn_default=tf32_cudnn_default,
            deterministic_algorithms=det_algos,
            bit_identical=bool(bit_identical), amp_ok=amp_ok)

# ===========================================================================
hdr(4, 'CAPACITY')
# ===========================================================================
# DenseContextU-Net peaks at 18.9 GiB at batch 8. Below that it micro-batches,
# which makes BatchNorm see 2 samples instead of 8 -- allowed, but a deviation
# you must state in Methods, and one you would rather not have.
free_gib = (p.total_memory - torch.cuda.memory_allocated()) / 1024**3
holds_dense = free_gib * 0.85 >= 18.9
line('usable VRAM', f'{free_gib:.1f} GiB')
line('DenseContextU-Net @8',
     'fits' if holds_dense else 'MICRO-BATCHES (BatchNorm sees 2, not 8)')
line('est. concurrent jobs', f'~{int(free_gib * 0.85 // 3)} at ~3 GiB each')
card.update(holds_dense_context=holds_dense,
            est_max_shards=int(free_gib * 0.85 // 3))

# ===========================================================================
hdr(5, 'THROUGHPUT AND CONCURRENCY SCALING')
# ===========================================================================
# A UNet of the same shape and scale as the real unet_v1 (~31M params) at the
# canonical batch 8 / 256 px, through a real DataLoader whose workers do CPU
# work comparable to the augmentation pipeline -- worker contention is what
# caps shard count, so it has to be present. Validated against the real
# repo benchmark on an RTX 4060: both report 3.64 it/s.
WORKER_SRC = textwrap.dedent('''
    import json, sys, time
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    WORKERS, BATCHES, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

    class Synth(Dataset):
        def __len__(self): return 1600
        def __getitem__(self, i):
            rng = np.random.default_rng(i)
            x = rng.random((256, 256), dtype=np.float32)
            x = (x - x.mean()) / (x.std() + 1e-6)
            x = np.rot90(x, i % 4).copy()
            y = (rng.random((256, 256)) * 4).astype(np.int64)
            return torch.from_numpy(x)[None], torch.from_numpy(y)

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                             nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1, bias=False),
                             nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self, bf=64, n=4):
            super().__init__()
            self.e = nn.ModuleList(); self.d = nn.ModuleList()
            self.u = nn.ModuleList(); ch, f = 1, bf
            for _ in range(n):
                self.e.append(block(ch, f)); ch = f; f *= 2
            self.b = block(ch, f)
            for _ in range(n):
                self.u.append(nn.ConvTranspose2d(f, f // 2, 2, 2))
                self.d.append(block(f, f // 2)); f //= 2
            self.o = nn.Conv2d(f, 4, 1); self.pool = nn.MaxPool2d(2)
        def forward(self, x):
            s = []
            for e in self.e:
                x = e(x); s.append(x); x = self.pool(x)
            x = self.b(x)
            for u, d, sk in zip(self.u, self.d, reversed(s)):
                x = d(torch.cat([u(x), sk], 1))
            return self.o(x)

    def main():
        dev = torch.device('cuda')
        m = UNet().to(dev).train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        crit = nn.CrossEntropyLoss(); scaler = torch.amp.GradScaler('cuda')
        dl = DataLoader(Synth(), batch_size=8, shuffle=True,
                        num_workers=WORKERS, pin_memory=True,
                        **({'prefetch_factor': 2} if WORKERS else {}))
        def step(b):
            x = b[0].to(dev, non_blocking=True); y = b[1].to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = crit(m(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        it = iter(dl)
        for _ in range(3):
            try: step(next(it))
            except StopIteration: it = iter(dl); step(next(it))
        torch.cuda.synchronize()
        t0 = time.perf_counter(); done = 0
        while done < BATCHES:
            try: b = next(it)
            except StopIteration: it = iter(dl); continue
            step(b); done += 1
        torch.cuda.synchronize()
        json.dump({'it_per_s': done / (time.perf_counter() - t0)}, open(OUT, 'w'))

    # Required under the spawn start method: workers re-import this module.
    if __name__ == '__main__':
        main()
''')
_wdir = '/content' if os.path.isdir('/content') else '.'
_wpath = os.path.join(_wdir, '_bench_worker.py')
open(_wpath, 'w').write(WORKER_SRC)

def run_n(n):
    procs, outs = [], []
    for i in range(n):
        o = os.path.join(_wdir, f'_bench_{n}_{i}.json')
        if os.path.exists(o):
            os.remove(o)
        outs.append(o)
        procs.append(subprocess.Popen(
            [sys.executable, _wpath, str(WORKERS), str(BATCHES), o],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    errs = [pr.communicate()[1] for pr in procs]
    rates = [json.load(open(o))['it_per_s'] for o in outs if os.path.exists(o)]
    if not rates:
        print(f'    {n:4d}  FAILED  {errs[0][-250:].decode(errors="ignore")}')
    return rates

print(f'  {WORKERS} dataloader workers per job\n')
print(f'  {"jobs":>5s} {"aggregate":>11s} {"per job":>9s} {"speedup":>8s} {"eff":>6s}')
scaling, base = {}, None
for n in CONCURRENCY:
    if n * (WORKERS + 1) > card['vcpu'] * 1.5:
        print(f'  {n:5d}  skipped: {n}x({WORKERS}+1) processes vs {card["vcpu"]} vCPU')
        continue
    rates = run_n(n)
    if not rates:
        continue
    agg = sum(rates)
    base = base or agg
    scaling[n] = {'aggregate': round(agg, 3), 'per_job': round(agg / len(rates), 3),
                  'speedup': round(agg / base, 3),
                  'efficiency': round((agg / base) / n, 3)}
    print(f'  {n:5d} {agg:10.2f} {agg/len(rates):9.2f} {agg/base:7.2f}x '
          f'{(agg/base)/n:6.0%}')
card['scaling'] = scaling

# ===========================================================================
hdr(6, 'VERDICT  -- copy these lines for each GPU')
# ===========================================================================
best_n, best = (max(scaling.items(), key=lambda kv: kv[1]['aggregate'])
                if scaling else (None, None))
n_runs = len(card.get('mamba2_runs', []))
print(f'  GPU                  {card["gpu"]}  ({sm}, {card["vram_gib"]} GiB, '
      f'{card["vcpu"]} vCPU)')
print(f'  shared mem           {shmem:,} B' if shmem else '  shared mem           ?')
print(f'  Mamba-2 variants     {n_runs}/10 RUN (measured)'
      if card.get('mamba2_runs') is not None else
      '  Mamba-2 variants     NOT TESTED (mamba-ssm missing)')
print(f'  kernel stack         triton={triton_ok} causal-conv1d={cc1d_ok} '
      f'mamba1={m1_ok}')
print(f'  DenseContext @ b8    {"fits" if holds_dense else "micro-batches"}')
if best:
    print(f'  best shards          {best_n}  ({best["speedup"]:.2f}x, '
          f'{best["efficiency"]:.0%} eff, {best["aggregate"]:.2f} it/s)')
print('''
DECIDE IN THIS ORDER:
  1 kernel stack -- any FAIL => R4/R5 impossible there. Disqualifying unless
    the build is fixable. This is the one that rules a card out.
  2 Mamba-2 variants -- this CHANGES a published claim. MORE running is more
    data and a better s4.3, but that section must be rewritten. FEWER running
    reproduces what you already have. Either is defensible; drifting between
    them without noticing is not.
  3 DenseContext fits -- avoids a BatchNorm deviation you would have to declare.
  4 best shards / throughput -- speed. Matters least.
  5 AVAILABILITY -- untestable here and it may decide everything: ALL of R1-R5
    must run on ONE card. Mixing cards reintroduces exactly the confound this
    revision removes, so a slower GPU you can hold beats a faster one you lose.
''')
json.dump(card, open(SAVE_TO, 'w'), indent=2, default=str)
print(f'Scorecard -> {SAVE_TO}   (copy to Drive before the runtime dies)')
