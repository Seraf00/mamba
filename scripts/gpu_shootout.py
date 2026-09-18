#!/usr/bin/env python3
"""Decide which GPU to run the whole programme on. Run it on each candidate.

All of R1-R5 must run on ONE card -- if R1 lands on an A100 and R4 on an H100,
base-vs-SSM is confounded with hardware, which is the exact defect this revision
exists to remove. So the choice is made once, and it should be made on measured
numbers rather than on specifications.

Four things actually differ between candidates, and only one of them is speed:

  shared memory   Decides the Mamba-2 failure set, which is a RESULT in Paper C
                  s4.3. The six failing variants request 328,192 B or 655,872 B
                  (Triton's own figures), above every candidate's limit --
                  Ada/Blackwell 101,376, A100 163,840, H100 227,328 -- so the
                  published {4 train, 6 fail} partition is expected on all of
                  them. Measured on Ada (sm_89) and Blackwell (sm_120): it held.
                  An earlier version of this docstring predicted a different
                  partition from a formula extrapolated below its two measured
                  points; the extrapolation was false.
  mamba-ssm       Without the CUDA fast path, SSM training is ~100x slower and
                  R4/R5 stop being feasible rather than merely slow. Newer
                  architectures may need a source build.
  concurrency     At batch 8 and 256 px these models leave the GPU mostly idle,
                  so throughput comes from running several at once, not from a
                  faster card. How well that scales is the number that decides
                  between "fast card, few shards" and "big card, many shards"
                  -- and it is the one thing you cannot read off a spec sheet.
  vCPU            The real cap on shard count. Not VRAM.

Writes a JSON scorecard per GPU. Run on each candidate, then compare:

    python scripts/gpu_shootout.py --out /content/drive/MyDrive/shootout_h100.json
    python scripts/gpu_shootout.py --compare shootout_*.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reference per-group cost in hours, from the sessions already on disk, at
# 8.3 s/epoch for unet_v1. Scaled by measured throughput below.
REF_HOURS = {'R1': 4.5, 'R2': 2.0, 'R3': 7.5, 'R4': 19.4, 'R5': 28.5}
REF_S_PER_EPOCH = 8.3

# bench_dataloader times TRAINING steps only. REF_S_PER_EPOCH comes from
# training_time_seconds / epochs_trained, which also includes the validation
# pass. Comparing them directly understates the projection by about a third:
# measured on the RTX 4060, training epochs ran 81 s while the benchmark
# reported 55.1 s for the same configuration. Scaling the benchmark by this
# factor makes the two comparable. It is a property of the train/val split
# sizes (1600/200), not of the card, so it transfers.
VALIDATION_OVERHEAD = 81.0 / 55.1

# Per-block shared memory each Mamba-2 variant's kernel REQUESTED, as reported
# by Triton in its own OutOfResources error (T10_shared_mem.tex, from
# results/hardware/ssm_shared_mem_ada.jsonl). None means the variant trained, so
# the only thing known is that its request fits a 101,376 B card -- it has run
# on two of them (Ada sm_89 and Blackwell sm_120).
#
# This replaces a formula, 1280*P + 512 with P the padded head dimension. It
# was fitted to the two measured points below and extrapolated downward, which
# predicted 164,352 B for head_dim 128. That variant then ran on a 101,376 B
# card, so the extrapolation was false -- and two points fit a two-parameter
# line exactly, so the fit was never evidence in the first place. Only measured
# requests are used now.
MAMBA2_MEASURED_REQUEST = {
    'mamba_dense_context_unet': None, 'mamba_deeplab': None,
    'mamba_nnunet': None, 'mamba_unet_resnet': None,
    'mamba_swin_unet': 328192, 'mamba_unet_v1': 328192,
    'mamba_transunet': 328192, 'pure_mamba_unet': 328192,
    'mamba_unet_v2': 655872, 'mamba_fpn': 655872,
}
# The smallest limit any variant marked None is known to fit.
KNOWN_FITS_BELOW = 101376


def device_max_shared_mem() -> int | None:
    """Max dynamic shared memory per block, in bytes.

    Triton is the authority because it is Triton's own limit the Mamba-2
    chunk-scan runs into, but it is not installed everywhere -- so fall back to
    torch's device properties, which expose the same opt-in figure.
    """
    try:
        import triton
        return int(triton.runtime.driver.active.utils
                   .get_device_properties(0)['max_shared_mem'])
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        p = torch.cuda.get_device_properties(0)
        for attr in ('shared_memory_per_block_optin',
                     'sharedMemPerBlockOptin',
                     'shared_memory_per_block'):
            v = getattr(p, attr, None)
            if v:
                return int(v)
    except Exception:  # noqa: BLE001
        pass
    return None


def probe_identity() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {'error': 'no CUDA device'}
    p = torch.cuda.get_device_properties(0)
    return {
        'gpu': torch.cuda.get_device_name(0),
        'vram_gib': round(p.total_memory / 1024 ** 3, 1),
        'compute_capability': f'{p.major}.{p.minor}',
        'sm_count': p.multi_processor_count,
        'max_shared_mem_bytes': device_max_shared_mem(),
        'vcpu': multiprocessing.cpu_count(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'tf32_default_on': bool(torch.backends.cuda.matmul.allow_tf32
                                or torch.backends.cudnn.allow_tf32),
    }


def probe_mamba() -> dict:
    """Is the CUDA fast path live? Blocking for R4/R5."""
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        return {'installed': False, 'fast_path': False,
                'note': 'not installed -- R4/R5 infeasible, R1-R3 unaffected'}
    try:
        from models.modules import MambaBlock
        blk = MambaBlock(dim=64, d_state=16)
        fast = bool(getattr(blk, 'use_fast_path', False))
        return {'installed': True, 'fast_path': fast,
                'note': '' if fast else
                        'PyTorch fallback (~100x slower) -- R4/R5 infeasible'}
    except Exception as e:  # noqa: BLE001
        return {'installed': True, 'fast_path': False,
                'note': f'{type(e).__name__}: {e}'}


def predict_mamba2(limit: int | None) -> dict:
    """Which Mamba-2 variants this card should run, from MEASURED requests.

    A prediction, and labelled as one: the request is a property of the Triton
    kernel in a given mamba-ssm build, so a different stack can move it. To
    know rather than predict, run scripts/colab_gpu_shootout_cell.py, which
    executes each configuration.
    """
    if not limit:
        return {'error': 'shared-memory limit unavailable (triton missing?)'}
    runs, fails, unknown = [], [], []
    for name, req in sorted(MAMBA2_MEASURED_REQUEST.items()):
        if req is None:
            # Known to fit KNOWN_FITS_BELOW; below that, not established.
            (runs if limit >= KNOWN_FITS_BELOW else unknown).append(name)
        else:
            (runs if req <= limit else fails).append(name)
    return {'limit_bytes': limit, 'runs': runs, 'fails': fails,
            'unknown': unknown, 'n_runs': len(runs), 'n_fails': len(fails),
            'basis': 'measured Triton requests, not a formula'}


def bench_once(data_dir: Path, workers: int, batches: int,
               out_json: Path, quiet: bool = True) -> dict | None:
    cmd = [sys.executable, str(ROOT / 'scripts' / 'bench_dataloader.py'),
           '--data-dir', str(data_dir), '--workers', str(workers),
           '--batches', str(batches), '--json', str(out_json)]
    if quiet:
        cmd.append('--quiet')
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not out_json.exists():
        print(f'    benchmark failed:\n{(r.stdout or r.stderr)[-600:]}')
        return None
    return json.loads(out_json.read_text())


def measure_concurrency(data_dir: Path, workers: int, batches: int,
                        counts: list[int], tmp: Path) -> dict:
    """Run N benchmarks at once and measure AGGREGATE throughput.

    This is the number my earlier projections guessed at (I assumed 75%
    efficiency, which was invented). One job leaves a large GPU mostly idle, so
    N jobs should approach N x throughput until either the SMs or the CPU
    saturate -- where that knee falls is what decides the shard count, and it
    differs per card.
    """
    tmp.mkdir(parents=True, exist_ok=True)
    out = {}
    for n in counts:
        procs, jsons = [], []
        t0 = time.perf_counter()
        for i in range(n):
            j = tmp / f'bench_n{n}_{i}.json'
            j.unlink(missing_ok=True)
            jsons.append(j)
            procs.append(subprocess.Popen(
                [sys.executable, str(ROOT / 'scripts' / 'bench_dataloader.py'),
                 '--data-dir', str(data_dir), '--workers', str(workers),
                 '--batches', str(batches), '--json', str(j), '--quiet'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for p in procs:
            p.wait()
        wall = time.perf_counter() - t0

        rates = []
        for j in jsons:
            if j.exists():
                rates.append(json.loads(j.read_text())['best']['it_per_s'])
        if not rates:
            out[n] = {'error': 'all children failed'}
            print(f'    {n:2d} concurrent: FAILED')
            continue

        agg = sum(rates)
        out[n] = {'n': n, 'per_job_it_per_s': round(sum(rates) / len(rates), 3),
                  'aggregate_it_per_s': round(agg, 3),
                  'children_ok': len(rates), 'wall_s': round(wall, 1)}
        print(f'    {n:2d} concurrent: {agg:6.2f} it/s aggregate '
              f'({sum(rates)/len(rates):5.2f} each)', flush=True)

    base = out.get(1, {}).get('aggregate_it_per_s')
    if base:
        for n, v in out.items():
            if 'aggregate_it_per_s' in v:
                v['speedup'] = round(v['aggregate_it_per_s'] / base, 2)
                v['efficiency'] = round(v['speedup'] / n, 3)
    return out


def project(concurrency: dict, s_per_epoch: float) -> dict:
    """Programme hours at each shard count, from measured scaling."""
    # s_per_epoch is training-only; REF_S_PER_EPOCH includes validation.
    scale = (s_per_epoch * VALIDATION_OVERHEAD) / REF_S_PER_EPOCH
    single_total = sum(REF_HOURS.values()) * scale
    rows = {}
    for n, v in sorted(concurrency.items()):
        if 'speedup' not in v:
            continue
        rows[n] = {'shards': n, 'speedup': v['speedup'],
                   'efficiency': v['efficiency'],
                   'total_hours': round(single_total / v['speedup'], 1),
                   'per_group_hours': {g: round(h * scale / v['speedup'], 1)
                                       for g, h in REF_HOURS.items()}}
    return {'single_job_total_hours': round(single_total, 1), 'by_shards': rows}


def run(args) -> int:
    ident = probe_identity()
    if 'error' in ident:
        print(f'FAIL: {ident["error"]}')
        return 1

    print('=' * 74)
    print(f'GPU SHOOTOUT  |  {ident["gpu"]}')
    print('=' * 74)
    print(f'  VRAM              {ident["vram_gib"]} GiB')
    print(f'  compute capability{ident["compute_capability"]:>7s}   '
          f'({ident["sm_count"]} SMs)')
    print(f'  max shared mem    {ident["max_shared_mem_bytes"]:,} B/block'
          if ident['max_shared_mem_bytes'] else '  max shared mem    unavailable')
    print(f'  vCPU              {ident["vcpu"]}')
    print(f'  torch             {ident["torch"]}  cuda {ident["cuda"]}')
    print(f'  TF32 default      {"ON" if ident["tf32_default_on"] else "off"}')

    print('\n--- mamba-ssm ---')
    mamba = probe_mamba()
    print(f'  installed={mamba["installed"]}  fast_path={mamba["fast_path"]}'
          + (f'\n  {mamba["note"]}' if mamba['note'] else ''))

    print('\n--- Mamba-2 variants this card can run (Paper C s4.3) ---')
    m2 = predict_mamba2(ident['max_shared_mem_bytes'])
    if 'error' in m2:
        print(f'  {m2["error"]}')
    else:
        print(f'  {m2["n_runs"]} run, {m2["n_fails"]} fail at a '
              f'{m2["limit_bytes"]:,} B limit')
        print(f'  runs : {", ".join(m2["runs"]) or "(none)"}')
        print(f'  fails: {", ".join(m2["fails"]) or "(none)"}')

    tmp = Path(args.tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)   # the child writes its JSON here
    print('\n--- single-job throughput ---')
    single = bench_once(Path(args.data_dir), args.workers, args.batches,
                        tmp / 'bench_single.json', quiet=True)
    if not single:
        print('  benchmark failed; cannot project timings')
        return 1
    s_per_epoch = single['best']['s_per_epoch']
    print(f'  {single["best"]["it_per_s"]:.2f} it/s, {s_per_epoch:.1f} s/epoch '
          f'(unet_v1, batch 8)')
    print(f'  reference sessions: {REF_S_PER_EPOCH} s/epoch  |  '
          f'RTX 4060 laptop: 81 s/epoch')

    counts = args.concurrency or [1, 2, 4, min(8, max(1, ident['vcpu'] //
                                                      (args.workers + 1)))]
    counts = sorted(set(c for c in counts if c >= 1))
    print(f'\n--- concurrency scaling ({args.workers} workers each) ---')
    conc = measure_concurrency(Path(args.data_dir), args.workers,
                               args.batches, counts, tmp)

    proj = project(conc, s_per_epoch)
    print(f'\n--- projected R1-R5 ({proj["single_job_total_hours"]} h at one job) ---')
    print(f'    {"shards":>6s} {"speedup":>8s} {"eff":>6s} {"total h":>9s}')
    for n, r in proj['by_shards'].items():
        print(f'    {n:6d} {r["speedup"]:7.2f}x {r["efficiency"]:6.0%} '
              f'{r["total_hours"]:8.1f}h')

    card = {'identity': ident, 'mamba_ssm': mamba, 'mamba2_prediction': m2,
            'single_job': single['best'], 'concurrency': conc,
            'projection': proj,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2))
    print(f'\nScorecard -> {out}')
    print('Run this on each candidate, then: '
          'python scripts/gpu_shootout.py --compare <files>')
    return 0


def compare(paths: list[str]) -> int:
    cards = []
    for p in paths:
        try:
            cards.append(json.loads(Path(p).read_text()))
        except Exception as e:  # noqa: BLE001
            print(f'skip {p}: {e}')
    if not cards:
        print('No scorecards to compare.')
        return 1

    print('=' * 100)
    print('GPU COMPARISON')
    print('=' * 100)
    hdr = f'{"":26s}' + ''.join(
        f'{c["identity"]["gpu"][:20]:>22s}' for c in cards)
    print(hdr)
    print('-' * 100)

    def row(label, fn):
        print(f'{label:26s}' + ''.join(f'{fn(c):>22s}' for c in cards))

    row('VRAM', lambda c: f'{c["identity"]["vram_gib"]} GiB')
    row('compute capability', lambda c: c['identity']['compute_capability'])
    row('vCPU', lambda c: str(c['identity']['vcpu']))
    row('shared mem / block',
        lambda c: f'{c["identity"]["max_shared_mem_bytes"]:,}'
        if c['identity']['max_shared_mem_bytes'] else '?')
    row('mamba-ssm fast path',
        lambda c: 'YES' if c['mamba_ssm']['fast_path'] else 'NO')
    row('Mamba-2 variants run',
        lambda c: f'{c["mamba2_prediction"].get("n_runs", "?")} of 10')
    row('s/epoch (1 job)', lambda c: f'{c["single_job"]["s_per_epoch"]:.1f}')

    best = {}
    for c in cards:
        rows = c['projection']['by_shards']
        if rows:
            k = min(rows.values(), key=lambda r: r['total_hours'])
            best[id(c)] = k
    row('best shard count',
        lambda c: str(best.get(id(c), {}).get('shards', '?')))
    row('scaling at that count',
        lambda c: f'{best.get(id(c), {}).get("speedup", 0):.2f}x')
    row('R1-R5 total',
        lambda c: f'{best.get(id(c), {}).get("total_hours", 0):.1f} h')

    print('-' * 100)
    print('\nWhat to weigh, in order:')
    print('  1. mamba-ssm fast path NO  -> R4/R5 are impossible there. '
          'Disqualifying unless you fix the build.')
    print('  2. Mamba-2 variants run    -> this CHANGES Paper C s4.3. More is '
          'more data, but it means rewriting that section.')
    print('  3. R1-R5 total             -> speed, which matters least of the '
          'three.')
    print('  4. Availability            -> untestable here, and it may decide '
          'it: all of R1-R5 must run on ONE card, so a slower GPU you can hold '
          'for the whole programme beats a faster one you get bumped off.')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=str(ROOT / 'data' / 'CAMUS'))
    ap.add_argument('--out', default='gpu_scorecard.json')
    ap.add_argument('--workers', type=int, default=3,
                    help='dataloader workers per job')
    ap.add_argument('--batches', type=int, default=40,
                    help='timed steps per benchmark')
    ap.add_argument('--concurrency', type=int, nargs='*', default=None,
                    help='shard counts to test (default: 1 2 4 and a '
                         'vCPU-derived cap)')
    ap.add_argument('--tmp-dir',
                    default=str(Path(os.environ.get('TMPDIR', '/tmp'))
                               / 'gpu_shootout'))
    ap.add_argument('--compare', nargs='+', default=None,
                    help='Compare scorecards instead of measuring')
    args = ap.parse_args()
    return compare(args.compare) if args.compare else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
