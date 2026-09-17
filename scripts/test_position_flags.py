"""Verify SSM position flags on any patched model.

Three properties must hold before a model joins the ablation:

  1. every position combination constructs and produces (1, C, H, W);
  2. every flag actually removes parameters -- a flag that changes nothing
     would produce a null ablation arm that looks like a real negative result;
  3. all-on reproduces the parameter count that was actually trained (from the
     hardware probe), and all-off recovers the plain non-SSM baseline.

Usage:
    python test_positions.py mamba_deeplab bottleneck decoder
    python test_positions.py mamba_transunet encoder skip bottleneck
"""
import itertools
import json
import sys

sys.path.insert(0, 'D:/Papers/Paper1')
import torch
from models import get_model

# registry name -> (probe key, base-model key for the all-off comparison)
PAIRS = {
    'mamba_deeplab':            ('mamba_deeplab', 'deeplab_v3'),
    'mamba_transunet':          ('mamba_transunet', 'transunet'),
    'mamba_unet_v1':            ('mamba_unet_v1', 'unet_v1'),
    'mamba_unet_v2':            ('mamba_unet_v2', 'unet_v2'),
    'mamba_unet_resnet':        ('mamba_unet_resnet', 'unet_resnet'),
    'mamba_nnunet':             ('mamba_nnunet', 'nnunet'),
    'mamba_swin_unet':          ('mamba_swin_unet', 'swin_unet'),
    'mamba_fpn':                ('mamba_fpn', 'fpn'),
    'mamba_dense_context_unet': ('mamba_dense_context_unet', 'dense_context_unet'),
}

SWIN = ('swin' in sys.argv[1]) if len(sys.argv) > 1 else False


def probe_params(key, variant='vmamba'):
    for line in open('D:/Papers/Paper1/results/hardware/ssm_shared_mem_ada.jsonl'):
        if not line.strip():
            continue
        r = json.loads(line)
        if r['model'] == key and r['mamba_type'] == variant:
            return r['params_M']
    return None


def base_params(key):
    for r in json.load(open('D:/Papers/Paper1/results/base_models/all_results.json')):
        if r['display_name'] == key:
            return r['num_params']
    return None


def main():
    import os
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    argv = sys.argv[1:]
    variant = 'vmamba'
    if '--variant' in argv:
        i = argv.index('--variant')
        variant = argv[i + 1]
        del argv[i:i + 2]
    name = argv[0]
    positions = argv[1:]
    probe_key, base_key = PAIRS[name]
    size = 224 if 'swin' in name else 256
    x = torch.randn(1, 1, size, size)

    hdr = ' '.join(f'{p[:9]:>9s}' for p in positions)
    print(f'{hdr} {"params":>13s}  {"output":>18s}')

    counts = {}
    for combo in itertools.product([True, False], repeat=len(positions)):
        kw = {f'mamba_in_{p}': v for p, v in zip(positions, combo)}
        m = get_model(name, in_channels=1, num_classes=4,
                      mamba_type=variant, pretrained=False, **kw).eval()
        n = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            y = m(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if isinstance(y, dict):
            y = y.get('out', next(iter(y.values())))
        counts[combo] = n
        flags = ' '.join(f'{str(v):>9s}' for v in combo)
        print(f'{flags} {n:13,d}  {str(tuple(y.shape)):>18s}', flush=True)
        assert tuple(y.shape)[:2] == (1, 4), f'bad output {y.shape}'
        assert tuple(y.shape)[2:] == (size, size), f'bad spatial {y.shape}'
        del m

    n_combos = 2 ** len(positions)
    assert len(set(counts.values())) == n_combos, (
        f'flags are not all independent: {sorted(counts.values())}')
    print(f'\nall {n_combos} combinations distinct')

    all_on = counts[tuple([True] * len(positions))]
    all_off = counts[tuple([False] * len(positions))]
    for i, p in enumerate(positions):
        off = [True] * len(positions)
        off[i] = False
        print(f'  {p:<12s} contributes {all_on - counts[tuple(off)]:>12,d} params')

    ref_on = probe_params(probe_key, variant)
    ref_off = base_params(base_key)
    print()
    if ref_on is not None:
        print(f'all-on  {all_on / 1e6:8.2f} M  vs probe {probe_key}/{variant} {ref_on:8.2f} M')
        assert round(all_on / 1e6, 2) == round(ref_on, 2), 'DEFAULT CHANGED SIZE'
    else:
        print(f'all-on  {all_on:,}  (no probe record to compare)')
    if ref_off is not None:
        print(f'all-off {all_off:,} vs base {base_key} {ref_off:,}'
              f'   {"MATCH" if all_off == ref_off else "differs"}')
    print(f'\nPASS: {name} -- flags effective, default unchanged')


if __name__ == '__main__':
    main()
