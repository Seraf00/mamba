"""Regression sweep: no default model may have changed size.

The position flags all default to the previously hard-wired behaviour, so every
model constructed with no flags must still have exactly the parameter count
recorded in the training artefacts. If any row differs, an edit changed the
trained configuration and the existing results would no longer be reproducible
from that code.
"""
import json
import os
import sys

# Works from Windows and from WSL, where the repo is under /mnt/d.
ROOT = next(p for p in ('D:/Papers/Paper1', '/mnt/d/Papers/Paper1')
            if os.path.isdir(p))
sys.path.insert(0, ROOT)
import torch
from models import get_model

# trained parameter counts, per session artefact
trained = {}
for sess in ('mamba_models', 'mamba2_models', 'vmamba_models'):
    for r in json.load(open(f'{ROOT}/results/{sess}/all_results.json')):
        if r.get('num_params'):
            trained[r['display_name']] = r['num_params']

# probe covers the configurations that never trained
probe = {}
for line in open(f'{ROOT}/results/hardware/ssm_shared_mem_ada.jsonl'):
    if line.strip():
        r = json.loads(line)
        probe[(r['model'], r['mamba_type'])] = r['params_M']

MODELS = ['mamba_unet_v1', 'mamba_unet_v2', 'mamba_unet_resnet', 'mamba_deeplab',
          'mamba_nnunet', 'mamba_dense_context_unet', 'mamba_swin_unet',
          'mamba_transunet', 'mamba_fpn', 'pure_mamba_unet']

print(f'{"model":26s} {"variant":8s} {"built":>13s} {"expected":>13s}  source   status')
bad = 0
for name in MODELS:
    for variant in ('mamba', 'mamba2', 'vmamba'):
        key = f'{name}_{variant}' if name != 'pure_mamba_unet' else f'pure_mamba_unet_{variant}'
        exp, src = trained.get(key), 'training'
        if exp is None:
            pm = probe.get((name, variant))
            if pm is None:
                continue
            exp, src = pm * 1e6, 'probe'
        try:
            m = get_model(name, in_channels=1, num_classes=4,
                          mamba_type=variant, pretrained=False)
            got = sum(p.numel() for p in m.parameters())
            del m
        except Exception as e:
            print(f'{name:26s} {variant:8s} {"ERROR":>13s} {exp:13,.0f}  {src:8s} {type(e).__name__}: {str(e)[:40]}')
            bad += 1
            continue
        ok = (got == exp) if src == 'training' else (round(got / 1e6, 2) == round(exp / 1e6, 2))
        if not ok:
            bad += 1
        print(f'{name:26s} {variant:8s} {got:13,d} {exp:13,.0f}  {src:8s} '
              f'{"ok" if ok else "*** CHANGED ***"}')

print()
if bad:
    sys.exit(f'FAIL: {bad} default configuration(s) changed')
print('PASS: every default model reproduces its trained parameter count')
