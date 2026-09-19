#!/usr/bin/env python3
"""Check a downloaded results_revision folder before trusting or deleting anything.

For every finished model:
  * best_model.pth loads, and its parameter count matches results.json
  * training_log.csv holds epochs 0..N-1 exactly once each (a resumed run must
    not have duplicated or lost epochs)
  * results.json says it trained the configured number of epochs
And for any model that was resumed after a disconnect, shows the resume point.

    python scripts/verify_downloaded_results.py results_revision
"""
import json
import sys
from pathlib import Path

import torch


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'results_revision')
    bad = 0
    for g in sorted(p for p in root.iterdir() if p.is_dir()):
        cfg = {}
        if (g / 'experiment_config.json').exists():
            cfg = json.loads((g / 'experiment_config.json').read_text())
        want_ep = cfg.get('epochs', 100)
        models = sorted(m for m in g.iterdir() if (m / 'results.json').exists())
        print(f'\n{g.name}: {len(models)} finished models')
        for m in models:
            res = json.loads((m / 'results.json').read_text())
            issues = []
            ck = m / 'best_model.pth'
            size = ''
            if not ck.exists():
                issues.append('no best_model.pth')
            else:
                try:
                    c = torch.load(ck, map_location='cpu', weights_only=False)
                    sd = c['model_state_dict']
                    n = sum(v.numel() for k, v in sd.items()
                            if v.dtype.is_floating_point
                            and 'running_' not in k and 'num_batches' not in k)
                    exp = res.get('trainable_params') or res.get('num_params')
                    if exp and abs(n - exp) / exp > 0.01:
                        issues.append(f'params {n:,} vs results.json {exp:,}')
                    size = f'{ck.stat().st_size / 1e6:7.1f} MB'
                except Exception as e:
                    issues.append(f'checkpoint unreadable ({type(e).__name__}: {e})')
            if res.get('epochs_trained') != want_ep:
                issues.append(f"epochs_trained {res.get('epochs_trained')} != {want_ep}")
            csv = m / 'training_log.csv'
            if csv.exists():
                lines = [l for l in csv.read_text().splitlines() if l.strip()]
                cols = lines[0].split(',')
                ei = cols.index('epoch')
                ep = [int(float(l.split(',')[ei])) for l in lines[1:]]
                if sorted(ep) != list(range(want_ep)):
                    issues.append(f'training_log.csv: {len(ep)} rows, '
                                  f'{len(set(ep))} unique epochs (want 0..{want_ep - 1} once each)')
            else:
                issues.append('no training_log.csv')
            bad += bool(issues)
            print(f"  {'ok ' if not issues else 'BAD'} {m.name:24s} {size}"
                  + ('' if not issues else '  <- ' + '; '.join(issues)))

        # resumed models, from the log
        txt = ''.join(l.read_text(errors='ignore') for l in sorted(g.glob('shard*.log')))
        pos = 0
        while True:
            i = txt.find('Resuming at epoch', pos)
            if i < 0:
                break
            t = txt.rfind('Training: ', 0, i)
            who = txt[t + len('Training: '):].split('\n', 1)[0].strip() if t >= 0 else '?'
            line = txt[i:].split('\n', 1)[0].strip()
            print(f'  resumed after a disconnect: {who} -- {line}')
            pos = i + 1

    print()
    print('ALL CHECKS PASSED' if not bad else f'{bad} model(s) with problems')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
