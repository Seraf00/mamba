"""Run the notebook's own _current() on logs shaped like the real ones."""
import json
import pathlib
import tempfile

nb = json.load(open(pathlib.Path(__file__).resolve().parents[1] / 'notebooks'
                    / 'colab_revision.ipynb', encoding='utf-8'))
src = next('\n'.join(c['source']) for c in nb['cells']
           if 'def _current' in ''.join(c['source']))
body = src[src.index('def _current'):
           src.index('    return model, epoch') + len('    return model, epoch')]
ns = {}
exec(body, ns)

d = pathlib.Path(tempfile.mkdtemp())
tq = ''.join(f'Epoch 38:  {k}%|#####     | {k}/200 [00:01<00:02, 45.0it/s, loss=0.3]\r'
             for k in range(0, 200, 2))
base = '===== session start =====\n' + '=' * 70 + '\nTraining: unet_v2\n' + '=' * 70 + '\n'
cases = [
    ('huge tqdm tail after the summary',
     base + 'Epoch 37/100 | Train Loss: 0.31 | Val Dice: 0.90\n' + tq * 60,
     ('unet_v2', '37/100')),
    ('summary right after a tqdm \\r',
     base + tq + 'Epoch 37/100 | Train Loss: 0.31 | Val Dice: 0.90\n',
     ('unet_v2', '37/100')),
    ('model started, no epoch finished yet',
     base + tq,
     ('unet_v2', '')),
    ('second model after a finished one',
     base + 'Epoch 100/100 | Train Loss: 0.1 | x\n' + 'Training: fpn\n' + tq,
     ('fpn', '')),
]
bad = 0
for name, text, want in cases:
    (d / 'shard0.log').write_text(text)
    got = ns['_current'](d)
    ok = got == want
    bad += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} -> {got}  (want {want})")
print('all passed' if not bad else f'{bad} failed')
