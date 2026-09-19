# Paste into a Colab cell. Read-only: lists, never deletes.
# 1) what is taking space on Drive, 2) the state of every revision group,
# 3) R1 vs R3 side by side.
import json, os, shutil, pathlib

MY = pathlib.Path('/content/drive/MyDrive')
REV = MY / 'Paper1' / 'results_revision'
LOCAL = pathlib.Path('/content/results')


def size(p):
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1e9


u = shutil.disk_usage(MY)
print(f'Drive: {u.used / 1e9:.1f} GB used of {u.total / 1e9:.1f} GB '
      f'({u.free / 1e9:.1f} GB free; Trash is not included here but still counts)')

print('\n== largest folders under MyDrive/Paper1 ==')
for d in sorted((MY / 'Paper1').iterdir()):
    if d.is_dir():
        pth = sum(f.stat().st_size for f in d.rglob('*.pth')) / 1e9
        print(f'  {d.name:24s} {size(d):7.2f} GB   (of which .pth {pth:.2f} GB)')
if (MY / 'CAMUS_public').exists():
    print(f'  {"../CAMUS_public":24s} {size(MY / "CAMUS_public"):7.2f} GB   (dataset -- keep)')

print('\n== revision groups on Drive ==')
groups = {}
for g in sorted(p for p in REV.iterdir() if p.is_dir()) if REV.exists() else []:
    models = [m for m in g.iterdir() if m.is_dir() and m.name not in ('evaluation', 'logs')]
    done = [m.name for m in models if (m / 'results.json').exists()]
    running = [m.name for m in models if (m / 'last.pth').exists() and m.name not in done]
    errs = []
    if (g / 'all_results.json').exists():
        try:
            errs = [r['display_name'] for r in json.load(open(g / 'all_results.json')) if 'error' in r]
        except Exception:
            pass
    evald = (g / 'evaluation' / 'evaluation_results.json').exists()
    off = (g / 'OFFLOADED.json').exists()
    print(f'  {g.name:18s} {len(done):3d} finished  {size(g):6.2f} GB'
          + (f' | training now: {", ".join(running)}' if running else '')
          + (f' | FAILED: {", ".join(errs)}' if errs else '')
          + (' | evaluated' if evald else ' | NOT evaluated')
          + (' | offloaded' if off else ''))
    groups[g.name] = g


def rows(g):
    for root in (LOCAL, REV):
        f = root / g / 'all_results.json'
        if f.exists():
            return {r.get('display_name'): r for r in json.load(open(f))}
    return {}


r1, r3 = rows('r1_canonical'), rows('r3_param_matched')
for name, rs in (('R1', r1), ('R3', r3)):
    if rs:
        print(f'\n== {name} ==')
        for n, r in rs.items():
            if 'error' in r:
                print(f'  {n:26s} ERROR {str(r["error"])[:80]}')
            else:
                print(f"  {n:26s} {r.get('epochs_trained', 0):>3} ep  val Dice "
                      f"{r.get('best_val_dice', 0):.4f}  {r.get('num_params', 0) / 1e6:6.1f} M  "
                      f"{r.get('training_time_seconds', 0) / 60:6.1f} min")
