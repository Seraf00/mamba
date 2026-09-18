# Paste into a Colab cell. Summarises finished groups from what is on disk
# (this machine first, then Drive), and checks that models trained in BOTH R1
# and R3 match -- same seed, same config, same GPU, pinned arithmetic, so they
# should be identical. Works with any version of the revision notebook.
import json, pathlib

ROOTS = [pathlib.Path('/content/results'),
         pathlib.Path('/content/drive/MyDrive/Paper1/results_revision')]


def show(g):
    d = next((r / g for r in ROOTS if (r / g / 'all_results.json').exists()), None)
    if d is None:
        print(f'\n{g}: no all_results.json yet')
        return {}
    rows = json.load(open(d / 'all_results.json'))
    print(f'\n== {g}: {len(rows)} models  ({d}) ==')
    out = {}
    for r in rows:
        n = r.get('display_name', '?')
        if 'error' in r:
            print(f'  {n:28s} ERROR  {str(r["error"])[:90]}')
            continue
        print(f"  {n:28s} {r.get('epochs_trained', 0):>3} ep  "
              f"val Dice {r.get('best_val_dice', 0):.4f}  "
              f"{r.get('num_params', 0) / 1e6:6.1f} M  "
              f"{r.get('training_time_seconds', 0) / 60:6.1f} min")
        out[n] = r
    return out


r1 = show('r1_canonical')
r3 = show('r3_param_matched')

common = sorted(set(r1) & set(r3))
if common:
    print('\n== same model trained in R1 and R3 (should be identical) ==')
    for n in common:
        a, b = r1[n].get('best_val_dice', 0), r3[n].get('best_val_dice', 0)
        ea, eb = r1[n].get('epochs_trained'), r3[n].get('epochs_trained')
        tag = 'IDENTICAL' if a == b else f'differs by {b - a:+.2e}'
        print(f'  {n:24s} R1 {a:.6f}  R3 {b:.6f}  {tag}'
              + ('' if ea == eb else f'  (epochs {ea} vs {eb})'))

wide = {n: r for n, r in r3.items() if n.endswith('_wide')}
if wide and r1:
    print('\n== widened control vs its baseline (val Dice) ==')
    for n, r in sorted(wide.items()):
        base = n[:-len('_wide')]
        if base in r1:
            print(f"  {base:20s} {r1[base].get('best_val_dice', 0):.4f} "
                  f"({r1[base].get('num_params', 0) / 1e6:.0f} M)  ->  wide "
                  f"{r.get('best_val_dice', 0):.4f} ({r.get('num_params', 0) / 1e6:.0f} M)"
                  f"  {r.get('best_val_dice', 0) - r1[base].get('best_val_dice', 0):+.4f}")
