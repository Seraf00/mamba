#!/usr/bin/env python3
"""Behavioural tests for early stopping and resume in training.Trainer.

Both of these were believed to work and did not:

  * "--early_stopping 0" disabled the EarlyStopping CALLBACK, but TrainingConfig
    defaults its own counter to on with patience 20, and train_all_models never
    passed the field -- so every run still stopped at patience 20. Found by
    reading the Trainer only after it had been declared fixed.
  * Resume did not exist: checkpoints lacked scaler and RNG state, train()
    always started at epoch 0, the CSV log was reopened with 'w', and the
    best-model callback forgot its score.

So these are tested by running the Trainer, not by reading it:

  1  early_stopping=False runs every epoch even when val Dice never improves
  2  early_stopping=True still stops (the counter works when asked)
  3  interrupted-and-resumed training is BIT-IDENTICAL to uninterrupted training
     -- final weights and full history
  4  the resumed CSV log has each epoch exactly once
  5  best_model.pth is never overwritten by a worse model after a resume
  6  last.pth is written atomically and removed when training finishes

Synthetic data, a tiny model, num_workers=0, pinned determinism. ~1 minute.

    python scripts/test_trainer_resume.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training import Trainer, TrainingConfig, CombinedLoss
from training.callbacks import ModelCheckpoint, CSVLogger
from utils import pin_determinism

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILS.append(msg)


class Tiny(nn.Module):
    """bn=False for the early-stopping tests: with lr=0 a BatchNorm layer still
    updates its running statistics in train mode, so validation Dice drifts and
    keeps resetting the patience counter -- which made an earlier version of
    this test pass for the wrong reason."""
    def __init__(self, bn=True):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1),
                                 nn.BatchNorm2d(8) if bn else nn.Identity(),
                                 nn.ReLU(), nn.Conv2d(8, 4, 1))

    def forward(self, x):
        return self.net(x)


def loaders():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 1, 32, 32, generator=g)
    y = torch.randint(0, 4, (32, 32, 32), generator=g)
    tr = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True, num_workers=0)
    va = DataLoader(TensorDataset(x[:8], y[:8]), batch_size=8, num_workers=0)
    return tr, va


class Killer:
    """Simulates a runtime dying at the end of a given epoch.

    Raises from on_epoch_end, i.e. AFTER that epoch's CSV row and best-model
    file are written but BEFORE the resume state is -- the worst place to die,
    because the on-disk artefacts then run ahead of last.pth.
    """
    def __init__(self, at_epoch):
        self.at = at_epoch

    def on_epoch_end(self, trainer, epoch, logs):
        if epoch == self.at:
            raise KeyboardInterrupt(f'simulated disconnect after epoch {epoch + 1}')


def make(save_dir, epochs, lr=1e-2, es=False, patience=1, resume_every=0,
         extra=(), bn=True):
    pin_determinism(0, 'full')           # same init on every construction
    model = Tiny(bn=bn)
    cfg = TrainingConfig(epochs=epochs, batch_size=8, learning_rate=lr,
                         scheduler='cosine', warmup_epochs=0, use_amp=(DEV == 'cuda'),
                         save_dir=str(save_dir), save_every=0,
                         resume_every=resume_every, device=DEV, num_workers=0,
                         early_stopping=es, patience=patience)
    tr, va = loaders()
    cbs = [ModelCheckpoint(save_dir=str(save_dir), monitor='val_dice', mode='max',
                           verbose=False),
           CSVLogger(filename=str(Path(save_dir) / 'training_log.csv')),
           *extra]
    return Trainer(model=model, train_loader=tr, val_loader=va,
                   criterion=CombinedLoss(dice_weight=1.0, ce_weight=1.0),
                   config=cfg, callbacks=cbs)


def csv_epochs(save_dir):
    lines = (Path(save_dir) / 'training_log.csv').read_text().splitlines()
    cols = lines[0].split(',')
    ei = cols.index('epoch')
    return [int(float(r.split(',')[ei])) for r in lines[1:] if r.strip()]


def main() -> int:
    print(f'device: {DEV}')
    root = Path(tempfile.mkdtemp(prefix='resume_test_'))
    try:
        # ---------------------------------------------------------------- 1, 2
        print('\n1-2. early stopping follows the config')
        # lr=0: nothing ever improves after the first epoch, so a live
        # patience-1 counter would stop the run at epoch 2.
        t = make(root / 'es_off', epochs=6, lr=0.0, es=False, patience=1, bn=False)
        h = t.train()
        flat = len(set(round(v, 12) for v in h['val_dice'])) == 1
        check(flat, 'control: val Dice is flat, so a live patience-1 counter '
                    'WOULD fire')
        check(len(h['train_loss']) == 6,
              f'early_stopping=False ran all 6 epochs (ran {len(h["train_loss"])})')

        t = make(root / 'es_on', epochs=6, lr=0.0, es=True, patience=1, bn=False)
        h = t.train()
        check(len(h['train_loss']) < 6,
              f'early_stopping=True, patience 1, stopped early '
              f'(ran {len(h["train_loss"])})')

        # ---------------------------------------------------------------- 3-6
        print('\n3-6. interrupted + resumed == uninterrupted')
        E = 6
        a = make(root / 'straight', epochs=E, resume_every=1)
        ha = a.train()
        wa = {k: v.detach().cpu().clone() for k, v in a.model.state_dict().items()}

        d = root / 'resumed'
        b = make(d, epochs=E, resume_every=2, extra=[Killer(at_epoch=2)])
        try:
            b.train()
        except KeyboardInterrupt as e:
            print(f'  ({e})')
        check((d / 'last.pth').exists(), 'last.pth exists after the interruption')
        check(not list(d.glob('*.tmp')), 'no partial .tmp file left behind')
        # Resume state is from epoch 2 (resume_every=2 -> written after epoch
        # index 1), but the CSV already holds epoch index 2 and best_model.pth
        # may be from it: the artefacts run ahead of the resume state.
        check(csv_epochs(d) == [0, 1, 2],
              f'before resume the CSV runs ahead of last.pth: {csv_epochs(d)}')

        c = make(d, epochs=E, resume_every=2)
        c.load_checkpoint(str(d / 'last.pth'))
        check(c.start_epoch == 2, f'resumes at epoch index 2 (got {c.start_epoch})')
        hc = c.train()

        wc = {k: v.detach().cpu() for k, v in c.model.state_dict().items()}
        same_w = all(torch.equal(wa[k], wc[k]) for k in wa)
        check(same_w, 'final weights bit-identical to the uninterrupted run')
        # Validation metrics exactly. The LOGGED train loss only to 1e-6: it is
        # a mean over pixels whose reduction order is not fully deterministic
        # on GPU, so the reported scalar can differ in the last float32 bit
        # between otherwise identical runs -- intermittently, even with no
        # resume involved. The gradient of a mean does not depend on the value
        # being averaged, which is why the weights above stay bit-identical.
        # Dice is built from integer pixel counts: exact. Both loss scalars are
        # float means with the same reduction-order caveat: 1e-6.
        check(ha['val_dice'] == hc['val_dice'],
              'validation Dice identical every epoch')
        check(np.allclose(ha['train_loss'], hc['train_loss'], rtol=1e-6, atol=0)
              and np.allclose(ha['val_loss'], hc['val_loss'], rtol=1e-6, atol=0),
              'logged train and val loss equal to 1e-6 every epoch')
        check(len(hc['train_loss']) == E, f'history covers all {E} epochs')
        check(csv_epochs(d) == list(range(E)),
              f'CSV has each epoch exactly once: {csv_epochs(d)}')
        check(not (d / 'last.pth').exists(), 'last.pth removed after finishing')

        best = torch.load(d / 'best_model.pth', map_location='cpu', weights_only=False)
        on_disk = best.get('best_score', best.get('best_val_dice'))
        check(abs(float(on_disk) - c.best_val_dice) < 1e-9,
              f'best_model.pth score ({float(on_disk):.6f}) matches reported '
              f'best_val_dice ({c.best_val_dice:.6f})')
        check(abs(c.best_val_dice - a.best_val_dice) < 1e-9,
              'best val Dice equals the uninterrupted run')
        # ---------------------------------------------------------------- 7-9
        print('\n7-9. divergence guard and bf16')

        class Poison:
            """Corrupt the weights at the end of epoch index 1, so epoch index
            2 produces a non-finite loss -- what fp16 overflow did to the
            widened DenseContextU-Net and FPN controls."""
            def on_epoch_end(self, trainer, epoch, logs):
                if epoch == 1:
                    with torch.no_grad():
                        for prm in trainer.model.parameters():
                            prm.fill_(float('inf'))

        dv = root / 'diverge'
        t = make(dv, epochs=6, resume_every=1, extra=[Poison()])
        h = t.train()
        check(t.diverged_epoch == 3, f'guard fires at the first non-finite epoch '
                                     f'(epoch {t.diverged_epoch})')
        check(len(h['train_loss']) == 3, f'training stopped there, not after 6 '
                                         f'({len(h["train_loss"])} epochs)')
        check(not (dv / 'last.pth').exists(), 'no NaN resume state left behind')
        bm = torch.load(dv / 'best_model.pth', map_location='cpu', weights_only=False)
        check(all(torch.isfinite(v).all() for v in bm['model_state_dict'].values()
                  if v.dtype.is_floating_point),
              'best_model.pth is from before the divergence (finite weights)')

        if DEV == 'cuda':
            pin_determinism(0, 'full')
            m = Tiny()
            cfg = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2,
                                 scheduler='cosine', warmup_epochs=0, use_amp=True,
                                 amp_dtype='bfloat16', save_dir=str(root / 'bf16'),
                                 save_every=0, device=DEV, num_workers=0,
                                 early_stopping=False)
            tr, va = loaders()
            tb = Trainer(model=m, train_loader=tr, val_loader=va,
                         criterion=CombinedLoss(dice_weight=1.0, ce_weight=1.0),
                         config=cfg, callbacks=[])
            hb = tb.train()
            check(tb.scaler is None, 'bf16 uses no GradScaler')
            check(all(np.isfinite(x) for x in hb['train_loss'])
                  and hb['train_loss'][-1] < hb['train_loss'][0],
                  f'bf16 trains: loss {hb["train_loss"][0]:.3f} -> {hb["train_loss"][-1]:.3f}')
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if FAILS:
        print(f'{len(FAILS)} FAILED:')
        for f in FAILS:
            print(f'  - {f}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
