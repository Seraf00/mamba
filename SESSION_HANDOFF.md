# Session handoff — CAMUS four-paper revision

**Read this first, then `REVISION_PROGRESS.md` (the detailed running log).**
A new chat session starts with no memory of the previous one except the files in
`~/.claude/projects/D--Papers-Paper1/memory/`, this repository, and any artifact
you are pointed at. This file is the bridge.

Last updated 2026-09-26 (Drive audited; R3b, R4b and R5 found finished).
Code on GitHub: `370e341` plus an uncommitted R5b cell.

## Where the work stands

Four manuscripts are being revised against two referee reports plus a
pre-submission audit. Training is being re-run from scratch on one pinned GPU,
because the original runs had three defects: early stopping truncated 20 of 48
runs, batch size was confounded with architecture (128 for base, 16 for SSM),
and EF was computed on the wrong grid.

| group | what | state |
|---|---|---|
| R1 | 9 baselines, canonical | **done**, downloaded to `results_revision/r1_canonical` |
| R2 | seeds 1 and 2 for 4 models | **done**, `results_revision/r2_seed{1,2}` |
| R3 | 6 widened parameter-matched controls | 4 float16 controls **done**; group files stale, see below |
| R3b | DenseContext + FPN in bf16 | **done** (17.8 h) |
| R4 | 29 SSM runs at batch 8 | **20 of 29**; 6 Mamba-2 fail as published (final), 3 recovered by R4b |
| R4b | 3 diverged SSM arms in bf16 + 3 controls | **done** (6.0 h) |
| R5 | 68 position-ablation arms | **65 of 68** (33.4 h); 3 diverged in fp16 |
| R5b | those 3 arms in bf16 | cell added, not run, ~1.2 h |
| evaluation | test Dice/HD95, EF, figures | **not run for any group** |
| tables | regenerate all four papers | blocked, see "Open work" |

**Training is finished apart from R5b.** Evaluation has never run, so every
number below is validation Dice.

Everything runs from `notebooks/colab_revision.ipynb` on a Colab **G4
(RTX PRO 6000 Blackwell)**. All of R1-R5 must stay on that one card and on
torch 2.11.0+cu128: mixing GPUs or stacks is the same class of confound the
revision exists to remove.

## Results so far (validation Dice; test Dice needs the Evaluation cells)

R1: DeepLabV3+ 0.9070, UNet-ResNet 0.9052, nnU-Net 0.9039, UNet-V2 0.9038,
TransUNet 0.9021, UNet-V1 0.9000, DenseContext 0.8560, FPN 0.8275, Swin 0.8139.

DeepLabV3+ went from last (0.8387, stopped at epoch 37) to first. Any prose
explaining its weakness as architectural must be rewritten.

R4 (19 of 29 arms, 19.2 h): best is mamba_deeplab_vmamba 0.9097, above every
baseline. But the gain tracks the parameter multiplier, not the SSM: where the
SSM block adds <=10% parameters the deltas are +0.0027 (deeplab), +0.0014
(unet_resnet) and **-0.0012** (nnunet), at or below the seed floor; the large
gains (fpn +0.0806, dense_context +0.0380) are all arms 2.5-5.4x their base.
**So R3b's fpn_wide and dense_context_unet_wide now decide the headline claim.**
The published Mamba-2 {4 train, 6 fail} partition replicated exactly. Four arms
newly diverged in fp16 (swin+mamba, swin+vmamba, transunet+vmamba,
pure_mamba+mamba) and need R4b. No seed estimate exists for DeepLabV3+ or its
SSM arm, so "SSM beats the best baseline" (a 0.0027 gap) is unsupported today.

**bfloat16 costs about 0.018 Dice.** R4b's three float16/bf16 pairs of the same
arm: TransUNet+Mamba -0.0199, TransUNet+VMamba -0.0209, Pure-Mamba+VMamba
-0.0124. Ten times the seed floor. Every bf16 row (R3b's two controls, R4b's
three recoveries, R5b's three arms) needs a precision column and a caveat.

**R3b splits the SSM claim.** fpn_wide 0.8940 bf16 at 169.1 M (~0.912 corrected)
against mamba_fpn 0.9081 -- capacity explains FPN's whole +0.0806 gain.
dense_context_unet_wide 0.8131 bf16 at 5.3 M (~0.831 corrected) against the base
0.8560 and the Mamba arm 0.8940 -- widening *hurts*, so capacity explains none
of DenseContext's +0.0380. The honest claim is architecture-specific.

**swin_unet_wide 0.7729** against base Swin 0.8139: widening by depth hurts. A
finding, not a control.

**fp16 divergence is not deterministic.** mamba_transunet_vmamba diverged at
epoch 28 once and finished at 0.9052 on a later attempt; the other divergence
epochs also moved. A single NaN run does not prove a configuration cannot train.

R2 seed floor (range over seeds 42/1/2): TransUNet 0.0004, nnU-Net 0.0011,
UNet-ResNet 0.0013, UNet-V1 0.0018. The top six span 0.0070, so the ranking is
mostly real. **Swin is the exception**: an earlier batch-8 100-epoch run scored
0.8445 against 0.8139 now, 20-30x the others' seed spread, and Swin was not in
R2. If any claim leans on Swin's rank, it needs seeds.

## Do this next

0. **Delete `results_revision/r3_param_matched/dense_context_unet_wide/` from
   Drive** and empty the Trash. It holds a `best_model.pth` and a 133-byte
   `training_log.csv` from a stopped float16 run and no `results.json`;
   `evaluate_all_models` scans for checkpoints, so left there it would evaluate
   a one-epoch model as a control. The real number is in R3b.
1. **Run the R3 cell exactly as the notebook builds it** (~5-10 min). All four
   controls resume-skip; the point is the side effect. The group's
   `all_results.json` currently holds 2 of the 4 rows, because a session started
   2026-09-26T07:24 with all six controls at float16 -- not the four the cell
   names -- overwrote it and then began retraining dense_context_unet_wide in
   float16 before being stopped. Without this re-walk `fill_tables.py` silently
   misses swin_unet_wide and transunet_wide.
   **Do not re-run R3b, R4 or R4b.** All three are complete.
2. **R5b** (~1.2 h): the three diverged position arms in bf16. The cell
   re-enters the same `r5_position` group, so the 65 finished arms resume-skip
   and the group files come out complete at 68 entries. A commented-out
   alternative in the same cell costs ~7.3 h and makes the affected within-model
   comparisons precision-consistent.
3. **The three Evaluation cells**, in that order (`evaluate_all_models` first --
   it rewrites `evaluation_results.json`, which `colab_session.py` injects
   into). About 115 models across seven groups. **This has never been timed:**
   run the R1 group first and multiply its rate by 12.8 rather than trusting the
   3-8 h guess.
4. Free Drive per group with `offload_check()` -> `verify_offload.py` ->
   `offload()`, then empty the Drive Trash. The programme needs 20-32 GB of
   checkpoints; a free Google account has 15 GB.

## Open work, not blocking training

**`fill_tables.py` is still wired to the pre-revision session names**
(`base_models`, `param_matched`, ...) and reads EF from fixed paths under
`results/yolo/`. Pointed at the new sessions it would pair NEW Dice with OLD EF
for every unchanged model name, silently. It must be rewired to
`results_revision/` before any table is regenerated. Nothing else blocks tables.

Also open: Paper B needs ~500 characters cut for its mandatory declarations;
author names/affiliation/funding TODOs; two missing citations; TRACE-1.2 and
1.3; LAT-1 re-timing; response letters.

## Things that cost hours to rediscover

- `--early_stopping 0` did **not** disable early stopping until 2026-09-18: the
  Trainer has its own counter. Every run now logs
  `Early stopping (effective): OFF`; check that line.
- The widened DenseContext and FPN controls **overflow in fp16** and go NaN
  (epoch 14 and 4). They did this in the original submission too, where early
  stopping at epoch 25 hid it. They now run in bf16, and a divergence guard
  stops any non-finite run at that epoch.
- TransUNet and Swin widen by **depth, not width**. Width matches the parameter
  count but the pretrained weights cannot load, so the control would be a
  randomly initialised network against a pretrained one.
- `--pin full` costs the heavy convolutional models 1.5-2x. R1 took 7.4 h, not
  the 3.5 h an unpinned benchmark predicted. Do not trust unpinned timings.
- Concurrency does not help: 8 jobs gave 1.06x of 1 on the G4. `NUM_SHARDS = 1`.
- Never predict the Mamba-2 shared-memory requirement from a formula; read
  Triton's own `Required:` bytes. The published {4 train, 6 fail} split
  replicates on Blackwell. All six failures are the shared-memory ceiling
  (327808 or 655488 vs 101376); the notebook's R4 prose claiming
  `mamba_unet_v2_mamba2` fails on causal-conv1d's multiple-of-8 rule is wrong
  and must be corrected in the builder.
- fp16 divergence is not confined to the R3 controls: four R4 SSM arms went
  NaN too, while sibling arms of the same architectures finished. Treat fp16 as
  the suspect for any NaN, and rerun in bf16.

## Map

- `notebooks/colab_revision.ipynb` — the only thing to run on Colab. Generated
  by `scripts/build_colab_notebook.py`; **edit the builder, never the .ipynb**.
- `REVISION_PROGRESS.md` — full running log, newest section last.
- `results_revision/` — downloaded results (git-ignored; 2.6 GB so far).
- `results/` — the ORIGINAL pre-revision sessions. Every current paper number
  comes from here. Do not delete.
- Tests, all runnable: `test_trainer_resume.py` (resume + guard + bf16),
  `test_notebook_sync.py` and `test_notebook_rungroup.py` (need rsync + CUDA,
  e.g. WSL `.venv-wsl`), `test_notebook_status.py`, `check_notebook_syntax.py`,
  `verify_downloaded_results.py`, `verify_offload.py`.

## Artifacts

- Referee response audit — https://claude.ai/artifact/9LVNaZT5oDp5kV2wLDupdn
- Four-paper revision plan — https://claude.ai/artifact/8MzK5d4CFHoc3Am65N9RLw
- Revision execution schedule — https://claude.ai/artifact/AGvfzwkEbdNp4fuHfsaZ8t
  (**stale**: its 61.8 h budget and item 06's shared-memory formula are both
  superseded by measurements above)
