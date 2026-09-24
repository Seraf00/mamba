# Session handoff — CAMUS four-paper revision

**Read this first, then `REVISION_PROGRESS.md` (the detailed running log).**
A new chat session starts with no memory of the previous one except the files in
`~/.claude/projects/D--Papers-Paper1/memory/`, this repository, and any artifact
you are pointed at. This file is the bridge.

Last updated 2026-09-22 (R4 results ingested). Code on GitHub: `9ac4b7d`.

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
| R3 | 6 widened parameter-matched controls | **3 of 6 done**; Swin to retrain, DenseContext+FPN to retrain in bf16 (R3b) |
| R4 | 29 SSM runs at batch 8 | **19 of 29 done**; 6 Mamba-2 fail as published, 4 need bf16 |
| R4b | 4 diverged SSM arms in bf16 (+2 controls) | cell added, not run, ~6 h |
| R5 | 68 position-ablation arms | not started, ~45 h |
| evaluation | test Dice/HD95, EF, figures | **not run for any group** |
| tables | regenerate all four papers | blocked, see "Open work" |

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

R2 seed floor (range over seeds 42/1/2): TransUNet 0.0004, nnU-Net 0.0011,
UNet-ResNet 0.0013, UNet-V1 0.0018. The top six span 0.0070, so the ranking is
mostly real. **Swin is the exception**: an earlier batch-8 100-epoch run scored
0.8445 against 0.8139 now, 20-30x the others' seed spread, and Swin was not in
R2. If any claim leans on Swin's rank, it needs seeds.

## Do this next

1. In Colab: run **R3** (Swin only, ~30 min), then **R3b** (DenseContext +
   FPN in bf16, ~6 h + ~10 h) -- R3b now decides the headline SSM claim, see
   below -- then **R4b** (the diverged SSM arms in bf16, ~6 h; cell exists,
   trains 6 arms because selection is a cross product -- 4 retries plus 2
   float16-vs-bf16 controls), then **R5**.
2. Run the three Evaluation cells after each group finishes, in that order
   (`evaluate_all_models` first -- it rewrites `evaluation_results.json`, which
   `colab_session.py` injects into).
3. Free Drive per group with `offload_check()` -> `verify_offload.py` ->
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
