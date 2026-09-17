# Handoff — executing the four-paper revision

Copy everything below the line into a fresh Claude Code session opened in
`D:\Papers\Paper1`. It is self-contained: it states what was verified, what
changed on this machine, and what to do first.

---

I have two referee reports on four related CAMUS manuscripts, and a completed
pre-submission audit. I now need to execute the revision. Please verify claims
against the artefacts rather than trusting this summary — the previous session
found three defects that neither referee caught, and it found them by reading
training logs, not prose.

## The four manuscripts

| | Paper | Path |
|---|---|---|
| A | Comprehensive Benchmark: CNNs to Transformers | `D:/Papers/Paper1/paper/` |
| B | How Close Can a Real-Time Detector Get? (ISBI) | `D:/Papers/Paper1/paper2/` |
| C | What Actually Moves Boundary Error? | `D:/Papers/Paper1/paper3/` |
| D | Selective State Space Models (Mamba) | `D:/Papers/Paper2/paper/` |

Prior work products, both current:

- Audit — what is wrong and why: https://claude.ai/code/artifact/4381b9ce-48f5-4b6b-addc-cf7b5c2f97b1
- Revision plan — what to do about it: https://claude.ai/code/artifact/3b9e2a66-2b72-4e51-995b-95743ed56034

## Established facts — verified against files, do not re-derive

**Three defects the referees missed.**

1. Three of Paper A's nine baselines never converged. Early stopping (patience
   20) fired while cosine LR was still near peak: `deeplab_v3` 37 epochs
   (best@16), `fpn` 38 (best@17), `unet_v2` 56. The same DeepLabV3+ in
   `results/param_matched/` runs 100 epochs and scores 0.9140 test Dice vs
   0.8602. The bottom of three leaderboards is a training artefact.
2. Batch size is confounded with the treatment. `experiment_config.json` per
   session: base_models=32, mamba/mamba2/vmamba=16, param_matched=8. There is
   no per-model batch fallback in `train_all_models.py`.
3. Three of six "parameter-matched widened" controls are the base architecture
   re-run — `param_config.json` picks `resnet34`/`resnet50`/`base_features=32`,
   all of which are already the defaults. They are accidental same-config
   replicates, spread up to 0.0117 Dice at fixed seed. That is the noise floor
   both referees demanded.

**The EF discrepancy is resolved.** Papers A and D compute EF from 256x256
predictions with native spacing (~10 ml ventricles). Paper C is correct.
`scripts/yolo/eval_baseline_ef.py` documents the fault in its own docstring.
Correct values are already on disk in `results/yolo/baseline_ef_native.json`;
the ground-truth oracle floor of 5.44% is in
`results/yolo/_EF_ORACLE/ef_oracle.json`. nnU-Net is 8.24%, not 5.52%.

**Other confirmed items.** Seed-mean YOLO Dice is 0.9113 (below TransUNet's
0.9121) while seed-mean HD95 3.809 mm stays best. Actual Dice range across 47
runs is 0.9035-0.9177, not the "0.913 to 0.918" in C section 4.3. Paper D's
`T6_failures.tex` is stale pre-HD95-fix output (values are ~2.2x low) and is
the one table `fill_tables.py` does not generate. Latency in Paper A's T6 and
in B/C Table 4 is attributed to two different GPUs. Per-class HD95/ASSD and IoU
already exist in `evaluation_results.json` — Paper A's "missing contributions"
are a table-generation task. `per_sample_dice` arrays (n=200) are stored, so
patient-level statistics need no GPU.

**The SSM hardware experiment is done.** All 20 configurations measured on the
local RTX 4060 (Ada, cc 8.9, ceiling 101,376 B):

- Probe: `scripts/ssm_shared_mem_probe.py`
- Records: `results/hardware/ssm_shared_mem_ada.jsonl`

Mamba-2 fails on exactly the same six architectures as on Colab and trains on
the same four — a reproduced result on a second GPU generation. Failures are
`OutOfResources` (hard kernel ceiling); the request is 328,192 B for models
spanning 68-242 M parameters, i.e. invariant to parameter count. VMamba trains
8 of 10 and its two failures are plain CUDA OOM, a different axis entirely.
Paper D's Equation 5 evaluates to 4,194,304 B, not the stated 256 KB, and its
substituted dimensions match no block in the study.

## Machine state changed by the previous session

- **WSL Ubuntu** `.venv-wsl/bin/python` (3.10, torch 2.9.1+cu128) now has
  `mamba_ssm` 2.2.6.post3, `causal_conv1d` 1.5.3.post1, triton 3.5.1, all from
  prebuilt wheels. All repo detection flags are True and all blocks report
  `use_fast_path=True`. Invoke from Git Bash as
  `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash <script>`.
- **Windows venv** gained `triton-windows==3.2.0.post21` only. `mamba_ssm` and
  `causal_conv1d` cannot be built there — the Dao-AILab sources put
  `#ifndef USE_ROCM` inside a macro expansion, which MSVC rejects.
- Measured VRAM at batch 32/256px: UNet-ResNet 2.2 GB, DeepLabV3+ 3.0, nnU-Net
  5.2, UNet-V1 6.5 all fit the 8 GB card; UNet-V2 8.9 and TransUNet 11.9 spill;
  FPN OOMs. At batch 8 everything fits except DenseContextU-Net (18.9 GB).
  Windows pages past VRAM silently instead of raising OOM.

## What I want you to do

**First, help me settle five decisions** — they gate everything and three of
them remove work. They are set out in section 02 of the revision plan: whether
Paper D keeps the parameter-matched claim, whether D adds a position ablation
or narrows its title, whether Paper B is submitted at all, which paper owns the
boundary-metric contribution, and whether Paper A stays "comprehensive".

**Then work the plan's phases in order.** Phase 3 (scripts and tables, no GPU)
can start immediately and in parallel with phase 1 retraining. The single most
important structural change is the consistency contract in section 03: every
number in every manuscript must come from a generator reading a named artefact,
with zero hand-typed values. Start by moving `T6_failures` into
`fill_tables.py` and grepping all four manuscripts for numeric literals inside
`tabular` environments.

Be adversarial about my claims. I would rather find problems now than after
submission.
