# Revision progress

Running log for the four-manuscript revision. One item at a time; each entry
records what was done, what verified it, and what it unblocked.

Protocol settled 2026-08-30: **batch 8, 100 epochs, early stopping off**, seeds
{0,1,2} where noted, 256 px (Swin 224, declared not like-for-like).
Decisions: D1 rebuild param-matched arm · D2 position ablation, 1D + VMamba,
**Option B** (add flags to all models) · D3 submit B and C with declaration ·
D4 Paper C owns boundary metrics · D5 reframe Paper A, retrain nine at batch 8.

---

## Done

### Consistency contract  (audit R1 / plan §03)
- `scripts/check_number_provenance.py` added. Classifies every `.tex` in all
  four manuscripts as generated or hand-maintained, flags measurement literals
  inside table environments. **52 -> 0 hard violations.**
- Moved into generators: `T6_failures` (Paper D), `T0_archs` (A),
  `T1_architectures` (D), plus two new tables `T10_shared_mem` (D) and
  `Y7_seedfloor` (C).
- `T0_prior_integration.tex` stays hand-maintained: citation table, no
  measurements.
- Verify with `python scripts/check_number_provenance.py --strict`.

### HD95-1  Paper D failure table
- `T6_failures` regenerates from the same JSONs as everything else. Was stale
  by 2.2-2.5x on HD95, and also wrong on Dice (0.8878 vs 0.8882) and EF
  (Swin-UNet 10.46 vs native 9.44).
- Gained an **Epochs** column: the two "1D divergence" rows ran 50 and 24
  epochs against a 100 budget while every recovery row ran 100.
- Still open: prose "1.92 mm" at `05_results.tex:98`.

### EQ-5  Paper D section 4.3
- Equation 5 replaced by the measurement; `T10_shared_mem` prints the Triton
  kernel's own reported requirement against the device ceiling.
- Both stated mitigations withdrawn, not just the A100 one: the code already
  runs `d_state=16`, below the library default and below what its own comments
  describe as the fix.
- Boundary expressed via max `head_dim` padded to a power of two, which
  explains the byte values as well as the failure set.

### EF-1  (desk half)
- `apply_native_ef()` overwrites `ef_metrics` at load time from
  `baseline_ef_native.json`. Structural, not per-site: the resized-grid block
  is now unreachable from any table or prose anchor. Models with no native
  measurement render `{---}` rather than a plausible wrong number.
- Paper A `T4_ef` repointed, oracle row added. nnU-Net 1st -> 5th
  (5.52 -> 8.24), UNet-V1 1st at 6.66, nothing beats the 5.44 floor.
- Six Paper A prose passages rewritten (abstract, intro bullet, results,
  discussion, two in conclusion). Prose lint clean at 8 anchors.
- Paper A builds at 30 pp. Its 8 overfull hboxes are **pre-existing** --
  a pristine HEAD build has the identical set.

### R5 pilot  `mamba_deeplab`  (decision D2, Option B)
- Genuine on/off for both SSM positions: `mamba_in_bottleneck` (the Mamba
  branch inside ASPP) and `mamba_in_decoder` (fusion after low-level concat).
  Model records `self.mamba_positions`.
- Verified: all four combinations construct and forward at (1,4,256,256);
  param counts all distinct; **all-on = 42,273,860 = the 42.27 M the probe
  recorded**; **all-off = 40,341,540 = base DeepLabV3+ exactly**.
- `load_state_dict` of the trained checkpoint fails on Windows for an unrelated
  pre-existing reason: without `mamba_ssm` the blocks take the pure-PyTorch
  path and nest one level deeper (`mamba.mamba_native.*`). Use WSL for any
  checkpoint-loading work.

### EF-1  (GPU half, base models)
- `eval_baseline_ef.py` now discovers models via `find_model_checkpoints`, so
  it covers every session, and sizes inputs with `get_img_size` (which catches
  `mamba_swin_unet_*` at 224 -- the old dict keyed on the nine base names would
  have evaluated those at 256).
- `scikit-image` (declared in `requirements.txt:19`) was not installed, so every
  EF call raised ImportError and silently skipped all 50 patients. Installed.
- **All nine baselines re-measured in this environment.** Old artefact backed up
  to `baseline_ef_native.PRE_RERUN_BACKUP.json`.

> **SUPERSEDED -- see "Determinism in the EF pipeline" and "Cross-environment
> test" below.** The drift recorded here was TF32 arithmetic, not the software
> environment, and it is removed by pinning. The ranking change was an artefact
> of comparing against an artefact of unknown provenance. Kept for the record of
> what was measured when.

**Superseded finding (2026-08-31): EF appeared not to reproduce across
environments.** Same checkpoints, same code, same split, same GPU -- only library
versions differed:

| model | old | new | delta |
|---|---|---|---|
| UNet-V1 | 6.66 | 6.56 | -0.10 |
| TransUNet | 6.76 | 7.06 | +0.30 |
| UNet-V2 | 7.72 | 7.50 | -0.22 |
| UNet-ResNet | 7.38 | 7.56 | +0.18 |
| DenseCtx | 9.74 | 9.38 | -0.36 |

DeepLabV3+, Swin-UNet and FPN reproduced bit-identically; the rest did not.
Max drift 0.36, mean 0.14, and a 3rd/4th place swap. **What this actually was:**
TF32 reduced-precision matmul, plus a comparison against an older artefact whose
provenance was unknown. With TF32 disabled the same nine models reproduce 8/9
exactly across OS, PyTorch and skimage versions. The "banded quantity" wording
this prompted has been replaced in Paper A by the precision finding.

The one durable lesson: the prose lint caught all four stale numbers the moment
the artefact changed, which is the consistency contract working as intended.

### R5 Option B  -- all nine SSM-capable models now have position flags
Every model takes `mamba_in_<position>` booleans defaulting to the previously
hard-wired behaviour, and records `self.mamba_positions`.

| model | positions | all-on == trained |
|---|---|---|
| `mamba_deeplab` | bottleneck, decoder | 42,273,860 (also all-off == base CNN exactly) |
| `mamba_transunet` | encoder, skip, bottleneck | 213,371,798 |
| `mamba_swin_unet` | encoder, decoder, bottleneck | 66,009,208 |
| `mamba_fpn` | skip, neck, bottleneck, decoder | 183,261,962 |
| `mamba_dense_context_unet` | encoder, skip, bottleneck | 5,390,500 |
| `mamba_unet_v1` | encoder, skip, bottleneck | already had them; all-off == base exactly |
| `mamba_unet_v2` | encoder, skip, bottleneck | 83,044,215 |
| `mamba_nnunet` | encoder, skip, bottleneck | 15,320,656 |
| `mamba_unet_resnet` | encoder, skip, bottleneck | 32,288,734 |

`pure_mamba_unet` is excluded: it is all-SSM by construction, so there is no
non-SSM fallback to ablate to.

Two defects found while doing this, both of which would have produced null
ablation arms that looked like real negative results:
- `use_multiscale_bottleneck` / `use_dual_bottleneck` / `use_global_bottleneck`
  select between two *Mamba* bottlenecks. None of them removes the SSM, so P1
  ("bottleneck is the best position") was untestable on those three models.
- `use_gated_skip` on `mamba_unet_resnet` is likewise a type switch, and the
  ungated branch concatenates -- turning it "off" made the model **larger**
  (-34,080 params). Now `mamba_in_skip` maps to a genuine on/off (+1,399,008).

Verification per model: every combination constructs and forwards at the right
shape; all combinations have distinct parameter counts (no null flags); the
default corner reproduces the trained count exactly. Regression sweep over all
ten models x three variants: **every `mamba` and `vmamba` default is exact**.

The `mamba2` defaults differ on Windows because `mamba_ssm` is absent and the
PyTorch fallback builds a different parameter set. **Confirmed environmental,
not a regression**: re-run under WSL with `mamba_ssm` 2.2.6.post3, all 10 models
x 3 variants pass, including every mamba2 row (e.g. `pure_mamba_unet` mamba2
136,764,484 vs probe 136.76 M). `pure_mamba_unet.py` was never edited and showed
the same Windows discrepancy, which is the independent check.

Promoted into the repo:
- `scripts/test_position_flags.py` -- per-model flag verification
  (`python scripts/test_position_flags.py mamba_fpn skip neck bottleneck decoder`)
- `scripts/verify_position_defaults.py` -- the 30-row regression sweep. Run it
  under WSL (`.venv-wsl/bin/python`); on Windows the mamba2 rows fail for the
  environmental reason above.

### Position ablation driver
`train_all_models.py --position_ablation` emits one arm per single position plus
a no-SSM arm, for every model in the new `POSITION_SETS` registry. Flags travel
as `extra_kwargs`, which the trainer already forwards to `get_model` and
persists into `results.json`, so the ablation table can be generated from the
checkpoints rather than parsed out of run names.

**Grid: 68 runs, ~28.5 h Colab-class** (priced from the measured per-epoch rates,
arms scaled at 0.65 of the full model). This is against the 18 h quoted for
Option A. Worst items: FPN/VMamba 4.8 h, DenseCtx/1D 3.0 h. With R1-R4 the
grand total is ~62 h, not the 51.6 h on the published schedule.

Cheapest honest cut: drop the `pos-none` arm on the six models where it
duplicates a base-model row R1 is retraining anyway -- 6 runs, roughly 5 h.


---

## In flight

Nothing running.

---

### TXT-1  prose and caption errors
- **Paper A average rank was backwards.** Section 4.6 said TransUNet had
  "middling average rank (4.09, sixth of nine)" and built an argument on rank
  not tracking Dice. Recomputed from `per_sample_dice`: TransUNet is **2.96,
  first of nine** at frame level and **2.68, first** at patient level, and the
  rank ordering agrees with the Dice ordering for all nine architectures. The
  CD diagram was right and the prose was wrong. Rewritten, and it now also
  reports that aggregating to 50 patients exchanges 2nd/3rd between nnU-Net
  (2.92) and UNet-V1 (2.82) -- which supports the paper's own "band, not a
  ranking" thesis.
- **Paper A recommendations were stale after the EF fix.** "nnU-Net ... lowest
  EF MAE, smallest Bland-Altman bias" is false on the corrected numbers
  (nnU-Net is 5th at 8.34). Now UNet-V1 lowest MAE 6.56, TransUNet smallest
  bias -0.54, with the caveat that they differ by less than run-to-run
  variation. Also "TransUNet is the highest-HD95 model" -> lowest, 4.08 mm.
- **Papers B and C Dice range.** Was "0.913 to 0.918 across every comparison".
  Now 0.9035-0.9177 across **32 distinct configurations** (all_runs.csv has 47
  rows but 15 are shared control arms). More importantly the *statistic*
  changed: comparing a range over 32 configs against a range over 3 seeds is
  n-biased. On SD, Dice varies **less** between configurations than between
  seeds (0.0037 vs 0.0045, ratio 0.83) while HD95 varies 1.46x more. The
  negative result is stronger stated this way.
- **Y2 caption** named the default configuration YOLO11n-seg; it is YOLO26n-seg.
  Fixed in `aggregate_results.py`, both papers regenerated.
- **Paper D ED/ES rationale was backwards.** "smaller chamber size ... at
  end-diastole" -- the ventricle is at maximum volume at ED. Now states the
  larger chamber and better blood-tissue contrast.
- Paper A's broken `efsec:results` reference is already gone.

### Paper B page budget -- standing constraint
Paper B sits at **exactly** ISBI's four-page limit with no slack. Verified by
building both wordings: the original passage gave 4 pages, a five-line
replacement gave 5. The corrected claim was fitted into 262 characters against
the original's 265. **Any addition to Paper B costs a page** -- the remaining
ADMIN-1 items (AI disclosure, the B-C declaration, citations for the uncited
comparators) are all additions and will need a trim pass elsewhere.


### Determinism in the EF pipeline  (adopted 2026-08-31)

`scripts/yolo/eval_baseline_ef.py` gained a `pin_determinism()` function partway
through the first EF run. **Provenance: not written by the assistant session
that ran the measurements**; confirmed intentional by the user and kept. Worth
recording because the artefacts it now produces will outlive the session.

Timeline established from log timestamps (UTC) against file mtimes (local +1):
the file was modified 06:46 local, i.e. 27 minutes into the vmamba session which
had started at 06:19. Python compiles a module at process start, so no session
used it -- verified empirically, not just argued: re-running nnU-Net under the
current code gives **8.32** against the **8.34** recorded by the run. The four
unpinned sessions are therefore mutually consistent, and are preserved in
`results/yolo/_unpinned_backup/`.

**Pinning is genuinely reproducible.** Three independent pinned runs of nnU-Net:

    mae=8.3200  r=0.782852  bias=0.7600   (x3, bit-identical)

The diagnosis in its docstring is better than the one I had written into Paper A.
The cause is cuDNN autotuning selecting different convolution algorithms between
runs; flipped borderline pixels and `largest_cc` are the *mechanism by which that
becomes visible*, not the cause. Paper A's "How finely EF can rank at all"
paragraph needs rewriting on that basis.

Scale check: pinning moves EF by 0.02 (within-machine). The cross-environment
swing measured earlier was 0.30 on TransUNet. Pinning addresses the smaller of
the two effects; whether it also fixes the larger one is what the cross-env test
answers.

Added: each output JSON now carries a `_determinism` block (seed, cuDNN flags,
TF32 state, CUBLAS_WORKSPACE_CONFIG, torch version, GPU) so an artefact records
the configuration that produced it. Keyed so `load_ef_native` skips it.

The script also exposes `--pin {full,cudnn,tf32,algos,off}`. A four-run sweep
(~15 min) would let Paper A name *which* control accounts for the drift rather
than asserting that a bundle fixes it. Not yet run.


**Determinism confirmed at full scale (2026-09-01).** Two *independent
processes*, both `--pin full`, produced bit-identical EF for all nine base
models:

    deeplab_v3 9.58 | dense_context_unet 9.38 | fpn 10.60 | nnunet 8.32
    swin_unet 9.44  | transunet 7.36 | unet_resnet 7.42 | unet_v1 6.62
    unet_v2 7.56                                        -- 9/9 agree

With the three nnU-Net repeats that is twelve independent confirmations. EF is
reproducible by construction under the pinned configuration.

**The unpinned numbers were worse than one model suggested.** I generalised from
nnU-Net's 0.02 delta and called the drift negligible. Measured across all nine:

| model | unpinned | pinned | delta |
|---|---|---|---|
| FPN-UNet | 9.72 | 10.60 | +0.88 |
| TransUNet | 6.76 | 7.36 | +0.60 |
| UNet-ResNet | 7.56 | 7.42 | -0.14 |
| DeepLabV3+ | 9.50 | 9.58 | +0.08 |
| nnU-Net | 8.34 | 8.32 | -0.02 |

FPN moves 0.88 -- far outside the band Paper A currently describes. **Every EF
number in all four manuscripts is provisional until the pinned run completes.**

**Concurrent writers.** A second process outside this session was writing
`results/yolo/*.json` at the same time (it rewrote `baseline_ef_native.json` at
23:28, adding `per_patient` arrays). It has since moved to
`results/yolo/_ablate/` doing a `--pin off` ablation on FPN. No data was lost --
`_unpinned_backup/` is intact -- and because pinned runs agree exactly an
interleaved pinned artefact would still be correct. Unpinned artefacts would
not have had that property.


**The oracle is not a floor on MAE.** It carries Bland-Altman bias +3.12 --
mask-derived EF systematically overestimates the recorded clinical value. A
model whose own bias runs the other way has its errors partially cancel:
`mamba_fpn_mamba` (bias -1.26) posts MAE 5.46 and r 0.867 against the oracle's
5.44 / 0.863, i.e. it matches the "floor" on MAE and exceeds it on correlation
without being more faithful to the anatomy.

Paper A's nine baselines are all far short on both statistics so its claim
stands, and a qualifying paragraph now says why. **Paper D must not present
5.46 next to "a 5.44 floor no model crosses"** -- that comparison has to be made
on correlation and bias, not MAE.

**TF32 moves the SSM models less than the baselines.** Max delta 0.46
(`mamba_unet_resnet_mamba`, `mamba_fpn_mamba`) against FPN-UNet's 0.88, and in
both directions. `mamba_fpn_mamba` 5.90 -> 5.46 while base FPN 9.72 -> 10.60, so
the apparent SSM advantage on FPN widened from 3.82 to 5.14 points -- still
confounded by base FPN being a 38-epoch unconverged run.


### Cross-environment test  (2026-09-01) -- pinning gives portability too

Nine base models, pinned, on Windows (torch 2.10.0, skimage 0.26.0) against WSL
(torch 2.9.1, skimage 0.25.2):

    8 of 9 identical | max delta 0.04 (TransUNet) | ranking identical

This **contradicts** the warning in `pin_determinism`'s docstring that skimage
version changes the answer. That was inferred from the *unpinned* comparison,
where TF32 dominated. With precision pinned the environment contributes at most
0.04. Before pinning the same comparison moved six of nine and swapped 3rd/4th.

So the honest claim for Paper A is stronger than planned: reproducible
bit-for-bit within an environment, and portable to within 0.04 across OS,
PyTorch version and skimage version. Paper A's prose updated -- it had stated
the pessimistic version.

### STAT-1 closed for Paper A
`T7_wilcoxon` now tests over 50 patients, not 200 frames. Frame ordering
verified empirically (patient-major, blocks of four: 2CH-ED, 2CH-ES, 4CH-ED,
4CH-ES), not assumed.

Result is an honest null: **27 of 36 pairs significant under either unit, 0
pairs change.** The referees are right about the estimand; it changes nothing
here because the surviving differences dwarf the extra power the wrong unit
buys. Stated that way in caption and prose.

The useful finding is the structure: the nine non-significant pairs form exactly
two cliques -- all 6 among the top four (TransUNet, nnU-Net, UNet-ResNet,
UNet-V1) and all 3 among the bottom three (DeepLabV3+, FPN-UNet,
DenseContextU-Net). Every other pair separates. Three tiers, not a ranking of
nine. Two of the lower clique are unconverged baselines, so that grouping may
dissolve after R1.

### FREE-1 closed
`T2_boundary` gained per-class HD95/ASSD; new `T8_iou` gives mean and per-class
IoU, wired into section 4.1. Caption caught a false claim of mine before it
shipped: epicardium is worst on **Dice** for all 9, but worst on **HD95** for
only 2 -- LA is worst for 5, and the two metrics disagree for 7 of 9. Stated as
the disagreement, which is a better argument for the paper than the claim I had
written.


### SEED-1 closed for Papers B and C  (+ LAT-1 first half)

Both papers claimed "the best value in both columns" from **seed 0** -- the best
of three -- tested over 200 frames. Fixing both defects:

| | frame level, seed 0 | patient level, seed mean |
|---|---|---|
| baselines separated | 8 of 9 | **5 of 9** |
| vs TransUNet | +0.0030, p=0.006 | **-0.0008, p=0.65** |

nnU-Net, UNet-V1 and UNet-ResNet lose significance, and the detector is *behind*
TransUNet on the Dice point estimate. HD95 survives cleanly: seed-mean 3.809 mm
against TransUNet's 4.081, a 0.271 mm margin, twice the paper's own 0.134 mm
seed floor.

Corrected claim, now in both papers: **best HD95 in the benchmark; Dice
statistically indistinguishable from the four leading baselines.**

Generators fixed, not just prose:
- `paired_dice` averages over `E9_seed*` and collapses to patient means.
- `table_stats` (Y4) duplicated the computation rather than calling
  `compute_sig`, so patching the latter alone left the published table
  unchanged -- caught because Y4 did not move.
- `table_main` (Y1) still showed seed 0 in bold as best-per-column, directly
  contradicting the corrected prose. Now one row, mean of three seeds.
  TransUNet takes bold Dice; the detector takes bold HD95.
- **LAT-1 (first half):** the Y1 latency column came from `latency_ms_mean`,
  which is validation wall time -- three seeds of one architecture differ
  41.6 / 45.7 / 29.2 ms, and seeds cannot change inference speed. Column
  removed; both papers already report latency properly from
  `latency_fair.json`.

Side effect: Paper B returned to 4 pages and Paper C dropped 23 -> 22.

**Bug I introduced and fixed:** the `_determinism` provenance block I added to
the EF artefacts was rendered as a table row by `make_journal_tables.table_ef`,
emitting a bare underscore and failing Paper C's build. `load_ef_native` skipped
it (it checks `ef_mae`), `table_ef` did not. Now filters `_`-prefixed keys.


### Hardware constraint  (measured 2026-09-01) -- the programme does not fit this machine

`unet_v1`, batch 8, 256 px, AMP. Three independent measurements agree:

| configuration | it/s |
|---|---|
| Windows, real training run (epochs 1/2/3 = 84/81/81 s) | 3.60 |
| Windows, `bench_dataloader.py`, workers 0/4/8/12/16 | 2.14 / 3.63 / 3.64 / 3.66 / 3.65 |
| WSL, same script, workers 4/8/12 | 3.50 / 3.47 / 3.45 |

Ruled out, each by measurement rather than argument:
- **worker count** -- flat from 4 to 16 on both platforms
- **`persistent_workers` / `prefetch_factor`** -- no effect either way
- **Windows vs WSL** -- WSL is marginally *slower* (0.97x), presumably `/mnt/d`
  I/O; spawn-vs-fork is not the issue
- **VRAM** -- only 2.8 of 8 GB used, so not paging

The GPU oscillating 0-87% is the train/validate cycle and the per-step pattern
of a small-batch U-Net, not a feedable input pipeline. **~3.6 it/s is what this
card does on this workload.**

Total epoch time (train + validation) is **81 s** against **8.3 s** on the
hardware that produced the reference sessions -- a **9.8x** gap. So:

| group | reference | this machine |
|---|---|---|
| R1 | 4.5 h | **~40 h** |
| R1+R2 | 6.5 h | **~55 h** |
| R1-R5 | 61.8 h | **~600 h (25 days continuous)** |

Three time estimates were given before this was measured (2x, then "workers
will fix it", then "WSL might fix it") and all three were wrong and optimistic,
because they extrapolated from artefacts produced on other hardware. The only
trustworthy figure is the measured 81 s/epoch.

`scripts/bench_dataloader.py` reproduces the real training rate to within 1%
(3.63 vs 3.60 it/s), so it can be used to re-check this for R2/R4 without
launching a training run.


## Open confounds -- do not write these up as clean results

**SSM EF advantage on FPN is confounded with convergence.** The WSL EF run gives
`mamba_fpn_mamba` 5.90% MAE against base FPN's 9.72% -- a 3.8-point swing, and
5.90 is only 0.46 above the 5.44 oracle floor. But base FPN is one of the three
unconverged runs (38 epochs at batch 32, best@17), while the Mamba arm ran to
100. So this compares a finished model against a truncated one. Same applies in
weaker form wherever a base row from `base_models` is compared against an SSM
row. Cannot be resolved until R1 retrains the baselines at batch 8.

**`pure_mamba_unet_mamba` EF is not comparable.** 42.29% MAE with **36 of 50
patients skipped** on empty LV predictions, r = 0.000. The figure is computed
over 14 patients. Report as undefined, not as a large error.
`mamba_swin_unet_mamba` (43.46%, r = 0.105) is the other collapsed run.

**EF environment sensitivity is narrower than first reported.** Two independent
environments (Windows/skimage 0.26.0, WSL/skimage 0.25.2, different OS and
PyTorch build) agree exactly on 7 of 9 base architectures; 3 are bit-identical
across all three runs including the original artefact. Only TransUNet (0.30) and
UNet-V2 (0.04) move between the two fresh runs, and the ranking is stable in
both. The 0.36 drift and rank swap reported earlier came from comparing against
the *original* artefact, whose provenance is unknown -- not from environment.
Paper A's prose was corrected accordingly. The authoritative artefact is now the
WSL run, which is the same environment the SSM rows are measured in.


## Next, in order

1. Run EF over `mamba_models`, `mamba2_models`, `vmamba_models` (23 models,
   ~3 h). **Must run under WSL**: without `mamba_ssm` the 1D models take the
   pure-PyTorch path, measured in `MAMBA_PERFORMANCE_FIX.md` at 44+ min/batch.
   Outputs land in `results/yolo/ef_native_<session>.json`; `load_ef_native`
   already reads them with canonical precedence.
2. Repoint Paper D's EF prose (6.54 / 5.26 / 5.52 / r 0.87 are all from the
   broken pipeline) once those artefacts exist.
3. R5 Option B on the remaining eight models, same pattern and same three
   checks per model.
4. Then the GPU run groups R1-R5 (51.6 h Colab-class, see the schedule).

## Notes that cost time once

- Heredocs mangle backslashes: write patch scripts with the Write tool.
- Model files carry trailing whitespace on blank lines, so substring anchors
  fail. `scratchpad/pos_patch.py` matches line-by-line with trailing space
  optional and asserts exactly one hit per edit.
- Background tasks piped through `grep`/`tail` produce no output until they
  exit -- check the process, not the log.

---

## Colab notebook rebuild + R3 (2026-09-05)

Decision recorded: the position ablation runs on **Paper 1's production flags**
(`POSITION_SETS`), not `Paper2/models/ablation_models.py`. The ablated models are
therefore the models the papers report. `decoder` exists on only 3 of 9, so a
bottleneck-vs-decoder claim rests on three architectures -- state that rather
than averaging the ragged grid.

### Code prerequisites (done, no GPU)

| item | change | verified |
|---|---|---|
| TF32 not pinned in training | `utils.misc.pin_determinism` + `--pin {full,cudnn,tf32,algos,off}`; recorded as `environment.determinism` | `--pin full` -> TF32 off both, deterministic on; default `off` leaves existing recipe untouched |
| micro-batch was unconditional | `PEAK_GIB_AT_BATCH8` + `_micro_batch_for()`, VRAM-aware | 8 GiB card -> mb=2; simulated 96 GiB -> None |
| no concurrency | `--num_shards/--shard` (round-robin over the same plan) + `--merge_shards` | 15 models -> 3 x 5; merge restores plan order, reports failures |
| R5 retrained R1+R4 | `--position_only` | 96 -> exactly 68 arms |

`set_seed` pins cuDNN algorithm *selection* but never touched arithmetic
precision, which is the larger effect: TF32 accounted for the whole 0.88-point
FPN EF shift.

### R3 rebuilt (`results/param_config_r3.json`)

Root causes in the shipped config: `fpn_channels` capped at 512 (hence the 57%
miss), and `swin_unet`/`transunet` marked unmatched because widening forfeits
pretrained weights.

**New finding beyond the audit.** The arm was aimed at the wrong models. Three
entries were no-ops because the override *equalled the default*
(`unet_resnet`=resnet34, `deeplab_v3`=resnet50, `nnunet`=bf=32), and for those
same models a control answers nothing: nnU-Net's Mamba variant is **smaller**
than its baseline (-1.9%), DeepLabV3+ +4.1%, UNet-ResNet +11.1%. Meanwhile
TransUNet (+98.1%) and Swin (+47.8%) -- where the gap is largest -- had no
control at all. Now gated on `MIN_INCREASE_PCT_FOR_CONTROL = 15`.

| control | widening | matched | vs target | was |
|---|---|---|---|---|
| `transunet_wide` | `vit_layers=26` | 201.32M | **-0.44%** | N/A |
| `fpn_wide` | `resnet101, fpn=1024` | 169.13M | **-2.29%** | -56.6% |
| `swin_unet_wide` | `embed_dim=114` | 58.96M | -4.67% | N/A |
| `unet_v1_wide` | `bf=96` | 69.82M | +1.67% | same |
| `unet_v2_wide` | `bf=104` | 87.12M | +4.91% | same |
| `dense_context_unet_wide` | `bf=200` | 5.29M | -1.86% | same |
| `unet_resnet`, `deeplab_v3`, `nnunet` | UNNEEDED | -- | -- | no-ops trained |

Six genuine controls where there were three; worst error 4.91% against 56.6%.

**TransUNet widens by depth, not width.** Width lands as close (`vit_dim=1128`,
-0.56%) but makes every ViT-B/16 tensor the wrong shape, so the control would be
a randomly initialised transformer against a pretrained one -- the extra
parameters would not be what the comparison measured. `_load_pretrained_vit` now
loads `min(n_layers, 12)` blocks and leaves the rest fresh, which is how the
Mamba variant adds capacity too; width is refused with a message rather than
crashing on a size mismatch. Depth also keeps `head_dim=64`, so the control does
not perturb the head-dimension analysis behind the Mamba-2 ceiling. Baseline
regression: 102,086,852 params, unchanged.

`param_config.json` (describing the *trained* session) is untouched; the rebuild
is a separate artefact until R3 actually runs.

### Notebook

`notebooks/colab_revision.ipynb`, generated by `scripts/build_colab_notebook.py`
(a .ipynb is JSON with escaped strings, so hand-editing is how flags drift).
33 cells, all 20 code cells parse (`scripts/check_notebook_syntax.py`).
R1-R5 as separate groups, canonical settings in one place, evaluation via
`eval_baseline_ef.py --pin full` and `colab_session.py`, then the three
generators and `check_number_provenance.py --strict`.

`notebooks/colab_training.ipynb` carries a SUPERSEDED banner: it is not stale but
actively wrong -- batch 128/16 and ES 20 throughout. Kept as the record of how
the sessions in `results/` were produced.

### Still open

- **Timing on the target card is unmeasured.** No reference session records its
  GPU (`device: cuda` only), so 61.8 h is anchored to unidentified hardware. At
  batch 8 the workload is launch-bound, not FLOP-bound, so a single job on an
  H100 is ~2-3x, not 10x; the gain is `NUM_SHARDS`. Run
  `scripts/bench_dataloader.py` on the actual instance before trusting any
  estimate.
- vCPU, not VRAM, caps concurrency: `NUM_SHARDS * (NUM_WORKERS+1)` vs core count.
- Paper B still needs ~500 characters cut for its declarations.
- ADMIN-1 author items; TRACE-1.2/1.3; LAT-1 re-timing.

---

## GPU decision: RTX PRO 6000 Blackwell ("G4") measured (2026-09-18)

Standalone battery `scripts/colab_gpu_shootout_cell.py`, run on Colab's G4:
sm_120, 188 SMs, 95 GB, 48 vCPU, torch 2.11.0+cu128, Triton 3.6.0,
mamba-ssm 2.3.2.post1. Max shared memory per block 101,376 B -- same as Ada.

| check | result |
|---|---|
| kernel stack | triton, causal-conv1d, Mamba-1, Mamba-2 all execute |
| Mamba-2 | **4 train / 6 fail -- replicates T10 and the original run exactly** |
| DenseContext @ batch 8 | fits, no micro-batching |
| determinism | bit-identical once pinned; TF32 default matmul=off, cuDNN=ON |
| 1 job | 45.03 it/s (12.4x the RTX 4060) |
| 8 jobs | 47.73 it/s aggregate -- 1.06x, 13% efficiency |

**Decision input:** G4 passes every disqualifying check. R1-R5 ~49 h at one job
(601.9 h measured locally x 3.64/45.03). Estimate is UNet-shaped; check R1's
first epochs before trusting it for SSM and heavy models.

### Corrections this forced -- all in chat/tools, none in the papers

1. **The 1280*P + 512 formula was extrapolated below its data.** Fitted to the
   two measured points (P=256 -> 328,192; P=512 -> 655,872), then extended down
   to predict 164,352 B for head_dim 128. G4 runs head_dim 128 on 101,376 B, so
   the extrapolation is false; and two points fit a two-parameter line exactly,
   so the "exact fit" was never evidence. The repo already contradicted it:
   `results/mamba2_models/all_results.json` shows the same four variants
   training. Claims withdrawn from chat: "Ada/A100 run 1 of 10", "three models
   miss A100 by 512 bytes", "H100 would force a s4.3 rewrite".
   **The papers never used the formula**: T10 reports Triton's own `Required:`
   bytes via `fill_tables._probe_required_bytes`, and no .tex contains it.
   From measured requests, all three candidate GPUs give the same {4, 6}: every
   failing variant needs >= 328,192 B, above even H100's 227,328.
   Fixed: `gpu_shootout.py` now predicts from `MAMBA2_MEASURED_REQUEST`; the
   Colab cell reports the kernel's own measured bytes.
2. **Concurrency does not scale.** Separate CUDA processes time-slice the GPU
   without MPS. Flat on both a 24-SM and a 188-SM card, so it was never about SM
   count. Notebook now defaults `NUM_SHARDS = 1`, `NUM_WORKERS = 8`.
3. **"Avoid G4" was overcautious**: the sm_120 stack installed and ran on
   Colab's default torch.
4. **DeepLabV3+ Dice.** 0.8602 -> 0.9140 is TEST Dice and matches T1; I had
   "corrected" it to 0.8387 -> 0.9076, which is VALIDATION Dice. Both real,
   different splits. Reader-facing text uses test Dice. On test Dice, five
   models that ran full length in both sessions differ by -0.007..+0.006;
   truncated DeepLabV3+ +0.054, UNet-V2 +0.017.

### New, separate from shared memory

`mamba_unet_v2` (d_model 341, d_inner 682) fails on G4 with a causal-conv1d
assertion -- channels must be a multiple of 8 -- before reaching the scan
kernel. The original run recorded a shared-memory failure for it. Same outcome,
different cause on a newer stack; d_model 341 (~1024/3) looks like a channel
split worth checking in the model code.

### Stale published artefact

Schedule artifact 4b1b7b35, item 06, states the formula "fits exactly at both
observed points" as evidence. Not republished -- needs the user's go-ahead.

---

## CRITICAL: early stopping was never off; resume built (2026-09-18)

**`--early_stopping 0` did not disable early stopping.** `training.Trainer`
has its own counter (`TrainingConfig.early_stopping=True, patience=20` by
default), separate from the EarlyStopping callback. `train_all_models.py` built
`TrainingConfig` without those fields, so the flag removed only the callback.
Every revision run would have stopped at patience 20 again, reproducing NEW-1.
It was declared fixed after checking only the callback. Pre-revision sessions
were unaffected (20 in both places). Fixed in both config constructions; each
model now logs `Early stopping (effective): OFF` from the config itself. Other
entry points checked: `train.py` (not in the pipeline), Paper2
`train_ablation.py` and Paper3 `train_refinements.py` pass the field explicitly.

**Resume, which did not exist.** Before this, a disconnect meant:
- restarting the interrupted model from epoch 0;
- `--resume_from` rewriting all_results.json without the finished models;
- the notebook crashing after every group at NUM_SHARDS=1 (merge found no shard
  files).

Now:
- `Trainer`: `start_epoch`; checkpoints add AMP scaler, RNG (torch/cuda/numpy/
  python), patience counter and cumulative elapsed time; atomic write
  (tmp + rename); rolling `last.pth` every `resume_every` epochs, deleted on
  completion; ModelCheckpoint's best score restored (no worse model overwriting
  best_model.pth); if best_model.pth is newer than the resume state, its score
  is adopted.
- `CSVLogger`: appends on resume, dropping rows the resume state does not
  cover (so a crash between the CSV write and last.pth does not duplicate
  epochs).
- `train_all_models`: `--resume` (skip models with results.json, continue from
  last.pth), `--resume_every` (5), `--checkpoint_every` (0 in the notebook;
  nothing reads periodic checkpoints). `training_time_seconds` is cumulative.
- Notebook: `--resume` in BASE_ARGS; sync to Drive every 10 min during training;
  `restore_from_drive()` at session start; own Drive folder `results_revision`;
  merge only when sharded; torch pinned to 2.11.0; a status cell per group.

Tests:
- `scripts/test_trainer_resume.py`: 15/15. Early stopping off/on with a
  flat-Dice control. Interrupted + resumed run bit-identical to uninterrupted
  (weights and full history) on GPU with AMP. CSV holds each epoch exactly once.
  best_model score consistent. last.pth cleaned up. Found and fixed an RNG-state
  device bug that would have crashed every resume.
- `scripts/test_notebook_sync.py` (WSL/Colab, needs rsync): 8/8, on the
  notebook's own extracted code. Correct files reach Drive; a finished model's
  stale last.pth is removed from Drive; saving an empty disk before restoring
  deletes nothing; restore reports only real interruptions.
- `scripts/test_resume_e2e.py`: real CAMUS + DeepLabV3+, process tree killed
  mid-model, same command relaunched twice (resume, then skip).

---

## Notebook review (2026-09-18)

Re-read every cell of `notebooks/colab_revision.ipynb` in run order, then
executed the settings + helper cells, extracted from the built notebook, against
the real trainer in WSL (`scripts/test_notebook_rungroup.py`). Found and fixed:

| # | problem | consequence | fix |
|---|---|---|---|
| 1 | evaluation ran `colab_session.py` BEFORE `evaluate_all_models.py` | the second rewrites `evaluation_results.json` and **erased the per-patient EF arrays** (Bland-Altman data) | reordered: evaluate -> EF -> colab_session |
| 2 | Tables cell ran on Colab | manuscripts are not there (Paper D is another repo); and `fill_tables.py` reads the OLD session names + fixed `results/yolo/` EF files, so it would **pair new Dice with old EF, without error** | cell removed; tables built locally. Generator rewiring is the open item below |
| 3 | R2 not evaluated | the seed floor, R2's whole purpose, had no test metrics | R2 added to SESSIONS |
| 4 | R3 retrained the 9 base models (15 runs) | ~1/3 of R3 wasted, duplicate baseline rows | `--wide_only` -> 6 runs |
| 5 | `evaluate_all_models.py` unpinned | test Dice/HD95 under cuDNN TF32 while training + EF pinned | `--pin` (default full) |
| 6 | sync filter lacked `.tex/.png/.pdf` | T5 table, overlays, Bland-Altman figures never left Colab | added |
| 7 | child stdout block-buffered | log and status line dozens of epochs late; buffered lines lost at disconnect | `PYTHONUNBUFFERED=1` |
| 8 | shard log opened with `'w'` | a relaunch erased the previous session's log | append + session header |
| 9 | status line parsed a 20 KB tail and matched tqdm's bar | model name vanished; epoch read "1:"; new model showed the previous model's epoch | whole-file parse on the summary marker, epoch must follow the latest `Training:` (`test_notebook_status.py`, 4/4) |
| 10 | no guard before R4/R5 | without mamba-ssm's fast path the trainer waits 10 s then trains ~100x slower | `_require_mamba_fast()` raises first |
| 11 | `pip install -r requirements.txt` included torch/mamba | a failed mamba build would abort the whole install | install everything else only |
| 12 | stale markdown | "set TORCH_SPEC=None", all Mamba-2 failures called shared-memory (UNet-V2 is causal-conv1d alignment on G4), no ordering note | rewritten |
| - | tqdm redraws | MB of progress lines per group synced to Drive | `TQDM_MININTERVAL=30` |

Verified by execution: dry-run model counts R1 9 / R2 4 / R3 6 / R4 29 /
R5 68; a real one-epoch group trains, exits 0, syncs to Drive, logs
`Early stopping (effective): OFF`; relaunch appends the log and skips the
finished model; mamba guard passes where mamba-ssm is installed.
`test_notebook_sync.py` 8/8 with the new filter.

### Open -- blocks TABLES, not training

`fill_tables.py` must be pointed at `results_revision/` sessions
(`r1_canonical`, `r2_seed*`, `r3_param_matched`, `r4_ssm_batch8`,
`r5_position`) and at each session's `baseline_ef_native.json`, instead of
`base_models`/`param_matched`/... and `results/yolo/*.json`. As it stands it
would silently attach OLD EF to NEW Dice for every unchanged model name.

---

## First results on G4, and what R3 broke (2026-09-19)

### R1 + R2 (downloaded to results_revision/, verified)

`scripts/verify_downloaded_results.py`: all 17 checkpoints load, parameter
counts match, every training_log.csv has epochs 0-99 exactly once. All groups:
batch 8, 100 epochs, early stopping effective OFF (0 early stops), --pin full,
TF32 off both, torch 2.11.0+cu128, RTX PRO 6000 Blackwell. R1 unet_v1 was
resumed after a real disconnect at epoch 6 -- clean.

Validation Dice (test Dice still to come from the Evaluation cells):
DeepLabV3+ 0.9070 (was 0.8387 truncated at 37 -> now FIRST), UNet-ResNet
0.9052, nnU-Net 0.9039, UNet-V2 0.9038, TransUNet 0.9021, UNet-V1 0.9000,
DenseContext 0.8560, FPN 0.8275, Swin 0.8139. FPN and Swin weak at full
length, so not an early-stopping artefact.

Seed floor (seeds 42/1/2, val Dice range): TransUNet 0.0004, nnU-Net 0.0011,
UNet-ResNet 0.0013, UNet-V1 0.0018. Top six span 0.0070. Swin: a previous
batch-8/100-epoch run scored 0.8445 vs 0.8139 now -- 20-30x the seed floor of
the others; Swin's seed variance is unmeasured (not in R2).

Measured time: R1 7.4 h (DenseCtx 164 min, FPN 147 min). --pin full costs the
heavy conv models 1.5-2x over the unpinned benchmark; the "~49 h" figure was
wrong. Budget R4 ~30 h, R5 ~45 h.

### R3 first run: 3 of 6 controls usable

| control | outcome |
|---|---|
| unet_v1_wide | 100 ep, 0.8999 (base 0.9000) |
| unet_v2_wide | 100 ep, 0.9041 (base 0.9038) |
| transunet_wide | 100 ep, 0.9035 (base 0.9021, seed range 0.0004) |
| swin_unet_wide | FAILED at construction: embed_dim=114 cannot load 96-wide Swin-Tiny |
| dense_context_unet_wide | NaN from epoch 14; ran 86 NaN epochs, ~6 h |
| fpn_wide | NaN from epoch 4; stopped by user at epoch ~32 (~6.3 min/epoch) |

Both NaN runs: train loss falling normally, then sudden NaN -> fp16 activation
overflow signature. Same controls diverged in the ORIGINAL submission too;
early stopping at epoch 25 hid it.

### Fixes

- **Divergence guard** (Trainer): non-finite train/val loss stops the model at
  that epoch, before best_model/last.pth are written; train_all_models renames
  best_model.pth -> diverged_best_model.pth, writes DIVERGED.json, raises (no
  results.json, so a relaunch retries).
- **--amp_dtype {float16,bfloat16}**: bf16 has fp32 range; no GradScaler. Default
  float16 path unchanged. Recorded per model as results.json `amp_dtype`.
- **Swin widening by depth**: [2,2,12,2] = 63.16 M, +2.12% (was embed_dim=114,
  -4.67%, unbuildable). Loader loads min(depth, 6) stage-3 blocks, refuses other
  widths with a message. Built WITH pretraining and trained fwd/bwd: OK.
- **param_match._verify_buildable**: every chosen control is constructed with
  pretraining on before being emitted; failures become BROKEN and are excluded.
- Notebook: R3 cell trains unet_v1/unet_v2/swin/transunet (3 skip via resume);
  new R3b group `r3b_param_matched_bf16` trains dense_context + fpn in bf16.
  Header time estimate corrected.
- test_trainer_resume.py: +guard, +bf16 checks. Found the logged train/val LOSS
  scalars differ intermittently at 1 float32 ulp between identical runs (reduction
  order); weights and Dice stay bit-identical. Test now exact on weights and
  Dice, 1e-6 on logged losses; 3/3 consecutive passes.

### Relaunch procedure for R3
Delete from Drive `results_revision/r3_param_matched/dense_context_unet_wide/`
and `.../fpn_wide/` (NaN runs; fpn's last.pth is a NaN state), empty Trash, pull,
reopen notebook, run setup/settings/section 4, then R3 cell, then R3b cell.
