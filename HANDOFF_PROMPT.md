# Handoff prompt — strategy session for a high-rank publication

Copy everything below the line into a fresh Claude Code session opened in
`D:\Papers\Paper1`. It is written to be self-contained: it states what exists,
what was measured, and what the open question is, so the new session does not
have to re-derive any of it.

---

I have completed a large empirical study on the CAMUS echocardiography
benchmark and need help deciding how to turn it into a high-impact publication.
Please start by reading the artefacts rather than trusting my summary — I want
the numbers verified independently.

## Read these first

- `scripts/yolo/README.md` — the pipeline and design decisions
- `scripts/yolo/experiments.yaml` — the machine-readable experiment plan (13 groups)
- `results/yolo/summary.csv` and `results/yolo/*/evaluation.json` — every run
- `results/yolo/encoding_ceiling.json`, `latency_fair.json`,
  `baseline_ef_native.json`, `_EF_ORACLE/ef_oracle.json`
- `paper2/` — finished 4-page IEEE ISBI paper
- `paper3/` — journal-length draft (Computers in Biology and Medicine format)
- `paper/` — a separate, earlier paper on Mamba-based CAMUS segmentation
  (context: same dataset, same 400/50/50 official split, nine baselines reused
  here)

## What was done

Real-time instance segmentation (YOLO26n-seg) applied to CAMUS, on the official
400/50/50 split (verified byte-identical to the released `subgroup_*.txt`
files). Roughly 50 training runs across 13 factors, all on one RTX 4060 Laptop
GPU. Evaluation reproduces the protocol used for nine semantic-segmentation
baselines: Dice/IoU on a 256x256 grid, HD95/ASSD at native resolution in mm.

## Key results (verify these against the artefacts)

**Positive**
- 0.9151 mean Dice, 3.74 mm HD95 at 2.7 M parameters — best in both columns
  against nine baselines (TransUNet 0.9121/4.08 at 102 M; nnU-Net 0.9099/4.27).
- Near-invariant to input resolution: 15.7 ms @256 px vs 15.5 ms @640 px, where
  every encoder-decoder scales 4-8x with pixel count. 1.5-23x faster at matched
  640 px. End-to-end 11.8 ms.
- Encoding ceiling: representing the annular myocardium as a filled epicardial
  disc and subtracting is near-lossless — 0.997 Dice, 0.308 mm HD95 (one pixel).

**Negative (the bulk of the study)**
- Measured seed floor: three seeds of one config span 0.134 mm HD95, and a
  paired Wilcoxon over the 200-frame test set calls that significant (p=0.035).
- Read against that floor, NONE of 13 factors improves on a stock configuration:
  encoding, vertex budget, model scale (2.7-27 M), generation, input resolution
  (512-960), mask-prototype resolution, augmentation recipe, pretraining, data
  scale (10.6x more frames), boundary-weighted loss, the three loss gains, the
  optimiser.
- Full-resolution mask prototypes are significantly WORSE (+0.157 mm, p=0.008).
- Only training-side factors register: removing augmentation costs 0.318 mm
  (p<0.001); COCO initialisation is worth 0.183 mm (p=0.009).

**Methodological findings**
- The official CAMUS EF code selects `find_contours(...)[0]` — first contour by
  position, not by area. Invisible on ground truth (single-component), but on
  predictions at native resolution it drives EF correlation from r=0.911 to
  r=0.030. A largest-component filter is mandatory.
- Computing volumes on a resampled 256x256 grid with native spacing gives
  ~10 ml ventricles instead of ~108 ml; EF partially survives because it is a
  ratio, so the fault is silent.
- Test-time augmentation is silently ignored by Ultralytics for segmentation
  models — it logs a warning and falls back to single-scale.
- Published inter-observer Dice on CAMUS is 0.899 (Leclerc et al., TMI 2019).
  Most published models, including ours, now exceed it.

## What I need from you

**1. Independent verification.** Re-derive the headline numbers from the
artefacts. Tell me where my claims are overstated, under-evidenced, or where a
reviewer would object. Be adversarial about this — I would rather find the
problems now.

**2. A literature review of the last two years (2024-2026) on CAMUS and cardiac
ultrasound segmentation.** For each relevant paper, record: the split used, the
metric definitions (native vs resampled boundary metrics), whether seeds or
variance are reported, the claimed numbers, and whether the comparison to prior
work is like-for-like. I am specifically interested in whether the
saturation/annotation-ceiling argument holds across the literature, and in
which claimed SOTA numbers are above inter-observer agreement. Treat
ResearchGate-only or unindexed preprints with suspicion — I was previously given
a list of "SOTA" papers of which several could not be verified and one reported
a right-ventricle score on a dataset that has no right-ventricle annotation.

**3. A publication strategy.** Given that my strongest results are negative and
methodological, tell me honestly:
   - Is this a reproducibility/benchmarking paper, or can it be reframed as a
     methods paper with a genuine novelty claim?
   - Which venue maximises impact-weighted acceptance probability? Consider
     journals (Medical Image Analysis, Computers in Biology and Medicine, IEEE
     JBHI, Computerized Medical Imaging and Graphics, MELBA) and conferences
     (MICCAI, ISBI, MIDL). MELBA and MIDL are explicitly receptive to
     reproducibility and negative results.
   - What is the minimum additional work that would lift this from "solid
     benchmark study" to "high-rank novelty"?

**4. A concrete novelty plan.** Based on the evidence — not on what is
fashionable — propose the highest-expected-value extensions. Note that my data
already rules out capacity-based approaches (scale, prototype resolution, more
data all null or negative), so proposals must target something the evidence
leaves open. Candidates I have considered, with my current reasoning:
   - Multi-task learning (segmentation + anatomical keypoints) — the one
     verified SOTA method on CAMUS uses this; adds information, not capacity.
   - Temporal/state-space modelling over the cardiac cycle — 17,006 annotated
     sequence frames are currently unused per-frame; but note that naively
     adding them as training data did NOT help.
   - Contour/vertex regression with a GNN (Curve-GCN / Deep Snake style) — the
     one architectural family my evidence does not rule out, since it changes
     the output representation rather than capacity.
   - Cross-dataset generalisation (CAMUS -> EchoNet-Dynamic / HMC-QU) — the
     axis where headroom demonstrably still exists, since in-distribution is
     annotation-bound.
   - Preprocessing (speckle reduction, gain normalisation) — untested, and
     training-side, which is where all my positive effects were.
   I am sceptical of KAN, mixture-of-experts, GANs and VLMs here: all are
   capacity or representation changes on a task where capacity changes did
   nothing, and KAN additionally costs latency, which undercuts the deployment
   argument.

**5. Hard constraints to respect in any proposal.**
   - One RTX 4060 Laptop GPU, 8 GB. A Docker ML stack sometimes contends for it.
   - Any new comparison MUST use >=3 seeds. My own data shows single-seed
     ablations on this benchmark produce false positives.
   - Any new evaluation MUST state the split and metric resolution explicitly.
   - I need this to be genuinely publishable, not padded. If the honest answer
     is that the current material is a solid benchmark paper and not a
     high-rank novelty paper, say so.
