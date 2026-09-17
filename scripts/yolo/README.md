# YOLO on CAMUS — Paper 2 (IEEE ISBI, 4 pages)

Real-time instance segmentation of LV endocardium, epicardium and left atrium,
evaluated against the Paper 1 baselines under an **identical protocol** so the
rows are directly comparable.

Everything here is built and validated end-to-end. **No training has been run
yet** — the GPU was busy with another workload. The commands below are the
whole remaining job.

---

## Quick start

Run the whole plan (resumable — safe to interrupt and relaunch):

```bash
./.venv/Scripts/python.exe scripts/yolo/run_experiments.py
```

Then regenerate tables and figures, and build the paper:

```bash
./.venv/Scripts/python.exe scripts/yolo/aggregate_results.py && ./.venv/Scripts/python.exe scripts/yolo/make_figures.py && cd paper2 && latexmk -pdf main.tex
```

See exactly what would run, without running it:

```bash
./.venv/Scripts/python.exe scripts/yolo/run_experiments.py --dry-run
```

Run one group at a time (recommended if you want results sooner):

```bash
./.venv/Scripts/python.exe scripts/yolo/run_experiments.py --groups E1_encoding E3_imgsz E4_maskratio
```

---

## Before you launch

The GPU was shared when this was built: a Docker stack (`vp_training`,
`vp_detection`, `vp_classification`, …) held **4.8 GB of the 8 GB VRAM at ~94%
utilisation**, and a 2-epoch smoke train made no measurable progress in ten
minutes. Two things follow.

1. **Free the GPU first**, or the sweep will take days longer than it should
   and the latency numbers in Table 1 will be meaningless (they are measured on
   the same device).
2. **Windows commit is tight.** The 26 GB pagefile lives on `C:`, which had
   1.3 GB free, leaving ~2.4 GB of commit headroom. That is what caused an
   early evaluation crash on a 12 MiB allocation. `wsl --shutdown` reclaims
   ~5.5 GB. The evaluation code was rewritten to stream its distance
   computations so it no longer needs the headroom, but training still does.

Batch sizes in `experiments.yaml` assume a **free** 8 GB card. If you keep the
other workload running, halve them.

---

## What the plan contains

`experiments.yaml` declares 9 groups → **27 result rows from 19 trainings**
(identical configurations shared between groups are hashed after defaults are
applied, trained once, and reused).

| Group | Question |
|---|---|
| E1 | How should the annular myocardium be encoded as a polygon? |
| E2 | How much do model scale and generation buy on 1600 frames? |
| E3 | Does higher input resolution reduce HD95? |
| E4 | Is the 1/4-resolution mask head the accuracy ceiling? |
| E5 | Do natural-image augmentations transfer to sector ultrasound? |
| E6 | Does COCO pretraining help on grayscale ultrasound? |
| E7 | Do the half-sequence frames buy accuracy at ED/ES? |
| E8 | How coarse can the training polygons be? |
| E9 | Best configuration, 3 seeds, for the comparison table |
| I1–I3 | Confidence threshold, TTA, FP16 — evaluated without retraining |

Rough cost on a free RTX 4060 Laptop: ~1–1.5 h per 100-epoch nano run at
640 px, so **roughly 20–30 h** for the full plan. `E7_seq` (~20k frames),
`E2_v26m` and `E4_mr1` (full-resolution prototypes) are the expensive ones.

---

## Which YOLO models

Checked against the installed **ultralytics 8.4.126**, which ships model
families v3/v5/v6/v8/v9/v10/11/12/26 and rt-detr. Segmentation weights that
actually download:

| Model | Params | Role |
|---|---|---|
| `yolo26n-seg` | 3.13 M | **default** — newest generation, end-to-end |
| `yolo26s-seg` | 11.51 M | E2 scale sweep |
| `yolo26m-seg` | 27.11 M | E2 scale sweep |
| `yolo11n-seg` | 2.88 M | E2 generation comparison |
| `yolov8n-seg` | 3.41 M | E2 generation comparison |
| `yolo12n-seg` | — | **no published -seg weights**; config only, scratch-only |
| `yolov9c-seg` | 27.90 M | available; not in the plan (no nano scale) |

YOLO26 is **end-to-end / NMS-free** (`end2end: True`, `reg_max: 1` in its
config): it emits a fixed instance set with no non-maximum suppression. That
removes a data-dependent post-processing stage from the latency budget, which
is exactly the property the deployment argument needs — hence the default,
rather than YOLO11.

Two consequences when reading results:

- The `I1_conf*` sweep still applies, but the NMS IoU threshold (`--iou`) is
  inert for an end-to-end model. Do not report it as a tuned knob.
- E2 separates the two effects on purpose: `v26n` vs `v11n` vs `v8n` is a
  matched-scale *generation* comparison, `v26n/s/m` is the *scale* sweep.

---

## Files

| File | Purpose |
|---|---|
| `prepare_yolo_dataset.py` | CAMUS NIfTI → YOLO polygons. Handles the annulus. |
| `yolo_common.py` | Polygon ↔ CAMUS 4-label map conversion. |
| `boundary_metrics.py` | Memory-safe HD95/ASSD, verified identical to `metrics/`. |
| `scoring.py` | Baseline-compatible scoring (256² overlap, native-res boundary). |
| `check_encoding_ceiling.py` | Upper bound on each encoding. **Already run.** |
| `train_yolo.py` | One training run; every ablated knob is a named flag. |
| `eval_yolo.py` | Native-resolution evaluation + per-sample records. |
| `run_experiments.py` | Executes `experiments.yaml`; resumable, dedupes configs. |
| `aggregate_results.py` | Emits `paper2/tables/Y*.tex`. |
| `make_figures.py` | Emits `paper2/figures/fig_*.pdf`. |

---

## Design decisions worth knowing

**The annulus.** CAMUS label 2 (myocardium) is a ring; a YOLO mask is a single
polygon, which has no hole. We train on the *filled* epicardial disc and
recover the myocardium as `epi \ endo` at inference, so the mask head only ever
produces blob shapes. The alternative — a seam-cut annulus polygon — is also
implemented and ablated (E1).

An "outer contour only" variant was implemented, then **removed**: the outer
boundary of the myocardial ring *is* the epicardial contour, so it is
mathematically identical to the filled encoding, not a distinct baseline. It
would have wasted a training run and added a misleading table row.

**The encoding ceiling is already measured** (`results/yolo/encoding_ceiling.json`):

| Encoding | n | Dice | HD95 | Myocardial target Dice |
|---|---|---|---|---|
| Filled (ours) | 32 | 0.9920 | 0.513 mm | 0.9953 |
| Filled (ours) | 96 | 0.9970 | **0.308 mm** | 0.9984 |
| Seam-cut | 32 | 0.9883 | 0.741 mm | **0.9671** |
| Seam-cut | 96 | 0.9968 | 0.308 mm | 0.9939 |

0.308 mm is *exactly one pixel* at CAMUS spacing — the encoding is essentially
lossless, so any gap to the U-Nets is the model's fault, not the polygons'.
The two encodings look identical on the composed label map (recomposition
re-carves the hole either way) and differ only on the raw myocardial target.

**Comparability.** `scoring.py` reproduces the Paper 1 protocol exactly rather
than improving on it: Dice/IoU on the 256×256 grid with ε=1e-6, HD95/ASSD at
native resolution with true NIfTI spacing. The project's ASSD is
boundary-to-*region*, which is non-standard; it is replicated verbatim so rows
are commensurable, and the textbook boundary-to-boundary value is reported
alongside as `masd_*`. Verified against `metrics/segmentation_metrics.py`:
HD95 to 2e-12 mm, ASSD exactly.

**Paired statistics.** Baseline per-frame Dice was stored in dataset order with
no identifiers. We reconstruct that order and *verify* it by checking that the
ED/ES subset means reproduce each baseline's reported per-phase means — all
nine reproduce to four decimals. If the check ever fails, the significance
markers are suppressed rather than reported on a mispaired sample.

---

## Paper

`paper2/` builds to 4 pages with `latexmk -pdf main.tex`. The layout was
verified against **mock full-size tables** so it will still fit once real
numbers land; the mock data has been deleted.

Numbers awaiting the sweep are wrapped in `\pending{...}` and render in red —
grep for them before submitting:

```bash
grep -rn "pending" paper2/sections paper2/main.tex
```
