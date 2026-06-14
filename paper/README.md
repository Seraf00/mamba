# Paper 1 — manuscript directory

LaTeX source for the Medical Image Analysis submission (Elsevier,
Harvard / author-year citations).

## Build

```bash
cd D:/Papers/Paper1/paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Or with latexmk:

```bash
latexmk -pdf main.tex
```

## Layout

```
paper/
├── main.tex              # Manuscript skeleton (elsarticle, Harvard refs)
├── references.bib        # Bibliography (verify DOIs before submission)
├── README.md             # this file
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methods.tex
│   ├── 04_results.tex
│   ├── 05_discussion.tex
│   └── 06_conclusion.tex
└── tables/
    ├── T0_archs.tex            # done (9 architectures, real param counts)
    ├── T1_main_dice.tex        # done (real per-class Dice for all 9)
    ├── T2_boundary.tex         # done (real HD95, ASSD in mm)
    ├── T3_edes.tex             # partial — HD95 done, Dice TODO
    ├── T4_ef.tex               # done (real EF MAE, r, bias)
    ├── T5_quality.tex          # PENDING — needs quality-grade subset eval
    ├── T6_efficiency.tex       # PENDING — re-run scripts/benchmark.py
    └── T7_wilcoxon.tex         # PENDING — extract from per-sample CSVs
```

## Status

### What is in the draft

All six sections are written with real measured numbers from
`/content/results/base_models/evaluation/*.json`. Tables T0, T1, T2, T4
contain the actual test-set results for all 9 base architectures
(TransUNet 0.9122 / nnU-Net 0.9099 / UNet-ResNet 0.9087 / ... /
DenseContextU-Net 0.8522).

### What is still pending (greppable as `\TODO{}` / `\PLACE{}`)

| Item | Where | Action | Time |
|---|---|---|---:|
| Author block | `main.tex` `\TODO{}` | fill names/affiliations | trivial |
| ED/ES Dice columns | `T3_edes.tex` | already in eval JSON, just paste | ~15 min |
| Per-quality-grade Dice | `T5_quality.tex` | small evaluation-script extension to subset by quality grade | ~1 h |
| Efficiency numbers | `T6_efficiency.tex` | re-run `scripts/benchmark.py` with latest code | ~15 min |
| Wilcoxon $9\times 9$ matrix | `T7_wilcoxon.tex` | extract from per-sample Dice arrays | ~30 min script |
| Bland--Altman figure | `figures/` | matplotlib from per-patient EF CSVs | ~30 min |
| Accuracy--cost Pareto figure | `figures/` | matplotlib scatter | ~15 min |
| Critical-difference figure | `figures/` | `Orange.evaluation` or custom matplotlib | ~30 min |
| Qualitative grid (5 rows × 3 cols) | `figures/` | save predictions on Good/Medium/Poor patients | ~30 min |
| References — DOIs and page ranges | `references.bib` | manual verification | ~1 h |

Before submission:
```bash
grep -r 'TODO\|PLACE\|REVIEW' sections/ tables/ main.tex
```
should return an empty list.
