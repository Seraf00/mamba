# Paper 1 — Submission Checklist (Medical Image Analysis)

_Status: data-complete, compiles to 22 pp, 0 errors. Only the items below remain._

Build: `cd D:/Papers/Paper1/paper && latexmk -pdf main.tex`
Pre-submission check (must return ONLY the items in §1 below):
```
grep -rn 'TODO\|PLACE\|REVIEW' main.tex sections/ tables/ | grep -v newcommand
```

---

## 1. MUST DO before submission (only you can)

| # | Item | Where | Time |
|---|------|-------|-----:|
| 1 | Author names | `main.tex` ~L52–60 | 5 min |
| 2 | Author emails (replace `*.TODO`) | `main.tex` `\ead{}` | 2 min |
| 3 | Affiliations / addresses | `main.tex` `\address[a/b]` | 5 min |
| 4 | GitHub repo URL (replace `github.com/TODO/repo`) | `main.tex` L138 | 2 min |
| 5 | Acknowledgements (funding, compute, dataset, clinical reviewers) | `main.tex` L141 | 10 min |
| 6 | Verify bib DOIs / page ranges | `references.bib` | ~1 h |

## 2. SHOULD DO (improves the paper, not strictly blocking)

| # | Item | Notes |
|---|------|-------|
| 7 | **Qualitative figure** (`fig:qualitative`) | Currently a framed placeholder. Assemble a 5×3 grid (4 top + 1 bottom architecture × Good/Med/Poor patient) from the per-model overlays in `results/explainability/<model>/`. Save as `figures/fig_qualitative.png` and swap the `\framebox` rule in `sections/04_results.tex`. |
| 8 | Read once end-to-end for flow | — |

## 3. DEFERRED to the extended/journal version (cut for this submission)

- **Quality-stratified Dice table** (was T5): the prose now says this is in the extended version. To add it, extend `evaluate_all_models.py` to subset per-patient Dice by the CAMUS quality grade.
- **Bland–Altman figure**: removed. To add, store `ef_pred`/`ef_true` per patient in `evaluate_all_models.py`, then `make_figures.py` will draw it.
- **ACDC / EchoNet cross-dataset** validation: mentioned as future work.

## 4. Figures currently embedded (auto-generated, real data)

- `fig_pareto.pdf` — accuracy vs parameters Pareto frontier
- `fig_cd_diagram.pdf` — critical-difference diagram (per-sample Dice)

Regenerate any figure with:
```
python ../scripts/make_figures.py --results_root ../results \
  --benchmark_csv ../results/benchmark_efficiency.csv \
  --paper1_figs figures
```

## 5. Tables currently embedded (auto-generated from local results)

T0 (architectures, static), T1 (main Dice), T2 (boundary mm), T3 (ED/ES),
T4 (EF), T6 (efficiency), T7 (Wilcoxon 9×9).
Regenerate with `scripts/fill_tables.py` (see its `--help`).

## 6. Key numbers (sanity-check these survive any re-run)

- Best base architecture: TransUNet, Dice 0.9122, HD95 1.82 mm
- nnU-Net: Dice 0.9099, HD95 1.93 mm, EF MAE 6.10 %, 15.6 M params (efficiency pick)
- HD95 ~1.9 mm vs published CAMUS nnU-Net ~4.3 mm (the headline)
- Honest caveat (already written): sub-0.01 Dice gaps are within training variance.
