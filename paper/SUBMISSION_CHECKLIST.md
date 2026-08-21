# Paper 1 — Submission Checklist (Medical Image Analysis)

_Status: **submission-ready except item 1.** 25 pp, compiles with 0 errors,
0 undefined references. All tables, figures, and prose numbers come from a
single evaluation run (2026-08-21, native-resolution boundary metrics)._

Build: `cd D:/Papers/Paper1/paper && latexmk -pdf main.tex`
Pre-submission check (must return only the `\TODO` in §1):
```
grep -rn 'TODO\|PLACE\|REVIEW' main.tex sections/ tables/ | grep -v newcommand
```

---

## 1. MUST DO before submission (only you can)

| # | Item | Where | Time |
|---|------|-------|-----:|
| 1 | **Funding sources / grant numbers**, and acknowledge any clinical collaborator who verified the EF pipeline. Delete the `\TODO` if not applicable. | `main.tex` L151 | 10 min |

Everything else previously on this list is done: author names, emails,
affiliations, repo URL, acknowledgements, qualitative figure,
quality-stratified table.

## 2. RESOLVED — HD95/ASSD spacing bug

Boundary metrics were computed on the resized 256x256 grid using the native
0.308 mm spacing, under-reporting every distance by ~2.2x. Fixed in
`evaluate_all_models.py` (commit `ecffda1`) and **re-run on 2026-08-21**
on a local RTX 4060; `boundary_resolution: native` is recorded in every
JSON as a self-verifying marker.

What changed in the paper:

| | before (buggy) | after (correct) |
|---|---|---|
| TransUNet HD95 | 1.82 mm | **4.08 mm** |
| nnU-Net HD95 | 1.93 mm | **4.27 mm** |
| Headline claim | "~2x better than the published 4.3 mm" | **matches** the published baseline |

Dice is unaffected (scale-free): TransUNet 0.9122 -> 0.9121. EF shifted
slightly because every metric now comes from one inference run rather than
being spliced across runs — nnU-Net EF MAE 6.10% -> **5.52%**, r 0.787 ->
**0.828**, which is now the paper's strongest clinical result.

The narrative was rewritten accordingly: protocol modernisation shows up in
**clinical agreement (EF down ~40% vs the 2019 baseline), not in contour
precision (HD95 at parity)**. Two rank flips were also propagated:
UNet-V1 now edges UNet-ResNet, DeepLabV3+ now edges FPN-UNet, and
UNet-ResNet — not TransUNet — has the highest EF correlation (0.832).

Guard against regression: `fill_tables.py` now anchors TransUNet and
nnU-Net **HD95** in the prose lint, not just Dice/EF. Run
`scripts/fill_tables.py --strict` — it fails the build on any
prose/table divergence.

## 3. DEFERRED to the extended/journal version

- **Bland–Altman figure**: still blocked, but for a smaller reason than
  before. All metrics now come from one run, so the consistency objection is
  gone; what is missing is that `evaluate_ef_biplane` stores only aggregates.
  `make_figures.py` prints the exact fix: store
  `ef_metrics['ef_pred']` and `['ef_true']`. One patch + one re-run enables it.
- **ACDC / EchoNet cross-dataset** validation: mentioned as future work.

## 4. Figures embedded (auto-generated, regenerated 2026-08-21)

- `fig_pareto.pdf` — accuracy vs parameters Pareto frontier
- `fig_cd_diagram.pdf` — critical-difference diagram (per-sample Dice)
- `fig_qualitative.png` — GT + 5 models x Good/Medium/Poor

Regenerate:
```
python ../scripts/make_figures.py --results_root ../results \
  --benchmark_csv ../results/benchmark_efficiency.csv --paper1_figs figures
```

## 5. Tables embedded (auto-generated, regenerated 2026-08-21)

T0 (architectures, static), T1 (main Dice + 95% CI), T2 (boundary mm),
T3 (ED/ES), T4 (EF + 95% LoA), T5 (quality-stratified Dice), T6 (efficiency),
T7 (Wilcoxon 9x9).

## 6. Bibliography

`references.bib` — one entry was **wrong, not merely incomplete**:
`baochen2023swinmae` credited Bao/Dong/Piao/Wei (the BEiT authors) for
Swin MAE. Corrected to Xu, Dai et al., _Computers in Biology and Medicine_
161:107037, with DOI. Remaining: 7 `@inproceedings` entries have empty
`pages` (cosmetic bibtex warnings, standard for proceedings).

## 7. Key numbers (sanity-check these survive any re-run)

- TransUNet: Dice 0.9121, HD95 4.08 mm, ASSD 0.82 mm, EF MAE 7.64%
- nnU-Net: Dice 0.9099, HD95 4.27 mm, EF MAE 5.52% (r = 0.828), 15.6 M params
- Published CAMUS nnU-Net baseline: HD95 ~4.3 mm, EF MAE ~9%
- Honest caveat (already written): sub-0.01 Dice gaps are within training
  variance; the top four are a statistical tie.

## 8. Backups

Pre-fix evaluation JSONs are preserved at
`results/<group>/evaluation_pre_hd95fix_backup/`. They are deliberately
renamed `*.PREFIX_BACKUP.json.bak` so that `fill_tables.py`'s recursive
search for `evaluation_results.json` cannot pick them up. **Do not rename
them back.** `results/` is gitignored, so these backups are the only copy
of the pre-fix numbers.
