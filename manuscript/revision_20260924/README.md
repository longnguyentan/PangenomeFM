# Working manuscript revision, 26 September 2026

This is a working scientific revision of the supplied LaTeX/PDF, not a final
submission. Original attachment hashes are in `source_provenance.json`.
Author grant numbers, CRediT contributions, named acknowledgements and a public
checkpoint archive remain unresolved. No placeholder funding details were invented.

## Completed changes

- Preserve the frozen v1 graph and NT methodology and all cached cCRE/SV anchors.
- Correct the SV target to insertion versus deletion among existing variants,
  including vector figure labels. Disclose ordered endpoint features and the
  legacy insertion pseudo-end convention.
- Correct the fusion gate, focal loss, scorer, split/sample counts, pinned NT
  preprocessing, complexity construction and verified donor aliases.
- Include completed EN-TEx findings: inconclusive P0/P1, modest assay-dependent
  P2 effects, and all prespecified sensitivities in the linked experiment report.
- Include the SV complexity result even though gain is smaller in high complexity.
  Label donor results as donor-stratified chromosome holdout, not unseen-donor fitting.
- Correct reconstruction interval weighting while preserving all ten published
  window-mean point estimates. The capacity table instead averages pooled-fold AP.
- Add the completed 120-run intrinsic window-scaling study in the supplement.
  Biological scaling, HG008 and a path-aware model remain unfinished.

## Rebuild

Run from the repository root using Python with NumPy, pandas, matplotlib and
PyMuPDF, plus a TeX distribution with `newtx`, `natbib` and the other packages
listed in `main.tex`:

```bash
python manuscript/revision_20260924/build_evidence.py
python manuscript/revision_20260924/correct_figure_labels.py
cd manuscript/revision_20260924
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The evidence builder checks source means and regenerates Figure 6 and Table S9.
The figure-label correction starts from preserved original vector figures and
records every replacement and file hash. It changes no plotted observations.
SV complexity, donor and scaling figures are copied from the committed compact
campaign outputs; their reproducible commands are in
`docs/FOUNDATION_CAMPAIGN_20260924.md`.

The reviewed PDF is exported locally to
`output/pdf/PangenomeFM_working_revision_20260926.pdf`. Compiled intermediates and
the duplicated PDF are ignored by Git; the complete LaTeX, bibliography, vector
figures, source data and regeneration scripts are version controlled.

## Placement recommendation

Keep the frozen cCRE/SV contrasts, degree-control limitation and corrected SV
complexity result in the main narrative. Retain full EN-TEx sensitivities,
donor/haplotype distributions and intrinsic scaling in the supplement. Promote
biological scaling or HG008 only after their complete-matrix and regression gates
pass. Keep path-aware claims out of the title until a compatible experiment and
frozen downstream benefit are actually demonstrated.
