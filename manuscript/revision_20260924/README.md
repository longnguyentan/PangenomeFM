# Working manuscript revision, evidence update 28 September 2026

This is a scientific working revision of the supplied manuscript. Original
attachment hashes are in `source_provenance.json`. Authors, affiliations, funding,
acknowledgements and administrative statements are preserved.

## Scientific scope

The original v1 cCRE and INS/DEL results remain identifiable. The update adds:

- The completed 120-run trained/random/handcrafted control matrix and the
  masking-induced endpoint-degree shortcut. Original reconstruction scores are
  retained with their limitation; one-hop baseline performance does not resolve it.
- All 450 completed EN-TEx runs, including RNA and the three additional assays,
  primary nulls, exposure-matched/assay-specific sensitivities and ascertainment
  caveats. Subtype cCRE rows are subsets of the binary probe, not new fitted tasks.
- Complete prospective HG008 refits, separate from failed historical replay;
  original TraitGym null/negative results and its classifier sensitivity; and
  COSIGT measured-quality prediction with its stronger constant reference.
- Natural-frequency INS/DEL/INV results, all 174,267 primary-chromosome events,
  explicit length/H controls and unresolved adjusted INV effects. The separate
  223-per-class common-support experiment remains reported.
- The original topology-pretrained representation **T** versus the new
  sequence-conditioned **E**. E passes its three-seed validation development gate;
  chromosome replication and the v1 comparison have no complete aggregate result
  here. No partial chromosome-test metrics were inspected or imported.
- Exact-test resolution (minimum two-sided p = 0.0625 for five nonzero fold
  differences, 0.125 for four), pointwise intervals and multiplicity caveats.

New results, methods and supplementary text are modular files named
`completed_*_20260928.tex`. The original figures remain unchanged. All figure
inputs are present; the captions identify the original-v1 architecture and scope.
No new plot or measurement was invented.

## Evidence bundle and rebuild

`evidence/completed_20260928/provenance.json` identifies every bundled compact
source table, original path, selection rule and SHA-256. The table builder
verifies these snapshots, completion receipts and paired-mean arithmetic before
formatting the rows. It never reads live experiments or recomputes confidence
intervals. The full contrast families include negative results.

From the repository root:

```bash
/opt/anaconda3/bin/python manuscript/revision_20260924/build_completed_evidence.py
cd manuscript/revision_20260924
/Library/TeX/texbin/latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The existing `build_evidence.py` and `correct_figure_labels.py` reproduce the
historical reconstruction/figure audit when NumPy, pandas, matplotlib and PyMuPDF
are available. They are separate from the completed-study table update and need
not rerun to compile this source. `build_completed_evidence.py` requires only
NumPy and pandas. The historical eight-row `entex_rows.tex` is preserved;
`main.tex` now imports the complete `entex_completed_rows.tex`.

This is a multi-file LaTeX project with a bibliography and external vector
figures. It was compiled using the existing local TeX distribution; no software
was installed. The built-in standalone-document compiler does not support these
additional project dependencies. Compile intermediates and QA images are ignored.
The reviewed PDF is `output/pdf/PangenomeFM_working_revision_20260928.pdf` within
this manuscript directory. The complete source is the editable artifact.

## Remaining evidence

Chromosome replication must finish and pass its predefined audits before results
are added. Fold B is separately development-exposed; the four-fold summary still
has historical label exposure. The original-v1 same-budget reference, independent
external E transfer, matched official graph-SSL architecture baselines, stronger
sequence context, donor-excluded evaluation, verified haplotype correspondence,
DUP/complex labels and an all-callable genotyping denominator remain outstanding.
Whole-graph NT completion and component-context preparation establish input
coverage, not biological improvement.

See `../../docs/MANUSCRIPT_EVIDENCE_AUDIT_20260928.md` for the change/evidence audit.
