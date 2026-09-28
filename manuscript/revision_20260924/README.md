# Method-focused manuscript revision, 28 September 2026

This is a scientific working revision of the supplied manuscript. Original
attachment hashes are in `source_provenance.json`; the latest LaTeX and three
meeting-note identities are in `method_revision_provenance.json`. Authors and
affiliations are preserved. The requested general T2T acknowledgement is added;
exact grant, consortium, contribution and release details remain author-confirmed.

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

Completed results, methods and supplementary text remain modular files named
`completed_*_20260928.tex`. The method-focused writing pass adds no measurements.
Biological figures and generated numerical rows remain unchanged. Figures 1 and 2
are rebuilt as PDF/SVG vectors from `build_method_figures.py`; their captions
distinguish original directed masking from the later reciprocal-mask repair.

The six-paragraph unheaded Introduction leads to five topical Results sections.
Methods opens with representation/architecture, masking/loss/optimization and
frozen pooling/conditional comparisons before resources. The named sequence
coverage Results heading explains truncation. The supplement has its own narrative
and reference list. See `../../docs/LATEX_COMMENT_RESPONSE_20260928.md` for all 118
comment lines, and `../../docs/METHOD_FIRST_REVISION_20260928.md` for the meeting
map, structural rationale and verification. Historical seeding and masking
qualifications are now explicit, without changing original performance numbers.

## Evidence bundle and rebuild

`evidence/completed_20260928/provenance.json` identifies every bundled compact
source table, original path, selection rule and SHA-256. The table builder
verifies these snapshots, completion receipts and paired-mean arithmetic before
formatting the rows. It never reads live experiments or recomputes confidence
intervals. The full contrast families include negative results.

From the repository root:

```bash
/opt/anaconda3/bin/python manuscript/revision_20260924/build_completed_evidence.py
/opt/anaconda3/bin/python manuscript/revision_20260924/build_method_figures.py
/opt/anaconda3/bin/python manuscript/revision_20260924/build_supplement_references.py
cd manuscript/revision_20260924
/Library/TeX/texbin/latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The existing `build_evidence.py` and `correct_figure_labels.py` reproduce the
historical reconstruction/figure audit when NumPy, pandas, matplotlib and PyMuPDF
are available. They are separate from the completed-study table update and need
not rerun to compile this source. `build_completed_evidence.py` requires only
NumPy and pandas. The historical eight-row `entex_rows.tex` is preserved;
`main.tex` imports the complete `entex_completed_rows.tex`.

`references.bib` is the maintained bibliography. Supplementary citations use
namespaced keys, and `build_supplement_references.py` derives their entries without
changing any bibliographic field. This prevents duplicate natbib labels and PDF
anchors while giving the supplement its own S-numbered references. `latexmk` runs
BibTeX for both lists automatically. Do not hand-edit the generated
`supplement_references.bib`. Figure builders require matplotlib; the bibliography
builder uses only the Python standard library.

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

## NMI exemplar writing pass

The current revision also uses the four supplied NMI papers as structural
examples. Results now follow method rationale, known SV types, regulatory reuse,
trained/random attribution and external limits. The complete H/random table is
main Table 2; the original reconstruction plot is Supplementary Fig. S1. There
are four main figures and two main tables. Numerical evidence and figure assets
are unchanged. See `../../docs/NMI_STRUCTURE_REVISION_20260928.md` for the
page-anchored review and the movement map.

All 118 original attached-LaTeX comment occurrences are restored in the included
comment-only `editorial_comment_archive.tex`. Existing inline comments remain;
new editorial rationale uses `% Long Note:`. Verify preservation with:

```bash
/opt/anaconda3/bin/python manuscript/revision_20260924/check_editorial_comments.py
```

`nmi_revision_provenance.json` records this pass separately from the prior
method-focused provenance. Both records describe writing revisions, not new model
results. Rebuild the same stable output PDF using the commands above.

## Comment completion and the open editor

The latest supplied 13 request groups have local `% Long Note:` dispositions:
12 Done, with R08 explicitly evidence-open for actual genotyping improvement.
All 38 original groups also have status responses in the included comment archive.
See `../../docs/LATEX_COMMENT_COMPLETION_20260928.md` for the current checklist.

The pdfLaTeX compression primitive is now engine-guarded. The native standalone
compiler proceeds past that error but cannot load companion project files. The
current source was checked with full-project draft-mode TeX, producing no new PDF.
The existing output PDF is the preceding NMI-writing snapshot; it is not a render
of the subsequent small prose/comment edits. The current editor remains open.
`comment_resolution_provenance.json` distinguishes these verification states.

## Downloadable source package

Run `python manuscript/revision_20260924/package_latex.py` from the repository
root. It creates `output/source/PangenomeFM_NMI_revised_LaTeX.zip`, with `main.tex`
at the ZIP root, all referenced vector figures, both bibliographies, supporting
TeX files, preserved comments, checklists and a per-file SHA-256 manifest. Upload
the ZIP to Overleaf and select pdfLaTeX/main.tex, or run `latexmk -pdf main.tex`
after extraction. The archive contains no compiled manuscript PDF.

`delivery_receipt_20260928.json` records the verified delivery. The latest pasted
full manuscript is byte-identical to the original reviewed attachment. Extraction
and a fresh draft-mode TeX/BibTeX build passed without creating a new PDF; the
comment checker also passed within the extracted package.
