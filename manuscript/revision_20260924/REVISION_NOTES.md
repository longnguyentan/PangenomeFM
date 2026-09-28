# Consolidated NMI revision — 28 September 2026

## Delivered source

`main.tex` is the single editable source for the article and supplement. All
numerical tables, bibliographies and professor comments are inline. Only ten
figure PDFs are external. Existing modular TeX files are retained as historical
repository material and are excluded from the delivery package.

The supplied `PangenomeFM_NMI_rewrite_20260928.zip` was reviewed against the open
manuscript and saved numerical evidence. This is an editorial/evidence review,
not a new experiment or a completed chromosome-replication result.

## Completed

- [x] Consolidate the article, 11 Supplementary Notes, Supplementary Methods,
  tables, separate reference lists and editorial archive into `main.tex`.
- [x] Keep the author's explicit restriction on findings: the abstract and the
  six-paragraph Introduction contain background, design and aims. The supplied
  rewrite's proposal to restore findings there was not adopted.
- [x] Preserve the six-paragraph Introduction, Macias-Velasco citation, integrated
  related work and Figure 1 citation. Methods opens with representation,
  architecture, masking, loss, optimization and frozen reuse.
- [x] Retain separate sequence-coverage Results, SV-type/stratum results, every
  EN-TEx assay and the negative or inconclusive external outcomes.
- [x] Keep trained/untrained controls prominent in main Table 2. Move the
  reconstruction-controls figure to the supplement to retain six main display
  items (four figures and two tables).
- [x] Verify the added control contrasts and junction-development measurements
  against the local CSV/JSON records listed in the provenance file.
- [x] Qualify development claims: near-chance performance of the tested
  incoming-only encoder is not proof that all such encoders cannot learn.
  Geometry/degree controls at chance do not mean all raw inputs are uninformative.
- [x] Correct the initialization claim in text and the development figure:
  topology-only historical checkpoints lack initial-state provenance;
  sequence-conditioned pairs have matching initialization records.
- [x] Remove “Reciprocal masking: later repair” from the native architecture
  diagram. Keep the relevant scientific masking limitation in Methods.
- [x] Add the missing regulatory panel **d**, using the original font and panel
  alignment. Numerical plot content is unchanged.
- [x] Check all 18 references added by the supplied rewrite against primary
  bibliographic records. Correct HyenaDNA author order, the DNABERT-2 title
  (“Genomes”), the Genome Biology article number for Tang et al. (203), and the
  PMLR page range for FakeEdge (56:1–56:19).
- [x] Preserve all 118 original professor comment occurrences, the 25-line
  follow-up, all earlier source comments and the supplied rewrite's comments.
  `% Long Note:` dispositions distinguish writing completed from evidence open.
- [x] Check the complete source in pdfLaTeX draft mode; no manuscript PDF was
  produced. Inspect all three corrected vector figures visually.

## Evidence checks and boundaries

The expanded control table uses `hr_analysis/paired_gains.csv`, checked against
`hr_analysis/summary.csv`. The retained contrasts include T and R after C+S+H,
their paired difference, confidence intervals, exact tests and multiplicity
adjustment. Negative cCRE trained-minus-untrained estimates remain visible.

The junction comparison uses 2,050 balanced candidates in 104 fold-A validation
regions. Verified AP values include incoming-only 0.500748 versus untrained
0.502783; bidirectional 0.789716 versus 0.625729; and sequence-conditioned
0.840748/0.869981 versus 0.679359/0.654747. The sequence-input boosting control is
0.734008. These are design-selection observations, not independent chromosome
replication. The junction biological development criterion passed in one of
three runs. The masked-feature encoder's three-run development table remains
separate; incomplete chromosome results are excluded.

Rounded table values can differ by 0.0001 when their displayed means are
subtracted. Paired deltas are calculated from full-precision source values,
not from already rounded table cells. No model or confidence interval was refit
for this revision.

The principal T remains the topology-pretrained model underlying the completed
biological matrices. The developmental, sequence-conditioned E is not promoted
to the principal model on partial evidence. Known-event SV classification is not
breakpoint detection. COSIGT quality prediction does not demonstrate improved
genotype calls. H accesses the full graph and is not a neighbourhood-matched
architectural ablation.

## Format and section decisions

The [NMI Article guide](https://www.nature.com/natmachintell/content), checked
28 September 2026, specifies a 150-word abstract, 3,500-word main text excluding
Methods/references/captions, and up to six main display items. The consolidated
source has approximately 3,360 main-text words and a 133-word abstract under
TeXcount, with six main display items. Counts and source hashes are recorded in
`consolidation_provenance.json`. Main references: 41; supplementary references: 11.

The guide normally uses an unheaded Introduction, whereas the author explicitly
requested clearly labelled sections. The Introduction heading is therefore
retained. The author also explicitly requested no findings in the abstract;
that request is retained despite the supplied NMI examples' different convention.
Neither formatting nor this editorial revision guarantees journal acceptance.

Main figure order: paradigms (1), architecture (2), structural variants (3),
regulatory elements (4). Supplementary figures are numbered automatically in
order of appearance; their historical asset filenames need not match the final
number. The supplement begins within the same document after the author
statements. It contains a labelled results/evidence body and separate Methods.

## Author or evidence confirmation still required

- [ ] Consortium authorship approval, order, membership and affiliations.
- [ ] NHGRI U24 and NIGMS R01 award numbers, recipients and acknowledgement wording.
- [ ] CRediT contributions and every author's competing-interest declaration.
- [ ] Final Shi laboratory repository, licence, weight hosting and archival DOI.
- [ ] Exact assembly/donor membership of the SV-resolution training graph; the
  separate full-resolution path inventory is not treated as that proof.
- [ ] Complete independent chromosome-level evaluation of the new objectives
  before any claim of a superior learned representation.
- [ ] Actual genotype-call improvement: it remains scientifically undemonstrated.

These items are not marked Done. Red author-input fields remain visible where
needed. An optional locus vignette is omitted because no prespecified selection
and verified tracks are supplied; it is not necessary to compile this manuscript.

## Primary records for the 18 added references

| Reference | Primary record |
|---|---|
| Enformer | [Nature Methods](https://www.nature.com/articles/s41592-021-01252-x) |
| Barabási–Albert | [Science DOI](https://doi.org/10.1126/science.286.5439.509) |
| GPN | [PNAS](https://www.pnas.org/doi/10.1073/pnas.2311219120) |
| FakeEdge | [PMLR](https://proceedings.mlr.press/v198/dong22a.html) |
| PanGenie | [Nature Genetics](https://www.nature.com/articles/s41588-022-01043-w) |
| Eizenga et al. | [Annual Review](https://www.annualreviews.org/content/journals/10.1146/annurev-genom-120219-080406) |
| Shortcut learning | [Nature Machine Intelligence](https://www.nature.com/articles/s42256-020-00257-z) |
| Probing controls | [ACL Anthology](https://aclanthology.org/D19-1275/) |
| Graph pretraining strategies | [OpenReview paper](https://openreview.net/pdf?id=HJlWWJSFDH) |
| Variational graph autoencoders | [Author preprint](https://arxiv.org/abs/1611.07308) |
| Minigraph | [Genome Biology](https://link.springer.com/article/10.1186/s13059-020-02168-z) |
| HeaRT | [NeurIPS proceedings](https://papers.neurips.cc/paper_files/paper/2023/hash/0be50b4590f1c5fdf4c8feddd63c4f67-Abstract-Datasets_and_Benchmarks.html) |
| Link prediction | [JASIST DOI](https://onlinelibrary.wiley.com/doi/10.1002/asi.20591) |
| HyenaDNA | [NeurIPS proceedings](https://papers.neurips.cc/paper_files/paper/2023/hash/86ab6927ee4ae9bde4247793c46797c7-Abstract-Conference.html) |
| Genome graphs | [Genome Research](https://genome.cshlp.org/content/27/5/665) |
| Giraffe | [Science DOI](https://doi.org/10.1126/science.abg8871) |
| Tang et al. | [Genome Biology](https://link.springer.com/article/10.1186/s13059-025-03674-8) |
| DNABERT-2 | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/b633e7052970b8f5aa1a69164d99e9e8-Abstract-Conference.html) |

## Compilation and delivery

The source package requires only `main.tex` and its `figures/` directory; run
pdfLaTeX three times, without BibTeX. The built-in single-file editor was checked
and stops at the first companion figure because it does not load additional
project files. Its current preview is therefore not a verified final rendering.
The existing editor was kept open. Full-project draft-mode checks resolve all
citations and references and report no box warnings. A fresh manuscript PDF was
not created as part of this request.
