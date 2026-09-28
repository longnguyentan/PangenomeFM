# Manuscript evidence audit — 28 September 2026

Scope: scientific updates to `manuscript/revision_20260924/`, preserving original
v1 numerical headlines and all authors, affiliations, funding and administrative
statements. No model training, live test-score inspection or result selection.

## Changes and source of truth

| Manuscript change | Completed evidence / protocol |
|---|---|
| Abstract and controls distinguish graph-derived information from learned-weight benefit; retain the masking-induced degree shortcut | `docs/MODEL_AND_DOWNSTREAM_STATUS_20260927.md`; `docs/MODEL_REPAIR_EXECUTION_20260927.md`; `results/foundation_evidence_20260927/hr_analysis/{audit.json,paired_gains.csv}` |
| Full H/random table: strict INS/DEL +0.005295 [0.002787, 0.010515]; one-hop inconclusive; strict cCRE favors random; no AP BH-confirmation | Same completed 120-run control report. Full-graph H has broader access than windowed T and is not a matched-context ablation |
| Correct EN-TEx scope to 450 runs; show RNA/ATAC/H3K4me3/H3K27me3 plus all P0 sensitivities; retain primary nulls and exposure/population caveats | `results/foundation_evidence_20260927/task_scorecard/completed_downstream_tasks.csv`, with original source paths retained per row |
| Natural-frequency INS/DEL/INV, all 174,267 primary-chromosome events; macro benefit after length/H, uncertain adjusted INV; retain matched population | `results/foundation_evidence_20260928/sv_type_natural_full_analysis/{audit.json,absolute.csv,contrasts.csv}`; `results/foundation_evidence_20260927/sv_type_full_analysis/contrasts.csv`; `configs/sv_type_{matched_20260927,natural_20260928}.json` |
| Replace stale HG008 stopped claim with complete prospective refits and wide null-crossing intervals; keep failed historical replay distinct | Scorecard and `results/foundation_evidence_20260927/hg008_prospective_refit/analysis/paired_gains.csv` |
| Add original TraitGym negative/null results, allele-score failure and separate classifier improvement without topology superiority | `traitgym_full_analysis`, `traitgym_allele_full_analysis`, `traitgym_probe_float64_full_analysis` under `results/foundation_evidence_20260927/`; corresponding downstream/probe study reports |
| Add COSIGT quality nulls, stronger training-median reference and validation-only fallback limits | `cosigt_quality_full_analysis`, `cosigt_fallback_full` under the same result root; `docs/COSIGT_GENOTYPING_QUALITY_20260927.md` |
| Separate original topology-pretrained T from sequence-conditioned E; show all three development seeds, random and coordinate controls; retain failed junction comparison | `results/foundation_evidence_20260928/masked_feature_full_analysis/{audit.json,paired_differences.csv}`; `masked_junction_full_reference/paired_per_seed.csv`; `configs/masked_nt_objective_20260928.json` |
| Explicitly pending full chromosome replication and same-budget v1 reference; no partial scores; separate development-exposed fold B | `configs/masked_nt_chromosome_replication_20260928.json`; `configs/masked_nt_v1_reference_20260928.json`; `docs/RESULT_REVIEW_20260928.md`; `docs/EXECUTION_20260928.md` |
| Explain five-fold minimum two-sided exact p=0.0625 and four-fold p=0.125; no bootstrap-to-BH equivalence | The fixed paired fold sign-flip protocol, retaining initialization seeds within genomic folds |

cCRE subclass rows are now explicitly subsets of binary-probe predictions. Known
SV classification is consistently separated from discovery, genotype calling and
allele-specific modeling. Original graph/cache denominators are separated:
751,237 graph segments, 303,425 downstream NT rows and 479,477 benchmark NT rows.
Whole-graph completion is input coverage, not evidence of a better model.

## Reproducible manuscript package

- `completed_results_20260928.tex`, `completed_methods_20260928.tex` and
  `completed_supplement_20260928.tex` keep the additions modular.
- `evidence/completed_20260928/provenance.json` records original source paths,
  original SHA-256 values, deterministic row-selection rules and bundled hashes.
  AP/primary-MAE families include unfavorable arms, not just favorable rows.
- `build_completed_evidence.py` validates source snapshot hashes, complete natural
  SV/development receipts, unique table-row selection and paired-mean identities,
  then writes six LaTeX row files. It formats archived intervals without refitting
  models or estimating new uncertainty. All five P1 tissue rows remain in the
  bundled scorecard, although the printed EN-TEx table shows their macro.
- Existing figures are present and preserved. Their captions identify the original
  v1 architecture and reconstruction limitations; no synthetic figure was added.
- New attribution checked against primary sources: [GraphMAE](https://arxiv.org/abs/2205.10803),
  [TraitGym author repository and citation](https://github.com/songlab-cal/TraitGym),
  [COSIGT primary article](https://link.springer.com/article/10.1186/s13059-026-04242-4).

## Verification

- Completed-source builder passed its hash, completion and arithmetic checks.
- Targeted Ruff passed for the new builder; scoped `git diff --check` passed.
- Existing local TeX tooling compiled the multi-file project with `latexmk -pdf`;
  no TeX software or package was installed. All citations and cross-references
  resolved. No overfull/underfull boxes or LaTeX warnings remain.
- Poppler rendered the PDF; the natural SV, complete EN-TEx, H/random, matched SV,
  transfer-boundary and development tables were visually inspected, along with
  page-layout overview images. Table text fits within the margins.
- Final local PDF: `manuscript/revision_20260924/output/pdf/PangenomeFM_working_revision_20260928.pdf`.
  Build intermediates/PDF/QA images are ignored; source and compact evidence are
  the version-controlled deliverable. This project needs bibliography and figure
  files, so the standalone native compiler is not applicable.

## Unresolved evidence

The chromosome matrix and v1 reference have no complete aggregate result in this
revision. Historical label exposure persists even after separately reporting fold
B. Development seeds are not independent chromosome replicates. The coordinate
ablation changes capacity, and the Q comparison changes epoch budget. The natural
SV extension has no random encoder; adjusted INV uncertainty remains. All studies
remain conditional on their callsets and feature definitions.

External E transfer, official graph-SSL architecture comparisons, stronger sequence
context, donor-excluded testing, allele/path correspondence, DUP/complex SV labels,
an all-callable genotyping denominator and measured effects of broader component
contexts remain outstanding. Input preparation and resource checks are not model
performance results. No foundation-model, final-architecture or journal-acceptance
claim was added.
