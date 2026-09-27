# Foundation-model evidence: remaining work and fixed extension

Plan fixed 27 September 2026 UTC. This is an exploratory continuation specified
after the primary results and the one-fold H/R diagnostic. It does not turn
previously inspected chromosome folds into untouched validation.

## Current answer

The planned core EN-TEx programme is complete: 330 fits for AS-prone cCREs,
three sensitivities, five enhancer tissues and CTCF/H3K27ac SNVs. The RNA extension
adds 30 completed fits; 60 saved SNV runs were re-evaluated for equal-locus and
donor/tissue consistency. P0/P1 and RNA are inconclusive globally. Equal-locus
H3K27ac effects remain positive; CTCF becomes inconclusive. Detailed estimates
are in [the lab-meeting brief](ENTEX_LAB_MEETING_20260929.md).

These are valid task-specific frozen-transfer measurements. A strong general
foundation-model claim remains unestablished. The completed single-fold controls
show substantial utility from an untrained encoder, and the v1 reconstruction
protocol exposes a masking cue. The next decision is whether learned weights
add information beyond graph architecture, coordinates, sequence and simple
graph features. More positive benchmarks cannot substitute for this comparison.

## Analysis plan

### A. Full trained/random/handcrafted control matrix

- Experimental units: cCRE loci or assembly-derived INS/DEL events. The original
  SV endpoint is insertion versus deletion, not detection or genotypability.
- Use all original five chromosome folds, seeds 42/314159/20260806, and strict/one-hop
  contexts. Match each random initialization seed to its trained control.
- Both encoders and the exact NT revision remain frozen. Fit the original
  standardized logistic probes using original train/validation/test partitions.
- Use one audited processing-v2 H cache from the canonical graph. Its construction
  is label-free; preserve source graph and cache checksums in every run receipt.
- Four feature comparisons: C+S, C+S+H, C+S+T/R, C+S+H+T/R. Preserve the other
  original feature sets too. Primary contrasts are trained-minus-random and
  topology gain conditional on H, with paired fold/seed AP and AUROC summaries.
- Validate identical example/label identities and invariant C+S/C+S+H predictions
  between trained/random runs before inference. Report calibration and convergence
  warnings, all results, fold-level contrasts and hierarchical intervals.
- This is retrospective exploratory validation: all five folds were inspected
  for v1. A fresh external dataset is needed for untouched confirmation.

`configs/hr_controls_20260927.json` contains 15 fold/seed jobs, each running
eight probes (2 models × 2 tasks × 2 contexts): **120 evaluations**. The existing
roadmap executor runs two bounded worker stages on GPUs 0 and 1; no new scheduler
is introduced. The earlier one-fold outputs stay archived separately.

### B. Additional EN-TEx assay panel

Use the already downloaded accessible-SNV source, SHA256
`e59a83a1595cbe12ba56714af79c297ac9f31b593e13ee44966c732d30d4daa8`.
The source audit observes 3,265,155 ATAC-seq, 1,659,748 H3K4me3, and 1,291,316
H3K27me3 measurements. They were selected before fitting for complementary
biological coverage, not observed performance. Other assays remain untested.

- The endpoint is the supplied significant allelic-imbalance call among measured
  heterozygous loci in each assay. These are not presence/absence or differential
  activity labels. Assay/donor/tissue provenance and read counts are retained.
- Locus repeats stay in their chromosome partition; the same segment C/K/S/T
  representations and seven feature combinations are reused. No encoder receives
  biological-label gradients. The graph and NT resources are pinned.
- At least 100 positives and 100 negatives are required in each original
  train/validation/test partition before fitting. Existing mapping/feature
  coverage gates apply; every exclusion and its class counts are reported.
- A fold-A/42/strict smoke checks executability and coverage only. It cannot be
  used to choose which assay advances: run all eligible assays and retain nulls.
- Each assay uses 5 folds × 3 seeds × 2 contexts: **90 additional runs** total.
  Summaries use existing paired hierarchical bootstrap, raw and normalized AP,
  AUROC, class prevalence and fixed graph-complexity strata. Fold sign-flip and
  BH adjustment over the six assay/context contrasts are exploratory sensitivities.
- Reuse equal-locus weighting and all count-eligible donor/tissue summaries from
  the original follow-up protocol. These evaluate repeated-measurement sensitivity,
  not donor-held-out fitting. Their rules were fixed before inspecting new smoke
  scores; thresholds match the earlier follow-up analysis.
- These tests predict per-measurement imbalance from static locus features;
  they do not identify the causal allele or predict an individual's haplotype.

The fixed plan is `configs/entex_extension_20260927.json`. Parsing is streamed
once into assay-specific Parquet caches. No source download is needed.

## Checklist and decision gates

- [x] Core EN-TEx + RNA + measurement-consistency results completed.
- [x] Full biological window-scaling analysis completed: 360 evaluations;
  task/context-dependent gains, not universal scaling.
- [x] One-fold H/R control matrix completed and archived, without multi-fold CIs.
- [x] Fix and test H branch-distance and balanced junction-candidate processing.
- [x] Implement the three-assay extension using existing preparation/probe code.
- [x] Fix assay panel, chromosome partitions and class-support gates before scores.
- [x] Prepare/map/inspect all three additional assay datasets on the exact graph.
  All source partition-support gates passed and every locus maps to one segment.
- [x] Complete all three seven-feature smoke runs and verify feature coverage.
  All have 100% joint C/K/S/T and measurement coverage; frozen encoders confirmed.
- [ ] Complete and summarize the 90-run additional EN-TEx matrix.
- [ ] Complete and audit the 120-evaluation H/R matrix.
- [ ] Establish full-genome v2 candidate coverage and geometry/visible-degree
  baselines under actual masking; the bounded audit still has a distance cue.
- [ ] Lock v2 model/objective choices using pretraining validation and run
  controlled frozen biological comparisons; no current v2 superiority claim.
- [ ] Match stronger/local sequence baselines on a common example universe.
- [ ] Resolve HG008 historical-probe replay (six of 30 fail the unchanged gate).
- [ ] Establish donor-excluded/path-compatible resources for strict population,
  haplotype ASE and methylation transfer. Hiding path labels alone is insufficient.
- [ ] Obtain independent variant-genotyping accuracy labels before genotypability
  experiments. The available inversion VCF alone is insufficient.
- [ ] Complete author-supplied administrative manuscript fields.

## Reproduction

From the review server worktree with the existing `pangenomefm-server` environment:

```bash
export PYTHONPATH=src:.
bash scripts/server/run_entex_extension_20260927.sh prepare
bash scripts/server/run_entex_extension_20260927.sh smoke
# After inspecting every smoke's audit and coverage:
bash scripts/server/run_entex_extension_20260927.sh matrix

# Execute these in separate worker sessions (GPUs 0 and 1):
CUDA_VISIBLE_DEVICES=0 python scripts/server/run_foundation_model_roadmap.py \
  --config configs/hr_controls_20260927.json --stage worker_0 \
  --result-root results/foundation_evidence_20260927/worker_0 \
  --execute --allow-gpu --fail-on-blocked
CUDA_VISIBLE_DEVICES=1 python scripts/server/run_foundation_model_roadmap.py \
  --config configs/hr_controls_20260927.json --stage worker_1 \
  --result-root results/foundation_evidence_20260927/worker_1 \
  --execute --allow-gpu --fail-on-blocked
```

Full source/feature matrices and individual predictions remain on the server.
Compact QC, summaries, figures and run provenance are versioned. The maintained
[project checklist](FOUNDATION_CAMPAIGN_CHECKLIST.md) and team brief must distinguish
completed results from running jobs and resource-dependent proposals.

## Observed preparation and launch receipt

The server execution worktree is `/home/tuv43532/PangenomeFM_evidence_20260927`
at `d2c0497`. It reuses exact source paths from the main checkout without modifying
the main or other LLM's branch. Two control workers use GPUs 0/1; smoke fits use
the CPU, and the planned full EN-TEx matrix reserves GPUs 2/3. About 647 GB of
disk space and 243 GiB of memory were available before launch.

| Assay | Measurements | Unique loci | Positive measurements | Prevalence | Mapping |
|---|---:|---:|---:|---:|---:|
| ATAC | 3,265,155 | 1,498,771 | 133,227 | 4.0803% | 100% |
| H3K4me3 | 1,659,748 | 265,099 | 74,771 | 4.5050% | 100% |
| H3K27me3 | 1,291,316 | 702,509 | 25,667 | 1.9877% | 100% |

Mapping success is data QC, not model performance. All three have zero unmapped
or multi-segment loci. Full-matrix results remain pending until their completed
audits, feature coverage and paired predictions have been checked.

The earlier eight-run fold-A/42 H/R matrix has now passed actual prediction
replay, matched example/label hashes and exact C+S/C+S+H score invariance. The
new auditor checks completeness before writing any inference. Reproduce it with:

```bash
python -m tasks.transfer.hr_control_report \
  --campaign-root /home/tuv43532/PangenomeFM_evidence_20260927/results/foundation_evidence_20260927/hr_controls \
  --out-dir results/foundation_evidence_20260927/hr_analysis
```

## Single-fold smoke checks (27 September, 07:12 UTC)

All three completed with seven feature sets, unchanged input hashes, 100% feature
and measurement coverage, and `encoder_training=false`. On fold A / seed 42 /
strict, Δ AUPRC was −0.000231 (ATAC), −0.000451 (H3K4me3), and −0.002823
(H3K27me3). These are execution checks with no multi-fold intervals. Every
predefined assay proceeds to the full matrix regardless of its smoke score.
The compact metrics and audit receipts are in
`results/entex/extension_20260927/smoke_qc/`.

## Full matrix launch and native v2 readiness plan

At 07:15 UTC on 27 September, the full predefined EN-TEx panel was launched
from the isolated reporting checkout at `cb21325`, using the data and output
roots in the execution checkout. GPUs 2/3 are assigned to it; the H/R workers
keep GPUs 0/1. As of the next progress check, 12/15 H/R fold/seed jobs had
completed (96/120 biological probe runs), with two final-fold jobs running.
The H/R report waits for all 15 successful receipts before replaying predictions.
No partial-matrix inference is presented.

### Native v2 audit: analysis plan fixed before its scores

**Decision:** whether the corrected junction objective has sufficient coverage
for a full v2 run, and whether coordinate geometry or visible degree alone can
solve its validation task. This concerns self-supervised connection labels;
no EN-TEx or other biological labels are used.

**Observed starting point:** the 72-window v2 smoke executes, but the earlier
20-window diagnostic used simplified masking and retained a distance cue.
A full model campaign remains premature.

**Implementation:** `tasks.transfer.junction_readiness` loads the same global
HPRC R2 index and all strict/one-hop manifest windows using the native training
loader and the smoke checkpoint's saved objective settings. The new optional
loader receipt reports exclusions without changing returned training arrays.
The auditor verifies balanced labels, endpoint multiplicities and unsplit
groups after node filtering. It requires every canonical chromosome in both
contexts to retain a window, and to have at least one native validation/test
window with four candidates (the native evaluator's minimum). Counts, failed
windows, graph/checkpoint/manifest and per-window source hashes are retained.

**Baselines:** fixed standardized logistic regression (C=1, balanced classes)
using geometry only, visible degree only, and their union. Geometry includes
the native signed offset/gap, orientation and same-coordinate-system flags;
cross-system offsets remain gated. Degree comes from the actual masked
message graph. Training uses complete packed groups, all internal held-out
positives masked, and one reproducible DropEdge draw per slice with the saved
rate. Validation uses native evaluation masks and no DropEdge. This reuses
the native protocol; it is not a replay of the historical training RNG trace.

**Partitions and interpretation:** fit only internal training candidates on
training chromosomes; report pooled AP/AUROC and native-style window-macro
AUROC on validation chromosomes/internal validation candidates. No score
direction or regularization search, and no held-out chromosome predictions.
Above-chance geometry identifies a remaining nuisance route; chance-level
linear controls alone do not establish the absence of more complex shortcuts.
These are validation diagnostics, with no biological superiority claim.

```bash
PYTHONPATH=src:. python -m tasks.transfer.junction_readiness \
  --checkpoint /home/tuv43532/PangenomeFM_junction_audit_20260927/results/v2_review_20260927/smoke_72/training/run_001/ckpt_strict__shared_dual_mscale3_orient_maskedq_jrbranching16_visdeg_splitseed20260806_heldout_chr1_chr6_chr11_chr16_chr21_val_chr2_chr7_chr12_chr17_chr22_ep1_pat1.pt \
  --manifest /home/tuv43532/PangenomeFM/server_workspace/data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv \
  --full-segments /home/tuv43532/PangenomeFM/server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz \
  --training-drop-edge-rate 0.1 \
  --out-dir results/foundation_evidence_20260927/junction_readiness_corrected
```

### Native audit failure and correction (27 September)

The first full-manifest audit stopped on
`slice_GRCh38_0_chr1_0_5000000_1hop`. Candidate generation produced 5,582
positive and 5,582 negative candidates, but row-wise node filtering removed
3,082 rows and broke endpoint balance in at least one retained group. The
canonical reverse-complement handle is not always represented in the
stored-direction encoder node set. Balanced class counts alone do not detect
this problem. The failed receipt and offending window are preserved in
`results/foundation_evidence_20260927/junction_readiness_failed/`.

**Correction:** for the junction objective only, the native loader now excludes
an entire masking group if any candidate endpoint is unavailable. It counts
missing-node rows, excluded groups, and valid partner rows excluded to preserve
balance. It does not add nodes/edges or change the graph release. Tests include
mixed forward/reverse link storage and check endpoint balance after native
filtering. The v1 row filtering path is unchanged. Historical v2 smoke results
remain archived as execution evidence; they are not validation of this corrected
loader and must not be used to promote a full v2 model.

The repeat uses a fresh output directory `junction_readiness_corrected`. Its
explicit `--training-drop-edge-rate 0.1` matches the proposed full v2 training
launcher; the old one-epoch smoke had DropEdge disabled. This override is fixed
before new baseline scores and recorded in the receipt. Geometry collection
also enables the native four-column pair geometry for the nuisance control;
checkpoint weights are not loaded into a model or evaluated.
