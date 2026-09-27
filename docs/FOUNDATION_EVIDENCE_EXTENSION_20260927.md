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
- [ ] Prepare/map/inspect all three additional assay datasets on the exact graph.
- [ ] Complete all three seven-feature smoke runs and verify feature coverage.
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
