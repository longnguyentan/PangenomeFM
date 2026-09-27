# PangenomeFM experiment and manuscript checklist

Last verified: 26 September 2026. Owner: this research task. A checked item means
its stated scope is complete, not that its scientific hypothesis was supported.
Server: existing Temple checkout; canonical graph and completed results are preserved.

## Completed EN-TEx programme

- [x] Inspect and validate sources; resolve legacy cCRE registry accessions.
- [x] P0: AS-prone cCRE preparation, mapping, feature QC, smoke and full matrix.
- [x] P0 sensitivities: exposure-matched, H3K27ac-only and CTCF-only analyses.
- [x] P0b: within-stratum comparisons, prevalence, normalized AP and AUROC.
- [x] P1: active versus explicitly repressed distal enhancers in selected tissues.
- [x] P2/P3: full accessible SNV source, CTCF and H3K27ac tasks. The supplied
  high-confidence file was RNA-only and was not substituted for these assays.
- [x] Complete 330 fold/seed/context jobs, paired summaries and figures.
- [x] Freeze completed result hashes and cached manuscript regression anchors.

Evidence: `docs/ENTEX_EXPERIMENTS.md`, `results/entex/v1/final_report/`,
`configs/completed_entex_regression_20260922.json`.
P0/P1 are inconclusive; P2 effects are modest and task/context dependent.

## EN-TEx lab meeting — Tuesday 29 September

- [x] Define exploratory follow-ups before computing their scores; preserve the
  original completed programme and explicitly identify this as post-primary work.
- [x] Implement equal-locus-weight evaluation and count-selected donor/tissue
  strata with fixed macro groups, exact prediction pairing and original-metric controls.
- [x] Complete both 30-run CTCF/H3K27ac follow-up summaries and all-group figures.
  The first diagnostic pass exposed the manuscript's probability-clipping convention;
  the corrected runs reproduced original unweighted metrics within 1e-12.
- [x] Prepare RNA ASE SNVs: 1,640,580 measurements, 466,867 loci, 23,546 positives.
- [x] Map all RNA loci to exactly one canonical segment; C/K/S/T coverage is 100%.
- [x] Finish RNA fold-A/42/strict smoke fit, seven feature sets; no full-matrix CI.
- [x] Finish the 30-run RNA matrix and complexity analysis. Both global ΔT
  intervals cross zero; the smoke result is not representative of all folds.
- [x] Create the separate team brief, `docs/ENTEX_LAB_MEETING_20260929.md`.
- [x] Add verified completed follow-up results and figures to the team brief.

## Structural interpretation and external transfer

- [x] Audit original SV target: **insertion versus deletion**, not SV detection.
  INS-only/DEL-only AP and AUROC are undefined; report class recall instead.
- [x] Prespecify complexity, size, allele-frequency and carrier-count strata,
  paired bootstrap, fold sign-flip tests and within-family BH adjustment.
- [x] Score all 30 archived SV runs across seven feature sets without refitting.
- [x] Finish secondary-metric summaries and vector/300-dpi structural figures.
- [x] Audit donor aliases: HG002 = NA24385; at least five overlapping donors.
- [x] Complete donor/haplotype-stratified chromosome-holdout analysis (30 runs).
- [x] Finish donor figures and absolute metrics; document that downstream
  training was pooled across donors, so this is not donor-held-out fitting.
- [ ] Establish strict unseen-donor/path transfer on a verified compatible
  donor-excluded graph. Hiding path labels alone is transductive.
- [x] HG008 source checksums, clonal filtering, exact graph mapping and smoke.
- [ ] HG008 full external summary: **24/30 passed; six stopped at the original
  HGSVC probe regression gate**. Diagnose before reporting a full result; do not
  relax the fixed 1e-4 AP tolerance after seeing external scores.

Evidence: `configs/structural_mechanism_v1.json`,
`results/foundation_campaign/20260924/{sv_strata,donor_sv}/`,
`docs/DOWNSTREAM_TRANSFER_V1.md`.
The SV complexity hypothesis was not supported: gain is positive in all three
groups but smaller in high than low complexity. Retain this result.

## Controlled pretraining-data scaling

- [x] Implement nested chromosome-balanced 12.5/25/50/100% window manifests;
  fixed validation/test universes, graph hashes and unchanged base architecture.
- [x] Test nesting, context pairing, leakage guards and deterministic selection.
- [x] Four one-epoch smoke jobs completed.
- [x] Four full-duration pilots completed: fold A, seed 42, strict context.
- [x] Evaluate four pilot checkpoints on SV, cCRE and CTCF (12 completed tasks).
- [x] Complete all 120 full-duration pretraining runs (4 fractions × 5 folds ×
  3 seeds × 2 contexts).
- [x] Complete frozen biological scaling evaluation: all 360 task evaluations.
  Exact baseline-score invariance passed; biological scaling is task/context dependent.
- [x] Summarize intrinsic scaling after auditing identical held-out targets:
  120 runs; monotonic mean reconstruction gain across four fractions.
- [x] Evaluate all three native SV complexity strata across fractions and produce paired figures.
- [x] Implement and test native SV-complexity scaling reports and paired fraction
  contrasts, reusing the original strata and without selecting bins using outcomes.
- [x] Correct the scaling aggregator: true balanced accuracy, exact Cartesian
  task/fraction/fold/seed/context completeness, saved prediction identity/label
  digests, metric replay, checkpoint provenance and baseline-invariance reports.
  Actual-prediction smoke checks passed for 21 feature/run rows across all three tasks.
- [x] Launch bounded dependent finalization for the running scaling workflow.
  It summarizes only after all 360 run audits are complete; upstream failures or
  a 24-hour timeout stop it. Status: `scaling_finalization_status.json` in the campaign root.
- [x] Audit parameter counts, training budgets, checkpoint selection and
  comparability to the newly trained 100% control (52,033 parameters; report
  variable epochs/window counts, not compute-matched scaling).

Server outputs: `results/foundation_campaign/20260924/scaling_full/`.
Single-fold pilots have no multi-fold CI and are not a completed scaling study.
Window scaling is not haplotype-diversity scaling or a scaling law.

## Path extension and optional functional benchmarks

- [x] Inspect existing GBZ extraction, donor splits and branch-choice modules.
- [x] Verify canonical SV graph has no paths; full-resolution path node IDs
  cannot be joined directly to canonical segment embeddings.
- [ ] Establish same-release node correspondence or a separately declared
  compatible path pilot, with valid observed-successor versus alternative-edge labels.
- [ ] Implement/run minimal path auxiliary objective using existing infrastructure.
- [ ] Run matched base/path, path-shuffle, capacity and context ablations.
- [ ] Evaluate frozen transfer; promote the extension only with verified benefit.
- [x] GTEx compact credible-set feasibility: matching leaves 92/34/42 loci in
  blood/liver/cortex, too small for the proposed benchmark. No fitting claimed.
- [ ] Optional graph-sensitive functional benchmark, only after structural priorities.
- [ ] Stronger local/long-context sequence baseline and frozen competing embeddings,
  only with a task-faithful common example universe and versioned resources.
- [ ] Donor-held-out methylation and paired haplotype ASE: existing runners need
  compatible path embeddings and matched donor-level labels/caches.
- [ ] Broader accessibility, histone, 3D-contact, molecular-QTL and variant-effect
  panel: conditional on compatible labels; no missing-data task is marked complete.
- [ ] Population-diversity scaling: requires ancestry/sample metadata and a
  donor-excluded graph protocol; ordinary window scaling does not satisfy it.

## Manuscript and reproducibility

- [x] Read new research notes, supplied LaTeX and PDF; preserve source hashes.
- [x] Retrieve prior completed cCRE subtype, degree-baseline and method-detail audits.
- [x] Resolve intrinsic reconstruction aggregation discrepancy: preserve headline
  window-weighted means and correct chromosome-block interval weighting.
- [x] Correct gate/loss/scorer, ordered SV pair representation, target naming,
  donor aliases, split counts and degree-control interpretation.
- [x] Integrate completed EN-TEx and structural findings with null results retained.
- [x] Add the completed intrinsic reconstruction scaling audit to the working
  supplement while withholding unfinished biological scaling/HG008/path claims.
- [x] Compile and visually verify revised PDF, figures, tables and citations.
- [ ] Final funding/author contributions/acknowledgements: awaiting author details.
- [x] Run scoped tests, cached regression, lint, compile and provenance checks.
- [x] Commit/push implementation increments and fast-forward the server checkout.
  The exact current commit is obtained with `git rev-parse HEAD`; do not confuse
  code synchronization with completion of the long-running experiments.

## Current access and next action

- [x] SSH reauthenticated; server status was successfully checked and the
  checkout was fast-forwarded. No passwords are stored in the project.
- [x] Complete label-free embedding and fixed-feature numerical diagnostics.
  Repeated GPU extraction differs by at most 2.7e-6 on the fixed subset; CPU/GPU
  differences are larger. Identical one-thread C+S+T fits repeat exactly, but four
  threads change AP by −0.000285 relative to one thread. The one-thread C+S refit
  differs from the historical score by +0.000438 AP. These identify numerical
  sensitivity without proving exact historical feature/fit reconstruction.
- [ ] Resolve historical HG008 replay before claiming a complete external result.
  The original fitted probes and historical embedding matrices were not retained;
  do not select a passing numerical setting or relax the 1e-4 gate. Compact
  diagnostics are under `results/downstream_v2/v1/qc/numerical_diagnostics/`.
  Detached biological jobs do not depend on the SSH window remaining connected.

Unrelated user edits are left intact. No passwords are stored in this checklist,
commands, repository or output artifacts.

## Independent v2 review — 27 September UTC / 26 September EDT

- [x] Preserve the other LLM's branch and uncommitted changes; work on
  `codex/v2-evidence-review-20260927` in an isolated worktree.
- [x] Verify prior results and read the professor-goal summary as supplied evidence.
- [x] Fix capped-eight branch-distance computation and missing-coordinate geometry.
- [x] Balance junction endpoint counts; preserve coordinate systems/orientations;
  enforce declared tolerances, stable canonical storage, and mandatory masking.
- [x] Test v1 compatibility and reviewed modules (64 scoped tests before cycle fix;
  cycle regression added separately). No biological encoder fine-tuning.
- [x] Run initial canonical HPRC audit: whole-span matching loses too many candidates;
  save the result and replace it with admissible balanced cycles, not looser labels.
- [ ] Complete and review cycle-sampler audit, retained counts and residual cues.
- [ ] Complete fold-A frozen H/R controls on SV and cCRE in both contexts.
- [ ] Run v2 training only after candidate and masking readiness gates pass.
- [ ] Fix a validation-only selection protocol; no fold used to choose settings
  may be described as an untouched test. Existing explored v1 folds are disclosed.
- [ ] Stronger sequence-model and allele-aware/genotypability benchmarks require
  versioned inputs and a shared evaluation universe; see the v2 review plan.

The review report and executable control command are in
`docs/V2_EVIDENCE_REVIEW_20260927.md`. Completion of this checklist does not imply
superiority or venue readiness; those depend on observed comparative evidence.
