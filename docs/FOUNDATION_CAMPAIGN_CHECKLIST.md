# PangenomeFM experiment and manuscript checklist

Last verified: 24 September 2026. Owner: this research task. A checked item means
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

## Structural interpretation and external transfer

- [x] Audit original SV target: **insertion versus deletion**, not SV detection.
  INS-only/DEL-only AP and AUROC are undefined; report class recall instead.
- [x] Prespecify complexity, size, allele-frequency and carrier-count strata,
  paired bootstrap, fold sign-flip tests and within-family BH adjustment.
- [x] Score all 30 archived SV runs across seven feature sets without refitting.
- [ ] Finish secondary-metric summaries and vector/300-dpi structural figures.
- [x] Audit donor aliases: HG002 = NA24385; at least five overlapping donors.
- [x] Complete donor/haplotype-stratified chromosome-holdout analysis (30 runs).
- [ ] Finish donor figures and absolute metrics; document that downstream
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
- [ ] Evaluate those frozen checkpoints on SV, high-complexity SV, cCRE and CTCF.
- [ ] Run the full replicate matrix and produce paired scaling figures.
- [ ] Audit parameter counts, training budgets, checkpoint selection and
  comparability to the newly trained 100% control.

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

## Manuscript and reproducibility

- [x] Read new research notes, supplied LaTeX and PDF; preserve source hashes.
- [x] Retrieve prior completed cCRE subtype, degree-baseline and method-detail audits.
- [ ] Resolve intrinsic reconstruction aggregation discrepancy before editing anchors.
- [ ] Correct gate/loss/scorer, ordered SV pair representation, target naming,
  donor aliases, split counts and degree-control interpretation.
- [ ] Integrate completed EN-TEx and structural findings with null results retained.
- [ ] Add scaling/HG008/path findings only after their completion gates pass.
- [ ] Compile and visually verify revised PDF, figures, tables and citations.
- [ ] Final funding/author contributions/acknowledgements: awaiting author details.
- [ ] Run scoped tests, cached regression, lint and provenance checks.
- [ ] Commit/push this increment and sync the server checkout through GitHub.

## Current access and next action

- [x] SSH reauthenticated; server status successfully checked this session.
- [ ] Diagnose six HG008 reconstruction failures while producing structural
  summaries and preparing frozen scaling evaluation. Detached jobs do not
  depend on the SSH window remaining connected.

Unrelated user edits are left intact. No passwords are stored in this checklist,
commands, repository or output artifacts.
