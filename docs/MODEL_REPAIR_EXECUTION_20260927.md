# Model repair and execution record — 27 September 2026

Scope: code, experiment validity, and measured performance; no manuscript changes.
Work is isolated on `codex/v2-evidence-review-20260927`. Historical resources,
results, and the other implementation worktree are preserved.

## Verified completion and remaining gates

- [x] Full trained/random/handcrafted control matrix: 120 evaluations; identical
  prediction identities and invariant baselines verified across 5 folds × 3 seeds.
- [x] Additional EN-TEx ATAC/H3K4me3/H3K27me3 panel: 90 runs plus donor,
  equal-locus, complexity, and multiple-comparison analyses.
- [x] Native junction audit: 1,216 canonical windows inspected; 1,137 retained;
  endpoint balance and complete chromosome coverage pass after whole-group filtering.
- [x] Diagnose residual coordinate shortcut on training candidates, before new training.
- [x] Implement optional typed bidirectional messages; hidden query reciprocals
  and reverse-complement copies are removed before message expansion.
- [x] Implement a low-capacity linear pair scorer on embedding products and
  absolute differences (linear endpoint concatenation cannot resolve balanced pairs).
- [x] Implement validation-only development mode, with no final test predictions.
- [x] Fix numeric GFA-name CSV round trips without changing segment order.
- [x] Implement optional signed-offset/contiguity-gap nuisance matching.
- [ ] Verify full canonical coverage and nuisance baselines under the new matching.
- [ ] Run bounded validation-only model comparison after candidate validity passes.
- [ ] Evaluate a selected model with frozen biological probes and matched random twins.
- [ ] Resolve historical HG008 replay failures or run a separately identified,
  prospective deterministic probe-refit protocol; never relabel a refit as replay.
- [ ] Measured genotypability labels and verified path-to-segment correspondence
  remain data requirements; FILTER flags and mismatched paths are not substitutes.

## New observed evidence

The completed control report is in
`results/foundation_evidence_20260927/hr_analysis/`.
For strict SV, trained minus random **after C+S+H** is +0.005294728 AUPRC
(bootstrap 95% CI +0.002787064 to +0.010515405). One-hop SV is +0.000418742
(CI crosses zero). cCRE strict favours random by 0.000759214; one-hop is
inconclusive. These are exploratory comparisons on previously inspected folds.
With only five chromosome clusters, exact two-sided sign-flip tests have a
minimum attainable p-value of 0.0625: bootstrap intervals must not be presented
as equivalent to a five-cluster exact-test rejection.

The degree-balanced junction task still has a substantial geometry shortcut:
fixed geometry-only logistic regression reaches validation AP 0.968555225
(strict) and 0.894021519 (one-hop). These are reconstruction diagnostics,
not new biological results. In **training** candidates, strict positive gaps
are zero in 71.0621% of cases, versus 0% of negatives; 41.7956% of negatives
have negative signed offsets, versus 0% of positives. Merely removing explicit
pair geometry from the scorer would leave this information in node inputs.
Source: `results/foundation_evidence_20260927/junction_readiness_corrected/`.

## Fixed candidate repair, specified before its audit results

`--junction_geometry_match signed_gap_bins` requires each same-coordinate-system
negative to match its positive's signed offset **and** signed contiguity gap in
fixed multiplicative bins with ratio 1.25. Zero is its own category; positive
and negative values cannot share a bin. Existing distance tolerances remain
additional constraints. Cross-system offsets are not subtracted; their existing
coordinate-system/orientation signatures and balanced marginals are preserved.
Only complete re-pairing cycles are emitted. No fallback relaxes matching when
reference-chain geometry makes the task impossible. Default `distance` mode is
unchanged for historical reproduction.

This is a nuisance-control experiment, not a claim of being shortcut-free.
Continuous geometry and degree must still be audited on validation chromosomes.
All feature tables and baseline scores are saved even if chromosome coverage
fails, so retention failure can be diagnosed without hiding exclusions.

## Development gate and bounded comparison (specified before new results)

For each context, require existing full chromosome coverage and at least one
native window with >=4 validation/test candidates on every canonical chromosome.
Additionally require >=200 validation candidates, and geometry-only and
geometry-plus-degree pooled validation AP/AUROC <=0.60. These are operational
screening gates, not tests of biological superiority. A failing context is not
promoted by relaxing thresholds or looking at biological held-out scores.

For eligible contexts, compare incoming/bidirectional messages crossed with the
existing scorer/linear interaction scorer. All use the repaired candidates,
visible-graph structure, identical fold-A train/validation chromosomes, seed 42,
48-dimensional embeddings, 2 layers, the same graph/checksum, and a fixed
10-epoch maximum with validation early stopping. No pair-geometry input is
added to these scorers; the nuisance controls still inspect geometry. No
biological labels enter pretraining. Record checkpoint and candidate hashes.
This small comparison is exploratory development and cannot support a final
best-model claim. Frozen downstream probes, matched random encoders, and the
complete chromosome matrix remain required after model selection.

## Reproduction

Run the existing native readiness module with the same canonical manifest,
segment file and smoke checkpoint used in `FOUNDATION_EVIDENCE_EXTENSION_20260927.md`,
adding `--junction-geometry-match signed_gap_bins
--junction-geometry-bin-ratio 1.25` and a fresh output directory. Never overwrite
`junction_readiness_corrected`, whose failure mechanism is part of the record.

Tests include legacy initialization identity, masked reverse-message absence,
successor gradients, checkpoint/random-twin compatibility, native CLI training,
validation-only scoring, geometry-bin matching, endpoint balance, and refusal
of impossible coordinate-matched negatives. Synthetic runs are execution tests,
not model performance evidence.

## Matching audit outcome (16:22 EDT)

The fixed matching completed all 1,216 windows. One-hop retains 603 windows
with chromosome coverage and >=4 native validation/test candidates per canonical
chromosome; fold-A controls have 66,170 training and 2,050 validation candidates.
Geometry AP/AUROC are exactly 0.5; geometry+degree AP is 0.506812 and AUROC
0.504322. It passes the declared operational gate. A second native audit with
bidirectional masking will check the corresponding changed visible graph before
launching that arm.

Strict retains only 12 windows across all chromosomes and only 138 training /
4 validation candidates in the fold-A control partitions. It fails the unchanged
coverage gate. The signed-geometry constraint has **not** been weakened. These
reference-only windows cannot currently support the proposed strict reconstruction
task. Only the eligible one-hop context proceeds to development; strict frozen
extraction can be studied subsequently with clear training-context attribution.
This does not replace any historical strict result.

## Separate prospective HG008 refit

Historical exact-probe replay remains unresolved because original fitted probes
were not saved and numerical reproduction fails the 1e-4 AP gate in six runs.
The new explicit `prospective_refit` protocol creates **new HGSVC-trained probes**,
with the same examples, chromosome folds, frozen encoders, features and fixed
logistic hyperparameters. Native BLAS is limited to one thread. Training and
calibration use HGSVC training/validation chromosomes only. Every fitted model is
saved, reloaded and checked for bit-exact predictions before external scoring;
its feature-matrix hash, input/checkpoint hashes and environment are recorded.
All 30 combinations run regardless of their historical replay differences.
Those differences stay in the outputs. The unchanged default historical protocol
still refuses scores outside 1e-4. Reports prohibit mixing the two protocols.

HG008 supplies no training, calibration, model-selection or stopping labels.
This remains one-genome, zero-shot **insertion-versus-deletion classification**;
it is not a cancer breakpoint-localization benchmark or replay of lost models.
Run in a fresh directory:

```bash
PYTHONPATH=src:. python scripts/server/run_hg008_refit_campaign.py \
  --out-root results/foundation_evidence_20260927/hg008_prospective_refit --execute
```

Bidirectional masking produced the same one-hop nuisance-control scores and
passed the same context-specific coverage gates. The combined audit receipts
remain `failed` because strict is ineligible; the pilot launcher checks the
predeclared one-hop gates explicitly and never changes those receipts.

Reproduce the fixed four-arm development run:

```bash
PYTHONPATH=src:. python scripts/server/run_junction_geometry_pilot.py \
  --incoming-audit results/foundation_evidence_20260927/junction_geometry_matched \
  --bidirectional-audit results/foundation_evidence_20260927/junction_geometry_bidirectional \
  --context 1hop --out-root results/foundation_evidence_20260927/geometry_pilot \
  --gpus 0 1 2 3 --execute
```

The launcher refuses overwritten outputs, mismatched graph/manifest hashes,
inadequate coverage, wrong masking configuration, or nuisance scores above the
fixed gate. Without `--execute` it writes the exact commands and gate receipts.
Training writes per-epoch loss and validation AUROC, atomic recovery checkpoints,
and final validation-only predictions. No test predictions are requested.
