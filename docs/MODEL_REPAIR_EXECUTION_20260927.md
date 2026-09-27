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
- [x] Audit coverage and nuisance baselines under new matching: one-hop passes the stated gate; strict fails and remains excluded.
- [x] Complete the bounded validation-only comparison and matched frozen-random controls.
- [ ] Evaluate a selected model with frozen biological probes and matched random twins.
- [x] Resolve historical HG008 replay failures or run a separately identified,
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

## Additional raw-input diagnostic (fixed before its results)

Chance performance for geometry/degree alone does not exclude nonlinear node
attribute cues. The optional `--include-node-controls` audit uses the model's
actual visible seven input features at both endpoints, plus fixed products and
absolute differences, without a graph encoder. It fits a fixed standardized
logistic probe and a fixed histogram-gradient-boosting probe (100 iterations,
7 leaves, learning rate 0.05, L2=1, no random internal validation/early stopping).
These fit training chromosomes only; validation labels are only scored. This
additional diagnostic neither changes the candidate matcher nor tunes pilot
hyperparameters. Failure means further objective investigation is needed before
claiming that the reconstruction requires learned graph context.

## Completed model-development comparison

All four runs completed on identical 2,050 validation candidates across 104
windows, with positive prevalence 0.5 and no held-out chromosome predictions.
Best checkpoint selection uses the native window-macro validation AUROC.

| Message direction | Pair head | Validation AP | Pooled AUROC | Window-macro AUROC |
|---|---|---:|---:|---:|
| incoming | existing MLP | 0.500748 | 0.501066 | 0.508325 |
| incoming | linear interactions | 0.530415 | 0.546088 | 0.576621 |
| bidirectional | existing MLP | 0.789716 | 0.786716 | 0.836070 |
| bidirectional | linear interactions | 0.740027 | 0.745120 | 0.794843 |

The fixed raw-input linear and boosting controls reach AP 0.570080 and 0.677132
respectively (macro AUROC 0.625505 and 0.752984). Thus geometry/degree balancing
alone does not make the task entirely attribute-independent. Bidirectional
message passing materially improves this bounded reconstruction pilot, but its
biological reuse remains untested; the subsequent frozen-random comparison is below.
These are development scores, not independent performance estimates.

## Frozen-random-backbone controls (fixed before their results)

Repeat the same four configurations and ten-epoch/early-stop budget, fitting
only the pair head with `--freeze_encoder`. Random backbones use the same initial
seed and architecture; encoder dropout is disabled to keep the representation
fixed. Candidate masking and DropEdge are retained. Hashes of every learned
backbone parameter must match exactly before/after fitting. This tests decoder
capacity and the architecture's random-feature contribution, while keeping
biological and held-out chromosome labels out of selection.

```bash
PYTHONPATH=src:. python scripts/server/run_junction_geometry_pilot.py \
  --incoming-audit results/foundation_evidence_20260927/junction_geometry_matched \
  --bidirectional-audit results/foundation_evidence_20260927/junction_geometry_bidirectional \
  --context 1hop --out-root results/foundation_evidence_20260927/geometry_random_controls \
  --gpus 0 1 2 3 --random-encoder-control --execute
```

## Completed frozen-random reconstruction controls

| Message direction | Pair head | Frozen-random AP | Trained AP | Trained minus random |
|---|---|---:|---:|---:|
| incoming | existing MLP | 0.502783 | 0.500748 | -0.002036 |
| incoming | linear interactions | 0.512805 | 0.530415 | +0.017610 |
| bidirectional | existing MLP | 0.625729 | 0.789716 | +0.163986 |
| bidirectional | linear interactions | 0.544846 | 0.740027 | +0.195181 |

All comparisons use the same fold-A reconstruction validation candidates. The
bidirectional existing-head model is selected by the previously declared native
validation macro-AUROC criterion (0.836070). This selection uses no biological
labels. Next, frozen cCRE/SV probes compare this checkpoint and its matched
random backbone with v1 and v1-random on **biological validation chromosomes**,
using C+S, C+S+T, C+S+H and C+S+H+T. No full biological test matrix is promoted
from reconstruction scores alone. The random control's T column means R.

The existing biological probe implementations now have an optional
`--validation-only` development mode. Training/chromosome exclusions and logistic
hyperparameters stay unchanged; no held-out features are scored. Predictions
are named `validation_predictions.csv.gz`, `n_test=0`, and scope is explicit.
Threshold/calibration also use validation, so validation F1/calibration are
optimistic development diagnostics; these are not independent test estimates.
The default historical test-scoring path remains the same. A unit test places
NaNs only in held-out features and permutes held-out labels: validation output
must remain unchanged and no held-out prediction may be attempted.

The cached manuscript regression still passes: cCRE gains +0.004008/+0.003406;
SV gains +0.034246/+0.040090. These are cached-result checks, not retraining.
The first regression invocation used the isolated worktree's absent import path;
rerunning with the original checkout's existing import directory passed.

A full local test run exposed a native OpenMP crash in sklearn histogram
binning after PyTorch tests. The small optional boosting diagnostic now scopes
OpenMP to one thread for fitting and prediction; its focused tests pass without
global environment overrides. This is a runtime correction, not a change to
model hyperparameters or selection criteria.

After the threading correction, the full local suite passes: **325 tests**.
Server access remains available. All 30 prospective HG008 refits have completed;
strict AP gain is +0.032525 (95% CI -0.053371 to +0.110254), and one-hop is
+0.064863 (-0.066739 to +0.212115). This 69-variant, one-genome result is
inconclusive. It does not repair or replace the historical replay receipts.

## Audited results and commands

The reconstruction table is recomputed from predictions, checks exact candidate
identities across all eight runs, verifies the unchanged random-backbone hashes,
and replays each selected checkpoint's native macro-AUROC. It selects
`bidirectional_default` without using biological labels.

```bash
PYTHONPATH=src:. python -m tasks.transfer.junction_pilot_report \
  --trained-root results/foundation_evidence_20260927/geometry_pilot \
  --random-root results/foundation_evidence_20260927/geometry_random_controls \
  --out-dir results/foundation_evidence_20260927/junction_pilot_analysis
```

Tables: `junction_pilot_analysis/validation_metrics.csv` and
`junction_pilot_analysis/paired_differences.csv`; figure:
`junction_pilot_analysis/junction_validation.{pdf,svg,png}` (all under
`results/foundation_evidence_20260927/`). No post-selection confidence intervals
are inferred from this single development fold/seed.

The 30-run HG008 prospective-refit outputs are now downloaded locally under
`results/foundation_evidence_20260927/hg008_prospective_refit/`. Saved logistic
models remain on the server at the same relative path in
`/home/tuv43532/PangenomeFM_refit_20260927`; local compact artifacts omit joblib
files. Each model was saved and reloaded before external scoring, with exact
prediction replay. `analysis/audit.json` verifies complete folds/seeds/contexts,
69 external variants, and 100% external feature coverage. Historical replay
differences remain in `analysis/original_probe_regression.csv`.

| HG008, new HGSVC-only refit | Mean C+S AP | Mean C+S+T AP | Paired gain, 95% CI |
|---|---:|---:|---:|
| Strict | 0.426009 | 0.458534 | +0.032525 [-0.053371, +0.110254] |
| One-hop | 0.426009 | 0.490872 | +0.064863 [-0.066739, +0.212115] |

This is insertion/deletion classification on one cancer genome. Neither gain
establishes general cancer prediction, and external labels will not tune a new
probe. Figure: `analysis/hg008_transfer.{png,svg}`; complete metric tables:
`analysis/per_run.csv`, `analysis/summary.csv`, `analysis/paired_gains.csv`.

### Biological validation running (17:03 EDT)

The server passed 27 focused tests before launching the eight frozen probe jobs
in tmux session `bioval` on socket `evidence-20260927`, using commit `16d7e9a`.
The authoritative command list is
`results/foundation_evidence_20260927/biological_validation/status.json` in
`/home/tuv43532/PangenomeFM_model_repair_20260927`.
It compares v1, v1-random, selected v2, and its matched frozen-random backbone,
for both cCRE and insertion/deletion, on fold A/seed 42/one-hop. The exact
manuscript graph, NT cache and provenance-checked H cache are reused.

Once all eight finish, run:

```bash
PYTHONPATH=src:. python -m tasks.transfer.development_report \
  --root results/foundation_evidence_20260927/biological_validation \
  --out-dir results/foundation_evidence_20260927/biological_validation_analysis
```

This report rejects changed locus/label identities, changed non-embedding
baseline predictions, incomplete matrices, or any test-partition prediction.
The historical H/R report shares the prediction audit; its test-scoring default
is preserved. Tests cover these failure cases and replay development metrics.
The full suite passes 327 tests; one additional metric-replay test also passes.

### Data limitations still requiring real inputs

The genotypability starting-point assessment is saved as
`results/foundation_evidence_20260927/genotypability_starting_point.json`.
The examined HPRC release summary reports self-genotyping, while the HGSVC3
workflow describes leave-one-out concordance without a located public table of
per-variant outcomes. Self-genotyping/FILTER flags are not the requested measured
leave-one-out labels. No substitute performance result is claimed.

## Sequence-cache coverage correction and repair

A fresh full-file audit found **751,237 canonical graph segments**, versus
303,425 contiguous IDs in the manuscript NT cache. The previously discussed
305,070 figure describes benchmark coverage, not the entire processed graph.
The attempted full-graph completion was deliberately interrupted after this
audit exposed 447,812 missing entries. Its logs/receipt are retained under
`whole_graph_nt/`; no completed whole-graph cache is claimed.

The completion tool now supports the exact union of unmasked link endpoints in
the canonical benchmark manifest, using the existing global-index/link mapper.
It includes all chromosomes and alternative nodes, retains all existing cache
entries, and reads no labels. A 10,000-new-segment guard refuses unexpectedly
broad inference before loading the sequence model. The native frozen NT
preparation script, pinned revision, pooling, base sampling and token limits are
reused; cached weights are required, and the original cache is never overwritten.

Cache merging now rejects different graph/preprocessing contracts, and can
recover those contracts from the original merged cache's checksum-verified
source-shard receipts. Six cache tests pass locally and the previous five pass
on the server. The full suite before the scope correction passed 330 tests.

```bash
PYTHONPATH=src:. python scripts/server/complete_node_sequence_fm_cache.py \
  --full-segments /home/tuv43532/PangenomeFM/server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz \
  --existing-cache /home/tuv43532/PangenomeFM/server_workspace/results/frozen_sequence_fm_cache_20260815/hprc_target_union_sequence_fm.npz \
  --manifest /home/tuv43532/PangenomeFM/server_workspace/data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv \
  --context 1hop --out-dir results/foundation_evidence_20260927/benchmark_nt \
  --device cuda --execute
```

This supports a future controlled sequence-conditioned graph experiment. Such
a model must be described as sequence-conditioned, and compared with matched
raw-input and random-encoder controls; it is not the existing topology-only T.
