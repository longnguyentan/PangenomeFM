# Model and downstream continuation — 28 September 2026

Verified at 01:37 EDT; live jobs may have progressed beyond this snapshot. This is an execution record, not a manuscript revision.
The [central checklist](MODEL_AND_DOWNSTREAM_STATUS_20260927.md) retains the
complete historical task inventory and all failed scientific gates.

## What this continuation resolves

| Work | Verified status | Evidence |
|---|---|---|
| Natural-frequency INS/DEL/INV preparation | Complete: 174,267 events, 147,429 anchors; all 298 primary-chromosome INV retained | Pinned [protocol](../configs/sv_type_natural_20260928.json) |
| Exact-graph mapping | Complete: 100%; one containing segment per anchor | [Mapping QC](../results/foundation_evidence_20260928/natural_sv_smoke/mapping_qc.json) |
| Missing chrY topology vectors | Fixed in all 30 caches; original values bitwise unchanged | [Completion receipt](../results/foundation_evidence_20260928/natural_sv_smoke/cache_completion.json), [all cache audits](../results/foundation_evidence_20260928/natural_sv_smoke/cache_extension_audits.json) |
| Natural-cohort smoke | Complete: 39/39 converged evaluations, independent prediction replay | [Metrics](../results/foundation_evidence_20260928/natural_sv_smoke/per_run.csv), [local replay](../results/foundation_evidence_20260928/natural_sv_smoke/local_replay.json) |
| Natural-cohort full matrix | Running: 15/30 runs complete, five chromosome shards; 1,170 evaluations planned | Server `sv_type_natural_shards/`, detailed commands below |
| New masked-feature pretraining objective | 8/12 pretraining runs complete; first-seed biological controls independently replayed; second-seed probing running | [Protocol](../configs/masked_nt_objective_20260928.json) |
| Whole-graph NT coverage | Missing 271,760 segments audited; completion queued after the model campaign | [Protocol](../configs/whole_graph_nt_completion_20260928.json) |
| Alternative-component context | Complete: 14,786 components partly covered by benchmark-union cache | [Coverage](../results/foundation_evidence_20260928/reference_component_context_20260928/coverage.csv) |
| DART-Eval accessibility task | Official schema/reference/split located; table download requires authentication | [Source audit](../results/foundation_evidence_20260928/dart_feasibility/source_audit.json), [403 receipt](../results/foundation_evidence_20260928/dart_feasibility/download.json) |

Running jobs are detached in tmux on the already authenticated server. No
passwords were transmitted or saved. The main checkout and other model worktrees
remain separate from this evidence checkout.

## 1. Natural-frequency SV type: preserve the whole denominator

The completed matched experiment used 223 events per class. Its newly running
counterpart retains **110,623 INS, 63,346 DEL and 298 INV** without length/class
subsampling. Both use the same first-affected-base anchor, original folds,
frozen v1 graph/NT representations and 13 feature sets. Event log length is the
separate L control. Repeated and mixed-class anchors remain in the data.

The first real smoke stopped before fitting because the frozen T cache lacked
segment 303424, used by **37 INS and 27 DEL events at 60 anchors**. C/K/S/H
coverage was already complete. All 30 original T caches have this gap. The
terminal chrY 55–60 Mb window has too few reconstruction candidates for the
historical loader; the structural graph is present.

The repair extracts missing vectors only from the original reference-overlapping
windows, with the original checkpoint and node-feature policy. It bypasses the
candidate-count condition during inference; it does not retrain, alter graph
releases or alter existing cached vectors. Strict caches append segments 303423
and 303424; one-hop caches append 303424. All 30 serialized caches pass bitwise
comparison of every original row. These are **separately identified coverage
extensions**, not silent replacements of the manuscript caches.

The first extension attempt exposed a JSON NumPy-integer serialization error.
The failed receipt and partial cache were retained, the serializer was fixed,
and the successful extension used a fresh output directory. No event was dropped
to clear either error.

### Independently verified smoke results

**Fold A, seed 42, strict only. Not a completed five-fold result; no CIs.**
These scores do not select features or change the declared full experiment.

| Class | Test prevalence | C+S AP | C+S+T AP | C+S+H AP | C+S+H+T AP |
|---|---:|---:|---:|---:|---:|
| DEL | 0.346734 | 0.473324 | 0.604763 | 0.585770 | 0.653511 |
| INS | 0.651411 | 0.743444 | 0.821476 | 0.785164 | 0.847424 |
| INV | 0.001854 | 0.007446 | 0.010437 | 0.013864 | 0.018557 |

All 39 feature/class fits converge. The INV AP values must be interpreted with
their rare-class prevalence. The full reporter also saves AUROC, normalized AP,
balanced accuracy, F1, precision/recall, per-class and macro paired intervals,
fold sign-flip tests and BH adjustment. This study does not contain a random
encoder arm and cannot isolate learned weights from architectural projection.

## 2. Test an alternative self-supervised signal

The junction candidate still fails two of its three-seed development contrasts.
The next experiment changes the pretraining target, without training on
biological labels or promoting that failed candidate.

The new objective follows the established masked-feature reconstruction and
latent re-masking principles of [GraphMAE](https://arxiv.org/abs/2205.10803).
It is an **objective adaptation on the native PangenomeFM encoder**, not an
official GraphMAE architecture reproduction. Grouping oriented handles and
standardizing frozen NT targets are explicit adaptations.

Fixed before the first fit:

- Native 48D, two-layer, four-head encoder; bidirectional graph messages,
  multiscale/orientation position encoding; unchanged frozen NT inputs.
- Mask 30% of segment IDs and both oriented handles together. Replace every
  input column at those handles with a learned mask token.
- Predict 512D NT vectors standardized using **unique training segments only**.
  Overlapping windows and orientations do not multiply their weight when
  estimating moments. The input NT cache remains unchanged.
- Linear projection, zero masked latent rows, one graph-attention decoder layer,
  then a linear output. Scaled cosine loss; no direct target-copy residual.
- Full structural adjacency remains visible because edges are not the target.
  Explicit position/orientation covariates remain; coordinate-only controls
  are consequently required.
- Four arms per seed: full trained, full frozen random, coordinate-only trained,
  coordinate-only frozen random. Same initial weights within each trained/random
  pair; random backbone hashes must remain unchanged.
- Seeds 42, 314159, 20260806; one-hop, original fold A. Twenty epochs maximum,
  patience five, checkpoint selection by reconstruction validation loss only.
- Native eligible training/validation windows; test windows are not loaded.
  The first runs contain **331 training and 134 validation windows**, with
  verified zero overlap in their segment IDs.
- Frozen downstream cCRE and INS/DEL probes, original matched example universe,
  C+S / C+S+E / C+S+H / C+S+H+E, uniform max_iter=4000. Validation only.

The first runs show lower reconstruction error for trained encoders. That is
an objective check, **not evidence of improved biological transfer**. All arms
and seeds will be retained irrespective of downstream scores. Three initializations
on one fold do not provide chromosome confidence intervals. The paired reporter
checks actual saved predictions, baseline identity, convergence and checkpoint
identity before summarizing any difference.

### First completed biological comparison

**Fold A validation, seed 42, one-hop only; no chromosome confidence intervals.**
All 32 feature evaluations converge, predictions replay, target hashes match,
and C+S / C+S+H predictions are bitwise identical across the four encoders.
E denotes the sequence-conditioned representation; the native `cst`/`csht`
CSV keys are compatibility aliases, not a claim that E is topology-only T.

| Task | C+S+H | + random E | + trained E | Trained minus random | Full trained minus coordinate-only trained |
|---|---:|---:|---:|---:|---:|
| INS versus DEL | 0.896261 | 0.900934 | 0.906437 | +0.005503 | +0.004536 |
| cCRE | 0.914653 | 0.919353 | 0.920763 | +0.001411 | +0.002730 |

The supplemental comparison against the original same-width junction-trained Q
also uses identical examples and bitwise C+S / C+S+H baselines. After H, the new
objective changes AP by **+0.000768 for SV and +0.000144 for cCRE**. Without H,
the corresponding changes are +0.004041 and -0.000006. This is not a comparison
against the 96D T+Q composite, and the 20-epoch versus 10-epoch budget is not
compute matched. All four arms, not only the best one, are included in the CSV.

The first supplemental report correctly stopped at a metadata identity check.
Inspection found a one-ULP difference (5.55e-17) in a previously serialized cCRE
prevalence, with identical target hashes, counts and baseline predictions. The
repair permits only 1e-15 absolute proportion roundoff; exact target and score
hash checks remain. No labels, predictions or metric values were changed.

[First-seed audited metrics](../results/foundation_evidence_20260928/masked_feature_seed42_analysis/audited_per_run.csv),
[all control differences](../results/foundation_evidence_20260928/masked_feature_seed42_analysis/paired_differences.csv),
[junction comparison](../results/foundation_evidence_20260928/masked_junction_seed42_reference/paired_per_seed.csv),
[control figure](../results/foundation_evidence_20260928/masked_feature_seed42_analysis/masked_feature_controls.pdf).
This is promising development evidence; the prespecified gate still requires
both tasks at all three seeds. The partial report explicitly says not promoted.

## 3. Correct the graph denominator and enable broader contexts

The exact existing processed graph has **751,237 segments and 1,097,658 links**,
not 305,070 segments. The latter scale describes the reference portion:
305,572 segments have GRCh38 reference tags; the downstream reference cache has
303,425 rows. The current benchmark NT cache contains 479,477 segments, or
**63.825% of the whole graph**. The original approximately 99.5% figure must not
be reused as whole-graph coverage.

A new label-free audit removes reference nodes and measures connected alternative
components. It finds 445,665 alternative segments in 135,902 components:

| Anchor class / cache coverage | Components | Alternative segments | Cached alternative segments |
|---|---:|---:|---:|
| Multiple anchors on one primary chromosome / complete | 120,257 | 129,125 | 129,125 |
| Multiple anchors on one primary chromosome / partial | 14,786 | 310,057 | 46,927 |
| Nonprimary reference contig / absent | 859 | 6,483 | 0 |

No unanchored, single-anchor or multiple-reference-contig components were found
in this graph. That is observed QC, not an assumption enforced by the code.
The largest component contains 20,761 alternative segments. This audit describes
cache membership across the union of current windows, not individual-window
coverage. Components are **not** directed bubbles or phased haplotype alleles.
A future context builder must retain orientation, check reference-anchor spans
and bound computation before making those claims.

Whole-graph NT completion is queued in a separate output directory, after the
current GPU campaign finishes. It adds **271,760** sequences using the exact
frozen model and manuscript preprocessing, offline. Estimated capped sequence
input is 112.6 million bases; raw output is 1.54 GB, with 679 GB free at audit.
The scope includes 4,972 previously uncached sequences longer than the original
6,000-base cap; their inherited end-sampling limitation remains explicit.

The job verifies graph/cache checksums, waits for free GPUs, checks disk space,
refuses existing output directories, and reopens the merged output to verify
exact IDs, finite values and byte-identical original vectors. It stops on a
failed dependency or a twelve-hour readiness timeout. It does not replace the
cache used by the active campaign or restore missing haplotype paths.

[Graph scope audit](../results/foundation_evidence_20260928/graph_scope_20260928.json),
[component audit](../results/foundation_evidence_20260928/reference_component_context_20260928/audit.json),
[cost audit](../results/foundation_evidence_20260928/whole_graph_nt_cost_20260928.json),
[completion protocol](../configs/whole_graph_nt_completion_20260928.json).

## 4. DART-Eval: concrete next dataset, permission pending

Task 3 predicts one of five cell-type labels at coordinate-anchored accessible
regions: GM12878, H1ESC, HEPG2, IMR90 and K562. The official generator uses
500 bp windows around ATAC peak summits. The processed schema is `chr`,
`input_start`, `input_end`, `elem_start`, `elem_end`, `is_peak`, `label`.
The official NT extractor uses GRCh38. These facts were verified in pinned
[publisher code](https://github.com/kundajelab/DART-Eval/tree/af2a86d666c35304257c2fa7e15180e1fbcabb01).

The original split uses validation chr6/chr21 and test chr5/chr10/chr14/chr18/
chr20/chr22. An adaptation using the existing PangenomeFM five-fold checkpoints
must be labelled as such. It cannot be presented as the official split or a
direct leaderboard comparison with the publisher's convolutional probing heads.

The actual processed table is **Synapse syn61788656, version 1**, file handle
173819470. Its metadata are publicly readable; its data-download endpoint
returns HTTP 403 for an anonymous client. The actual rows, counts, label
distribution and graph coverage are therefore **not yet verified**. No synthetic
performance or substitute labels have been created. An authorized local/server
copy is the next dependency; credentials should not be sent in chat.

## Reproducible execution

Dedicated server checkout:
`/home/tuv43532/PangenomeFM_evidence_report_20260927`.
All new native results live below its `results/foundation_evidence_20260927/`.
Versioned compact snapshots in this repository are under
`results/foundation_evidence_20260928/`, keeping mutable server status files out
of Git synchronization.

```bash
# From the dedicated checkout; original data paths are recorded in launch.json.
export PYTHONPATH=src:.

# Full natural SV recovery: append-only cache completion -> smoke -> five shards
# -> exact 30-job merge -> saved-prediction report. Fresh output roots required.
python results/foundation_evidence_20260927/sv_type_natural_recovery2_driver/run.py

# Masked-feature campaign: 12 pretraining runs -> 24 frozen validation probes.
# The launch record contains the exact existing template checkpoint and caches.
python results/foundation_evidence_20260927/masked_feature_driver/launch.py

# Supplemental objective comparison after the full model report completes:
python -m tasks.transfer.masked_junction_reference \
  --config configs/masked_nt_junction_reference_20260928.json \
  --masked-report results/foundation_evidence_20260927/masked_feature_full_analysis \
  --out-dir <fresh-junction-comparison>

# Audited dependency job: supplemental report -> whole-graph NT completion.
# Already queued on the server; use a fresh runtime/output root for replay.
python results/foundation_evidence_20260927/post_campaign_driver/run.py

# Read-only durable-job status:
tmux -L evidence-20260927 list-sessions

# When complete, independently rerun either reporter to a fresh output root:
python -m tasks.transfer.sv_type_report --root <natural-full-root> \
  --examples <preparation>/natural_events.parquet --out-dir <fresh-report>
python -m tasks.transfer.masked_feature_report --root <masked-feature-root> \
  --out-dir <fresh-report>
```

The natural-SV archived launcher pins native commit `113c2ce`; check it out for
an exact replay. The masked-feature launcher records its native commit in
`launch.json` (first launch `ce9587f`). Do not rerun a launcher over active or
completed output directories. The scripts refuse to overwrite them.
On a fresh checkout, copy the archived driver directory from
`results/foundation_evidence_20260928/` to its recorded runtime location under
`results/foundation_evidence_20260927/` before execution. Source data and exact
server resources remain prerequisites; launchers do not download replacements.

## Tests and remaining checklist

- [x] Full local suite: 428 tests; two subsequent component tests also pass (430 unique tests). The report metadata-only follow-up passes its targeted test.
- [x] Initial eleven server tests, then nine cache/report tests and two context tests pass; Ruff and compile checks pass.
- [x] Real natural-SV smoke independently replayed: all 39 evaluations.
- [x] Existing manuscript regression checks remain in the passing suite.
- [ ] Complete natural-SV fitting, full replay, paired uncertainty and plots.
- [x] Complete and independently replay the first seed of masked-feature biological controls and the supplemental junction-Q comparison.
- [x] Audit whole-graph scope, alternative-component coverage and NT completion cost.
- [ ] Complete the queued full-graph frozen NT cache and verify preserved values.
- [ ] Complete all three seeds of masked-feature trained/random/coordinate comparisons. Retain failed
  gates if any; do not choose only favorable tasks or seeds.
- [ ] Only after adequate development evidence, freeze a chromosome-replication
  protocol. Current v2 has not established a universally better final model.
- [ ] Obtain authorized DART rows, verify labels/coordinates/coverage, then
  prespecify an appropriately labelled comparison.
- [ ] Official graph-SSL architecture and stronger complete sequence-embedding
  controls remain separate from this native-backbone objective adaptation.
- [ ] DUP/complex class labels, per-variant PanGenie concordance/callability,
  all-tested QTL universes and verified haplotype correspondence remain data
  dependencies. Completed COSIGT/GTEx feasibility checks do not resolve them.

The EN-TEx panel, HG008 prospective refits, original scaling, all three TraitGym
studies, matched SV types and COSIGT studies remain complete. Their null and
negative results are retained in the central scorecard; these new experiments
do not overwrite them or guarantee a stronger model.
