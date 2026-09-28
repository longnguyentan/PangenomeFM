# Model and downstream continuation — 28 September 2026

Verified at 02:37 EDT; live jobs may have progressed beyond this snapshot. This is an execution record, not a manuscript revision.
The [central checklist](MODEL_AND_DOWNSTREAM_STATUS_20260927.md) retains the
complete historical task inventory and all failed scientific gates.

## What this continuation resolves

| Work | Verified status | Evidence |
|---|---|---|
| Natural-frequency INS/DEL/INV preparation | Complete: 174,267 events, 147,429 anchors; all 298 primary-chromosome INV retained | Pinned [protocol](../configs/sv_type_natural_20260928.json) |
| Exact-graph mapping | Complete: 100%; one containing segment per anchor | [Mapping QC](../results/foundation_evidence_20260928/natural_sv_smoke/mapping_qc.json) |
| Missing chrY topology vectors | Fixed in all 30 caches; original values bitwise unchanged | [Completion receipt](../results/foundation_evidence_20260928/natural_sv_smoke/cache_completion.json), [all cache audits](../results/foundation_evidence_20260928/natural_sv_smoke/cache_extension_audits.json) |
| Natural-cohort smoke | Complete: 39/39 converged evaluations, independent prediction replay | [Metrics](../results/foundation_evidence_20260928/natural_sv_smoke/per_run.csv), [local replay](../results/foundation_evidence_20260928/natural_sv_smoke/local_replay.json) |
| Natural-cohort full matrix | Complete: 30/30 runs, 1,170 converged and replayed evaluations, zero exclusions | [Full audit](../results/foundation_evidence_20260928/sv_type_natural_full_analysis/audit.json) |
| New masked-feature pretraining objective | Complete: 12 pretraining runs, 24 frozen probes / 96 feature evaluations; three-seed gate passes | [Protocol](../configs/masked_nt_objective_20260928.json) |
| Whole-graph NT coverage | Complete: all 751,237 graph segments, original 479,477 rows byte-identical | [Protocol](../configs/whole_graph_nt_completion_20260928.json) |
| Chromosome replication | Running: first fold-A test probes; fixed 5 folds × 3 seeds × 4 arms | [Protocol](../configs/masked_nt_chromosome_replication_20260928.json) |
| Alternative-component context | Complete: 14,786 components partly covered by benchmark-union cache | [Coverage](../results/foundation_evidence_20260928/reference_component_context_20260928/coverage.csv) |
| DART-Eval accessibility task | Official schema/reference/split located; table download requires authentication | [Source audit](../results/foundation_evidence_20260928/dart_feasibility/source_audit.json), [403 receipt](../results/foundation_evidence_20260928/dart_feasibility/download.json) |

Running jobs are detached in tmux on the already authenticated server. No
passwords were transmitted or saved. The main checkout and other model worktrees
remain separate from this evidence checkout.

## 1. Natural-frequency SV type: preserve the whole denominator

The completed matched experiment used 223 events per class. Its completed
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

### Completed natural-frequency matrix

**Five original chromosome folds × three seeds × two contexts; all 174,267 events
retained.** Independent server prediction replay verifies all 1,170 evaluations;
every fit converged. This resolves the missing-chrY-feature blocker without
changing the denominator. Prevalence in the complete callset is 63.48% INS,
36.35% DEL and 0.171% INV; these are callset frequencies, not population frequencies.

| Macro AUPRC or paired gain | Strict | One-hop |
|---|---:|---:|
| C+S | 0.406955 | 0.406955 |
| C+S+T | 0.474012 | 0.458608 |
| ΔT given C+S, 95% CI | +0.067057 [0.062310, 0.070836] | +0.051653 [0.047341, 0.055334] |
| C+S+H | 0.455130 | 0.455130 |
| C+S+H+T | 0.499834 | 0.472585 |
| ΔT given C+S+H, 95% CI | +0.044704 [0.040045, 0.049689] | +0.017455 [0.014850, 0.020240] |
| C+S+L+H | 0.508173 | 0.508173 |
| C+S+L+H+T | 0.555836 | 0.523160 |
| ΔT given C+S+L+H, 95% CI | +0.047664 [0.039885, 0.055252] | +0.014988 [0.007719, 0.021421] |

L is measured event log length; H is the existing label-free graph-statistic
cache. The INS/DEL improvements persist after both controls. INV is different:
C+S AP is 0.006907, C+S+H is 0.020660, and adding event length raises C+S+L+H
to 0.174329. Its incremental T after all controls is +0.015190 strict
(CI -0.004323 to +0.036473) and -0.001538 one-hop
(CI -0.019657 to +0.011729). Thus the data **do not establish improved inversion
prediction beyond length and cheap structure**, despite the positive macro result.

The matched and natural-frequency experiments answer different conditional
questions and both remain reported. The larger natural-cohort gains must not be
presented as an improvement to model weights: both use the existing frozen v1
encoder. There is no random-encoder arm in this extension. The exact two-sided
fold sign-flip minimum is 0.0625 with only five independent folds; these macro
contrasts have BH q=0.0741. Positive pointwise bootstrap intervals are not
multiplicity-controlled confirmation. The task classifies already ascertained
SV events and does not demonstrate discovery or genotyping accuracy.

[Per-run results](../results/foundation_evidence_20260928/sv_type_natural_full_analysis/per_run.csv),
[all absolute metrics](../results/foundation_evidence_20260928/sv_type_natural_full_analysis/absolute.csv),
[paired intervals and multiplicity](../results/foundation_evidence_20260928/sv_type_natural_full_analysis/contrasts.csv),
[figure](../results/foundation_evidence_20260928/sv_type_natural_full_analysis/sv_type_topology_gains.pdf).

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

### Completed three-seed biological development comparison

**Fold-A validation, one-hop; 12 pretraining runs, 24 frozen probe runs and all
96 feature evaluations completed.** All fits converge and saved metrics replay.
Targets, split sizes and baseline C+S / C+S+H predictions match across arms.
Random encoders preserve their initial weights exactly. E denotes the
sequence-conditioned masked-feature representation; native `cst`/`csht` are
compatibility keys, not a claim that this is topology-only T.

| Seed | SV trained − random after C+S+H | cCRE trained − random after C+S+H | SV full − coordinate trained | cCRE full − coordinate trained |
|---|---:|---:|---:|---:|
| 42 | +0.005503 | +0.001411 | +0.004536 | +0.002730 |
| 314159 | +0.006468 | +0.002178 | +0.007598 | +0.002027 |
| 20260806 | +0.004855 | +0.001920 | +0.006302 | +0.002359 |
| Descriptive mean | +0.005609 | +0.001836 | +0.006145 | +0.002372 |

The originally declared trained-minus-random gate passes all six checks. The
separate full-versus-coordinate contrasts are also positive at every seed.
This is development evidence for learned sequence-conditioned representations;
seeds are not independent genomic replicates, so **no chromosome CI** is supplied.
Removing a stream also removes capacity; this is not an exact equal-capacity
edge-rewiring ablation. F1 and balanced accuracy are not uniformly better in
every seed, and all secondary metrics remain in the tables.

The supplemental same-width junction-Q comparison verifies bitwise identical
C+S / C+S+H scores and targets. Masked-feature E minus Q after H averages
**+0.001877 SV / +0.000099 cCRE**. Without H, it averages
**+0.006154 SV / -0.000145 cCRE**. Thus it does not uniformly dominate Q.
The 20-epoch versus 10-epoch budgets are not compute matched, and this is not a
comparison against the 96D T+Q composite. Every arm and seed is retained.
The initial prevalence identity check failed at a one-ULP serialization
roundoff (5.55e-17); only a 1e-15 absolute proportion tolerance was added, with
exact target/count/baseline-score checks preserved. No scores or labels changed.

[Completed audit](../results/foundation_evidence_20260928/masked_feature_full_analysis/audit.json),
[per-run metrics](../results/foundation_evidence_20260928/masked_feature_full_analysis/audited_per_run.csv),
[all paired controls](../results/foundation_evidence_20260928/masked_feature_full_analysis/paired_differences.csv),
[control figure](../results/foundation_evidence_20260928/masked_feature_full_analysis/masked_feature_controls.pdf),
[all junction comparisons](../results/foundation_evidence_20260928/masked_junction_full_reference/paired_per_seed.csv).

### Fixed chromosome replication: running

The [replication protocol](../configs/masked_nt_chromosome_replication_20260928.json)
was committed before any new chromosome-test results. After reviewing the
completed gate and full-versus-coordinate results, the detached
`masked_chromosome` job was launched. After whole-graph NT completion released
the GPUs, it advanced into fold-A/seed-42 test probing. The explicit
[review receipt](../results/foundation_evidence_20260928/masked_chromosome_replication_driver/development_review.json)
pins development audit SHA-256
`3e05b7840c26b328300db07b758eb70a979f677626b17cff771a0218eaf8d6a2`.
This authorizes an experiment, not an architecture superiority claim.

The fixed matrix retains all four arms across five original folds and three
seeds: **60 encoder instances** (12 existing fold-A encoders reused, 48 new
pretraining runs), **120 frozen biological probe runs / 480 feature evaluations**.
Model width, objective, optimizer, input cache, features and convergence budget
remain fixed. New folds receive architecture metadata only, never trained
weights. The new whole-graph cache is not substituted into this experiment.
Saved classifier/scaler/calibration artifacts are enabled for future reuse.

Fold-A development validation used the chromosomes in **fold B's test set**.
The primary follow-up summary therefore uses folds A/C/D/E; fold B and the
full five-fold matrix are separate sensitivity scopes. This does not erase
historical v1 label exposure, so no untouched-external-validation claim is made.
The existing paired hierarchical bootstrap, fold sign-flip and BH routines are
reused. The matrix is still running; completed individual probes do not constitute a
full chromosome-replication result. No partial test-driven model changes are made.

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

A separate graph-wide check finds **zero direct links between distinct reference
contigs**, and zero alternative components attached to multiple reference contigs.
Together these rule out graph paths crossing GRCh38 contigs in this exact graph.
The audit does not establish donor independence.
[Separation receipt](../results/foundation_evidence_20260928/reference_contig_separation_20260928.json).

No unanchored, single-anchor or multiple-reference-contig components were found
in this graph. That is observed QC, not an assumption enforced by the code.
The largest component contains 20,761 alternative segments. This audit describes
cache membership across the union of current windows, not individual-window
coverage. Components are **not** directed bubbles or phased haplotype alleles.
A future context builder must retain orientation, check reference-anchor spans
and bound computation before making those claims.

Whole-graph NT completion is **complete** in a separate output directory:
**751,237 × 512 finite float32 vectors**, including 271,760 appended segments.
All 479,477 original serialized rows are byte-identical, with exact canonical
ID coverage. The completed SHA-256 is
`5c07f9bd8b46e4104c7c787fd96dae7cdf14ab8207807a87495e3e0365d257f3`.

The exact original frozen model, revision and preprocessing were reused offline.
The inherited 6,000-base end-sampling/1,000-token limitation remains; 4,972 new
segments exceed that raw sequence cap. This is representation infrastructure,
not a biological result or a verified haplotype/bubble representation. No
original cache was overwritten and the active replication still uses its
prespecified benchmark-union cache. The four workers completed and the merged
serialized cache passed the independent extension audit.

[Completion and preservation receipt](../results/foundation_evidence_20260928/whole_graph_nt_completion/status.json),
[cache provenance](../results/foundation_evidence_20260928/whole_graph_nt_completion/whole_graph_nt.npz.audit.json).
Server output: `results/foundation_evidence_20260927/whole_graph_nt_completion_20260928/whole_graph_nt.npz`.

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

## 5. Check the proposed larger SV resource before defining new labels

The public 1,218-genome release supplies **587,779 GRCh38 INS/DEL** calls;
614,522 is its CHM13 count. It therefore does not directly resolve the missing
DUP/complex type labels. [UCSC release documentation](https://hgdownload.soe.ucsc.edu/gbdb/hs1/hubs/public/lrSv1kLin.html).

I inspected the actual compressed GRCh38 VCF header, the pinned author repository
and its Zenodo inventory. The header contains 1,218 genotype sample columns and
GT/allele-length/breakpoint-position fields, plus allele frequency/count and HWE
annotations. It does not declare per-variant measured genotyping F1/concordance.
The repository describes a short-read F1 selection criterion, but its public
file inventory contains scripts rather than those outcomes or an all-tested
phenotype denominator. AF, NS and HWE must not substitute for measured accuracy.
[Author repository](https://github.com/jiadong324/1KG_LongRead_SV/tree/400552c4980277f580d123ab5b6bf9d41874de07).

The release mixes HPRC/HGSVC and other long-read genomes. A future external
INS/DEL evaluation must explicitly remove training-call/donor overlap before
claiming independence. Neither released associations alone nor untested SVs can
define an association-negative class. These are task-design requirements inferred
from the verified release structure, not measured model performance.

The candidate compressed VCF is 306,045,265 bytes; only its header and first-row
metadata were read. No full callset download, full QC or fitting was performed.
[Zenodo release](https://zenodo.org/records/22000872),
[actual header/inventory audit](../results/foundation_evidence_20260928/lin1218_feasibility/source_audit.json).

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

# Fixed chromosome replication; the launch helper pins the reviewed gate SHA.
# Already running; do not start a duplicate in its output directory.
python results/foundation_evidence_20260927/masked_chromosome_replication_driver/launch.py

# Read-only durable-job status:
tmux -L evidence-20260927 list-sessions

# When complete, independently rerun either reporter to a fresh output root:
python -m tasks.transfer.sv_type_report --root <natural-full-root> \
  --examples <preparation>/natural_events.parquet --out-dir <fresh-report>
python -m tasks.transfer.masked_feature_report --root <masked-feature-root> \
  --out-dir <fresh-report>
python -m tasks.transfer.masked_replication --root <completed-replication-root> \
  --out-dir <fresh-replication-report>
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

- [x] Full local suite: **436 passed**, including cached manuscript checks, replication gates and exact saved-probe replay; 14 known warnings. Targeted Ruff passes.
- [x] Server: latest 19 probe/persistence/replication tests pass, in addition to earlier cache/context checks; Ruff and compile checks pass.
- [x] Real natural-SV smoke independently replayed: all 39 evaluations.
- [x] Existing manuscript regression checks remain in the passing suite.
- [x] Complete natural-SV fitting, full replay, paired uncertainty and plots: 30 runs, 1,170 evaluations, zero exclusions.
- [x] Complete and independently replay all three seeds of masked-feature biological controls and the supplemental junction-Q comparison.
- [x] Audit whole-graph scope, alternative-component coverage and NT completion cost.
- [x] Complete the full-graph frozen NT cache: 751,237 rows, every original row byte-identical.
- [x] Complete all three seeds of masked-feature trained/random/coordinate comparisons; all six original development gates pass.
- [x] Freeze/test the chromosome-replication protocol before observing new chromosome-test scores.
- [x] Review the completed gate and launch fixed chromosome replication in tmux, now running after cache completion.
- [ ] Complete/replay all 120 chromosome-test probes, then assess improvement. Current v2 has not established a universally better final model.
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

## Saved-probe repair

The native cCRE/INS-DEL runners now support `--save-probes`; default behavior is
unchanged. The shared, previously tested HG008 persistence helper was moved to
`evaluation.probe_artifacts` and reused. Each optional artifact stores the
training-fitted sklearn pipeline (including scaling), validation temperature
and threshold, feature dimension, partition metadata and encoder identity.
Serialization is immediately checked against the original raw predictions.
Random-uniform and prevalence baselines are not misrepresented as fitted models.
The fixed chromosome replication opts in; ongoing development fits retain their
original command lines. Two tests show bitwise equality of all default/opt-in
metrics and predictions, plus exact raw/calibrated/classification replay after
reload. This prevents repeating the historical missing-probe external-replay
blocker; it does not retrospectively recover lost historical classifiers.

[Scientific result review and remaining limitations](RESULT_REVIEW_20260928.md).

The DART source audit also examined the official reconstruction route. Labels
require replicate ATAC counts and differential-accessibility analysis; consensus
peak membership is not an equivalent label. Neither local nor server Synapse
configuration is present (existence-only check, no credentials inspected).
The alternate route needs the exact BAM/count inputs and final label processing;
an authorized processed table remains the smallest dependency to resolve.
[Author workflow](https://github.com/kundajelab/DART-Eval/blob/af2a86d666c35304257c2fa7e15180e1fbcabb01/README.md#task-3-discriminating-cell-type-specific-elements),
[reconstruction feasibility audit](../results/foundation_evidence_20260928/dart_feasibility/raw_reconstruction_audit.json).
