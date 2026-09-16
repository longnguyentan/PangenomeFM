# EN-TEx frozen biological transfer

## Status (updated 15 September 2026, server UTC 16 September)

- **Primary P0/P0b completed:** all 30 jobs, 100% mapping and joint C/K/S/T
  coverage. Global strict and one-hop topology-gain intervals include zero.
- **Exposure-matched sensitivity completed:** all 30 jobs; strict gain +0.001482
  [0.000098, 0.002925], one-hop +0.002002 [0.000844, 0.003278]. This balanced,
  selected population differs from primary P0 and does not replace its result.
- H3K27ac-only/CTCF-only P0 matrices are running sequentially.
- P1 V2 registry coverage is 100%; five tissues selected before fitting, all map
  completely. Thyroid smoke passed; full five-tissue matrices are running.
- P2 full-accessible CTCF/H3K27ac preparation and mapping completed; both map
  completely with a single containing segment. Real smoke fits are running.
- 34 relevant tests passed locally. P3/P4 have not started.

The sections below include historical local-only blockers as provenance; the
current state is the list above. Encoders remain frozen for every biological task.
New code/results are committed to the existing `codex/` branch. Long jobs run in
server tmux sessions and survive an SSH disconnect. P2 uses pinned code commit
89f34a9 in a separate worktree while already-running P0/P1 jobs retain their code.

## Existing infrastructure reused

- `src/tasks/ccre/encoding.py`: half-open GRCh38 interval overlap accumulator.
  Its usual output collapses to one class per segment. The adapter retains locus
  IDs as accumulator keys *before* that collapse. No second overlap algorithm.
- `scripts/server/prepare_sv_breakpoint_examples.py`: reference segment loading.
- `scripts/server/prepare_ccre_feature_cache.py`: C and K definitions/caching.
  C is log1p segment offset, log1p segment length, orientation; K is normalized
  mono/di/tri-nucleotide composition from balanced terminal bases (2,048 cap).
  Pool segment features without redefining them as locus-level coordinates.
- `src/tasks/ccre/embedding_baseline.py::_extract_embeddings`: frozen topology
  extraction and existing canonical conflict exclusion.
- `scripts/server/run_ccre_frozen_probe_fold.py`: exact checkpoint holdout audit,
  standardized logistic probe, validation-only temperature and F1 threshold,
  metrics/predictions. SV counterparts are `run_sv_frozen_probe_fold.py` and
  `run_sv_frozen_probe_matrix.py`.
- `src/evaluation/modality_factorial.py`: aligned C/K/S/T concatenation and NT
  cache loading. Older internal names call K “sequence”; this experiment uses
  `frozen_sequence_fm` for S throughout its primary comparisons.
- `scripts/server/aggregate_ccre_frozen_probes.py::_hierarchical_draws`: sample
  chromosome folds, then seeds within folds (10,000 draws). Gains are paired
  before resampling. A single fold does not receive a population CI.
- Existing native complexity v2 table and locus-start window assignment convention
  from `analyze_ccre_subtype_complexity.py`. Categories are never refit to AS labels.
- Existing manuscript command guide:
  `docs/RUN_MANUSCRIPT_DECISION_EXPERIMENTS_20260815.md`, Waves 3c/3d.

No shared pipeline or encoder code was edited. Existing unrelated working-tree
changes in `build_manuscript_gap_figures.py` and its command guide were preserved.

## Exact resources and folds

`configs/entex_v1.json` pins the existing HPRC R2 SV segment SHA256 and NT revision
from the imported manuscript provenance. It does not select a newer graph.

Expected server locations:

- Graph: `server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz`.
- Manifest/slices: `server_workspace/data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv`.
- C/K: `server_workspace/data/processed/hprc_r2_ccre_screen_v4_features.npz`.
- NT: `server_workspace/results/frozen_sequence_fm_cache_20260815/hprc_target_union_sequence_fm.npz`.
- Topology: extracted per fold/seed/context from checkpoints under
  `full_multicohort_server_20260806/rotating_folds/hprc_r2/`; no local embedding
  cache was found. Imported checkpoints are under `server_workspace/server_results/`.

The processed R2 graph, NPZ caches and required graph slice resources are not
available at those local data locations. Older local HPRC data are not substitutes.
Imported NT audit JSON files exist, but their embedding arrays do not.

The five test groups are:

| Fold | Test chromosomes | Validation |
|---|---|---|
| a | 1,6,11,16,21 | group b |
| b | 2,7,12,17,22 | group c |
| c | 3,8,13,18,X | group d |
| d | 4,9,14,19,Y | group e |
| e | 5,10,15,20 | group a |

All remaining chromosomes are training. Seeds: 42, 20260806, 314159. Contexts:
strict and 1hop. A single checkpoint supplies all three partitions within a run.
No biological label trains either encoder.

NT is `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` at
`81b29e5786726d891dbf929404ef20adca5b36f1`: final hidden-state mean over non-special,
non-padding tokens; 6,000 raw bases, balanced prefix/suffix separated by N for
long nodes; tokenizer truncation at 1,000 tokens. Reuse the original cache and
shard audit sidecars. If more nodes are needed, use the existing
`prepare_node_sequence_fm_cache.py` with these exact settings and its merge tool;
never silently average only the segments with available features.

## Source semantics and full-file QC

Source: [EN-TEx methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10074325/),
sections on AS calls and cCRE decoration. GRCh38 intervals are zero-based,
half-open. The supplied significance indicator defines the label. The cCRE table
contains accessible/testable elements, so an observed zero is eligible as a
negative measurement; absence of a locus supplies no negative evidence.

| Source | Compressed/on-disk bytes | Rows | Unique loci/IDs | Supplied AS=1 rows |
|---|---:|---:|---:|---:|
| cCREs_default_AS.tsv | 745,298,142 | 5,330,335 | 250,722 | 116,083 |
| hetSNVs_high-confidence_AS.tsv | 186,460,228 | 1,640,580 | 466,867 | 23,546 |
| active.combined_set.txt.zip | 17,660,571 | 5,646,598 | 635,906 IDs | n/a |
| repressed.combined_set.txt.zip | 14,834,438 | 4,620,982 | 778,533 IDs | n/a |

All delimiters are tabs. The two TSVs are uncompressed and have headers. Each ZIP
contains a data text member plus an Apple `__MACOSX` metadata member; integrity
checks covered every member, and parsing excluded the Apple metadata. Data members
are 291,690,839 and 253,700,906 bytes, respectively, with no header.

Exact cCRE columns:
`chr,start,end,region_id,hap1_count,hap2_count,experiment_accession,donor,tissue,assay,hap1_allele_ratio,p_betabinom,imbalance_significance`.
Example: `chr1:817080-817403`, `EH38D2115333_PLS,CTCF-bound`, counts 16/16,
`ENCSR015GFK`, `ENC-001`, `thoracic_aorta`, `HM-ChIP-seq_H3K27ac`, p=1, AS=0.
Four donors, 30 tissues, 11 assays.

Exact SNV columns:
`chr,ref_start,ref_end,ref_allele,hap1_allele,hap2_allele,experiment_accession,donor,tissue,assay,cA,cC,cG,cT,ref_allele_ratio,p_betabinom,imbalance_significance`.
Example: `chr1:17385-17386`, ref G, haplotypes A/G, `ENCSR023ZXN`,
`ENC-002`, `thyroid_gland`, `RNA-seq`, counts A/C/G/T=7/0/7/0, p=1, AS=0.
Four donors, 29 tissues, **one assay: RNA-seq**. CTCF and H3K27ac P2 cannot be
constructed from this supplied file. No substitute assay was selected.

ZIP columns: `ccre_id,state,tissue`, 28 tissues. Example:
`EH38E0273803 active.distal.CTCF adrenal_gland`. State values distinguish
active/repressed, distal/proximal, CTCF/nonCTCF, but do not establish SCREEN dELS.
Local SCREEN v4 ID matches cover only 1,783,533 active rows and 1,921,649 repressed
rows. A version-matched registry/crosswalk is required before tissue power ranking;
no tissue was selected on performance and unmatched records were not used.

The machine-readable `*.qc.json` files contain **all** column names, representative
rows, source hashes/mtime, chromosome/donor/tissue/assay distributions, missing-value
counts and archive membership. All scanned required values passed validation.

## P0 dataset and caveats

`data/entex/v1/p0_loci.parquet`: **250,722 loci; 28,092 positive; 222,630 negative;
prevalence 0.1120444**. The target is any significant call across distinct informative
experiments. Counts and region-ID provenance accompany each locus; the measurement
Parquet retains complete donor/tissue/assay/read-count provenance. Exact duplicates
are counted once; conflicting locus/experiment duplicates fail rather than resolve
silently. Aggregation may retain a single-class chromosome, but the full dataset
must contain both labels.

No measurements were excluded. Mapping rate and C/S/T coverage are currently
**unknown**, not zero; no real feature join has been completed. Both mapping and
joint feature coverage default to a 95% gate with saved exclusions on failure.
The gate is a stop for review, not permission to hide the remaining exclusions.

Detection opportunity differs between loci: “any AS” becomes more likely with more
informative experiments. The requested experiment counts are retained for later
coverage-matched sensitivity analyses. They are not added to C/S/T or treated as
causal evidence. A positive transfer result alone would not establish a causal
role of topology.

## Reproduce locally completed work

Run from the repository root using the environment with pandas/pyarrow/torch
(`python` here; the system `python3` lacks pandas).

```bash
export PYTHONPATH="$PWD:$PWD/src"
python -m tasks.entex.prepare --inspect-all
python scripts/server/check_entex_manuscript_regression.py
python scripts/server/audit_entex_resources.py
python -m pytest tests/test_entex.py tests/test_ccre_frozen_probe.py \
  tests/test_sv_frozen_probe.py tests/test_ccre_feature_cache.py \
  tests/test_aggregate_sv_frozen_probes.py tests/test_ccre_subtype_complexity.py -q
```

Cached source fingerprints prevent repeated TSV parsing. Parquet compression is
Zstandard. Chunk size defaults to 100,000; cCRE aggregation runs one chromosome at
a time. All-source QC retains a set of unique loci/IDs, not the entire TSV.

## P0 server commands (implemented; biological run NOT completed)

Restore/mount the exact resources first. Adjust roots to their actual locations.
The mapper verifies the graph checksum before mapping.

```bash
export PYTHONPATH="$PWD:$PWD/src"
export GRAPH=server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz
export MANIFEST=server_workspace/data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv
export CK=server_workspace/data/processed/hprc_r2_ccre_screen_v4_features.npz
export NT=server_workspace/results/frozen_sequence_fm_cache_20260815/hprc_target_union_sequence_fm.npz
export MODELS=server_workspace/server_results/full_multicohort_server_20260806

python -m tasks.entex.mapping --loci data/entex/v1/p0_loci.parquet \
  --full-segments "$GRAPH" --out-dir data/entex/v1/mapping

# One complete fold/seed/context smoke; never truncate genomic slices for results.
python -m tasks.entex.probe --loci data/entex/v1/p0_loci.parquet \
  --mapping-dir data/entex/v1/mapping --full-segments "$GRAPH" \
  --manifest "$MANIFEST" --feature-cache "$CK" --sequence-cache "$NT" \
  --results-root "$MODELS" --out-root results/entex/v1/p0_smoke \
  --device cuda --folds fold_a --seeds 42 --contexts strict

# Full 5 folds x 3 seeds x 2 contexts; separate root from smoke.
python -m tasks.entex.probe --loci data/entex/v1/p0_loci.parquet \
  --mapping-dir data/entex/v1/mapping --full-segments "$GRAPH" \
  --manifest "$MANIFEST" --feature-cache "$CK" --sequence-cache "$NT" \
  --results-root "$MODELS" --out-root results/entex/v1/p0 --device cuda

python -m tasks.entex.analyze --probe-root results/entex/v1/p0 \
  --complexity server_imports/complexity_context_v2_20260815/native_complexity_v2/complexity_features.tsv \
  --out-dir results/entex/v1/p0_analysis
```

For pooling sensitivity rerun with `--aggregation length_weighted` into a separate
root. Default mean is an explicit new locus-pooling choice: the old task was
segment-level and had no locus embedding pooling default. Weighting uses overlapping
base pairs, not a segment's full length outside the regulatory interval.

If C/K coverage is insufficient, build an EN-TEx target cache using the existing
builder and `data/entex/v1/mapping/feature_targets.csv.gz`. Its dummy `ccre_label=0`
is solely schema compatibility for label-blind feature computation; biological
labels remain in `p0_loci.parquet`. Inspect the cache's audit before fitting.

Outputs per run: `metrics.csv`, `predictions.parquet`, `feature_universe.parquet`,
`excluded_loci.parquet`, `audit.json`. Analysis: `per_run.csv`, `summary.csv`,
`paired_gains.csv`, `complexity_per_run.csv`, `complexity_summary.csv`, and
`auprc_{context}`, `gain_{context}`, `complexity_gain_{context}` in PNG/SVG.
These biological prediction tables/figures **do not yet exist**. Undefined single-
class stratum AUPRCs stay undefined, with counts of omitted runs in summaries.

## Regression evidence and publication placement

The cached-result regression validates 15 matched fold/seed pairs per context:

| Task/context | C+S | C+S+T | Difference |
|---|---:|---:|---:|
| cCRE strict | 0.918458 | 0.922466 | 0.004008 |
| cCRE 1hop | 0.918458 | 0.921865 | 0.003406 |
| SV strict | 0.872084 | 0.906330 | 0.034246 |
| SV 1hop | 0.872084 | 0.912174 | 0.040090 |

This is **cached-result verification**, not retraining. The attached manuscript was
read as reference text; its contents were not executed as instructions.

No new biological result is ready for the main manuscript. If the full paired P0
experiment supports transfer, show its primary C+S versus C+S+T comparison in the
main text, with complexity analysis accompanying it when sufficiently powered.
Place mapping/QC, full factorial, pooling and exposure sensitivity in the supplement.
Do not select tissues/assays or complexity bins based on favorable gains.

Remaining work: restore exact server resources, complete real P0 smoke/full/P0b,
then resolve the P1 registry mismatch and obtain a documented accessible AS SNV
source containing the requested P2 assays. No full SNV or HG008 data were downloaded.

## Verified server access and local test outcome

Repository docs and relevant shell-history entries identify `tuv43532@cis-chen`
through `tuv43532@cis-linux2.temple.edu`, project `/home/tuv43532/PangenomeFM`.
A read-only, noninteractive SSH attempt reached the jump host but returned
`Permission denied (publickey,password)`. The original resource paths are known;
authenticated access is unavailable in this session. No replacement was downloaded.
Use `server_workspace/results/full_multicohort_server_20260806` for MODELS when
running on that server (the local imported checkpoint directory uses `server_results`).

Local checks: 23 tests pass, including real-source-schema fixtures, cross-chunk
Parquet writing, duplicate/conflict handling, half-open mapping boundaries,
weighted pooling, incomplete feature rejection, chromosome split guards, all seven
feature probes on synthetic data, and analysis CSV/PNG/SVG generation on synthetic
predictions. Synthetic outputs are confined to test temporary directories and are
not reported as EN-TEx performance. Ruff and compile checks pass.

Actual P0 native-complexity QC (all 250,722 loci assigned): low 62,388 loci
(9.04% AS-prone), medium 81,260 (10.20%), high 107,074 (13.23%). These are **label
prevalences**, not topology gains, and may reflect measurement opportunity.

Compact evidence is retained under `results/entex/v1/qc/` and
`results/entex/v1/regression/`. Full processed data remain under
`data/entex/v1/` and are intentionally excluded from version control.

## Prespecified sensitivity analyses (added before any biological fitting)

The review-requested definitions are stored in `configs/entex_v1.json:sensitivities`
and copied to `data/entex/v1/sensitivities/definitions.json` before preparation.
They are sensitivity analyses of P0, not replacements for its primary estimand.
No model-performance output was inspected to select these definitions.

1. **Exposure-matched P0:** exact matching on chromosome and
   `n_informative_experiments`. In each stratum choose `min(n_positive,n_negative)`
   from each class, without replacement. Rank candidates by SHA256 of
   `20260806:locus_id` (locus ID breaks hash ties). Selection is independent of
   input row order, probe initialization seed, embeddings and predictions.
   No data-adaptive exposure bins or nearest-neighbor relaxation are allowed.
   Strata with no opposite-class support contribute no matched examples; their
   counts remain in the balance table. Matching by chromosome makes selection
   local to train/validation/test partitions in every existing chromosome fold.
   **Apply after joint C/K/S/T coverage** and before probe fitting, using exactly
   the same selected rows for all feature sets in a run. Save every excluded locus
   and the per-chromosome/exposure balance table. Do not reuse the pre-feature
   matched preview as the input universe. Feature coverage gates still apply to
   the complete original P0 dataset *before* matching. Different fold/context
   coverage may change the eligible subset; it must remain audited.
2. **H3K27ac-only:** filter raw cached measurements to the exact assay
   `HM-ChIP-seq_H3K27ac`, then recompute supplied-call unions, informative experiment
   counts, donor/tissue counts and labels at each locus. A locus without a measured
   H3K27ac experiment is absent, not negative.
3. **CTCF-only:** the same procedure for `TF-ChIP-seq_CTCF`. A locus can be a primary
   P0 positive and an assay-specific negative if its significant measurements were
   in other assays; the tests explicitly verify this behavior.

The assays retain all accessible loci of that assay, without exposure matching.
These are three separate predefined analyses; do not silently add crossed analyses
or choose an assay-specific matching rule after inspecting model gains. Exposure
matching controls measurement **count**, not read depth or donor/tissue identity.
Those residual limitations must remain explicit. Counts are never added to C/S/T.

### Real-data preparation results

| Dataset | Measurements | Loci | Positive | Negative | Prevalence |
|---|---:|---:|---:|---:|---:|
| H3K27ac | 1,181,650 | 146,914 | 6,801 | 140,113 | 0.0462924 |
| CTCF | 1,075,083 | 119,426 | 7,756 | 111,670 | 0.0649440 |
| Exposure-matched preview | primary P0 | 50,654 | 25,327 | 25,327 | 0.5 |

The preview excludes 2,765 positives and 197,303 negatives. This is intentional
matching attrition, recorded separately from mapping/feature loss. It retains
90.2% of positives. The preview has not been fitted; final post-feature matched
counts are unknown. All assay-specific mapping, fitting, and topology gains remain
**not run**, blocked on the same original server resources.

### P0b metrics

Raw AP remains the primary endpoint. For each existing complexity stratum, the
analysis verifies identical C+S and C+S+T locus/label universes, then reports:

- raw AUPRC;
- AUROC;
- `normalized_ap = (AUPRC - positive_prevalence) / (1 - positive_prevalence)`.

Normalized AP has baseline 0 for constant scores and 1 for perfect ranking; it can
be negative. This rescales performance relative to prevalence, but does **not** make
AP fully prevalence invariant. All ranking metrics remain undefined for a
single-class stratum. Use within-stratum paired gains; do not interpret observed
label prevalence gradients as evidence that topology helps. Raw APs from the
50%-prevalence matched dataset are not directly comparable with primary P0 APs.

`complexity_metric_summary.csv` adds means, SDs and fold-then-seed CIs for both
feature sets and their paired difference for all three metrics. Existing
`complexity_summary.csv` and raw-AP figures remain backward-compatible. Counts,
prevalence and undefined-run counts accompany the estimates. Analyze each
sensitivity in its own output root; mixed subtask roots fail validation.

### Reproduction commands

Local preparation reads the existing Parquet cache; it does not reparse or
redownload the 745 MB TSV. Existing sensitivity outputs are protected against
overwriting; use a new versioned output directory for an intentional redefinition.

```bash
export PYTHONPATH="$PWD:$PWD/src"
python -m tasks.entex.sensitivity
```

On the original server, set GRAPH, MANIFEST, CK and NT as above, and use:

```bash
export MODELS=server_workspace/results/full_multicohort_server_20260806

# Uses the original P0 loci/mapping; matching happens only after feature coverage.
python -m tasks.entex.probe --sensitivity exposure_matched \
  --loci data/entex/v1/p0_loci.parquet --mapping-dir data/entex/v1/mapping \
  --full-segments "$GRAPH" --manifest "$MANIFEST" \
  --feature-cache "$CK" --sequence-cache "$NT" --results-root "$MODELS" \
  --out-root results/entex/v1/p0_exposure_matched --device cuda

# Assay labels and mapping are independently audited for each measured universe.
for assay in h3k27ac ctcf; do
  python -m tasks.entex.mapping \
    --loci "data/entex/v1/sensitivities/${assay}_loci.parquet" \
    --full-segments "$GRAPH" --out-dir "data/entex/v1/mapping_${assay}"
  python -m tasks.entex.probe --sensitivity "$assay" \
    --loci "data/entex/v1/sensitivities/${assay}_loci.parquet" \
    --mapping-dir "data/entex/v1/mapping_${assay}" \
    --full-segments "$GRAPH" --manifest "$MANIFEST" \
    --feature-cache "$CK" --sequence-cache "$NT" --results-root "$MODELS" \
    --out-root "results/entex/v1/p0_${assay}" --device cuda
done

for sensitivity in exposure_matched h3k27ac ctcf; do
  python -m tasks.entex.analyze \
    --probe-root "results/entex/v1/p0_${sensitivity}" \
    --complexity server_imports/complexity_context_v2_20260815/native_complexity_v2/complexity_features.tsv \
    --out-dir "results/entex/v1/p0_${sensitivity}_analysis"
done
```

For a smoke run, append `--folds fold_a --seeds 42 --contexts strict` to the probe
command and use a separate `_smoke` output root. Full sensitivity runs use the same
5 folds × 3 seeds × 2 contexts, frozen encoders, classifier and hyperparameters as
primary P0. Keep the primary and all three predefined sensitivity results in the
report regardless of sign or statistical significance. These sensitivity results
belong in the supplement, referenced alongside any eventual main-text P0 claim.

Sensitivity validation: 26 tests pass in the combined EN-TEx/existing-probe suite;
Ruff, compile and diff-whitespace checks pass. All four cached manuscript regression
comparisons remain unchanged. Real-data checks confirm exact class balance in all
3,739 retained chromosome/exposure strata. Every assay-positive locus is primary-P0
positive, and assay exposure never exceeds primary exposure. Complete compact QC,
fixed definitions, balance counts and test logs are in
`results/entex/v1/qc/sensitivities/`; processed sensitivity Parquets are in
`data/entex/v1/sensitivities/`.


## Frozen topology cache and server execution

Pass `--topology-cache-root results/entex/v1/topology_cache` to primary P0 and all
three sensitivities. Run primary P0 first for each fold/seed/context so its requested
segment superset is cached. This wrapper calls the unchanged manuscript extractor;
it never reads biological labels. Checkpoint, graph and manifest hashes, seed,
context, and canonical-conflict policy must match on reuse. Missing extracted nodes
remain missing for coverage accounting. A mismatched cache or a request beyond its
target universe fails. Cache writes are locked and array/sidecar files are replaced
atomically; a partially written cache fails rather than being silently reused.
The optional cache has synthetic tests for exact numerical reuse, subset coverage,
missing nodes and provenance mismatch. Large NPZ caches are not committed.

Verified server environment: Python 3.11.16, torch 2.6.0+cu124, pandas 3.0.5,
pyarrow 25.0.1, scikit-learn 1.9.0; four idle NVIDIA RTX A5000 GPUs at inspection.
The package versions describe the actual new run and are not asserted to equal the
August manuscript environment. Existing probe types/settings are unchanged.

The default tmux socket failed before executing any work. A separate socket
`tmux -L entex-20260915` successfully ran mapping and launched the smoke probe.
Server logs and exit files live under `results/entex/v1/server/` and the smoke
output root is `results/entex/v1/p0_smoke`. Inspect `smoke.exit`, `smoke.log` and the
per-run audit before launching full fitting. Never infer success from dispatch.


## Completed full primary P0 (15 September 2026)

All 30 fold/seed/context jobs completed successfully on the original server.
P0/P0b summaries and PNG/SVG figures are in `results/entex/v1/p0_analysis/`.
Strict paired topology gain: +0.000208, 95% CI [-0.000317, +0.000746].
One-hop: -0.000654, 95% CI [-0.002092, +0.000466]. Neither supports a
clear global improvement. Strict complexity-stratum intervals all include zero;
one-hop low/medium strata have negative intervals. These are descriptive subgroup
results, not multiplicity-adjusted discoveries. Sequence provides a positive gain
over C+T in both contexts. Preserve all predefined sensitivities regardless of sign.

## P1 registry resolution and prepared tissues

V3 failed ID coverage. ENCODE V2 file ENCFF924IMH matches 100% of both source
archives (5,646,598 active and 4,620,982 repressed rows). Its official metadata is
https://www.encodeproject.org/files/ENCFF924IMH/ and the pinned SHA256 is in config.
`tasks.entex.registry` handles V2 BED11 and V3 BED6 accession/class columns
explicitly. Compact coverage evidence is in `results/entex/v1/qc/registry_v2/`.

P1 requires registry dELS AND supplied distal state. Only explicitly active/repressed
rows are labeled. Conflicting states within locus/tissue are excluded and counted;
these exclusions do not imply an inferred biological state. Select five tissues by
largest smaller-class count (minimum 1,000/class), then total count and name, before
fitting. Separate tissue classifiers use the same chromosome folds and frozen probe.

| Tissue | Loci | Active | Repressed | Conflicting loci excluded |
|---|---:|---:|---:|---:|
| thyroid_gland | 260898 | 115611 | 145287 | 32569 |
| tibial_nerve | 233508 | 101438 | 132070 | 17605 |
| body_of_pancreas | 227552 | 100064 | 127488 | 25486 |
| gastroesophageal_sphincter | 205788 | 94745 | 111043 | 14565 |
| Peyers_patch | 234737 | 93921 | 140816 | 26773 |

Preparation completed; P1 fitting has not yet run. All 28 tissue counts and source
hashes are in `results/entex/v1/qc/p1/`. Reproduce preparation with:

```bash
PYTHONPATH=.:src python -m tasks.entex.registry --registry data/encode/legacy_v2/ENCFF924IMH.bed.gz --out-dir results/entex/v1/qc/registry_v2 --source-url https://www.encodeproject.org/files/ENCFF924IMH/
PYTHONPATH=.:src python -m tasks.entex.enhancer
```

For each selected tissue, map `data/entex/v1/p1/<tissue>_loci.parquet` using the
existing EN-TEx mapper. Use `run_entex_probe_matrix.py --task p1 --subtask <tissue>`
with its loci/mapping and the same manuscript resources. Set
`--cache-all-reference-targets --topology-cache-root results/entex/v1/topology_cache_all_reference`
to reuse frozen embeddings across tissues. Keep primary P0's existing cache intact.
After all selected tissues finish, `python -m tasks.entex.tissues --probe-root
results/entex/v1/p1 --preparation-audit data/entex/v1/p1/audit.json --complexity
<existing native v2 table> --out-dir results/entex/v1/p1_analysis` produces per-tissue
results and equally weighted tissue means within fold/seed, followed by the existing
hierarchical bootstrap. Missing tissue pairs fail validation.

Validation: 30 tests passed across EN-TEx and existing cCRE/SV probe/cache/aggregation
suites. P1 tests cover BED column differences, conflicting labels, sample-count
selection, and complete paired tissue coverage. Large data and caches stay out of Git.


## P2 accessible SNV assay preparation and probe adapter

The original high-confidence file remains RNA-only. The official full accessible
call set was downloaded to the server after checking storage (688 GB free), response
length and file presence. The HTTPS endpoint had a certificate error; the public
HTTP endpoint served the source. SHA256:
`e59a83a1595cbe12ba56714af79c297ac9f31b593e13ee44966c732d30d4daa8`.
Downloaded bytes: 2,675,757,003. This benchmark is explicitly the **default accessible
call set**, not the high-confidence subset. No optional whole-dataset modeling is
performed: parsing validates all rows but retains only the two requested assays.

Preparation scanned 22,310,439 measurements across 12 assays in 200,000-row chunks.
CTCF: 2,391,015 measurements / 596,650 loci, 83,911 positives (3.5094%).
H3K27ac: 3,168,525 / 713,418 loci, 74,833 positives (2.3618%). No duplicate
SNV/experiment rows were removed; conflicting records would fail. Supplied calls
provide labels; accessible nonsignificant observations are the negatives.

```bash
PYTHONPATH=.:src python -m tasks.entex.snv --source data/entex/sources/hetSNVs_default_AS.tsv
```

Map each `<assay>_loci.parquet` with the existing mapper. The probe requires one
containing segment and rejects ambiguous multi-segment SNVs. Run with `--task p2
--subtask <assay> --measurements data/entex/v1/p2/<assay>_measurements.parquet`, the
matching unique locus/mapping files, and the exact same graph/checkpoint/NT caches.
Use the all-reference topology cache from P1. Measurement occurrences remain
separate labels; every occurrence of a coordinate inherits its chromosome partition.

To avoid allocating repeated multi-GB feature matrices, `measurement_probe.py`
collapses identical training locus/label copies with occurrence weights. The scaler
uses original occurrence weights and logistic class weights use original row class
totals. Model type, C, solver, stopping settings, calibration and validation threshold
selection are reused unchanged. Validation/test metrics are over individual original
measurements; they are not a locus-level union-label task. Paired bootstrap remains
fold then seed, not an independent-row bootstrap. A numerical test compares raw
probabilities and scaler moments to the expanded original pipeline (absolute
probability tolerance 2e-5). Comparator checks use unique measurement IDs and accept
repeated loci. Source read counts, significance and assay metadata are retained as
provenance and are never classifier inputs.

34 tests pass across the EN-TEx/existing-probe suite. Actual P2 preparation QC is in
`results/entex/v1/qc/p2/preparation.json`; P2 fitting has not yet completed. The P1
single-fold thyroid smoke completed with 99.9992% joint feature coverage (2 loci
excluded); it is not a full tissue result. See `qc/p1_server_smoke/`.


P2 representation limitation: measurements at the same locus receive the same
C/S/T score, irrespective of donor or tissue. This tests transferable locus
representation against assay-specific measurement labels; it does not model
sample-specific allelic direction, donor genotype effects, or expression magnitude.
Read depth and AS-test p-values are not input features.

Exact password-free shell commands used on the original server are archived under
`results/entex/v1/qc/server_commands/`. Output directories are protected against
reuse; choose new output roots for reruns. The pinned P2 worktree can be recreated
from its recorded commit. Do not interpret partial run coverage snapshots as complete
matrices; completion requires 30 successful jobs and complete paired fold/seed keys.
