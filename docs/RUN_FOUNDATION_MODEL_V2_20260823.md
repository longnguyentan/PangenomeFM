# PangenomeFM foundation-model roadmap: implementation and server runbook

## Scope

This implementation preserves PangenomeFM v1 and adds a separate v2 pilot
path. It does not relabel experiments that require unavailable data as
completed.

Implemented now:

- joint topology plus raw-sequence-token or frozen-sequence-profile encoding;
- coordinate, population, and haplotype conditioning;
- modality dropout and interpretable fusion weights;
- ordered path pooling;
- edge, masked-token, path-order, branch-choice, and cross-graph objectives;
- an audited NPZ pilot builder;
- a same-example frozen-embedding adapter for DeepGene, Nucleotide
  Transformer, and other sequence models;
- a restart-safe roadmap planner/runner with CPU/GPU/expense/external-data
  gates; and
- focused tests and a deterministic synthetic end-to-end smoke test.

Reused existing implementation:

- canonical reverse-complement candidate identity and masking;
- rotating chromosome folds and shared/cross-graph training;
- graph complexity and external-locus infrastructure;
- cCRE and SV frozen probes;
- path indexing, donor splits, path materialization, and branch-choice data;
- methylation, ASE, cross-graph alignment, QTL/GWAS, calibration, and capacity
  workflows; and
- GPU matrices, postprocessing, packaging, and provenance capture.

Data-gated rather than falsely marked complete:

- a donor-excluded HPRC graph;
- official same-sample PangenomeX CNV evaluation;
- additional donor/chromosome methylation and long-context sequence profiles;
- phased ASE labels;
- accessibility, histone/CTCF, 3D, molecular-QTL, phenotype, and clinical
  labels; and
- new population pangenomes.

## 1. Server checkout and environment

Use the repository environment rather than installing comparator dependencies
into it.

```bash
cd ~/PangenomeFM
git status --short
git pull --ff-only

conda env update -n PangenomeFM -f environment-server.yml --prune
conda activate PangenomeFM

export PYTHONPATH="$PWD/src:$PWD"
export MPLCONFIGDIR="$PWD/server_workspace/matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"
```

Keep DeepGene and PangenomeX in separate environments because their published
dependency stacks differ from PangenomeFM.

Define high-capacity locations:

```bash
export REPO_ROOT="$HOME/PangenomeFM"
export PANGENOMEFM_DATA_ROOT="$REPO_ROOT/server_workspace/data"
export PANGENOMEFM_RESULTS_ROOT="$REPO_ROOT/server_workspace/results"
export CUDA_GPUS=0,1,2,3
```

## 2. Inspect the complete roadmap without running it

```bash
PYTHONPATH=src:. python scripts/server/run_foundation_model_roadmap.py \
  --plan \
  --result-root "$PANGENOMEFM_RESULTS_ROOT/foundation_model_v2_roadmap_20260823"
```

The generated `execution_plan.json` classifies every task as `ready`,
`completed`, `blocked`, or `designed_data_gated`. A data-gated row is an honest
research prerequisite, not a software failure.

## 3. Run correctness tests and the synthetic v2 smoke test

The launcher defaults to the inexpensive `00_preflight` stage:

```bash
TMUX_SESSION=pangenomefm-foundation-v2 \
  bash scripts/server/launch_foundation_model_roadmap.sh
```

Monitor it with:

```bash
tmux attach -t pangenomefm-foundation-v2
```

or:

```bash
tail -f \
  "$PANGENOMEFM_RESULTS_ROOT/foundation_model_v2_roadmap_20260823/roadmap.log"
```

Direct equivalent:

```bash
PYTHONPATH=src:. python scripts/server/run_foundation_model_roadmap.py \
  --execute --stage 00_preflight \
  --result-root "$PANGENOMEFM_RESULTS_ROOT/foundation_model_v2_roadmap_20260823"
```

The smoke result is a software test and has
`promotable_to_primary_result=false`.

## 4. Freeze PangenomeFM v1 with the corrected data contract

This is the expensive canonical all-chromosome rerun. Verify storage and the
server plan before launching it.

```bash
export PANGENOMEFM_CONFIG="$PWD/configs/server_full_multicohort_20260806.json"
export DOWNLOAD_PROFILE=analysis-full
export DOWNLOAD_JOBS=2
export PREP_JOBS=2
export CONTEXT_JOBS=2
export CUDA_GPUS=0,1,2,3

TMUX_SESSION=pangenomefm-v1-canonical \
  bash scripts/server/launch_full_tmux.sh
```

Resume by running the same underlying pipeline after inspecting its completion
state:

```bash
bash scripts/server/run_full_server_pipeline.sh
```

Do not delete historical runs. The versioning utilities preserve new runs in
new directories.

## 5. Build the canonical same-example benchmark

Point to the canonical regenerated benchmark rather than the historical
candidate source:

```bash
export PFM_BENCHMARK_MANIFEST="$PANGENOMEFM_DATA_ROOT/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv"
export PFM_FULL_SEGMENTS="$PANGENOMEFM_DATA_ROOT/processed/hprc_r2_sv/full_segments.csv.gz"

PYTHONPATH=src:. python scripts/build_comparative_benchmark_pilot.py \
  --manifest "$PFM_BENCHMARK_MANIFEST" \
  --full-segments "$PFM_FULL_SEGMENTS" \
  --chromosome chr22 \
  --fold fold_b \
  --contexts strict 1hop \
  --out-dir "$PANGENOMEFM_RESULTS_ROOT/foundation_model_v2_roadmap_20260823/same_example_benchmark" \
  --overwrite
```

Inspect these outputs before training any adapted comparator:

```bash
python -m json.tool \
  "$PANGENOMEFM_RESULTS_ROOT/foundation_model_v2_roadmap_20260823/same_example_benchmark/benchmark_audit.json"
```

Do not use `--require-exact-all`: DeepGene is approximate and PangenomeX is
contextual for this task by design.

## 6. Run frozen sequence-model adapters on identical examples

Create one NPZ per frozen model with:

```text
ids          Unicode/string array [nodes]
embeddings   float array [nodes, dimensions]
```

Then run:

```bash
export SHARED_BENCHMARK="$PANGENOMEFM_RESULTS_ROOT/foundation_model_v2_roadmap_20260823/same_example_benchmark/benchmark_manifest.csv"
export DEEPGENE_ENDPOINT_EMBEDDINGS=/path/to/deepgene_endpoint_embeddings.npz
export NT_ENDPOINT_EMBEDDINGS=/path/to/nucleotide_transformer_endpoint_embeddings.npz

PYTHONPATH=src:. python -m evaluation.paired_embedding_baseline \
  --manifest "$SHARED_BENCHMARK" \
  --embeddings "$DEEPGENE_ENDPOINT_EMBEDDINGS" \
  --method-name DeepGene-adapted \
  --out-dir "$PANGENOMEFM_RESULTS_ROOT/foundation_model_v2_roadmap_20260823/deepgene_adapted_edge"

PYTHONPATH=src:. python -m evaluation.paired_embedding_baseline \
  --manifest "$SHARED_BENCHMARK" \
  --embeddings "$NT_ENDPOINT_EMBEDDINGS" \
  --method-name NucleotideTransformer-frozen \
  --out-dir "$PANGENOMEFM_RESULTS_ROOT/foundation_model_v2_roadmap_20260823/nt_adapted_edge"
```

The adapter selects regularization on validation examples, refits on
train-plus-validation, and preserves test examples for one final evaluation.
Its output explicitly states that it is not a published-task reproduction.

## 7. Prepare donor/path-resolved examples

Index paths and construct donor-disjoint splits using the existing server
tools. Example:

```bash
export PFM_PATH_METADATA="$PANGENOMEFM_DATA_ROOT/processed/hprc_r2_paths/path_metadata.tsv"

PYTHONPATH=src:. python scripts/server/build_path_resolved_donor_splits.py \
  --path-metadata "$PFM_PATH_METADATA" \
  --dataset hprc_r2 \
  --out-dir "$PANGENOMEFM_RESULTS_ROOT/foundation_model_v2_roadmap_20260823/donor_path_splits" \
  --seed 20260823
```

After path-positive transitions have been materialized:

```bash
export PFM_PATH_EDGE_TABLE="$PANGENOMEFM_RESULTS_ROOT/path_examples_20260809/hprc_r2_all/positive_path_edges.csv.gz"
export PFM_PATH_GFA="$PANGENOMEFM_DATA_ROOT/raw/hprc/v2.0/hprc-v2.0-mc-grch38.gfa.gz"
export PFM_PATH_GRAPH_AUDIT="$PANGENOMEFM_DATA_ROOT/processed/hprc_r2_full_resolution/segment_index_audit.json"

PYTHONPATH=src:. python scripts/server/build_path_branch_choice_examples.py \
  --positive-edges "$PFM_PATH_EDGE_TABLE" \
  --gfa "$PFM_PATH_GFA" \
  --graph-audit "$PFM_PATH_GRAPH_AUDIT" \
  --alternatives-per-positive 5 \
  --out-dir "$PANGENOMEFM_RESULTS_ROOT/foundation_model_v2_roadmap_20260823/path_branch_choices"
```

These negatives mean “not the observed next branch for this focal traversal.”
They must never be described as globally absent biological edges.

## 8. Assemble a real multimodal v2 pilot

Required frozen topology NPZ:

```text
ids          stable node IDs
embeddings   PangenomeFM v1 topology profiles
```

Required raw sequence-token NPZ:

```text
ids       the same stable ID namespace
tokens    0=PAD, 1=A, 2=C, 3=G, 4=T, 5=N, 6=MASK
```

The edge table must contain stable endpoint IDs, binary labels, and frozen
`train`, `validation`, or `test` assignments. Optional tables add coordinates,
population, haplotype, ordered paths, branch choices, and cross-graph anchors.

```bash
export PFM_V1_TOPOLOGY_NPZ=/path/to/topology_embeddings.npz
export PFM_SEQUENCE_TOKENS_NPZ=/path/to/sequence_tokens.npz
export PFM_V2_EDGE_TABLE="$SHARED_BENCHMARK"
export PFM_V2_NODE_TABLE=/path/to/node_metadata.tsv
export PFM_V2_PATH_MEMBERSHIP=/path/to/path_membership.tsv.gz
export PFM_CROSS_GRAPH_ANCHORS=/path/to/hprc_hgsvc_same_locus_anchors.tsv
export PFM_BRANCH_TABLE="$PANGENOMEFM_RESULTS_ROOT/foundation_model_v2_roadmap_20260823/path_branch_choices/branch_choice_candidates.csv.gz"
export PFM_V2_PILOT_DIR="$PANGENOMEFM_RESULTS_ROOT/foundation_model_v2_roadmap_20260823/v2_pilot_input"

PYTHONPATH=src:. python scripts/build_multimodal_pilot_npz.py \
  --topology-npz "$PFM_V1_TOPOLOGY_NPZ" \
  --sequence-npz "$PFM_SEQUENCE_TOKENS_NPZ" \
  --sequence-key tokens \
  --edge-table "$PFM_V2_EDGE_TABLE" \
  --node-table "$PFM_V2_NODE_TABLE" \
  --node-id-column node_id \
  --coordinate-columns coordinate_start coordinate_end segment_length orientation_code \
  --population-column superpopulation \
  --haplotype-column haplotype \
  --node-split-column split \
  --path-membership "$PFM_V2_PATH_MEMBERSHIP" \
  --path-id-column path_id \
  --path-node-column node_id \
  --path-position-column path_position \
  --path-split-column split \
  --branch-table "$PFM_BRANCH_TABLE" \
  --cross-graph-anchors "$PFM_CROSS_GRAPH_ANCHORS" \
  --out-dir "$PFM_V2_PILOT_DIR" \
  --require-complete
```

Column names are explicit parameters. Change them to the audited source schema
instead of renaming data silently.

## 9. Train the real multimodal pilot

```bash
export PFM_V2_PILOT_NPZ="$PFM_V2_PILOT_DIR/multimodal_pilot.npz"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. \
python -m training.multimodal_pretrain \
  --input-npz "$PFM_V2_PILOT_NPZ" \
  --out-dir "$PANGENOMEFM_RESULTS_ROOT/foundation_model_v2_roadmap_20260823/v2_real_pilot" \
  --device cuda \
  --hidden-dim 128 \
  --n-heads 8 \
  --sequence-layers 4 \
  --path-layers 2 \
  --epochs 100 \
  --patience 20 \
  --edge-weight 1.0 \
  --masked-token-weight 1.0 \
  --path-order-weight 0.25 \
  --branch-choice-weight 0.5 \
  --cross-graph-weight 0.1
```

This full-batch trainer is the objective-interaction pilot. Do not use it for a
whole-genome scale claim. Promotion requires connection to the existing lazy,
chromosome-scale loaders plus donor-excluded graphs.

## 10. Run selected roadmap stages through the planner

After setting the variables required by those stages:

```bash
PYTHONPATH=src:. python scripts/server/run_foundation_model_roadmap.py \
  --execute \
  --stage 20_benchmark \
  --allow-external \
  --result-root "$PANGENOMEFM_RESULTS_ROOT/foundation_model_v2_roadmap_20260823"
```

GPU pilot:

```bash
PYTHONPATH=src:. python scripts/server/run_foundation_model_roadmap.py \
  --execute \
  --task v2_real_pilot \
  --allow-gpu --allow-expensive \
  --result-root "$PANGENOMEFM_RESULTS_ROOT/foundation_model_v2_roadmap_20260823"
```

Run all currently ready tasks only after inspecting the plan:

```bash
RUN_READY_STAGES=1 ALLOW_GPU=1 ALLOW_EXPENSIVE=1 ALLOW_EXTERNAL=1 \
TMUX_SESSION=pangenomefm-foundation-v2-all-ready \
bash scripts/server/launch_foundation_model_roadmap.sh
```

Blocked tasks remain recorded and are skipped; the runner does not invent
inputs to satisfy them.

## 11. Methylation and ASE continuation

The existing methylation pipeline remains the reproducible entry point:

```bash
export GRAPHGENOMEFM_PYTHON="$(command -v python)"
export GRAPHGENOMEFM_PUBLIC_DATA_ROOT="$PANGENOMEFM_DATA_ROOT/public"
export GRAPHGENOMEFM_RESULT_ROOT="$PANGENOMEFM_RESULTS_ROOT/haplotype_methylation_chr8_v2_repeat"
export GRAPHGENOMEFM_CHECKPOINT=/path/to/frozen_pangenomefm_checkpoint.pt
bash scripts/run_chr8_haplotype_cohort.sh
```

Do not replace the locked gate thresholds. The next run must add the
validation-only shrinkage/rank-aware model, long-context sequence comparator,
and independent chromosomes/donors before expanding assays.

For a prepared ASE pair table and path embeddings:

```bash
PYTHONPATH=src:. python -m tasks.haplotype.ase \
  --pairs /path/to/phased_ase_pairs.csv.gz \
  --embeddings /path/to/v2_path_embeddings.npz \
  --out-dir "$PANGENOMEFM_RESULTS_ROOT/foundation_model_v2_roadmap_20260823/ase_heldout_donor" \
  --n-splits 5 \
  --seed 20260823
```

## 12. PangenomeX task-faithful evaluation

PangenomeX cannot consume the endpoint-edge benchmark. Create a separate
same-sample CNV study containing:

- shallow-WGS BAMs;
- CNV truth VCFs;
- reference FASTA and chromosome lengths;
- the required phylogeny; and
- PangenomeFM locus embeddings mapped to the same CNV candidates.

Run the official PangenomeX repository in its own pinned environment, then
compare:

```text
official PangenomeX
official inputs + PangenomeFM locus profiles
official inputs + matched sequence profiles
official inputs + sequence + PangenomeFM profiles
```

No executable official command is placed in the roadmap until the exact cohort
and external checkout pass their own license, checksum, and entry-point audit.
This prevents an apparently successful but scientifically invalid edge-to-CNV
conversion.

## 13. Promotion gates

A v2 result can enter the main manuscript only when all relevant gates pass:

1. no chromosome, donor, path, example-ID, or reverse-complement leakage;
2. trainable sequence, topology-only, and joint variants use identical units;
3. validation selects settings and test is touched once;
4. complete donor/population groups are held out;
5. at least three seeds and all prespecified folds finish;
6. frozen probes improve more than one independent biological task;
7. calibration and negative results are retained;
8. runtime, memory, environment, manifest hashes, checkpoints, and predictions
   are archived; and
9. external-model results are labeled exact, adapted, or contextual.

The target milestone is not the synthetic smoke test. It is a jointly
pretrained sequence-topology-path model that improves frozen downstream probes
and transfers to genuinely unseen donors.
