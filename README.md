# Graphgenome

**A Graph Genome Foundation Model for Pangenome Link Prediction and Structural Variation Discovery**

## Overview

This project builds a **graph attention network (GAT)** that learns structural representations of the [Human Pangenome Reference Consortium (HPRC)](https://humanpangenome.org/) pangenome graph. The core task is **link prediction** (predicting whether edges exist between DNA segments in a GFA-format variation graph) as a pre-training objective for downstream functional genomics and structural variation analysis.

The pangenome graph encodes genomic diversity across 47 ancestrally diverse human genomes as a variation graph, where linear (backbone) regions represent shared sequence and **bubble structures** represent sites of structural variation (insertions, deletions, inversions, duplications). By learning on the graph directly — rather than serializing it into a flat sequence — the model captures topological signals that linear genome models miss.

## Current Code Organization

The repository is moving to a simple dataset-directory convention:

- `data/<dataset>/full_segments.csv`
- `data/<dataset>/full_links.csv`
- optional derived folders such as `data/<dataset>/benchmark/` and
  `data/<dataset>/ccre/`

The clean repo uses a flat `src/` layout so collaborators can browse the code
without stepping through an extra package folder. The top-level `data/`
directory is reserved for real datasets, while Python dataset/parsing code lives
inside `src/data/`:

```text
src/
  graphgenomefm.py  # keeps `python -m graphgenomefm` working
  cli.py
  pipeline.py
  metadata.py
  graph/       # graph table IO, slicing, features, negative sampling
  models/      # DualStreamPangenomeGAT, GAT utilities, heads, losses
  training/    # foundation-model pretraining
  tasks/ccre/  # cCRE labeling, binary task, baselines, GAT classifier
  analysis/    # network and latent-space analysis utilities
  data/        # dataset layout, GFA parser, benchmark construction
  utils/       # run versioning and small helpers
data/
  hprc/
    full_segments.csv
    full_links.csv
```

Starting from HPRC:

```bash
pip install -e ".[training,test]"
python -m graphgenomefm check-data --data-dir data/hprc
python -m graphgenomefm make-benchmark --data-dir data/hprc --no-network-analysis --no-viz
python -m graphgenomefm pretrain --data-dir data/hprc --device cpu
```

Parse a raw GFA/rGFA graph into the table schema expected by the current
training code:

```bash
python -m graphgenomefm parse-gfa \
  --gfa /path/to/graph.gfa.gz \
  --out-dir data/hgsvc3
```

See `docs/RUN_HPRC.md` for the current step-by-step workflow and
`docs/PSB_EXPERIMENT_PLAN.md` for the paper-facing experiment plan.

### Key results

| Metric                          | Value                      | Notes                                                   |
| ------------------------------- | -------------------------- | ------------------------------------------------------- |
| Cross-chromosome strict AUC     | **0.980** (mean)           | Range 0.957–0.995 across held-out chromosomes           |
| Cross-chromosome 1-hop AUC      | **0.995** (mean)           | Range 0.991–0.997                                       |
| Baseline (LR, distance-matched) | 0.515 strict / 0.840 1-hop | Graph model provides +0.465 strict improvement          |
| Generalization gap              | ~0                         | Held-out chromosome AUC matches or exceeds training AUC |
| Model size                      | ~100K parameters           | Compact; no large-scale pre-training required           |

## Table of Contents

- [Background](#background)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Experimental History](#experimental-history)
- [Validated Results](#validated-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [References](#references)

## Background

### The pangenome graph

A single linear reference genome (GRCh38) cannot represent the full spectrum of human genetic variation. The HPRC pangenome graph addresses this by encoding 47 diverse human genomes as a **variation graph** in rGFA format:

- **Segments (S lines):** DNA sequences of variable length, each with a chromosome assignment (SN), genomic offset (SO), and reference status (SR = 0 for GRCh38 reference, SR > 0 for alt haplotypes)
- **Links (L lines):** Directed edges between segments, connecting the back end of one to the front end of another, with orientation (+/−) indicating whether the segment or its reverse complement is used
- **Bubbles:** Regions where multiple paths diverge from and reconverge to the reference backbone — these encode structural variants (insertions, deletions, inversions)

### Why graphs, not sequences?

Existing genomic foundation models (DNABERT, DNABERT-2, Nucleotide Transformer, DeepGene) either train on a single reference genome or serialize the pangenome into flat token sequences, discarding the graph topology. DeepGene (Zhang et al., 2024) takes the most advanced approach — serializing the Minigraph DAG and applying RoPE-based Transformers — but still flattens the graph structure.

Our approach preserves the graph and uses **Graph Attention Networks** to learn directly from the topology. This allows the model to:

1. See multiple haplotype paths simultaneously (not just one serialized traversal)
2. Learn from bubble structure (branching nodes, convergence points, path diversity)
3. Capture structural variation patterns that are invisible in linear representations

### Link prediction as pre-training

Link prediction — predicting whether an edge should exist between two nodes — serves as a self-supervised pre-training objective. The trained model produces **node embeddings** that encode both sequential position and structural context, which can then be transferred to downstream tasks (functional element prediction, SV detection, epigenetic mark classification).

## Architecture

### DualStreamPangenomeGAT

The model uses a **dual-stream** design that separates two types of information, following the insight that both positional and topological signals are needed for pangenome understanding:

```
Raw node features (7-dim: LN, SO, SR, degree, is_ref, branching_frac, ...)
                │
                ▼
         Node Encoder (MLP)
                │
     ┌──────────┴──────────┐
     │                     │
     ▼                     ▼
Stream A: Linear        Stream B: Graph
(RoPE self-attention)   (GAT on edges)
     │                     │
     │  Genomic offset     │  Edge topology
     │  → rotated Q,K      │  → neighbor aggregation
     │  → relative         │  → branching-aware
     │    distance          │    attention
     │                     │
     └──────────┬──────────┘
                ▼
          Fusion Gate
    g · h_linear + (1−g) · h_graph
    (learned per-node sigmoid)
                │
                ▼
         LayerNorm + Residual
                │
                ▼
       Edge Predictor (MLP)
    [h_u ∥ h_v] → P(edge exists)
```

**Stream A (Linear-Positional):** Self-attention with Rotary Position Embedding (RoPE) applied to the SO (segment offset) genomic coordinates. RoPE encodes absolute positions into Q and K vectors via rotation, but the resulting dot products depend only on the _difference_ between positions — giving the model relative positional encoding. We use multi-scale RoPE with 3 frequency bands (local ~100bp, structural ~10kb, chromosomal ~1Mb) and orientation-aware encoding that incorporates strand direction.

**Stream B (Graph-Topological):** Standard GAT attention operating on the actual graph edges. Captures branching structure, bubble topology, hub connectivity, and multi-path diversity. Uses adaptive attention windowing based on the local branching fraction.

**Fusion Gate:** A learned per-node sigmoid gate that decides how much to weight each stream. Nodes in linear backbone regions tend to weight Stream A; nodes in complex bubble regions tend to weight Stream B. Validated as essential — removing it eliminates strict-closure gains.

### Architectural choices (validated experimentally)

| Component            | Choice                               | Justification                                         |
| -------------------- | ------------------------------------ | ----------------------------------------------------- |
| Positional encoding  | Multi-scale RoPE (3 bands)           | +0.032 strict AUC over single-scale                   |
| Attention window     | Per-slice adaptive (branching-aware) | Best 1-hop on high-branching slices                   |
| Orientation encoding | Strand bit in RoPE rotation phase    | +0.005 strict, broad improvement                      |
| Stream fusion        | Learned sigmoid gate                 | Cross-attention fusion was detrimental (−0.08 strict) |
| Dimensions           | hidden_dim=48, n_heads=4, n_layers=2 | Finer positional resolution > more parameters         |
| Loss                 | Focal BCE (γ=2.0, α=0.25)            | Better calibration on hard negatives                  |
| Regularization       | DropEdge (rate=0.1)                  | Prevents topology memorization                        |

### What didn't work

| Experiment                             | Result                                  | Reason                                               |
| -------------------------------------- | --------------------------------------- | ---------------------------------------------------- |
| Cross-attention fusion                 | 1-hop collapse on high-branching slices | O(N²) attention too expensive                        |
| Population conditioning                | No strict gain                          | Insufficient population signal without full metadata |
| k-hop edge expansion                   | Large AUC drops on chr8, chr19          | Over-smoothing on already-branchy regions            |
| Bubble-focused training                | chr20 collapse                          | Filtered out informative linear slices               |
| Topology-derived attention temperature | No gain over flat                       | GAT self-regulates attention internally              |

## Data Pipeline

### Input data

| File                       | Description                                             | Size           |
| -------------------------- | ------------------------------------------------------- | -------------- |
| `data/hprc/full_segments.csv` | All segments from HPRC Minigraph rGFA                | ~480K segments |
| `data/hprc/full_links.csv`    | All links (edges) between segments                   | ~548K links    |

13,717 unique SN (sequence name) values exist in the data: 24 GRCh38 reference chromosomes plus thousands of sample-specific alt-haplotype contigs.

### Benchmark construction

Benchmarks are constructed by slicing the pangenome graph into 50kb genomic windows:

**benchmark_v3** (current):

- 240 slices total: 10 per chromosome × 24 chromosomes × 2 closure types
- **Strict closure:** Only GRCh38 backbone edges (pure reference topology)
- **1-hop closure:** Backbone + all edges to/from immediate alt-haplotype neighbors (captures bubble structure)
- Windows are randomly sampled and may overlap (60–99% within-chromosome overlap identified)
- Cross-chromosome held-out validation eliminates overlap leakage

**benchmark_v4** (in progress):

- Non-overlapping tiled windows to eliminate within-chromosome leakage
- Generator script: `scripts/09_make_benchmark_v4.py`

### Negative sampling

For each positive edge (real link in the graph), a corresponding negative edge is generated:

- **1-hop negatives:** Random node pairs from the same 1-hop subgraph that are not connected — tests whether the model understands local topology
- **Strict negatives:** Distance-matched node pairs from the reference backbone — controls for genomic coordinate proximity, testing genuine structural signal

The strict negatives are the scientifically meaningful evaluation: they remove the "coordinate shortcut" that inflates simpler baselines. Overall AUC (averaging strict and 1-hop) is dominated by the easier 1-hop task and is misleading; **strict AUC is the primary reported metric.**

### Pipeline scripts

```
bootstrap.py           →  Auto-detect data files, build benchmark slices + manifest
05_network_analysis.py →  Compute graph statistics per slice (branching_frac, degree dist, etc.)
03_hardneg_edgepred*.py → Logistic regression baseline with distance-matched negatives
06_gat_train.py        →  Per-slice GAT training (original, 240 independent models)
06b_shared_train.py    →  Cross-slice shared training (breakthrough paradigm)
07_ensemble.py         →  GAT+LR ensemble
08_multiseed_ensemble.py → Multi-seed robustness analysis
09_make_benchmark_v4.py →  Non-overlapping tiled benchmark generator
```

## Experimental History

The project progressed through several phases, each building on the previous findings. This section documents the full arc from initial baseline through the validated final result.

### Phase 1: Baseline establishment

**Logistic Regression (LR) baseline** using hand-crafted features: segment length (LN), offset (SO), reference status (SR), degree, delta-SO (coordinate distance between endpoints), orientation match.

| Metric     | LR (random neg) | LR (distance-matched) |
| ---------- | --------------- | --------------------- |
| 1-hop AUC  | 0.711           | 0.840                 |
| Strict AUC | 0.970           | 0.515                 |

The LR strict score collapses from 0.970 to 0.515 when distance-matched negatives are used — proving the high score was entirely due to the delta-SO coordinate shortcut, not genuine structural learning.

**Initial GAT (per-slice, graph-only):** 1-hop AUC = 0.912, strict AUC = 0.500. The GAT learns meaningful topology for 1-hop but cannot solve strict because strict graphs are nearly linear with no structural diversity to exploit.

### Phase 2: Dual-stream architecture (per-slice)

Following the professor's advice to add positional encoding (referencing DeepGene's RoPE approach), the dual-stream architecture was developed and tested in several ablation rounds:

**Round 1 — Stream design:**

- Dual + RoPE + gate: strict +0.013 over baseline
- Dual + sinusoidal + gate: strict +0.016 (sinusoidal was sufficient for 50kb windows)
- Dual + RoPE + no gate (additive): strict gains collapsed — fusion gate is essential
- All variants regressed on 1-hop (−0.018) due to O(N²) linear attention on large graphs

**Rounds A–C — Component refinement:**

- Adaptive attention window (branching-aware): best 1-hop on high-branching slices
- Multi-scale RoPE (3 frequency bands): +0.032 strict over single-scale
- Orientation-aware RoPE: +0.005 strict, broad improvement
- Cross-attention fusion: abandoned (−0.08 strict, 1-hop collapse)
- k-hop expansion: abandoned (large regressions on chr8, chr19)

**Rounds D1–D2 — Best per-slice configuration:**

- D2 (full dual-stream, all features): strict AUC = 0.592
- 48 slices training 100+ epochs achieved 0.893 strict
- 124 slices stopping at ≤35 epochs achieved 0.513 (chance)
- Correlation: epochs_run ↔ test_auc = **r = 0.849**

### Phase 3: Diagnosis (data starvation)

The diagnostic analysis revealed the fundamental bottleneck was **not architecture but data volume**:

| Epoch bin | N slices | Strict AUC | Interpretation                  |
| --------- | -------- | ---------- | ------------------------------- |
| ≤26       | 53       | 0.521      | At chance — insufficient signal |
| 27–35     | 71       | 0.507      | At chance                       |
| 36–75     | 64       | 0.514      | At chance                       |
| 76–100    | 4        | 0.676      | Starting to learn               |
| 101–200   | 48       | **0.893**  | Near 1-hop performance          |

Each slice has only ~190 positive training edges — far too few for the model to learn. The architecture was sound; the training paradigm was the bottleneck.

### Phase 4: Shared training

**Solution:** Instead of training 240 independent models on ~190 edges each, train **one shared model** across all slices simultaneously (~45,000 edges total).

**Script:** `scripts/06b_shared_train.py` with warmup + cosine LR schedule, gradient accumulation, per-slice adaptive window_k, and `--test_chrs` for cross-chromosome held-out validation.

### Phase 5: Leakage identification and control

Within benchmark_v3, slices on the same chromosome overlap by 60–99%. Shared training on overlapping slices conflates two signals: genuine generalization and memorization of overlapping regions.

**Solution:** Cross-chromosome held-out validation — train on 23 chromosomes, hold out 1, evaluate on the held-out. Zero overlap possible between training and test data.

## Validated Results

### Cross-chromosome held-out strict AUC

| Held-out chromosome    | Strict AUC | 1-hop AUC | N held-out slices |
| ---------------------- | ---------- | --------- | ----------------- |
| chr1                   | 0.957      | 0.991     | 10                |
| chr8                   | 0.983      | 0.997     | 10                |
| chr16                  | 0.978      | 0.994     | 10                |
| chr19                  | 0.995      | 0.997     | 10                |
| chrY                   | 0.976      | 0.993     | 10                |
| **Mean**               | **0.980**  | **0.995** |                   |
| Multi (chr1+chr8+chrY) | 0.986      | —         | 30                |

All 50 held-out slices exceed 0.93. The generalization gap (held-out minus train-chr AUC) ranges from −0.007 to +0.009 — effectively zero, confirming the model learns generalizable pangenome structure rather than memorizing local topology.

### Performance progression summary

| Stage                                    | Strict AUC | What changed                               |
| ---------------------------------------- | ---------- | ------------------------------------------ |
| LR baseline (distance-matched)           | 0.515      | Coordinate shortcut removed                |
| GAT per-slice (graph-only)               | 0.500      | No positional signal for strict            |
| Dual-stream per-slice (D2)               | 0.592      | Positional + topological, but data-starved |
| **Shared training (cross-chr held-out)** | **0.980**  | One model, all slices, 45K edges           |

### Key findings

1. **Data starvation was the fundamental bottleneck.** Per-slice models with ~190 edges each were underpowered. Shared training across all slices was the paradigm shift.
2. **The dual-stream architecture provides essential inductive bias.** The graph stream captures topology; the linear stream captures position. The fusion gate routes information per-node. Removing any component degrades performance.
3. **Strict closure is the scientifically meaningful metric.** Overall AUC is dominated by easier 1-hop slices. Strict closure with distance-matched negatives isolates genuine structural learning from coordinate shortcuts.
4. **Cross-chromosome generalization is strong.** Test AUC matches or exceeds training AUC — the model learns universal pangenome patterns, not chromosome-specific memorization.
5. **Flat attention temperature outperforms all topology-derived schedules.** The GAT self-regulates attention internally; external rules add no value.

## Project Structure

```
graphgenome-fm/
├── src/
│   ├── graphgenomefm.py              # Stable `python -m graphgenomefm` command
│   ├── cli.py                        # Main command-line interface
│   ├── pipeline.py                   # Runs migrated modules behind the CLI
│   ├── metadata.py                   # Experiment metadata helpers
│   ├── data/                         # Dataset layout, GFA parsing, benchmark build
│   ├── graph/                        # Graph IO, slicing, features, negative sampling
│   ├── models/                       # DualStreamPangenomeGAT, heads, losses
│   ├── training/                     # Shared foundation-model pretraining
│   ├── tasks/ccre/                   # cCRE labeling, binary task, baselines, GAT
│   ├── evaluation/                   # Metrics and split helpers
│   ├── analysis/                     # Network and latent-space analysis
│   └── utils/                        # Run versioning and small helpers
│
├── data/
│   ├── hprc/full_segments.csv        # Not committed
│   ├── hprc/full_links.csv           # Not committed
│   └── encode/GRCh38-human-cCREs.bed # Not committed
│
├── results/                          # Not committed
├── docs/
├── tests/
├── pyproject.toml
└── README.md
```

## Installation

### Requirements

- Python 3.10+
- PyTorch (CPU or CUDA)
- pandas, numpy, scikit-learn

**No graph library abstraction** (e.g., PyG, DGL) is used — the GAT is implemented in pure PyTorch for full control over the attention mechanism and positional encoding.

### Setup

```bash
git clone https://github.com/<org>/graphgenome-fm.git
cd graphgenome-fm
git checkout dev-ver1

# Create environment
conda create -n shilab python=3.10 pytorch pandas numpy scikit-learn -c pytorch
conda activate shilab
pip install -e ".[training,test]"

# Verify imports
python -c "from models.dual_stream_gat import DualStreamPangenomeGAT; print('OK')"
python -c "from models.losses import focal_bce_loss; print('OK')"
python -m graphgenomefm --help
```

### Data

Place HPRC data files in `data/hprc/`:

```bash
data/
├── hprc/
│   ├── full_segments.csv
│   └── full_links.csv
└── encode/
    └── GRCh38-human-cCREs.bed
```

## Usage

### Step 1: Check HPRC data

```bash
python -m graphgenomefm check-data --data-dir data/hprc
```

### Step 2: Build benchmark windows

```bash
python -m graphgenomefm make-benchmark \
  --data-dir data/hprc \
  --out-dir data/hprc/benchmark \
  --n-windows 10 \
  --window-bp 50000 \
  --no-network-analysis \
  --no-viz
```

### Step 3: Shared foundation-model pretraining

```bash
python -m graphgenomefm pretrain \
  --data-dir data/hprc \
  --benchmark-dir data/hprc/benchmark \
  --out-dir results/hprc/pretrain \
  --test-chrs chr1 chr8 chr16 chr19 chrY \
  --epochs 100 \
  --patience 20 \
  --device cpu
```

Use `--device mps` on Apple Silicon or `--device cuda` on a CUDA server.

### Step 4: Label ENCODE cCREs

```bash
python -m graphgenomefm label-ccre \
  --data-dir data/hprc \
  --encode-bed data/encode/GRCh38-human-cCREs.bed \
  --out-dir data/hprc/ccre
```

### Step 5: Run cCRE tasks

```bash
python -m graphgenomefm ccre-binary-baseline \
  --data-dir data/hprc \
  --test-chrs chr8 chr19 chr22 \
  --val-chr chr16 \
  --out-dir results/hprc/ccre_binary_baseline
```

```bash
python -m graphgenomefm ccre-gat \
  --data-dir data/hprc \
  --benchmark-dir data/hprc/benchmark \
  --test-chrs chr8 chr19 chr22 \
  --val-chrs chr16 \
  --out-dir results/hprc/ccre_gat \
  --epochs 60 \
  --patience 15 \
  --device cpu
```

## Roadmap

### Completed

- [x] Data pipeline: rGFA parsing → benchmark slicing → manifest generation
- [x] LR baseline with distance-matched negative sampling
- [x] Graph-only GAT (per-slice)
- [x] Dual-stream architecture (RoPE + GAT + fusion gate)
- [x] Multi-scale RoPE, orientation-aware RoPE, adaptive attention window
- [x] Focal loss, DropEdge regularization
- [x] Cross-slice shared training paradigm
- [x] Cross-chromosome held-out validation (5 chromosomes + multi-chr)
- [x] Data starvation diagnosis (r=0.849 correlation)
- [x] Within-chromosome overlap analysis (60–99%)
- [x] Benchmark v4 generator (non-overlapping tiled windows)
- [x] Multi-seed ensemble script
- [x] Full project report

### Near-term (in progress)

- [ ] Run benchmark_v4 (non-overlapping tiled windows)
- [ ] Full 24-chromosome leave-one-out cross-validation
- [ ] Node embedding extraction script (`11_extract_embeddings.py`)
- [ ] ENCODE cCRE data pipeline — map functional elements to pangenome graph
- [ ] ENCODE functional element classifier (graph embeddings vs linear baselines)

### Medium-term

- [ ] SV anomaly detection — score edges by predicted probability, overlap with HPRC VCF ground truth
- [ ] SV complexity stratification (simple deletions → complex nested events)
- [ ] NOMAD SV cross-reference (contact Shi Fan)
- [ ] DAZ region methylation check (contact Mark Lopez / Kyle)
- [ ] DeepGene GUE benchmark integration (Epigenetic Marks Prediction, Promoter Detection)
- [ ] Graph vs linear vs hybrid ablation table for downstream tasks

### Long-term

- [ ] Foundation model pre-training (masked node prediction on full pangenome graph)
- [ ] Population stratification experiments (AFR/EUR/EAS via `--pop_table`)
- [ ] Synthetic bubble experiments if a new generator is added under `src/analysis/`
- [ ] Multi-species pangenome extension
- [ ] Paper submission (target: PSB)

## Key Design Decisions

### Why pure PyTorch (no PyG/DGL)?

The GAT implementation uses raw PyTorch tensor operations rather than graph library abstractions. This gives full control over the attention mechanism, positional encoding injection, and fusion gate — all of which required custom modifications that would be awkward to implement through a library API.

### Why cross-slice shared training?

Per-slice training creates 240 independent models, each with only ~190 training edges. The correlation between training duration and test performance (r=0.849) proved this was a data volume problem, not an architecture problem. Shared training pools all ~45,000 edges into one model, resolving the starvation.

### Why strict closure as primary metric?

1-hop closure includes immediate neighbors, making the task partly solvable by coordinate proximity alone. Strict closure uses distance-matched negatives that eliminate this shortcut. The LR baseline proves the point: it scores 0.970 on strict with random negatives but collapses to 0.515 with distance-matched negatives.

### Why not serialize the graph (like DeepGene)?

DeepGene serializes the pangenome DAG into token sequences and feeds them to a Transformer. This loses the graph topology — the model sees only one linearized traversal, not the full multi-path structure. Our GAT preserves the graph and attends over actual edges, giving it access to bubble structure, branching patterns, and path diversity simultaneously.

### Population lookup design

Population metadata should be loaded at runtime from a user-supplied TSV/CSV via `--pop_table`, not hard-coded. This keeps the code portable across HPRC versions and other cohorts.

## Output conventions

- All experiment outputs go to `results/`
- Each run is auto-versioned into `run_NNN/` subdirectories via `utils.versioning`
- CSV filenames self-document active experiment flags (e.g., `strict_results__shared_dual_mscale3_orient_adpwk32a4_focal2_0_dedge0_1_heldout_chr1_ep100_pat20.csv`)
- Main commands use `python -m graphgenomefm ...`

## References

- **DeepGene** (Zhang et al., 2024): Pan-genome graph Transformer with RoPE. Our RoPE implementation draws from their positional encoding approach, but we preserve graph topology rather than serializing. [bioRxiv 2024.04.24.590879](https://doi.org/10.1101/2024.04.24.590879)
- **HPRC draft pangenome** (Liao et al., 2023): The pangenome reference used as input data. [Nature 617, 312–324](https://doi.org/10.1038/s41586-023-05896-x)
- **Minigraph** (Li et al., 2020): Construction method for the rGFA pangenome graph. [Genome Biology 21, 265](https://doi.org/10.1186/s13059-020-02168-z)
- **RoPE / RoFormer** (Su et al., 2024): Rotary position embedding theory. [Neurocomputing 568, 127063](https://doi.org/10.1016/j.neurocomp.2023.127063)
- **DNABERT-2** (Zhou et al., 2023): Multi-species genome foundation model and GUE benchmark. [arXiv:2306.15006](https://arxiv.org/abs/2306.15006)
- **Nucleotide Transformer** (Dalla-Torre et al., 2023): Large-scale genomic foundation model. [bioRxiv 2023.01.11.523679](https://doi.org/10.1101/2023.01.11.523679)
