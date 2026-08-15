# PangenomeFM next-stage Phase 0 audit

Audit date: 2026-08-14
Branch inspected: `dev-exp`
Evidence policy: `VERIFIED COMPLETE` requires inspectable local source outputs. A terminal ledger without copied outputs is `COMPLETE BUT NOT VERIFIED`.

## Executive decision

The current topology, transfer, and release result bundle is mature enough to preserve and consolidate; it should not be rerun by default. The requested three-way edge-reconstruction comparison is **not an exact same-task benchmark**:

- PangenomeFM natively predicts masked oriented graph relations.
- DeepGene is a pan-genome-serialized sequence Transformer pretrained with masked language modeling and fine-tuned for sequence classification. It can receive approximately matched endpoint sequences only through a new downstream adapter; that is not its published task.
- PangenomeX is a two-layer GCN for large-CNV detection from shallow whole-genome sequencing. Its published inputs are BAM-derived CNV candidates, truth CNVs, a reference, and a phylogeny—not pangenome adjacency candidates.

The pilot therefore has three tiers: PangenomeFM `exact`, DeepGene `approximately matched`, and PangenomeX `contextual`. The fairness audit must fail if a caller requires all three methods to be exact. No cross-paper score will be represented as a same-data comparison.

## 1. Repository structure

| Area | Role | Audit result |
|---|---|---|
| `src/` | graph processing, model, training, evaluation, downstream tasks | 143 Python source files; actively structured and reusable |
| `scripts/` | local analyses, paper figures, server/HPC orchestration | extensive; reuse preferred |
| `configs/` | cohort, release, folds, ablation, capacity and server definitions | rotating folds and seeds already frozen |
| `data/` | local HPRC/HGSVC graphs, slices, annotations | about 42 GiB; do not duplicate |
| `results/`, `analysis/` | local runs and canonical full-server analysis | about 3.9 GiB of results plus the verified analysis bundle |
| `server_workspace/` | imported server results/checkpoints and execution records | about 1.2 GiB |
| `paper/`, `docs/` | manuscript, figures, reports and group-discussion ledgers | manuscript and figure infrastructure exist |
| `tests/` | unit/integration tests | about 80 test files |

Repository-wide inventory at audit time: 17,202 non-`.git` files (51.24 GB), 238 checkpoint files (75.36 MB locally counted), and 317 summary JSON files. Existing uncommitted manuscript/document edits were treated as user-owned and left untouched.

## 2. Data processing and graph modules

- `src/data/`: GFA parsing, validation, benchmarking and canonicalization.
- `src/graph/`: graph I/O, slicing, features, negative sampling, QC and validation.
- `src/graph/slicing.py` defines the canonical segment index and strict versus endpoint-expanded induced subgraphs.
- `scripts/audit_benchmark_context.py` already audits interval overlap, exposure, structural context and split duplication.
- Server scripts already prepare full released graphs, deterministic 5 Mb pretraining tiles, evaluation windows and path metadata.

Status: **VERIFIED COMPLETE** for the preprocessing needed by the current graph-reconstruction experiments.

## 3. Model

- `src/models/dual_stream_gat.py`: coordinate/context and graph streams plus learned fusion.
- `src/models/gat.py`, `attention_window.py`, `rope.py`: GAT, branching-aware attention and positional/orientation machinery.
- `src/training/pretrain.py`: shared cross-slice relation reconstruction, focal loss, DropEdge, rotating chromosome holdouts, prediction export and checkpoints.
- The encoder consumes topology and graph-derived/coordinate attributes. Nucleotide strings are not model tokens.

Status: **VERIFIED COMPLETE** for the current topology-native encoder.

## 4. Training, evaluation, splits and masking

- Five rotating chromosome folds, 24 chromosomes and seeds `42`, `314159`, `20260806` are frozen in `configs/server_full_multicohort_20260806.json`.
- The trainer previously used `--seed` for split/order RNGs but did not seed PyTorch model initialization or DropEdge. Repeated identical smoke commands produced different scores. Historical rows are independent completed executions, but the numeric seed labels are not reproducible model-initialization seeds. Global Python/NumPy/PyTorch seeding and deterministic backend settings are now applied for future runs.
- Evaluation and aggregation code uses chromosome-aware folds; random candidate splits remain an inner loader mechanism rather than the principal generalization claim.
- Exact positive directed edges were masked before message passing. The Phase 0 audit found that a materialized reverse-complement traversal could remain in canonicalized bidirected graphs. Main full-server runs used uncanonicalized released SV graphs and contained no detected reverse-complement duplicate link rows. The 120 rotating prediction files contain 23,360,676 exported rows with zero exact candidate duplicates within or across their exported partitions. Reverse-equivalent candidate identity cannot be reconstructed from those dense local IDs alone, so this is evidence against—not a proof of absence of—historical leakage.
- The local matched 50 kb source benchmark contains 3,311 exact and 3,368 reverse-equivalent repeated candidate rows across 431/480 slices. These were created before the new audit and can cross the old inner row-wise partitions. The chr22 pilot drops 127 such rows before splitting (1,438 requested rows to 1,311 unique biological relations). The loader now deduplicates exact and reverse-equivalent candidates and rejects conflicting labels.
- The whole-source audit also found 28 orientation-equivalent endpoint pairs with conflicting positive/negative labels across 12/480 slices. None occur in the selected chr22 pilot examples, but they occur in chromosomes needed to train/validate fold B. Full fold-B execution on this local 50 kb source is therefore **BLOCKED** until candidate files are regenerated. The loader fails rather than choosing labels silently.

Status: rotating splits **VERIFIED COMPLETE**; historical score aggregation **VERIFIED COMPLETE but NEEDS RERUN for claims tied to the named initialization seeds**; historical main-run masking **VERIFIED for exact rows with residual reverse-equivalence uncertainty**; canonical reverse-orientation masking **NEEDS RERUN for affected canonicalized/local inner-split runs**, not automatically for the main uncanonicalized chromosome-held-out bundle.

## 5. Existing datasets

| Dataset | Locally/server verified inventory | Intended use | Status |
|---|---:|---|---|
| HPRC R2 SV graph | 751,237 segment rows; 1,097,658 links locally; server manifest: 231 donors/462 haplotypes | primary/current HPRC graph | VERIFIED COMPLETE |
| HGSVC3 SV graph | server manifest: 728,443 segments, 1,053,557 links, 65 donors/130 haplotypes | second released construction and transfer | VERIFIED COMPLETE |
| Official HGSVC3 + HPRC-v1 graph | 934,154 segments, 1,356,160 links | integrated-construction stress test | VERIFIED COMPLETE |
| HPRC R1.1 | 389,065 segments, 562,222 links | release transfer only | VERIFIED COMPLETE |
| HPRC/HGSVC path indexes | 135,927,476 / 107,125,789 segments; 73,897 / 29,130 W records | future path-resolved work | PARTIAL |

Four donors overlap HPRC R2 and HGSVC3: `HG00733`, `HG02818`, `NA19036`, and `NA19240`. Primary aggregate graphs are retained; transfer must be called cross-graph/cross-release rather than independent-donor transfer.

## 6. Experiment outputs and completion state

The checksummed primary bundle is `server_workspace/server_results/full_multicohort_server_20260806`; its local paper source reports 186 aggregated prediction inputs with zero aggregation failures. The canonical analysis is `analysis/full_multicohort_20260809`.

| Result family | Evidence | Status |
|---|---|---|
| final regimes | 30/30 jobs, zero failures | VERIFIED COMPLETE |
| rotating chromosome folds | 120/120 jobs, zero failures | VERIFIED COMPLETE |
| frozen cohort transfer | 18/18 jobs, zero failures | VERIFIED COMPLETE |
| release transfer | 12/12 jobs, zero failures | VERIFIED COMPLETE |
| shared HPRC R2 + HGSVC3 training | per-seed/per-chromosome tables present | VERIFIED COMPLETE |
| context exposure | coordinate-matched audit present; zero pairs under 10% exposure caliper | VERIFIED COMPLETE |
| rotating architecture ablations | local queue summaries are dry-run plans | MISSING / PLANNED |
| capacity scaling | dry-run plan with `execute=false` | MISSING / PLANNED |
| cCRE feature cache | ledger reports 303,425 nodes | COMPLETE BUT NOT VERIFIED |
| cCRE cross-fit | ledger reports 30/30 and aggregate complete; metrics not imported | COMPLETE BUT NOT VERIFIED |
| SV truth/example preparation | 176,231 eligible; 173,969 mapped (98.716%); 167,685 feature nodes | COMPLETE BUT NOT VERIFIED |
| SV seed-42 pilot | ledger reports 5/5 | COMPLETE BUT NOT VERIFIED |
| SV full matrix/aggregate | session finished, matrix and aggregate audit absent | PARTIAL |
| donor-excluded transfer | graph/path-union gate not verified | BLOCKED |
| QTL/GWAS matched enrichment | prespecified regions and six-covariate controls absent | BLOCKED |

The locally regenerated headline AUPRC/AUROC summaries agree with the expected values in the implementation brief. Core and expanded contexts remain separate tasks: 50 kb evaluation windows expose about 1.73–1.81x nodes and 2.07–2.19x links; 5 Mb pretraining tiles expose about 1.56–1.59x nodes and 1.84–1.89x links. These are different benchmark families and should not be mixed.

## 7. Downstream pipelines

- cCRE: label mapping, aligned baselines, frozen embedding extraction, per-fold matrix execution and hierarchical aggregation already exist under `src/tasks/ccre/` and `scripts/server/`.
- SV: truth audit, breakpoint-to-node mapping, frozen-probe folds, full matrix and stratified aggregation already exist under `scripts/server/`.
- Haplotype/methylation/functional signal modules exist under `src/tasks/haplotype/`; they are not a current blocker.
- QTL/GWAS resources are processed, but no valid matched enrichment analysis is yet frozen.

## 8. Existing figures

- `scripts/make_paper_figures.py` writes PNG/PDF/SVG and contains reusable visual primitives.
- `scripts/make_proposed_paper_pngs.py` contains revised draft panels, but several outputs are PNG-only and the conceptual story does not match the new Figure 1/2 specification.
- `analysis/full_multicohort_20260809` already contains all-chromosome and transfer PNG/PDF plots.
- `docs/group_discussion_20260811` contains additional meeting figures.

Needed: a simple three-paradigm Figure 1 and a separate detailed pretraining/reuse Figure 2, each generated as SVG/PDF/PNG with source-table traceability.

## 9. Comparators

### DeepGene

- Full name: *DeepGene: An Efficient Foundation Model for Genomics based on Pan-genome Graph Transformer*.
- Authors: Xiang Zhang, Mingjie Yang, Xunhang Yin, Yining Qian and Fei Sun.
- Preprint: bioRxiv 2024.04.24.590879.
- Official repository: `https://github.com/wds-seu/DeepGene`; audited commit `486343e5212361d6cd7ed03c624f430ed3d5f02e`.
- Published input/output: HPRC v1.0 graph serialization and BPE-tokenized DNA for MLM pretraining; full-parameter downstream sequence classification on GUE/LPD.
- The paper compares sequence baselines including DNABERT, DNABERT-2 and Nucleotide Transformer.
- Existing local `src/tasks/ccre/serialized_baseline.py` is explicitly not a full DeepGene reproduction.

Integration status: **PARTIAL adapter only; exact link task BLOCKED** by task mismatch, missing official checkpoint, legacy dependency/GPU assumptions and hard-coded released-data paths.

### PangenomeX

- Full name: *PangenomeX: a graph convolutional network-based pangenome framework for unbiased population-scale genomic variation analysis*.
- Authors: Zhengfa Xue, Yu Wang, Xuwen Wang, Jiajing Yuan, Lin Wang, Jingyu Zeng, Xin Jin, Huanhuan Zhu and Jiayin Wang.
- Publication: *Briefings in Bioinformatics* 26(5), bbaf550 (2025).
- Official repository: `https://github.com/Nevermore233/PangenomeX`; audited commit `1b2b193e76a83e917098c0fc9086967467b9e8be`.
- Published input/output: BAM-derived CNV candidates, CNV truth, reference/length files and a Newick phylogeny; node-classification CNV report.

Integration status: **MISSING for edge reconstruction; contextual only**. A task-faithful comparison would require a separate shallow-WGS large-CNV benchmark, not endpoint-pair conversion.

## 10. HPC/SLURM infrastructure

The repository already has reusable queue matrices, execution-state JSON, recovery checkpoints, GPU allocation, and `scripts/server/slurm_gpu_matrix.sbatch`. The correct next gate is a local/single-job pilot, output validation and resource measurement before generating any full method matrix.

Status: **VERIFIED COMPLETE** for PangenomeFM/cCRE/SV orchestration; comparator environments remain unbuilt.

## 11. Manuscript material

`paper/psb2026_graphgenomefm_prof_revision.tex` already has Introduction, Related Work, Methods, Results and Discussion material, but also duplicate Discussion/Conclusion blocks and active user edits. It should not be automatically rewritten. The current analysis bundle has a report, claim-evidence matrix, reviewer-risk table and reproducibility notes that can seed a new manuscript-support layer.

## 12. Missing pieces and minimal implementation

1. Versioned comparator registry and explicit exact/approximate/contextual tiers.
2. HPRC R2 chr22 pilot example manifest with fold-aware assignments and endpoint provenance.
3. Per-method conversion/fairness audit that reports zero exact conversion for incompatible tasks and can fail on `--require-exact-all`.
4. Deterministic per-window complexity features; a transparent performance-independent score fit on strict/core windows; frozen tertile thresholds; an editable external-region catalogue.
5. Reverse-complement query-edge masking and regression tests.
6. Canonical primary result TSV/Parquet generated from the verified analysis bundle.
7. New Figure 1/2 scripts and panel-level results manifest.
8. Manuscript outline/placeholders that do not modify the active LaTeX draft.
9. Import and checksum the cCRE/SV server output matrices before manuscript claims.

## Minimal pilot design

- Dataset: local HPRC graph corresponding to the existing matched 50 kb benchmark (historical local label; the run manifest must retain exact source path rather than silently calling it a different release).
- Chromosome: chr22, the test chromosome of `fold_b`.
- Contexts: strict/core and endpoint-expanded as separate prediction tasks over the same ten coordinate windows.
- Examples: every valid candidate endpoint pair in the selected slice files; stable IDs; fixed split seed `20260806`; model seeds `42`, `314159`, `20260806`.
- PangenomeFM: exact input conversion.
- DeepGene: endpoint sequences can be inventoried for an approximately matched supervised adapter, but the official model cannot produce an edge score unchanged.
- PangenomeX: contextual status only; no endpoint conversion.
- Expected pilot artifacts: `benchmark_manifest.tsv/parquet`, `benchmark_audit.tsv/json`, conversion failures, method status, runtime template, checksums and commands.

The chr22 manifest itself passes. A chr22-only inner-split smoke can test the loader/model path, but it is not held-out generalization evidence. A proper fold-B pilot must wait for the 28 source label conflicts to be removed by candidate regeneration.

No full 24-chromosome comparative matrix should be generated until the audit proves an exact common task for at least two methods or the paper explicitly approves an approximate comparison.

## Complexity implementation plan

Immediately computable from existing slice tables: visible node/link counts, node/link density per kb, simple-graph degree, branching count/fraction, components, largest-component fraction, density, cycle rank, node-length statistics, reference/alternate fractions using the graph's `SR` tag, orientation-row fraction and candidate class balance. Path entropy, bubbles, named SV counts and inversion overlap are unavailable in these slice tables and will remain explicitly missing until verified annotations/path materializations are joined.

The version-1 score uses strict/core windows only for fitting, robust median/MAD normalization, equal weights and fixed tertiles. It uses edge density, branching fraction, maximum degree and cycle-rank density. The `SR`-based alternate fraction is reported but excluded because it has zero variance in this local fit population. The score does not read predictions or performance. The resulting locus category is then joined to both strict and expanded contexts.

## Figure and manuscript-support plan

- Reuse the matplotlib style/primitives but generate a new conceptual Figure 1 and detailed Figure 2 in a separate `paper/figures/next_stage/` directory.
- Reuse the verified all-chromosome/transfer source tables for Figure 3 later.
- Leave Figure 4 performance panels blocked until a valid comparator pilot exists; complexity examples may be drafted now.
- Leave Figure 5 blocked until copied cCRE/SV aggregate outputs pass audit.
- Create `paper/next_stage/` with a readable result outline, explicit placeholders and a machine-readable panel manifest; leave the active manuscript unchanged.
