# Run Prof-Suggested Experiments

This document lists the new runnable commands without starting any expensive
jobs automatically.

If the package is not installed in your environment, prefix individual commands
with `PYTHONPATH=src`.

## Full Queue

```bash
PYTHONPATH=src DEVICE=cuda PYTHON_BIN=python bash scripts/run_prof_suggested_experiments.sh
```

Use `DEVICE=cpu` or `DEVICE=mps` if needed.

## Architecture Ablations

```bash
python -m graphgenomefm pretrain --data-dir data/hprc --benchmark-dir data/hprc/benchmark_paper_hardneg --out-dir results/hprc/ablation_coordinate_only --stream-mode coordinate --test-chrs chr1 chr8 chr19 chrY --val-chrs chr16 --device cuda
python -m graphgenomefm pretrain --data-dir data/hprc --benchmark-dir data/hprc/benchmark_paper_hardneg --out-dir results/hprc/ablation_graph_only --stream-mode graph --test-chrs chr1 chr8 chr19 chrY --val-chrs chr16 --device cuda
python -m graphgenomefm pretrain --data-dir data/hprc --benchmark-dir data/hprc/benchmark_paper_hardneg --out-dir results/hprc/ablation_dual_no_gate --no-fusion-gate --test-chrs chr1 chr8 chr19 chrY --val-chrs chr16 --device cuda
```

## Aligned cCRE Baselines

All-node linear-reference baseline:

```bash
python -m graphgenomefm ccre-aligned-baseline --data-dir data/hprc --node-labels data/hprc/ccre/run_003/node_labels.csv.gz --out-dir results/hprc/ccre_allnode_coordinate_binary_logistic --method logistic --feature-set coordinate --label-scheme binary --evaluation-universe all --test-chrs chr8 chr19 chr22 --val-chrs chr16
```

All-node linearized-graph baseline:

```bash
python -m graphgenomefm ccre-aligned-baseline --data-dir data/hprc --node-labels data/hprc/ccre/run_003/node_labels.csv.gz --out-dir results/hprc/ccre_allnode_linearized_binary_logistic --method logistic --feature-set linearized_graph --label-scheme binary --evaluation-universe all --test-chrs chr8 chr19 chr22 --val-chrs chr16
```

Benchmark-window-aligned logistic baseline:

```bash
python -m graphgenomefm ccre-aligned-baseline --data-dir data/hprc --node-labels data/hprc/ccre/run_003/node_labels.csv.gz --out-dir results/hprc/ccre_window_structural_binary_logistic --method logistic --feature-set structural --label-scheme binary --evaluation-universe benchmark_windows --benchmark-manifest data/hprc/benchmark_paper_hardneg/manifest.csv --closures strict 1hop --test-chrs chr8 chr19 chr22 --val-chrs chr16
```

Reduced cCRE grouping:

```bash
python -m graphgenomefm ccre-aligned-baseline --data-dir data/hprc --node-labels data/hprc/ccre/run_003/node_labels.csv.gz --out-dir results/hprc/ccre_allnode_structural_group5_logistic --method logistic --feature-set structural --label-scheme group5 --evaluation-universe all --test-chrs chr8 chr19 chr22 --val-chrs chr16
```

Category-specific binary task:

```bash
python -m graphgenomefm ccre-aligned-baseline --data-dir data/hprc --node-labels data/hprc/ccre/run_003/node_labels.csv.gz --out-dir results/hprc/ccre_allnode_category_enhancer_like_logistic --method logistic --feature-set structural --label-scheme category_binary --positive-group enhancer_like --evaluation-universe all --test-chrs chr8 chr19 chr22 --val-chrs chr16
```

## Reduced-Label cCRE GAT

Scratch:

```bash
python -m graphgenomefm ccre-gat --data-dir data/hprc --benchmark-dir data/hprc/benchmark_paper_hardneg --node-labels data/hprc/ccre/run_003/node_labels.csv.gz --out-dir results/hprc/ccre_gat_scratch_group5 --task group5 --test-chrs chr8 chr19 chr22 --val-chrs chr16 --device cuda
```

Frozen pretrained:

```bash
python -m graphgenomefm ccre-gat --data-dir data/hprc --benchmark-dir data/hprc/benchmark_paper_hardneg --node-labels data/hprc/ccre/run_003/node_labels.csv.gz --out-dir results/hprc/ccre_gat_frozen_group5 --task group5 --pretrained-checkpoint results/hprc/pretrain_paper_hardneg/run_001/ckpt_strict__shared_dual_mscale3_orient_adpwk32a4_focal2.0_dedge0.1_heldout_chr1_chr8_chr19_chrY_val_chr16_ep100_pat20.pt --freeze-backbone --keep-is-grch38 --test-chrs chr8 chr19 chr22 --val-chrs chr16 --device cuda
```

Fine-tuned pretrained:

```bash
python -m graphgenomefm ccre-gat --data-dir data/hprc --benchmark-dir data/hprc/benchmark_paper_hardneg --node-labels data/hprc/ccre/run_003/node_labels.csv.gz --out-dir results/hprc/ccre_gat_finetune_group5 --task group5 --pretrained-checkpoint results/hprc/pretrain_paper_hardneg/run_001/ckpt_strict__shared_dual_mscale3_orient_adpwk32a4_focal2.0_dedge0.1_heldout_chr1_chr8_chr19_chrY_val_chr16_ep100_pat20.pt --keep-is-grch38 --test-chrs chr8 chr19 chr22 --val-chrs chr16 --device cuda
```

## Frozen Embedding Baseline

```bash
python -m graphgenomefm ccre-embedding-baseline --data-dir data/hprc --benchmark-dir data/hprc/benchmark_paper_hardneg --checkpoint results/hprc/pretrain_paper_hardneg/run_001/ckpt_strict__shared_dual_mscale3_orient_adpwk32a4_focal2.0_dedge0.1_heldout_chr1_chr8_chr19_chrY_val_chr16_ep100_pat20.pt --node-labels data/hprc/ccre/run_003/node_labels.csv.gz --out-dir results/hprc/ccre_embedding_strict_binary_logistic --method logistic --label-scheme binary --closure strict --test-chrs chr8 chr19 chr22 --val-chrs chr16 --device cuda
```

## HGSVC Imputation

Without newer comparison graph:

```bash
python -m graphgenomefm impute-edges --data-dir data/hgsvc3_expanded --benchmark-dir data/hgsvc3_expanded/benchmark_chr19_chr21_chr22_chrY_hardneg --checkpoint results/hprc/pretrain_paper_hardneg/run_001/ckpt_1hop__shared_dual_mscale3_orient_adpwk32a4_focal2.0_dedge0.1_heldout_chr1_chr8_chr19_chrY_val_chr16_ep100_pat20.pt --out-dir results/hgsvc3/graph_imputation_1hop --closure 1hop --candidate-label 0 --top-k 1000 --device cuda
```

With newer comparison graph from Tomoya:

```bash
python -m graphgenomefm impute-edges --data-dir data/hgsvc3_expanded --benchmark-dir data/hgsvc3_expanded/benchmark_chr19_chr21_chr22_chrY_hardneg --checkpoint results/hprc/pretrain_paper_hardneg/run_001/ckpt_1hop__shared_dual_mscale3_orient_adpwk32a4_focal2.0_dedge0.1_heldout_chr1_chr8_chr19_chrY_val_chr16_ep100_pat20.pt --out-dir results/hgsvc3/graph_imputation_1hop_with_new_graph --closure 1hop --candidate-label 0 --top-k 1000 --comparison-segments path/to/new/full_segments.csv.gz --comparison-links path/to/new/full_links.csv.gz --device cuda
```
