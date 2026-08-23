# Run Prof-Suggested Experiments

This runbook is for the current submission-gap queue. It replaces the older
commands that kept `is_grch38` for pretrained cCRE GAT compatibility. The new
default is leakage-safe cCRE evaluation: both `SR` and `is_grch38` are excluded
from cCRE inputs.

If the package is not installed in the active environment, use `PYTHONPATH=src`.

## Full Queue

Prepare the hard-negative HPRC/HGSVC benchmarks and base HPRC checkpoints if
they are not present:

```bash
PYTHONPATH=src DEVICE=cuda PYTHON_BIN=python bash scripts/prepare_prof_experiment_inputs.sh
```

Run the full submission-gap queue:

```bash
PYTHONPATH=src DEVICE=cuda PYTHON_BIN=python bash scripts/run_prof_suggested_experiments.sh
```

Useful overrides:

```bash
CCRE_FEATURE_POLICY=leakage_safe
HPRC_NODE_LABELS=data/hprc/ccre/run_003/node_labels.csv.gz
HGSVC_NEW_SEGMENTS=/path/to/new/full_segments.csv.gz
HGSVC_NEW_LINKS=/path/to/new/full_links.csv.gz
SEEDS="7 42 1337"
RUN_ALLNODE_SERIALIZED=1
```

For a remote server without `tmux`:

```bash
mkdir -p logs
nohup bash -lc 'source ~/.bashrc; conda activate PangenomeFM; cd ~/PangenomeFM; PYTHONPATH=src DEVICE=cuda PYTHON_BIN=python bash scripts/run_prof_suggested_experiments.sh' > logs/prof_experiments_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## What The Queue Runs

1. Leakage-safe all-node cCRE baselines: coordinate, structural, linearized-graph, MLP, random forest.
2. Leakage-safe benchmark-window cCRE baselines on the same node universe as the graph/window methods.
3. Leakage-safe cCRE GAT scratch, frozen-pretrained, and fine-tuned runs. The pretrained loader adapts the old 7-feature checkpoint input layer to the leakage-safe 5-feature cCRE input.
4. DeepGene-style serialized-graph Transformer baselines. These serialize nodes by chromosome and coordinate and consume no graph edges.
5. HGSVC imputation candidate scoring. If `HGSVC_NEW_SEGMENTS` and `HGSVC_NEW_LINKS` are set, the same command computes precision/recall-at-K against the newer graph.
6. Degree-matched distance-negative HPRC benchmark and training.
7. Non-overlapping hard-negative HPRC benchmark and training.
8. Seed-variance HPRC reruns plus reliability curves from pooled predictions.

## Individual Commands

Leakage-safe fair window baseline:

```bash
PYTHONPATH=src python -m graphgenomefm ccre-aligned-baseline \
  --data-dir data/hprc \
  --node-labels data/hprc/ccre/run_003/node_labels.csv.gz \
  --out-dir results/hprc/ccre_safe_window_structural_binary_logistic \
  --method logistic \
  --feature-set structural \
  --feature-policy leakage_safe \
  --label-scheme binary \
  --evaluation-universe benchmark_windows \
  --benchmark-manifest data/hprc/benchmark_paper_hardneg/manifest.csv \
  --closures strict 1hop \
  --test-chrs chr8 chr19 chr22 \
  --val-chrs chr16
```

Leakage-safe pretrained GAT:

```bash
PYTHONPATH=src python -m graphgenomefm ccre-gat \
  --data-dir data/hprc \
  --benchmark-dir data/hprc/benchmark_paper_hardneg \
  --node-labels data/hprc/ccre/run_003/node_labels.csv.gz \
  --out-dir results/hprc/ccre_safe_gat_frozen_binary \
  --task binary \
  --pretrained-checkpoint results/hprc/pretrain_paper_hardneg/run_001/ckpt_strict__shared_dual_mscale3_orient_adpwk32a4_focal2.0_dedge0.1_heldout_chr1_chr8_chr19_chrY_val_chr16_ep100_pat20.pt \
  --freeze-backbone \
  --feature-policy leakage_safe \
  --test-chrs chr8 chr19 chr22 \
  --val-chrs chr16 \
  --device cuda
```

DeepGene-style serialized-graph baseline:

```bash
PYTHONPATH=src python -m graphgenomefm ccre-serialized-baseline \
  --data-dir data/hprc \
  --benchmark-dir data/hprc/benchmark_paper_hardneg \
  --node-labels data/hprc/ccre/run_003/node_labels.csv.gz \
  --out-dir results/hprc/ccre_safe_serialized_window_binary \
  --label-scheme binary \
  --evaluation-universe benchmark_windows \
  --closures strict 1hop \
  --feature-policy leakage_safe \
  --test-chrs chr8 chr19 chr22 \
  --val-chrs chr16 \
  --device cuda
```

HGSVC imputation with a newer comparison graph:

```bash
PYTHONPATH=src python -m graphgenomefm impute-edges \
  --data-dir data/hgsvc3_expanded \
  --benchmark-dir data/hgsvc3_expanded/benchmark_chr19_chr21_chr22_chrY_hardneg \
  --checkpoint results/hprc/pretrain_paper_hardneg/run_001/ckpt_1hop__shared_dual_mscale3_orient_adpwk32a4_focal2.0_dedge0.1_heldout_chr1_chr8_chr19_chrY_val_chr16_ep100_pat20.pt \
  --out-dir results/hgsvc3/graph_imputation_1hop_with_new_graph \
  --closure 1hop \
  --candidate-label 0 \
  --top-k 1000 \
  --comparison-segments /path/to/new/full_segments.csv.gz \
  --comparison-links /path/to/new/full_links.csv.gz \
  --device cuda
```

Degree-matched negative benchmark:

```bash
PYTHONPATH=src python -m graphgenomefm make-benchmark \
  --data-dir data/hprc \
  --out-dir data/hprc/benchmark_paper_degree_matched \
  --targets all \
  --closures strict 1hop \
  --negative-sampler distance_matched \
  --negative-degree-matched \
  --n-windows 10 \
  --no-network-analysis \
  --no-viz
```

Non-overlapping hard-negative benchmark:

```bash
PYTHONPATH=src python -m graphgenomefm make-benchmark \
  --data-dir data/hprc \
  --out-dir data/hprc/benchmark_paper_hardneg_nonoverlap \
  --targets all \
  --closures strict 1hop \
  --negative-sampler distance_matched \
  --non-overlapping-windows \
  --n-windows 10 \
  --no-network-analysis \
  --no-viz
```

Reliability curves from pooled predictions:

```bash
python scripts/make_reliability_curves.py \
  --inputs hgsvc_strict=results/hgsvc3/external_eval_hardneg_strict/run_004/pooled_predictions.csv.gz hgsvc_1hop=results/hgsvc3/external_eval_hardneg_1hop/run_003/pooled_predictions.csv.gz \
  --out-dir results/reliability_curves
```

Summaries:

```bash
python scripts/summarize_submission_gap_status.py
python scripts/summarize_experiment_status.py
```
