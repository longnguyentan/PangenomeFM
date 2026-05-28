#!/usr/bin/env bash
set -euo pipefail

# Full experiment queue requested after the 2026-05-27 Prof meeting.
# This script is intentionally runnable but not automatically launched by Codex.
# Override variables at invocation time, e.g.:
#   DEVICE=cuda PYTHON_BIN=.venv/bin/python bash scripts/run_prof_suggested_experiments.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
export PYTHONPATH="${PYTHONPATH:-src}"

HPRC_DATA_DIR="${HPRC_DATA_DIR:-data/hprc}"
HPRC_BENCHMARK="${HPRC_BENCHMARK:-data/hprc/benchmark_paper_hardneg}"
HPRC_NODE_LABELS="${HPRC_NODE_LABELS:-data/hprc/ccre/run_003/node_labels.csv.gz}"
ENCODE_BED="${ENCODE_BED:-data/encode/GRCh38-human-cCREs.bed}"

HPRC_STRICT_CKPT="${HPRC_STRICT_CKPT:-results/hprc/pretrain_paper_hardneg/run_001/ckpt_strict__shared_dual_mscale3_orient_adpwk32a4_focal2.0_dedge0.1_heldout_chr1_chr8_chr19_chrY_val_chr16_ep100_pat20.pt}"
HPRC_1HOP_CKPT="${HPRC_1HOP_CKPT:-results/hprc/pretrain_paper_hardneg/run_001/ckpt_1hop__shared_dual_mscale3_orient_adpwk32a4_focal2.0_dedge0.1_heldout_chr1_chr8_chr19_chrY_val_chr16_ep100_pat20.pt}"

HGSVC_DATA_DIR="${HGSVC_DATA_DIR:-data/hgsvc3_expanded}"
HGSVC_BENCHMARK="${HGSVC_BENCHMARK:-data/hgsvc3_expanded/benchmark_chr19_chr21_chr22_chrY_hardneg}"
HGSVC_NEW_SEGMENTS="${HGSVC_NEW_SEGMENTS:-}"
HGSVC_NEW_LINKS="${HGSVC_NEW_LINKS:-}"

TEST_CHRS=(chr8 chr19 chr22)
VAL_CHRS=(chr16)
TEST_SNS=(GRCh38#0#chr8 GRCh38#0#chr19 GRCh38#0#chr22)
VAL_SNS=(GRCh38#0#chr16)

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 2
  fi
}

require_file "$HPRC_DATA_DIR/full_segments.csv"
require_file "$HPRC_DATA_DIR/full_links.csv"
require_file "$HPRC_BENCHMARK/manifest.csv"
require_file "$HPRC_NODE_LABELS"
require_file "$HPRC_STRICT_CKPT"
require_file "$HPRC_1HOP_CKPT"
require_file "$HGSVC_DATA_DIR/full_segments.csv.gz"
require_file "$HGSVC_BENCHMARK/manifest.csv"

echo "[prof] 1/7 Architecture ablations for HPRC link prediction"
"$PYTHON_BIN" -m graphgenomefm pretrain \
  --data-dir "$HPRC_DATA_DIR" \
  --benchmark-dir "$HPRC_BENCHMARK" \
  --out-dir results/hprc/ablation_coordinate_only \
  --stream-mode coordinate \
  --test-chrs chr1 chr8 chr19 chrY \
  --val-chrs chr16 \
  --epochs 100 \
  --patience 20 \
  --device "$DEVICE"

"$PYTHON_BIN" -m graphgenomefm pretrain \
  --data-dir "$HPRC_DATA_DIR" \
  --benchmark-dir "$HPRC_BENCHMARK" \
  --out-dir results/hprc/ablation_graph_only \
  --stream-mode graph \
  --test-chrs chr1 chr8 chr19 chrY \
  --val-chrs chr16 \
  --epochs 100 \
  --patience 20 \
  --device "$DEVICE"

"$PYTHON_BIN" -m graphgenomefm pretrain \
  --data-dir "$HPRC_DATA_DIR" \
  --benchmark-dir "$HPRC_BENCHMARK" \
  --out-dir results/hprc/ablation_dual_no_gate \
  --stream-mode full \
  --no-fusion-gate \
  --test-chrs chr1 chr8 chr19 chrY \
  --val-chrs chr16 \
  --epochs 100 \
  --patience 20 \
  --device "$DEVICE"

echo "[prof] 2/7 All-node cCRE linear/linearized/simple baselines"
for feature_set in coordinate structural linearized_graph; do
  "$PYTHON_BIN" -m graphgenomefm ccre-aligned-baseline \
    --data-dir "$HPRC_DATA_DIR" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_allnode_${feature_set}_binary_logistic" \
    --method logistic \
    --feature-set "$feature_set" \
    --label-scheme binary \
    --evaluation-universe all \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}"
done

for method in mlp random_forest; do
  "$PYTHON_BIN" -m graphgenomefm ccre-aligned-baseline \
    --data-dir "$HPRC_DATA_DIR" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_allnode_structural_binary_${method}" \
    --method "$method" \
    --feature-set structural \
    --label-scheme binary \
    --evaluation-universe all \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}"
done

echo "[prof] 3/7 Reduced cCRE label groups and category-specific binary baselines"
for scheme in group3 group4 group5 full9; do
  "$PYTHON_BIN" -m graphgenomefm ccre-aligned-baseline \
    --data-dir "$HPRC_DATA_DIR" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_allnode_structural_${scheme}_logistic" \
    --method logistic \
    --feature-set structural \
    --label-scheme "$scheme" \
    --evaluation-universe all \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}"
done

for group in enhancer_like promoter_like tf_ctcf_associated open_chromatin dels pels pls; do
  "$PYTHON_BIN" -m graphgenomefm ccre-aligned-baseline \
    --data-dir "$HPRC_DATA_DIR" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_allnode_category_${group}_logistic" \
    --method logistic \
    --feature-set structural \
    --label-scheme category_binary \
    --positive-group "$group" \
    --evaluation-universe all \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}"
done

echo "[prof] 4/7 Fair benchmark-window baselines aligned to current cCRE GAT universe"
for feature_set in coordinate structural linearized_graph; do
  "$PYTHON_BIN" -m graphgenomefm ccre-aligned-baseline \
    --data-dir "$HPRC_DATA_DIR" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_window_${feature_set}_binary_logistic" \
    --method logistic \
    --feature-set "$feature_set" \
    --label-scheme binary \
    --evaluation-universe benchmark_windows \
    --benchmark-manifest "$HPRC_BENCHMARK/manifest.csv" \
    --closures strict 1hop \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}"
done

echo "[prof] 5/7 Reduced-label cCRE GAT experiments"
for scheme in group3 group4 group5; do
  "$PYTHON_BIN" -m graphgenomefm ccre-gat \
    --data-dir "$HPRC_DATA_DIR" \
    --benchmark-dir "$HPRC_BENCHMARK" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_gat_scratch_${scheme}" \
    --task "$scheme" \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}" \
    --epochs 60 \
    --patience 15 \
    --device "$DEVICE"

  "$PYTHON_BIN" -m graphgenomefm ccre-gat \
    --data-dir "$HPRC_DATA_DIR" \
    --benchmark-dir "$HPRC_BENCHMARK" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_gat_frozen_${scheme}" \
    --task "$scheme" \
    --pretrained-checkpoint "$HPRC_STRICT_CKPT" \
    --freeze-backbone \
    --keep-is-grch38 \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}" \
    --epochs 60 \
    --patience 15 \
    --device "$DEVICE"

  "$PYTHON_BIN" -m graphgenomefm ccre-gat \
    --data-dir "$HPRC_DATA_DIR" \
    --benchmark-dir "$HPRC_BENCHMARK" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_gat_finetune_${scheme}" \
    --task "$scheme" \
    --pretrained-checkpoint "$HPRC_STRICT_CKPT" \
    --keep-is-grch38 \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}" \
    --epochs 60 \
    --patience 15 \
    --device "$DEVICE"
done

echo "[prof] 6/7 Frozen GraphGenome-FM embedding baselines"
for method in logistic mlp; do
  "$PYTHON_BIN" -m graphgenomefm ccre-embedding-baseline \
    --data-dir "$HPRC_DATA_DIR" \
    --benchmark-dir "$HPRC_BENCHMARK" \
    --checkpoint "$HPRC_STRICT_CKPT" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_embedding_strict_binary_${method}" \
    --method "$method" \
    --label-scheme binary \
    --closure strict \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}" \
    --device "$DEVICE"
done

for scheme in group3 group4 group5; do
  "$PYTHON_BIN" -m graphgenomefm ccre-embedding-baseline \
    --data-dir "$HPRC_DATA_DIR" \
    --benchmark-dir "$HPRC_BENCHMARK" \
    --checkpoint "$HPRC_STRICT_CKPT" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_embedding_strict_${scheme}_logistic" \
    --method logistic \
    --label-scheme "$scheme" \
    --closure strict \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}" \
    --device "$DEVICE"
done

echo "[prof] 7/7 HGSVC graph-imputation candidate scoring"
comparison_args=()
if [[ -n "$HGSVC_NEW_SEGMENTS" && -n "$HGSVC_NEW_LINKS" ]]; then
  comparison_args=(--comparison-segments "$HGSVC_NEW_SEGMENTS" --comparison-links "$HGSVC_NEW_LINKS")
fi

"$PYTHON_BIN" -m graphgenomefm impute-edges \
  --data-dir "$HGSVC_DATA_DIR" \
  --benchmark-dir "$HGSVC_BENCHMARK" \
  --checkpoint "$HPRC_1HOP_CKPT" \
  --out-dir results/hgsvc3/graph_imputation_1hop \
  --closure 1hop \
  --candidate-label 0 \
  --top-k 1000 \
  --device "$DEVICE" \
  "${comparison_args[@]}"

echo "[prof] Done. Summarize results with:"
echo "  $PYTHON_BIN scripts/summarize_experiment_status.py"
