#!/usr/bin/env bash
set -euo pipefail

# Full experiment queue for the submission gaps Prof flagged.
# This script is intentionally runnable but not automatically launched by Codex.
# Override variables at invocation time, e.g.:
#   DEVICE=cuda PYTHON_BIN=.venv/bin/python bash scripts/run_prof_suggested_experiments.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
FEATURE_POLICY="${CCRE_FEATURE_POLICY:-leakage_safe}"
SEEDS="${SEEDS:-7 42 1337}"
RUN_ALLNODE_SERIALIZED="${RUN_ALLNODE_SERIALIZED:-1}"
RUN_MASKED_PRETRAIN="${RUN_MASKED_PRETRAIN:-0}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

HPRC_DATA_DIR="${HPRC_DATA_DIR:-data/hprc}"
HPRC_BENCHMARK="${HPRC_BENCHMARK:-data/hprc/benchmark_paper_hardneg}"
HPRC_DEGREE_BENCHMARK="${HPRC_DEGREE_BENCHMARK:-data/hprc/benchmark_paper_degree_matched}"
HPRC_TILED_BENCHMARK="${HPRC_TILED_BENCHMARK:-data/hprc/benchmark_paper_hardneg_nonoverlap}"
HPRC_NODE_LABELS="${HPRC_NODE_LABELS:-}"

HPRC_PRETRAIN_OUT="${HPRC_PRETRAIN_OUT:-results/hprc/pretrain_paper_hardneg}"
HPRC_HEURISTIC_OUT="${HPRC_HEURISTIC_OUT:-results/hprc/link_heuristics_hardneg}"
HPRC_STRICT_CKPT="${HPRC_STRICT_CKPT:-}"
HPRC_1HOP_CKPT="${HPRC_1HOP_CKPT:-}"

HGSVC_DATA_DIR="${HGSVC_DATA_DIR:-data/hgsvc3_expanded}"
HGSVC_BENCHMARK="${HGSVC_BENCHMARK:-data/hgsvc3_expanded/benchmark_chr19_chr21_chr22_chrY_hardneg}"
HGSVC_NEW_SEGMENTS="${HGSVC_NEW_SEGMENTS:-}"
HGSVC_NEW_LINKS="${HGSVC_NEW_LINKS:-}"

TEST_CHRS=(chr8 chr19 chr22)
VAL_CHRS=(chr16)
PRETRAIN_TEST_CHRS=(chr1 chr8 chr19 chrY)
PRETRAIN_VAL_CHRS=(chr16)

require_file() {
  local path="$1"
  if [[ -z "$path" || ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    echo "hint: run scripts/prepare_prof_experiment_inputs.sh first, or override the relevant path variable." >&2
    exit 2
  fi
}

latest_file_in_runs() {
  local root="$1"
  local pattern="$2"
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  find "$root" -path "*/$pattern" -type f 2>/dev/null | sort | tail -n 1
}

if [[ -z "$HPRC_STRICT_CKPT" ]]; then
  HPRC_STRICT_CKPT="$(latest_file_in_runs "$HPRC_PRETRAIN_OUT" 'ckpt_strict__*.pt')"
fi
if [[ -z "$HPRC_1HOP_CKPT" ]]; then
  HPRC_1HOP_CKPT="$(latest_file_in_runs "$HPRC_PRETRAIN_OUT" 'ckpt_1hop__*.pt')"
fi
if [[ -z "$HPRC_NODE_LABELS" ]]; then
  HPRC_NODE_LABELS="$(latest_file_in_runs "$HPRC_DATA_DIR/ccre" 'node_labels.csv.gz')"
fi

require_file "$HPRC_DATA_DIR/full_segments.csv"
require_file "$HPRC_DATA_DIR/full_links.csv"
require_file "$HPRC_BENCHMARK/manifest.csv"
require_file "$HPRC_NODE_LABELS"
require_file "$HPRC_STRICT_CKPT"
require_file "$HPRC_1HOP_CKPT"
require_file "$HGSVC_DATA_DIR/full_segments.csv.gz"
require_file "$HGSVC_BENCHMARK/manifest.csv"

echo "[prof] 1/8 Leakage-safe all-node and fair-window cCRE feature baselines"
for feature_set in coordinate structural linearized_graph; do
  "$PYTHON_BIN" -m graphgenomefm ccre-aligned-baseline \
    --data-dir "$HPRC_DATA_DIR" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_safe_allnode_${feature_set}_binary_logistic" \
    --method logistic \
    --feature-set "$feature_set" \
    --feature-policy "$FEATURE_POLICY" \
    --label-scheme binary \
    --evaluation-universe all \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}"

  "$PYTHON_BIN" -m graphgenomefm ccre-aligned-baseline \
    --data-dir "$HPRC_DATA_DIR" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_safe_window_${feature_set}_binary_logistic" \
    --method logistic \
    --feature-set "$feature_set" \
    --feature-policy "$FEATURE_POLICY" \
    --label-scheme binary \
    --evaluation-universe benchmark_windows \
    --benchmark-manifest "$HPRC_BENCHMARK/manifest.csv" \
    --closures strict 1hop \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}"
done

for method in mlp random_forest; do
  "$PYTHON_BIN" -m graphgenomefm ccre-aligned-baseline \
    --data-dir "$HPRC_DATA_DIR" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_safe_allnode_structural_binary_${method}" \
    --method "$method" \
    --feature-set structural \
    --feature-policy "$FEATURE_POLICY" \
    --label-scheme binary \
    --evaluation-universe all \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}"
done

echo "[prof] 2/8 Leakage-safe reduced cCRE groups and category-specific baselines"
for scheme in group3 group4 group5 full9; do
  "$PYTHON_BIN" -m graphgenomefm ccre-aligned-baseline \
    --data-dir "$HPRC_DATA_DIR" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_safe_allnode_structural_${scheme}_logistic" \
    --method logistic \
    --feature-set structural \
    --feature-policy "$FEATURE_POLICY" \
    --label-scheme "$scheme" \
    --evaluation-universe all \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}"
done

for group in enhancer_like promoter_like tf_ctcf_associated open_chromatin dels pels pls; do
  "$PYTHON_BIN" -m graphgenomefm ccre-aligned-baseline \
    --data-dir "$HPRC_DATA_DIR" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_safe_allnode_category_${group}_logistic" \
    --method logistic \
    --feature-set structural \
    --feature-policy "$FEATURE_POLICY" \
    --label-scheme category_binary \
    --positive-group "$group" \
    --evaluation-universe all \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}"
done

echo "[prof] 3/8 Leakage-safe graph-native cCRE GAT runs"
for scheme in binary group3 group4 group5 full9; do
  "$PYTHON_BIN" -m graphgenomefm ccre-gat \
    --data-dir "$HPRC_DATA_DIR" \
    --benchmark-dir "$HPRC_BENCHMARK" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_safe_gat_scratch_${scheme}" \
    --task "$scheme" \
    --feature-policy "$FEATURE_POLICY" \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}" \
    --epochs 60 \
    --patience 15 \
    --device "$DEVICE"

  "$PYTHON_BIN" -m graphgenomefm ccre-gat \
    --data-dir "$HPRC_DATA_DIR" \
    --benchmark-dir "$HPRC_BENCHMARK" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_safe_gat_frozen_${scheme}" \
    --task "$scheme" \
    --pretrained-checkpoint "$HPRC_STRICT_CKPT" \
    --freeze-backbone \
    --feature-policy "$FEATURE_POLICY" \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}" \
    --epochs 60 \
    --patience 15 \
    --device "$DEVICE"

  "$PYTHON_BIN" -m graphgenomefm ccre-gat \
    --data-dir "$HPRC_DATA_DIR" \
    --benchmark-dir "$HPRC_BENCHMARK" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_safe_gat_finetune_${scheme}" \
    --task "$scheme" \
    --pretrained-checkpoint "$HPRC_STRICT_CKPT" \
    --feature-policy "$FEATURE_POLICY" \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}" \
    --epochs 60 \
    --patience 15 \
    --device "$DEVICE"
done

echo "[prof] 4/8 DeepGene-style serialized-graph Transformer baseline"
for scheme in binary group3; do
  "$PYTHON_BIN" -m graphgenomefm ccre-serialized-baseline \
    --data-dir "$HPRC_DATA_DIR" \
    --benchmark-dir "$HPRC_BENCHMARK" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_safe_serialized_window_${scheme}" \
    --label-scheme "$scheme" \
    --evaluation-universe benchmark_windows \
    --closures strict 1hop \
    --feature-policy "$FEATURE_POLICY" \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}" \
    --epochs 30 \
    --patience 8 \
    --device "$DEVICE"
done

if [[ "$RUN_ALLNODE_SERIALIZED" == "1" ]]; then
  "$PYTHON_BIN" -m graphgenomefm ccre-serialized-baseline \
    --data-dir "$HPRC_DATA_DIR" \
    --node-labels "$HPRC_NODE_LABELS" \
    --out-dir "results/hprc/ccre_safe_serialized_allnode_binary" \
    --label-scheme binary \
    --evaluation-universe all \
    --feature-policy "$FEATURE_POLICY" \
    --test-chrs "${TEST_CHRS[@]}" \
    --val-chrs "${VAL_CHRS[@]}" \
    --epochs 30 \
    --patience 8 \
    --device "$DEVICE"
fi

echo "[prof] 5/8 HGSVC imputation candidate scoring and optional newer-build validation"
if [[ -n "$HGSVC_NEW_SEGMENTS" && -n "$HGSVC_NEW_LINKS" ]]; then
  "$PYTHON_BIN" -m graphgenomefm impute-edges \
    --data-dir "$HGSVC_DATA_DIR" \
    --benchmark-dir "$HGSVC_BENCHMARK" \
    --checkpoint "$HPRC_1HOP_CKPT" \
    --out-dir results/hgsvc3/graph_imputation_1hop \
    --closure 1hop \
    --candidate-label 0 \
    --top-k 1000 \
    --device "$DEVICE" \
    --comparison-segments "$HGSVC_NEW_SEGMENTS" \
    --comparison-links "$HGSVC_NEW_LINKS"
else
  "$PYTHON_BIN" -m graphgenomefm impute-edges \
    --data-dir "$HGSVC_DATA_DIR" \
    --benchmark-dir "$HGSVC_BENCHMARK" \
    --checkpoint "$HPRC_1HOP_CKPT" \
    --out-dir results/hgsvc3/graph_imputation_1hop \
    --closure 1hop \
    --candidate-label 0 \
    --top-k 1000 \
    --device "$DEVICE"
fi

echo "[prof] 6/9 Topology-only link-prediction heuristic baselines"
"$PYTHON_BIN" -m graphgenomefm link-heuristics \
  --data-dir "$HPRC_DATA_DIR" \
  --benchmark-dir "$HPRC_BENCHMARK" \
  --out-dir "$HPRC_HEURISTIC_OUT" \
  --target-chrs "${PRETRAIN_TEST_CHRS[@]}" \
  --split test

if [[ "$RUN_MASKED_PRETRAIN" == "1" ]]; then
  echo "[prof] 6b/9 Query-edge-masked hard-negative pretraining rerun"
  "$PYTHON_BIN" -m graphgenomefm pretrain \
    --data-dir "$HPRC_DATA_DIR" \
    --benchmark-dir "$HPRC_BENCHMARK" \
    --out-dir results/hprc/pretrain_paper_hardneg_masked_query \
    --test-chrs "${PRETRAIN_TEST_CHRS[@]}" \
    --val-chrs "${PRETRAIN_VAL_CHRS[@]}" \
    --epochs 100 \
    --patience 20 \
    --save-predictions \
    --mask-query-edges \
    --device "$DEVICE"
fi

echo "[prof] 7/9 Degree-matched negative benchmark and training"
"$PYTHON_BIN" -m graphgenomefm make-benchmark \
  --data-dir "$HPRC_DATA_DIR" \
  --out-dir "$HPRC_DEGREE_BENCHMARK" \
  --targets all \
  --closures strict 1hop \
  --negative-sampler distance_matched \
  --negative-degree-matched \
  --n-windows 10 \
  --no-network-analysis \
  --no-viz

"$PYTHON_BIN" -m graphgenomefm pretrain \
  --data-dir "$HPRC_DATA_DIR" \
  --benchmark-dir "$HPRC_DEGREE_BENCHMARK" \
  --out-dir results/hprc/pretrain_paper_degree_matched \
  --test-chrs "${PRETRAIN_TEST_CHRS[@]}" \
  --val-chrs "${PRETRAIN_VAL_CHRS[@]}" \
  --epochs 100 \
  --patience 20 \
  --save-predictions \
  --device "$DEVICE"

echo "[prof] 8/9 Non-overlapping tiled hard-negative benchmark and training"
"$PYTHON_BIN" -m graphgenomefm make-benchmark \
  --data-dir "$HPRC_DATA_DIR" \
  --out-dir "$HPRC_TILED_BENCHMARK" \
  --targets all \
  --closures strict 1hop \
  --negative-sampler distance_matched \
  --non-overlapping-windows \
  --n-windows 10 \
  --no-network-analysis \
  --no-viz

"$PYTHON_BIN" -m graphgenomefm pretrain \
  --data-dir "$HPRC_DATA_DIR" \
  --benchmark-dir "$HPRC_TILED_BENCHMARK" \
  --out-dir results/hprc/pretrain_paper_hardneg_nonoverlap \
  --test-chrs "${PRETRAIN_TEST_CHRS[@]}" \
  --val-chrs "${PRETRAIN_VAL_CHRS[@]}" \
  --epochs 100 \
  --patience 20 \
  --save-predictions \
  --device "$DEVICE"

echo "[prof] 9/9 Seed variance and reliability curves"
read -r -a SEED_LIST <<< "$SEEDS"
for seed in "${SEED_LIST[@]}"; do
  "$PYTHON_BIN" -m graphgenomefm pretrain \
    --data-dir "$HPRC_DATA_DIR" \
    --benchmark-dir "$HPRC_BENCHMARK" \
    --out-dir "results/hprc/pretrain_paper_hardneg_seed_${seed}" \
    --test-chrs "${PRETRAIN_TEST_CHRS[@]}" \
    --val-chrs "${PRETRAIN_VAL_CHRS[@]}" \
    --epochs 100 \
    --patience 20 \
    --seed "$seed" \
    --save-predictions \
    --device "$DEVICE"
done

reliability_inputs=()
for seed in "${SEED_LIST[@]}"; do
  strict_preds="$(latest_file_in_runs "results/hprc/pretrain_paper_hardneg_seed_${seed}" 'strict_pooled_predictions*.csv.gz')"
  onehop_preds="$(latest_file_in_runs "results/hprc/pretrain_paper_hardneg_seed_${seed}" '1hop_pooled_predictions*.csv.gz')"
  [[ -n "$strict_preds" ]] && reliability_inputs+=("hprc_seed${seed}_strict=$strict_preds")
  [[ -n "$onehop_preds" ]] && reliability_inputs+=("hprc_seed${seed}_1hop=$onehop_preds")
done
hgsvc_strict_preds="$(latest_file_in_runs "results/hgsvc3/external_eval_hardneg_strict" 'pooled_predictions.csv.gz')"
hgsvc_1hop_preds="$(latest_file_in_runs "results/hgsvc3/external_eval_hardneg_1hop" 'pooled_predictions.csv.gz')"
[[ -n "$hgsvc_strict_preds" ]] && reliability_inputs+=("hgsvc_strict=$hgsvc_strict_preds")
[[ -n "$hgsvc_1hop_preds" ]] && reliability_inputs+=("hgsvc_1hop=$hgsvc_1hop_preds")
if [[ "${#reliability_inputs[@]}" -gt 0 ]]; then
  "$PYTHON_BIN" scripts/make_reliability_curves.py \
    --inputs "${reliability_inputs[@]}" \
    --out-dir results/reliability_curves
fi

"$PYTHON_BIN" scripts/summarize_submission_gap_status.py
"$PYTHON_BIN" scripts/summarize_experiment_status.py

echo "[prof] Done."
