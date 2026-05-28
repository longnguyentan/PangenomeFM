#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

HPRC_DATA_DIR="${HPRC_DATA_DIR:-data/hprc}"
HGSVC_DATA_DIR="${HGSVC_DATA_DIR:-data/hgsvc3}"
HGSVC_GFA="${HGSVC_GFA:-data/hgsvc3/hgsvc3-2024-02-23-mc-chm13.sv.gfa.gz}"
HGSVC_EVAL_DATA_DIR="${HGSVC_EVAL_DATA_DIR:-data/hgsvc3_expanded}"
NODE_LABELS="${NODE_LABELS:-data/hprc/ccre/run_003/node_labels.csv.gz}"

WINDOW_BP="${WINDOW_BP:-50000}"
N_WINDOWS="${N_WINDOWS:-10}"
NEGATIVE_TOL_BP="${NEGATIVE_TOL_BP:-1000}"
NEGATIVE_TOL_FRAC="${NEGATIVE_TOL_FRAC:-0.10}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
PRETRAIN_PATIENCE="${PRETRAIN_PATIENCE:-20}"
CCRE_EPOCHS="${CCRE_EPOCHS:-60}"
CCRE_PATIENCE="${CCRE_PATIENCE:-15}"

HPRC_HARDNEG_BENCHMARK="${HPRC_HARDNEG_BENCHMARK:-data/hprc/benchmark_paper_hardneg}"
HPRC_HARDNEG_PRETRAIN_OUT="${HPRC_HARDNEG_PRETRAIN_OUT:-results/hprc/pretrain_paper_hardneg}"
FORCE_RERUN_HPRC_HARDNEG_PRETRAIN="${FORCE_RERUN_HPRC_HARDNEG_PRETRAIN:-0}"

HGSVC_HARDNEG_BENCHMARK="${HGSVC_HARDNEG_BENCHMARK:-$HGSVC_EVAL_DATA_DIR/benchmark_chr19_chr21_chr22_chrY_hardneg}"
HGSVC_EXTERNAL_STRICT_OUT="${HGSVC_EXTERNAL_STRICT_OUT:-results/hgsvc3/external_eval_hardneg_strict}"
HGSVC_EXTERNAL_1HOP_OUT="${HGSVC_EXTERNAL_1HOP_OUT:-results/hgsvc3/external_eval_hardneg_1hop}"
HGSVC_TARGETS=(${HGSVC_TARGETS:-chr19 chr21 chr22 chrY})

CCRE_BENCHMARK="${CCRE_BENCHMARK:-data/hprc/benchmark_paper}"
CCRE_BINARY_GAT_SCRATCH_OUT="${CCRE_BINARY_GAT_SCRATCH_OUT:-results/hprc/ccre_binary_gat_scratch}"
CCRE_BINARY_GAT_FROZEN_OUT="${CCRE_BINARY_GAT_FROZEN_OUT:-results/hprc/ccre_binary_gat_pretrained_frozen}"
CCRE_BINARY_GAT_FINETUNE_OUT="${CCRE_BINARY_GAT_FINETUNE_OUT:-results/hprc/ccre_binary_gat_pretrained_finetune}"
FORCE_RERUN_CCRE_BINARY_GAT="${FORCE_RERUN_CCRE_BINARY_GAT:-0}"

# Default to the original strong strict checkpoint for cCRE transfer.
# Set CCRE_PRETRAINED_CKPT=hardneg to use the strict hard-negative checkpoint generated in this script.
CCRE_PRETRAINED_CKPT="${CCRE_PRETRAINED_CKPT:-results/hprc/pretrain_paper/run_003/ckpt_strict__shared_dual_mscale3_orient_adpwk32a4_focal2.0_dedge0.1_heldout_chr1_chr8_chr19_chrY_val_chr16_ep100_pat20.pt}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_next_wave_experiments_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[next-wave] repo: $REPO_ROOT"
echo "[next-wave] python: $PYTHON_BIN"
echo "[next-wave] device: $DEVICE"
echo "[next-wave] log: $LOG_FILE"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[next-wave] ERROR: required file not found: $path" >&2
    exit 2
  fi
}

latest_run_dir() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  find "$root" -maxdepth 1 -type d -name 'run_*' 2>/dev/null | sort | tail -n 1
}

latest_file() {
  local root="$1"
  local pattern="$2"
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  find "$root" -maxdepth 1 -type f -name "$pattern" 2>/dev/null | sort | tail -n 1
}

latest_summary_exists() {
  local root="$1"
  local run
  run="$(latest_run_dir "$root")"
  [[ -n "$run" && -f "$run/summary.json" ]]
}

latest_pretrain_run() {
  local root="$1"
  local run
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  while IFS= read -r run; do
    if compgen -G "$run/ckpt_strict__*.pt" >/dev/null && compgen -G "$run/ckpt_1hop__*.pt" >/dev/null; then
      printf '%s\n' "$run"
      return 0
    fi
  done < <(find "$root" -maxdepth 1 -type d -name 'run_*' 2>/dev/null | sort -r)
}

if [[ "$DEVICE" == "cuda" ]]; then
  "$PYTHON_BIN" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("CUDA requested but torch.cuda.is_available() is false.", file=sys.stderr)
    print("Install CUDA-enabled PyTorch or rerun with DEVICE=cpu/mps.", file=sys.stderr)
    sys.exit(2)
print("CUDA OK:", torch.cuda.get_device_name(0))
PY
fi

require_file "$HPRC_DATA_DIR/full_segments.csv"
require_file "$HPRC_DATA_DIR/full_links.csv"
require_file "$HGSVC_GFA"
require_file "$NODE_LABELS"

echo
echo "[next-wave] 1/8 Build HPRC distance-matched benchmark"
"$PYTHON_BIN" -m graphgenomefm make-benchmark \
  --data-dir "$HPRC_DATA_DIR" \
  --out-dir "$HPRC_HARDNEG_BENCHMARK" \
  --targets all \
  --closures strict 1hop \
  --n-windows "$N_WINDOWS" \
  --window-bp "$WINDOW_BP" \
  --negative-sampler distance_matched \
  --negative-tol-bp "$NEGATIVE_TOL_BP" \
  --negative-tol-frac "$NEGATIVE_TOL_FRAC" \
  --no-network-analysis \
  --no-viz

echo
echo "[next-wave] 2/8 Run HPRC hard-negative pretraining"
HARDNEG_RUN="$(latest_pretrain_run "$HPRC_HARDNEG_PRETRAIN_OUT")"
STRICT_CKPT=""
ONEHOP_CKPT=""
if [[ -n "$HARDNEG_RUN" ]]; then
  STRICT_CKPT="$(latest_file "$HARDNEG_RUN" 'ckpt_strict__*.pt')"
  ONEHOP_CKPT="$(latest_file "$HARDNEG_RUN" 'ckpt_1hop__*.pt')"
fi
if [[ "$FORCE_RERUN_HPRC_HARDNEG_PRETRAIN" == "1" || ! -f "$STRICT_CKPT" || ! -f "$ONEHOP_CKPT" ]]; then
  "$PYTHON_BIN" -m graphgenomefm pretrain \
    --data-dir "$HPRC_DATA_DIR" \
    --benchmark-dir "$HPRC_HARDNEG_BENCHMARK" \
    --out-dir "$HPRC_HARDNEG_PRETRAIN_OUT" \
    --test-chrs chr1 chr8 chr19 chrY \
    --val-chrs chr16 \
    --epochs "$PRETRAIN_EPOCHS" \
    --patience "$PRETRAIN_PATIENCE" \
    --seed "$SEED" \
    --device "$DEVICE"

  HARDNEG_RUN="$(latest_pretrain_run "$HPRC_HARDNEG_PRETRAIN_OUT")"
  STRICT_CKPT="$(latest_file "$HARDNEG_RUN" 'ckpt_strict__*.pt')"
  ONEHOP_CKPT="$(latest_file "$HARDNEG_RUN" 'ckpt_1hop__*.pt')"
else
  echo "[next-wave] Reusing completed hard-negative pretraining run: $HARDNEG_RUN"
fi
require_file "$STRICT_CKPT"
require_file "$ONEHOP_CKPT"
echo "[next-wave] strict checkpoint: $STRICT_CKPT"
echo "[next-wave] 1hop checkpoint:   $ONEHOP_CKPT"

if [[ "$CCRE_PRETRAINED_CKPT" == "hardneg" ]]; then
  CCRE_PRETRAINED_CKPT="$STRICT_CKPT"
fi
require_file "$CCRE_PRETRAINED_CKPT"
echo "[next-wave] cCRE transfer checkpoint: $CCRE_PRETRAINED_CKPT"

echo
echo "[next-wave] 3/8 Parse/build expanded HGSVC3 distance-matched benchmark"
if [[ -f "$HGSVC_EVAL_DATA_DIR/full_segments.csv.gz" && -f "$HGSVC_EVAL_DATA_DIR/full_links.csv.gz" ]]; then
  echo "[next-wave] Reusing expanded HGSVC3 cleaned tables: $HGSVC_EVAL_DATA_DIR"
else
  "$PYTHON_BIN" -m graphgenomefm parse-gfa \
    --gfa "$HGSVC_GFA" \
    --out-dir "$HGSVC_EVAL_DATA_DIR" \
    --target-prefix 'id=CHM13|' \
    --target-sns "${HGSVC_TARGETS[@]}" \
    --include-link-neighbors
fi
"$PYTHON_BIN" -m graphgenomefm check-data --data-dir "$HGSVC_EVAL_DATA_DIR"
"$PYTHON_BIN" -m graphgenomefm make-benchmark \
  --data-dir "$HGSVC_EVAL_DATA_DIR" \
  --out-dir "$HGSVC_HARDNEG_BENCHMARK" \
  --target-prefix 'id=CHM13|' \
  --targets "${HGSVC_TARGETS[@]}" \
  --closures strict 1hop \
  --n-windows "$N_WINDOWS" \
  --window-bp "$WINDOW_BP" \
  --negative-sampler distance_matched \
  --negative-tol-bp "$NEGATIVE_TOL_BP" \
  --negative-tol-frac "$NEGATIVE_TOL_FRAC" \
  --no-network-analysis \
  --no-viz

echo
echo "[next-wave] 4/8 External HGSVC3 strict eval"
"$PYTHON_BIN" -m graphgenomefm eval-external \
  --data-dir "$HGSVC_EVAL_DATA_DIR" \
  --benchmark-dir "$HGSVC_HARDNEG_BENCHMARK" \
  --checkpoint "$STRICT_CKPT" \
  --out-dir "$HGSVC_EXTERNAL_STRICT_OUT" \
  --closure strict \
  --split all \
  --seed "$SEED" \
  --device "$DEVICE"

echo
echo "[next-wave] 5/8 External HGSVC3 1-hop eval"
"$PYTHON_BIN" -m graphgenomefm eval-external \
  --data-dir "$HGSVC_EVAL_DATA_DIR" \
  --benchmark-dir "$HGSVC_HARDNEG_BENCHMARK" \
  --checkpoint "$ONEHOP_CKPT" \
  --out-dir "$HGSVC_EXTERNAL_1HOP_OUT" \
  --closure 1hop \
  --split all \
  --seed "$SEED" \
  --device "$DEVICE"

echo
echo "[next-wave] 6/8 Binary cCRE scratch GAT"
if [[ "$FORCE_RERUN_CCRE_BINARY_GAT" == "1" ]] || ! latest_summary_exists "$CCRE_BINARY_GAT_SCRATCH_OUT"; then
  "$PYTHON_BIN" -m graphgenomefm ccre-gat \
    --data-dir "$HPRC_DATA_DIR" \
    --benchmark-dir "$CCRE_BENCHMARK" \
    --node-labels "$NODE_LABELS" \
    --out-dir "$CCRE_BINARY_GAT_SCRATCH_OUT" \
    --task binary \
    --test-chrs chr8 chr19 chr22 \
    --val-chrs chr16 \
    --epochs "$CCRE_EPOCHS" \
    --patience "$CCRE_PATIENCE" \
    --seed "$SEED" \
    --device "$DEVICE"
else
  echo "[next-wave] Reusing completed scratch binary cCRE GAT run."
fi

echo
echo "[next-wave] 7/8 Binary cCRE frozen pretrained GAT"
if [[ "$FORCE_RERUN_CCRE_BINARY_GAT" == "1" ]] || ! latest_summary_exists "$CCRE_BINARY_GAT_FROZEN_OUT"; then
  "$PYTHON_BIN" -m graphgenomefm ccre-gat \
    --data-dir "$HPRC_DATA_DIR" \
    --benchmark-dir "$CCRE_BENCHMARK" \
    --node-labels "$NODE_LABELS" \
    --out-dir "$CCRE_BINARY_GAT_FROZEN_OUT" \
    --task binary \
    --pretrained-checkpoint "$CCRE_PRETRAINED_CKPT" \
    --freeze-backbone \
    --keep-is-grch38 \
    --test-chrs chr8 chr19 chr22 \
    --val-chrs chr16 \
    --epochs "$CCRE_EPOCHS" \
    --patience "$CCRE_PATIENCE" \
    --seed "$SEED" \
    --device "$DEVICE"
else
  echo "[next-wave] Reusing completed frozen binary cCRE GAT run."
fi

echo
echo "[next-wave] 8/8 Binary cCRE fine-tuned pretrained GAT"
if [[ "$FORCE_RERUN_CCRE_BINARY_GAT" == "1" ]] || ! latest_summary_exists "$CCRE_BINARY_GAT_FINETUNE_OUT"; then
  "$PYTHON_BIN" -m graphgenomefm ccre-gat \
    --data-dir "$HPRC_DATA_DIR" \
    --benchmark-dir "$CCRE_BENCHMARK" \
    --node-labels "$NODE_LABELS" \
    --out-dir "$CCRE_BINARY_GAT_FINETUNE_OUT" \
    --task binary \
    --pretrained-checkpoint "$CCRE_PRETRAINED_CKPT" \
    --keep-is-grch38 \
    --test-chrs chr8 chr19 chr22 \
    --val-chrs chr16 \
    --epochs "$CCRE_EPOCHS" \
    --patience "$CCRE_PATIENCE" \
    --seed "$SEED" \
    --device "$DEVICE"
else
  echo "[next-wave] Reusing completed fine-tuned binary cCRE GAT run."
fi

echo
echo "[next-wave] Regenerate master experiment status table"
"$PYTHON_BIN" scripts/summarize_experiment_status.py

echo
echo "[next-wave] Done."
echo "[next-wave] Hard-negative pretrain run: $HARDNEG_RUN"
echo "[next-wave] Log: $LOG_FILE"
