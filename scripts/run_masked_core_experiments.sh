#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cpu}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
BENCHMARK_DIR="${BENCHMARK_DIR:-data/hprc/benchmark_paper_hardneg}"
DATA_DIR="${DATA_DIR:-data/hprc}"
OUT_ROOT="${OUT_ROOT:-results/hprc}"

COMMON_ARGS=(
  -m graphgenomefm pretrain
  --data-dir "$DATA_DIR"
  --benchmark-dir "$BENCHMARK_DIR"
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --device "$DEVICE"
  --mask-query-edges
  --save-predictions
)

echo "[masked-core] DEVICE=$DEVICE EPOCHS=$EPOCHS PATIENCE=$PATIENCE"

PYTHONPATH=src "$PYTHON_BIN" "${COMMON_ARGS[@]}" \
  --out-dir "$OUT_ROOT/pretrain_masked_train_full"

PYTHONPATH=src "$PYTHON_BIN" "${COMMON_ARGS[@]}" \
  --stream-mode coordinate \
  --out-dir "$OUT_ROOT/pretrain_masked_train_coordinate"

PYTHONPATH=src "$PYTHON_BIN" "${COMMON_ARGS[@]}" \
  --stream-mode graph \
  --out-dir "$OUT_ROOT/pretrain_masked_train_graph"

PYTHONPATH=src "$PYTHON_BIN" "${COMMON_ARGS[@]}" \
  --no-fusion-gate \
  --out-dir "$OUT_ROOT/pretrain_masked_train_nogate"
