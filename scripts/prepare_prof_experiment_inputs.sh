#!/usr/bin/env bash
set -euo pipefail

# Prepare shared inputs required by scripts/run_prof_suggested_experiments.sh.
# This builds the distance-matched HPRC benchmark, the base HPRC pretraining
# checkpoints used by frozen-transfer tasks, and the expanded HGSVC benchmark
# when the raw HGSVC GFA is available.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

HPRC_DATA_DIR="${HPRC_DATA_DIR:-data/hprc}"
HPRC_BENCHMARK="${HPRC_BENCHMARK:-data/hprc/benchmark_paper_hardneg}"
HPRC_PRETRAIN_OUT="${HPRC_PRETRAIN_OUT:-results/hprc/pretrain_paper_hardneg}"
HPRC_NODE_LABELS="${HPRC_NODE_LABELS:-}"
ENCODE_BED="${ENCODE_BED:-data/encode/GRCh38-human-cCREs.bed}"

HGSVC_GFA="${HGSVC_GFA:-data/hgsvc3/hgsvc3-2024-02-23-mc-chm13.sv.gfa.gz}"
HGSVC_DATA_DIR="${HGSVC_DATA_DIR:-data/hgsvc3_expanded}"
HGSVC_BENCHMARK="${HGSVC_BENCHMARK:-data/hgsvc3_expanded/benchmark_chr19_chr21_chr22_chrY_hardneg}"
HGSVC_TARGETS=(${HGSVC_TARGETS:-chr19 chr21 chr22 chrY})

WINDOW_BP="${WINDOW_BP:-50000}"
N_WINDOWS="${N_WINDOWS:-10}"
NEGATIVE_TOL_BP="${NEGATIVE_TOL_BP:-1000}"
NEGATIVE_TOL_FRAC="${NEGATIVE_TOL_FRAC:-0.10}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
PRETRAIN_PATIENCE="${PRETRAIN_PATIENCE:-20}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[prepare] missing required file: $path" >&2
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

echo "[prepare] repo: $REPO_ROOT"
echo "[prepare] python: $PYTHON_BIN"
echo "[prepare] device: $DEVICE"

require_file "$HPRC_DATA_DIR/full_segments.csv"
require_file "$HPRC_DATA_DIR/full_links.csv"

if [[ -f "$HPRC_BENCHMARK/manifest.csv" ]]; then
  echo "[prepare] Reusing HPRC benchmark: $HPRC_BENCHMARK/manifest.csv"
else
  echo "[prepare] Building HPRC distance-matched benchmark"
  "$PYTHON_BIN" -m graphgenomefm make-benchmark \
    --data-dir "$HPRC_DATA_DIR" \
    --out-dir "$HPRC_BENCHMARK" \
    --targets all \
    --closures strict 1hop \
    --n-windows "$N_WINDOWS" \
    --window-bp "$WINDOW_BP" \
    --negative-sampler distance_matched \
    --negative-tol-bp "$NEGATIVE_TOL_BP" \
    --negative-tol-frac "$NEGATIVE_TOL_FRAC" \
    --no-network-analysis \
    --no-viz
fi

STRICT_CKPT="$(latest_file_in_runs "$HPRC_PRETRAIN_OUT" 'ckpt_strict__*.pt')"
ONEHOP_CKPT="$(latest_file_in_runs "$HPRC_PRETRAIN_OUT" 'ckpt_1hop__*.pt')"

if [[ -f "$STRICT_CKPT" && -f "$ONEHOP_CKPT" ]]; then
  echo "[prepare] Reusing HPRC checkpoints:"
  echo "[prepare]   strict: $STRICT_CKPT"
  echo "[prepare]   1hop:   $ONEHOP_CKPT"
else
  echo "[prepare] Running base HPRC hard-negative pretraining"
  "$PYTHON_BIN" -m graphgenomefm pretrain \
    --data-dir "$HPRC_DATA_DIR" \
    --benchmark-dir "$HPRC_BENCHMARK" \
    --out-dir "$HPRC_PRETRAIN_OUT" \
    --test-chrs chr1 chr8 chr19 chrY \
    --val-chrs chr16 \
    --epochs "$PRETRAIN_EPOCHS" \
    --patience "$PRETRAIN_PATIENCE" \
    --seed "$SEED" \
    --device "$DEVICE"

  STRICT_CKPT="$(latest_file_in_runs "$HPRC_PRETRAIN_OUT" 'ckpt_strict__*.pt')"
  ONEHOP_CKPT="$(latest_file_in_runs "$HPRC_PRETRAIN_OUT" 'ckpt_1hop__*.pt')"
fi

require_file "$STRICT_CKPT"
require_file "$ONEHOP_CKPT"

if [[ -z "$HPRC_NODE_LABELS" ]]; then
  HPRC_NODE_LABELS="$(latest_file_in_runs "$HPRC_DATA_DIR/ccre" 'node_labels.csv.gz')"
fi

if [[ -f "$HPRC_NODE_LABELS" ]]; then
  echo "[prepare] Reusing cCRE node labels: $HPRC_NODE_LABELS"
elif [[ -f "$ENCODE_BED" ]]; then
  echo "[prepare] Mapping ENCODE cCRE intervals to HPRC graph nodes"
  "$PYTHON_BIN" -m graphgenomefm label-ccre \
    --data-dir "$HPRC_DATA_DIR" \
    --encode-bed "$ENCODE_BED" \
    --out-dir "$HPRC_DATA_DIR/ccre"
  HPRC_NODE_LABELS="$(latest_file_in_runs "$HPRC_DATA_DIR/ccre" 'node_labels.csv.gz')"
else
  echo "[prepare] Skipping cCRE labeling because ENCODE BED is missing: $ENCODE_BED"
fi

if [[ -f "$HGSVC_DATA_DIR/full_segments.csv.gz" && -f "$HGSVC_DATA_DIR/full_links.csv.gz" ]]; then
  echo "[prepare] Reusing HGSVC cleaned tables: $HGSVC_DATA_DIR"
elif [[ -f "$HGSVC_GFA" ]]; then
  echo "[prepare] Parsing expanded HGSVC graph"
  "$PYTHON_BIN" -m graphgenomefm parse-gfa \
    --gfa "$HGSVC_GFA" \
    --out-dir "$HGSVC_DATA_DIR" \
    --target-prefix 'id=CHM13|' \
    --target-sns "${HGSVC_TARGETS[@]}" \
    --include-link-neighbors
else
  echo "[prepare] Skipping HGSVC parse because raw GFA is missing: $HGSVC_GFA"
fi

if [[ -f "$HGSVC_DATA_DIR/full_segments.csv.gz" && -f "$HGSVC_DATA_DIR/full_links.csv.gz" ]]; then
  if [[ -f "$HGSVC_BENCHMARK/manifest.csv" ]]; then
    echo "[prepare] Reusing HGSVC benchmark: $HGSVC_BENCHMARK/manifest.csv"
  else
    echo "[prepare] Building HGSVC distance-matched benchmark"
    "$PYTHON_BIN" -m graphgenomefm make-benchmark \
      --data-dir "$HGSVC_DATA_DIR" \
      --out-dir "$HGSVC_BENCHMARK" \
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
  fi
fi

echo "[prepare] Done."
echo "[prepare] Use these checkpoint overrides if needed:"
echo "export HPRC_STRICT_CKPT=\"$STRICT_CKPT\""
echo "export HPRC_1HOP_CKPT=\"$ONEHOP_CKPT\""
if [[ -f "$HPRC_NODE_LABELS" ]]; then
  echo "export HPRC_NODE_LABELS=\"$HPRC_NODE_LABELS\""
fi
