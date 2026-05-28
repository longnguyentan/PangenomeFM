#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
ALLOW_DEVICE_FALLBACK="${ALLOW_DEVICE_FALLBACK:-0}"

HPRC_DATA_DIR="${HPRC_DATA_DIR:-data/hprc}"
HGSVC_DATA_DIR="${HGSVC_DATA_DIR:-data/hgsvc3}"
ENCODE_BED="${ENCODE_BED:-data/encode/GRCh38-human-cCREs.bed}"
HGSVC_GFA="${HGSVC_GFA:-data/hgsvc3/hgsvc3-2024-02-23-mc-chm13.sv.gfa.gz}"

HPRC_BENCHMARK="${HPRC_BENCHMARK:-data/hprc/benchmark_paper}"
HGSVC_BENCHMARK="${HGSVC_BENCHMARK:-data/hgsvc3/benchmark_chr22}"

WINDOW_BP="${WINDOW_BP:-50000}"
N_WINDOWS="${N_WINDOWS:-10}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
PRETRAIN_PATIENCE="${PRETRAIN_PATIENCE:-20}"
CCRE_EPOCHS="${CCRE_EPOCHS:-60}"
CCRE_PATIENCE="${CCRE_PATIENCE:-15}"
RUN_TESTS="${RUN_TESTS:-1}"
FORCE_REBUILD_HPRC_BENCHMARK="${FORCE_REBUILD_HPRC_BENCHMARK:-0}"
FORCE_REBUILD_HGSVC_DATA="${FORCE_REBUILD_HGSVC_DATA:-0}"
FORCE_REBUILD_HGSVC_BENCHMARK="${FORCE_REBUILD_HGSVC_BENCHMARK:-0}"

PRETRAIN_OUT="${PRETRAIN_OUT:-results/hprc/pretrain_paper}"
HGSVC_EXTERNAL_STRICT_OUT="${HGSVC_EXTERNAL_STRICT_OUT:-results/hgsvc3/external_eval_chr22_strict}"
HGSVC_EXTERNAL_1HOP_OUT="${HGSVC_EXTERNAL_1HOP_OUT:-results/hgsvc3/external_eval_chr22_1hop}"
CCRE_BINARY_OUT="${CCRE_BINARY_OUT:-results/hprc/ccre_binary_baseline}"
CCRE_MULTI_OUT="${CCRE_MULTI_OUT:-results/hprc/ccre_baseline}"
CCRE_GAT_SCRATCH_OUT="${CCRE_GAT_SCRATCH_OUT:-results/hprc/ccre_gat_scratch}"
CCRE_GAT_FROZEN_OUT="${CCRE_GAT_FROZEN_OUT:-results/hprc/ccre_gat_pretrained_frozen}"
CCRE_GAT_FINETUNE_OUT="${CCRE_GAT_FINETUNE_OUT:-results/hprc/ccre_gat_pretrained_finetune}"

LOG_DIR="${LOG_DIR:-logs}"
LOG_TO_FILE="${LOG_TO_FILE:-1}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_lab_experiments_$(date +%Y%m%d_%H%M%S).log"
if [[ "$LOG_TO_FILE" == "1" ]]; then
  if (: > >(true)) 2>/dev/null; then
    exec > >(tee -a "$LOG_FILE") 2>&1
  else
    echo "[run] WARNING: shell does not support tee process substitution here; continuing without file logging."
    echo "[run] Set LOG_TO_FILE=0 to silence this warning."
  fi
fi

echo "[run] repo: $REPO_ROOT"
echo "[run] python: $PYTHON_BIN"
echo "[run] log: $LOG_FILE"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[run] ERROR: required file not found: $path" >&2
    exit 2
  fi
}

latest_run_dir() {
  local root="$1"
  find "$root" -maxdepth 1 -type d -name 'run_*' 2>/dev/null | sort | tail -n 1
}

latest_file() {
  local root="$1"
  local pattern="$2"
  find "$root" -type f -name "$pattern" 2>/dev/null | sort | tail -n 1
}

require_file "$HPRC_DATA_DIR/full_segments.csv"
require_file "$HPRC_DATA_DIR/full_links.csv"
require_file "$ENCODE_BED"
require_file "$HGSVC_GFA"

detect_best_device() {
  "$PYTHON_BIN" - <<'PY'
import torch
if torch.cuda.is_available():
    print("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
}

check_requested_device() {
  local requested="$1"
  "$PYTHON_BIN" - "$requested" <<'PY'
import sys
import torch

requested = sys.argv[1]
if requested == "cuda":
    ok = torch.cuda.is_available()
    detail = torch.version.cuda or "Torch was not compiled with CUDA enabled"
elif requested == "mps":
    ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    detail = "MPS backend available" if ok else "MPS backend unavailable"
elif requested == "cpu":
    ok = True
    detail = "CPU"
else:
    ok = False
    detail = f"unknown device {requested}"

print(f"{int(ok)}\t{detail}")
PY
}

if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(detect_best_device)"
else
  DEVICE_STATUS="$(check_requested_device "$DEVICE")"
  DEVICE_OK="${DEVICE_STATUS%%$'\t'*}"
  DEVICE_DETAIL="${DEVICE_STATUS#*$'\t'}"
  if [[ "$DEVICE_OK" != "1" ]]; then
    BEST_DEVICE="$(detect_best_device)"
    if [[ "$ALLOW_DEVICE_FALLBACK" == "1" ]]; then
      echo "[run] WARNING: requested DEVICE=$DEVICE is unavailable: $DEVICE_DETAIL"
      echo "[run] Falling back to DEVICE=$BEST_DEVICE because ALLOW_DEVICE_FALLBACK=1"
      DEVICE="$BEST_DEVICE"
    else
      echo "[run] ERROR: requested DEVICE=$DEVICE is unavailable: $DEVICE_DETAIL" >&2
      echo "[run] Best available device in this environment appears to be: $BEST_DEVICE" >&2
      echo "[run] To run locally without CUDA: DEVICE=auto bash scripts/run_lab_experiments.sh" >&2
      echo "[run] To force CPU: DEVICE=cpu bash scripts/run_lab_experiments.sh" >&2
      echo "[run] On a CUDA lab node, install a CUDA-enabled PyTorch build, then rerun." >&2
      echo "[run] Official selector: https://pytorch.org/get-started/locally/" >&2
      echo "[run] Example CUDA 12.8 install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128" >&2
      echo "[run] Example CUDA 12.6 install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126" >&2
      echo "[run] Or permit fallback with: ALLOW_DEVICE_FALLBACK=1 DEVICE=cuda bash scripts/run_lab_experiments.sh" >&2
      exit 2
    fi
  fi
fi
echo "[run] device: $DEVICE"

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "[run] Running unit tests"
  "$PYTHON_BIN" -m pytest -q
fi

echo "[run] Checking HPRC data"
"$PYTHON_BIN" -m graphgenomefm check-data --data-dir "$HPRC_DATA_DIR"

echo "[run] Building corrected all-chromosome HPRC benchmark"
if [[ -f "$HPRC_BENCHMARK/manifest.csv" && "$FORCE_REBUILD_HPRC_BENCHMARK" != "1" ]]; then
  echo "[run] Reusing existing HPRC benchmark manifest: $HPRC_BENCHMARK/manifest.csv"
else
  "$PYTHON_BIN" -m graphgenomefm make-benchmark \
    --data-dir "$HPRC_DATA_DIR" \
    --out-dir "$HPRC_BENCHMARK" \
    --targets all \
    --n-windows "$N_WINDOWS" \
    --window-bp "$WINDOW_BP" \
    --no-network-analysis \
    --no-viz
fi

echo "[run] Running full HPRC shared pretraining"
"$PYTHON_BIN" -m graphgenomefm pretrain \
  --data-dir "$HPRC_DATA_DIR" \
  --benchmark-dir "$HPRC_BENCHMARK" \
  --out-dir "$PRETRAIN_OUT" \
  --val-chrs chr16 \
  --test-chrs chr1 chr8 chr19 chrY \
  --epochs "$PRETRAIN_EPOCHS" \
  --patience "$PRETRAIN_PATIENCE" \
  --seed "$SEED" \
  --device "$DEVICE"

PRETRAIN_RUN="$(latest_run_dir "$PRETRAIN_OUT")"
if [[ -z "$PRETRAIN_RUN" ]]; then
  echo "[run] ERROR: no pretraining run directory found under $PRETRAIN_OUT" >&2
  exit 2
fi
STRICT_CKPT="$(latest_file "$PRETRAIN_RUN" 'ckpt_strict*.pt')"
HOP1_CKPT="$(latest_file "$PRETRAIN_RUN" 'ckpt_1hop*.pt')"
require_file "$STRICT_CKPT"
require_file "$HOP1_CKPT"
echo "[run] strict checkpoint: $STRICT_CKPT"
echo "[run] 1hop checkpoint:   $HOP1_CKPT"

echo "[run] Parsing HGSVC3 chr22 target plus direct neighbors"
if [[ -f "$HGSVC_DATA_DIR/full_segments.csv.gz" && -f "$HGSVC_DATA_DIR/full_links.csv.gz" && "$FORCE_REBUILD_HGSVC_DATA" != "1" ]]; then
  echo "[run] Reusing existing HGSVC3 cleaned tables under $HGSVC_DATA_DIR"
else
  "$PYTHON_BIN" -m graphgenomefm parse-gfa \
    --gfa "$HGSVC_GFA" \
    --out-dir "$HGSVC_DATA_DIR" \
    --target-sns chr22 \
    --target-prefix 'id=CHM13|' \
    --include-link-neighbors
fi

echo "[run] Checking HGSVC3 chr22 cleaned data"
"$PYTHON_BIN" -m graphgenomefm check-data --data-dir "$HGSVC_DATA_DIR"

echo "[run] Building HGSVC3 chr22 benchmark"
if [[ -f "$HGSVC_BENCHMARK/manifest.csv" && "$FORCE_REBUILD_HGSVC_BENCHMARK" != "1" ]]; then
  echo "[run] Reusing existing HGSVC3 benchmark manifest: $HGSVC_BENCHMARK/manifest.csv"
else
  "$PYTHON_BIN" -m graphgenomefm make-benchmark \
    --data-dir "$HGSVC_DATA_DIR" \
    --out-dir "$HGSVC_BENCHMARK" \
    --targets chr22 \
    --target-prefix 'id=CHM13|' \
    --n-windows "$N_WINDOWS" \
    --window-bp "$WINDOW_BP" \
    --no-network-analysis \
    --no-viz
fi

echo "[run] External HPRC-to-HGSVC3 strict evaluation"
"$PYTHON_BIN" -m graphgenomefm eval-external \
  --data-dir "$HGSVC_DATA_DIR" \
  --benchmark-dir "$HGSVC_BENCHMARK" \
  --checkpoint "$STRICT_CKPT" \
  --out-dir "$HGSVC_EXTERNAL_STRICT_OUT" \
  --split all \
  --device "$DEVICE"

echo "[run] External HPRC-to-HGSVC3 1-hop evaluation"
"$PYTHON_BIN" -m graphgenomefm eval-external \
  --data-dir "$HGSVC_DATA_DIR" \
  --benchmark-dir "$HGSVC_BENCHMARK" \
  --checkpoint "$HOP1_CKPT" \
  --out-dir "$HGSVC_EXTERNAL_1HOP_OUT" \
  --split all \
  --device "$DEVICE"

echo "[run] Mapping ENCODE cCRE labels to HPRC graph nodes"
"$PYTHON_BIN" -m graphgenomefm label-ccre \
  --data-dir "$HPRC_DATA_DIR" \
  --encode-bed "$ENCODE_BED" \
  --out-dir "$HPRC_DATA_DIR/ccre"

echo "[run] Running binary cCRE logistic baseline"
"$PYTHON_BIN" -m graphgenomefm ccre-binary-baseline \
  --data-dir "$HPRC_DATA_DIR" \
  --test-chrs chr8 chr19 chr22 \
  --val-chr chr16 \
  --out-dir "$CCRE_BINARY_OUT" \
  --seed "$SEED"

echo "[run] Running multiclass cCRE logistic baseline"
"$PYTHON_BIN" -m graphgenomefm ccre-baseline \
  --data-dir "$HPRC_DATA_DIR" \
  --test-chrs chr8 chr19 chr22 \
  --val-chr chr16 \
  --out-dir "$CCRE_MULTI_OUT" \
  --seed "$SEED"

echo "[run] Running scratch cCRE GAT"
"$PYTHON_BIN" -m graphgenomefm ccre-gat \
  --data-dir "$HPRC_DATA_DIR" \
  --benchmark-dir "$HPRC_BENCHMARK" \
  --test-chrs chr8 chr19 chr22 \
  --val-chrs chr16 \
  --out-dir "$CCRE_GAT_SCRATCH_OUT" \
  --epochs "$CCRE_EPOCHS" \
  --patience "$CCRE_PATIENCE" \
  --seed "$SEED" \
  --device "$DEVICE"

echo "[run] Running frozen pretrained cCRE GAT"
"$PYTHON_BIN" -m graphgenomefm ccre-gat \
  --data-dir "$HPRC_DATA_DIR" \
  --benchmark-dir "$HPRC_BENCHMARK" \
  --test-chrs chr8 chr19 chr22 \
  --val-chrs chr16 \
  --out-dir "$CCRE_GAT_FROZEN_OUT" \
  --pretrained-checkpoint "$STRICT_CKPT" \
  --freeze-backbone \
  --keep-is-grch38 \
  --epochs "$CCRE_EPOCHS" \
  --patience "$CCRE_PATIENCE" \
  --seed "$SEED" \
  --device "$DEVICE"

echo "[run] Running fine-tuned pretrained cCRE GAT"
"$PYTHON_BIN" -m graphgenomefm ccre-gat \
  --data-dir "$HPRC_DATA_DIR" \
  --benchmark-dir "$HPRC_BENCHMARK" \
  --test-chrs chr8 chr19 chr22 \
  --val-chrs chr16 \
  --out-dir "$CCRE_GAT_FINETUNE_OUT" \
  --pretrained-checkpoint "$STRICT_CKPT" \
  --keep-is-grch38 \
  --epochs "$CCRE_EPOCHS" \
  --patience "$CCRE_PATIENCE" \
  --seed "$SEED" \
  --device "$DEVICE"

echo "[run] DONE"
echo "[run] Main outputs:"
echo "  HPRC pretrain:              $PRETRAIN_RUN"
echo "  HGSVC strict external eval: $(latest_run_dir "$HGSVC_EXTERNAL_STRICT_OUT")"
echo "  HGSVC 1-hop external eval:  $(latest_run_dir "$HGSVC_EXTERNAL_1HOP_OUT")"
echo "  cCRE binary baseline:       $(latest_run_dir "$CCRE_BINARY_OUT")"
echo "  cCRE multiclass baseline:   $(latest_run_dir "$CCRE_MULTI_OUT")"
echo "  cCRE scratch GAT:           $(latest_run_dir "$CCRE_GAT_SCRATCH_OUT")"
echo "  cCRE frozen pretrained:     $(latest_run_dir "$CCRE_GAT_FROZEN_OUT")"
echo "  cCRE fine-tuned pretrained: $(latest_run_dir "$CCRE_GAT_FINETUNE_OUT")"
