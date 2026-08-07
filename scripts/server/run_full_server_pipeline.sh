#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${PANGENOMEFM_CONFIG:-${REPO_ROOT}/configs/server_full_multicohort_20260806.json}"
PROFILE="${DOWNLOAD_PROFILE:-analysis-full}"
DOWNLOAD_JOBS="${DOWNLOAD_JOBS:-2}"
PREP_JOBS="${PREP_JOBS:-2}"
CONTEXT_JOBS="${CONTEXT_JOBS:-2}"
CUDA_GPUS="${CUDA_GPUS:-0}"

: "${PANGENOMEFM_DATA_ROOT:?Export PANGENOMEFM_DATA_ROOT to high-capacity server storage.}"
: "${PANGENOMEFM_RESULTS_ROOT:?Export PANGENOMEFM_RESULTS_ROOT to a result/checkpoint filesystem.}"

RESULT_DIR="${PANGENOMEFM_RESULTS_ROOT}/full_multicohort_server_20260806"
mkdir -p "${RESULT_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${RESULT_DIR}/matplotlib_cache"

echo "[1/11] Server preflight"
python scripts/server/preflight_server.py \
  --data-root "${PANGENOMEFM_DATA_ROOT}" \
  --results-root "${RESULT_DIR}" \
  --profile "${PROFILE}" \
  --strict

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  echo "[2/11] Resumable downloads and checksum validation"
  python scripts/server/download_full_data.py \
    --data-root "${PANGENOMEFM_DATA_ROOT}" \
    --profile "${PROFILE}" \
    --jobs "${DOWNLOAD_JOBS}"
else
  echo "[2/11] Download skipped by SKIP_DOWNLOAD=1; verifying existing files"
  python scripts/server/download_full_data.py \
    --data-root "${PANGENOMEFM_DATA_ROOT}" \
    --profile "${PROFILE}" \
    --jobs "${DOWNLOAD_JOBS}" \
    --verify-only
fi

echo "[3/11] Capture software and hardware environment"
python scripts/capture_execution_environment.py \
  --out "${RESULT_DIR}/execution_environment.json"
python scripts/server/audit_server_plan.py \
  --config "${CONFIG}" \
  --out "${RESULT_DIR}/server_plan_audit.json"

echo "[4/11] Parallel parsing, GBZ path indexing, validation, and benchmark generation"
python scripts/server/run_parallel_prepare.py \
  --config "${CONFIG}" \
  --jobs "${PREP_JOBS}" \
  --execute

if [[ "${SKIP_SCALING:-0}" != "1" ]]; then
  echo "[5/11] Measured one-epoch scaling pilots"
  FIRST_GPU="${CUDA_GPUS%%,*}"
  for DATASET in hprc_r2 hgsvc3; do
    if [[ "${DATASET}" == "hprc_r2" ]]; then
      MANIFEST="${PANGENOMEFM_DATA_ROOT}/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv"
      SEGMENTS="${PANGENOMEFM_DATA_ROOT}/processed/hprc_r2_sv/full_segments.csv.gz"
    else
      MANIFEST="${PANGENOMEFM_DATA_ROOT}/benchmarks/hgsvc3_pretrain_5mb_paired/manifest.csv"
      SEGMENTS="${PANGENOMEFM_DATA_ROOT}/processed/hgsvc3_sv/full_segments.csv.gz"
    fi
    CUDA_VISIBLE_DEVICES="${FIRST_GPU}" python scripts/run_scaling_benchmark.py \
      --manifest "${MANIFEST}" \
      --full-segments "${SEGMENTS}" \
      --out-dir "${RESULT_DIR}/scaling/${DATASET}" \
      --closure 1hop \
      --train-chromosomes chr1 chr2 chr3 \
      --validation-chromosome chr21 \
      --test-chromosome chr22 \
      --fixed-evaluation-slices 2 \
      --train-counts 4 8 16 32 \
      --device cuda \
      --batch-size 512 \
      --lazy-tensorize \
      --seed 20260806
  done
else
  echo "[5/11] Scaling pilots skipped by SKIP_SCALING=1"
fi

if [[ "${SKIP_FINAL:-0}" != "1" ]]; then
  echo "[6/11] Final all-chromosome models: 5 regimes x 3 seeds x 2 contexts"
  python scripts/server/run_gpu_matrix.py \
    --phase final --config "${CONFIG}" --gpus "${CUDA_GPUS}" --execute
fi

if [[ "${SKIP_FOLDS:-0}" != "1" ]]; then
  echo "[7/11] Rotating chromosome holdouts: 4 regimes x 5 folds x 3 seeds x 2 contexts"
  python scripts/server/run_gpu_matrix.py \
    --phase folds --config "${CONFIG}" --gpus "${CUDA_GPUS}" --execute
fi

if [[ "${SKIP_TRANSFER:-0}" != "1" ]]; then
  echo "[8/11] Separate HPRC/HGSVC and integrated-graph transfer tests"
  python scripts/server/run_gpu_matrix.py \
    --phase transfer --config "${CONFIG}" --gpus "${CUDA_GPUS}" --execute
fi

if [[ "${SKIP_RELEASE:-0}" != "1" ]]; then
  echo "[9/11] HPRC graph-release transfer tests (R1.1 versus R2; never pooled)"
  python scripts/server/run_gpu_matrix.py \
    --phase release --config "${CONFIG}" --gpus "${CUDA_GPUS}" --execute
fi

echo "[10/11] Context/leakage audits, chromosome statistics, confidence intervals, and figures"
python scripts/server/run_server_postprocessing.py \
  --config "${CONFIG}" \
  --context-jobs "${CONTEXT_JOBS}" \
  --n-boot 2000

if [[ "${SKIP_PACKAGE:-0}" != "1" ]]; then
  echo "[11/11] Checksummed server-to-local result bundle (checkpoints excluded)"
  python scripts/server/package_server_results.py \
    --config "${CONFIG}" \
    --require-complete
else
  echo "[11/11] Packaging skipped by SKIP_PACKAGE=1"
fi

echo "Full server workflow complete: ${RESULT_DIR}"
