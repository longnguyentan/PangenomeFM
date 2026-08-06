#!/usr/bin/env bash
#SBATCH --job-name=pangenomefm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-29
#SBATCH --output=slurm/pangenomefm_%A_%a.out
#SBATCH --error=slurm/pangenomefm_%A_%a.err

set -euo pipefail
: "${PANGENOMEFM_REPO:?Export PANGENOMEFM_REPO in the submission environment.}"
: "${PANGENOMEFM_DATA_ROOT:?Export PANGENOMEFM_DATA_ROOT in the submission environment.}"
: "${PANGENOMEFM_RESULTS_ROOT:?Export PANGENOMEFM_RESULTS_ROOT in the submission environment.}"
PHASE="${PHASE:-final}"
CONFIG="${PANGENOMEFM_CONFIG:-${PANGENOMEFM_REPO}/configs/server_full_multicohort_20260806.json}"

cd "${PANGENOMEFM_REPO}"
export PYTHONPATH="${PANGENOMEFM_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
python scripts/server/run_gpu_matrix.py \
  --phase "${PHASE}" \
  --config "${CONFIG}" \
  --gpus 0 \
  --job-index "${SLURM_ARRAY_TASK_ID}" \
  --execute
