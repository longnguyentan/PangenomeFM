#!/bin/bash
set -euo pipefail
cd /home/tuv43532/PangenomeFM
export PATH=/home/tuv43532/miniconda3/envs/pangenomefm-server/bin:$PATH
export PYTHONPATH="$PWD:$PWD/src" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec > results/entex/v1/server/full_p0.log 2>&1
trap 'printf "%s\n" "$?" > results/entex/v1/server/full_p0.exit' EXIT
python -m pytest tests/test_entex.py -q
python scripts/server/run_entex_probe_matrix.py --loci data/entex/v1/p0_loci.parquet --mapping-dir data/entex/v1/mapping --full-segments server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz --manifest server_workspace/data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv --feature-cache server_workspace/data/processed/hprc_r2_ccre_screen_v4_features.npz --sequence-cache server_workspace/results/frozen_sequence_fm_cache_20260815/hprc_target_union_sequence_fm.npz --results-root server_workspace/results/full_multicohort_server_20260806 --out-root results/entex/v1/p0 --topology-cache-root results/entex/v1/topology_cache --gpus 0,1,2,3
