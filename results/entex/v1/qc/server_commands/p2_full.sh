#!/bin/bash
set -euo pipefail
cd /home/tuv43532/PangenomeFM
export PATH=/home/tuv43532/miniconda3/envs/pangenomefm-server/bin:$PATH
export PYTHONPATH="/tmp/pangenomefm-entex-code-89f34a9:/tmp/pangenomefm-entex-code-89f34a9/src" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec > results/entex/v1/server/p2_full.log 2>&1
trap 'printf "%s\n" "$?" > results/entex/v1/server/p2_full.exit' EXIT
# Avoid oversubscribing the server while the P0 sensitivity matrix is active.
for gate in p2_smoke sensitivities; do
  while [ ! -f "results/entex/v1/server/${gate}.exit" ]; do sleep 30; done
  test "$(cat "results/entex/v1/server/${gate}.exit")" = 0
done
python /tmp/pangenomefm-entex-code-89f34a9/scripts/server/run_entex_probe_matrix.py --task p2 --subtask ctcf --measurements data/entex/v1/p2/ctcf_measurements.parquet --loci data/entex/v1/p2/ctcf_loci.parquet --mapping-dir data/entex/v1/p2/mapping_ctcf --full-segments server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz --manifest server_workspace/data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv --feature-cache server_workspace/data/processed/hprc_r2_ccre_screen_v4_features.npz --sequence-cache server_workspace/results/frozen_sequence_fm_cache_20260815/hprc_target_union_sequence_fm.npz --results-root server_workspace/results/full_multicohort_server_20260806 --out-root results/entex/v1/p2/ctcf --topology-cache-root results/entex/v1/topology_cache_all_reference --cache-all-reference-targets --gpus 0,1,2,3
python -m tasks.entex.analyze --probe-root results/entex/v1/p2/ctcf --complexity server_workspace/results/complexity_context_v2_20260815/native_complexity_v2/complexity_features.tsv --out-dir results/entex/v1/p2_analysis/ctcf
python /tmp/pangenomefm-entex-code-89f34a9/scripts/server/run_entex_probe_matrix.py --task p2 --subtask h3k27ac --measurements data/entex/v1/p2/h3k27ac_measurements.parquet --loci data/entex/v1/p2/h3k27ac_loci.parquet --mapping-dir data/entex/v1/p2/mapping_h3k27ac --full-segments server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz --manifest server_workspace/data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv --feature-cache server_workspace/data/processed/hprc_r2_ccre_screen_v4_features.npz --sequence-cache server_workspace/results/frozen_sequence_fm_cache_20260815/hprc_target_union_sequence_fm.npz --results-root server_workspace/results/full_multicohort_server_20260806 --out-root results/entex/v1/p2/h3k27ac --topology-cache-root results/entex/v1/topology_cache_all_reference --cache-all-reference-targets --gpus 0,1,2,3
python -m tasks.entex.analyze --probe-root results/entex/v1/p2/h3k27ac --complexity server_workspace/results/complexity_context_v2_20260815/native_complexity_v2/complexity_features.tsv --out-dir results/entex/v1/p2_analysis/h3k27ac
