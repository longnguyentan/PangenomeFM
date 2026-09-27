#!/usr/bin/env bash
# Reuse canonical sources/caches from the existing main checkout.
set -euo pipefail
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
main_root=${PANGENOMEFM_MAIN_CHECKOUT:-/home/tuv43532/PangenomeFM}
panel_data=${PANGENOMEFM_PANEL_DATA:-data/entex/extension_20260927}
panel_out=${PANGENOMEFM_PANEL_OUT:-results/entex/extension_20260927}
graph="$main_root/server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz"
common=(
  --config configs/entex_v1.json --task p2
  --full-segments "$graph"
  --manifest "$main_root/server_workspace/data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv"
  --feature-cache "$main_root/server_workspace/data/processed/hprc_r2_ccre_screen_v4_features.npz"
  --sequence-cache "$main_root/server_workspace/results/frozen_sequence_fm_cache_20260815/hprc_target_union_sequence_fm.npz"
  --results-root "$main_root/server_workspace/results/full_multicohort_server_20260806"
  --topology-cache-root "$main_root/results/entex/v1/topology_cache_all_reference"
  --cache-all-reference-targets
)
case "${1:-}" in
  prepare)
    python -m tasks.entex.snv --source "$main_root/data/entex/sources/hetSNVs_default_AS.tsv" \
      --assays atac h3k4me3 h3k27me3 --out-dir "$panel_data"
    python -m tasks.entex.panel --data-dir "$panel_data"
    for assay in atac h3k4me3 h3k27me3; do
      python -m tasks.entex.mapping --loci "$panel_data/${assay}_loci.parquet" \
        --full-segments "$graph" --out-dir "$panel_data/mapping_$assay"
    done
    ;;
  smoke|matrix)
    for assay in atac h3k4me3 h3k27me3; do
      args=("${common[@]}" --subtask "$assay" --loci "$panel_data/${assay}_loci.parquet"
        --measurements "$panel_data/${assay}_measurements.parquet" --mapping-dir "$panel_data/mapping_$assay")
      if [[ "$1" == smoke ]]; then
        python -m tasks.entex.probe "${args[@]}" --out-root "$panel_out/smoke/$assay" \
          --device cpu --folds fold_a --seeds 42 --contexts strict
      else
        python -c 'import json,sys; from pathlib import Path; a=json.loads(Path(sys.argv[1]).read_text()); assert a["status"] == "complete"; assert a["joint_feature_coverage"] >= .95' \
          "$panel_out/smoke/$assay/fold_a/seed_42/strict/audit.json"
        python scripts/server/run_entex_probe_matrix.py "${args[@]}" \
          --out-root "$panel_out/matrix/$assay" --gpus "${PANGENOMEFM_PANEL_GPUS:-2,3}"
        python -m tasks.entex.analyze --probe-root "$panel_out/matrix/$assay" \
          --complexity "$main_root/server_workspace/results/complexity_context_v2_20260815/native_complexity_v2/complexity_features.tsv" \
          --out-dir "$panel_out/analysis/$assay"
        python -m tasks.entex.measurement_followups --assay "$assay" \
          --protocol configs/entex_extension_20260927.json \
          --probe-root "$panel_out/matrix/$assay" --out-dir "$panel_out/followups/$assay"
      fi
    done
    if [[ "$1" == matrix ]]; then
      python -m tasks.entex.followup_report --assays atac h3k4me3 h3k27me3 \
        --root "$panel_out/followups" --out-dir "$panel_out/report"
    fi
    ;;
  *) printf '%s\n' 'Usage: bash scripts/server/run_entex_extension_20260927.sh {prepare|smoke|matrix}' >&2; exit 2 ;;
esac
