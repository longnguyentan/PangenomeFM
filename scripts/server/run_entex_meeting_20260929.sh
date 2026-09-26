#!/usr/bin/env bash
# From the repository root, with the existing pangenomefm-server env activated.
set -euo pipefail
export PYTHONPATH=src:.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
entex_data=data/entex/meeting_20260929/rna
entex_out=results/entex/meeting_20260929
entex_graph=server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz
common=(
  --config configs/entex_v1.json --task p2 --subtask rna
  --loci "$entex_data/rna_loci.parquet"
  --measurements "$entex_data/rna_measurements.parquet"
  --mapping-dir "$entex_data/mapping" --full-segments "$entex_graph"
  --manifest server_workspace/data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv
  --feature-cache server_workspace/data/processed/hprc_r2_ccre_screen_v4_features.npz
  --sequence-cache server_workspace/results/frozen_sequence_fm_cache_20260815/hprc_target_union_sequence_fm.npz
  --results-root server_workspace/results/full_multicohort_server_20260806
  --topology-cache-root results/entex/v1/topology_cache_all_reference
  --cache-all-reference-targets
)
case "${1:-}" in
  rna-smoke)
    if [[ ! -f "$entex_data/mapping/mapping_qc.json" ]]; then
      python -m tasks.entex.mapping --loci "$entex_data/rna_loci.parquet" \
        --full-segments "$entex_graph" --out-dir "$entex_data/mapping"
    fi
    python -m tasks.entex.probe "${common[@]}" --out-root "$entex_out/rna_smoke" \
      --device cpu --folds fold_a --seeds 42 --contexts strict
    ;;
  rna-matrix)
    python -c 'import json; from pathlib import Path; a=json.loads(Path("results/entex/meeting_20260929/rna_smoke/fold_a/seed_42/strict/audit.json").read_text()); assert a["status"] == "complete"'
    python scripts/server/run_entex_probe_matrix.py "${common[@]}" \
      --out-root "$entex_out/rna" --gpus 0
    python -m tasks.entex.analyze --probe-root "$entex_out/rna" \
      --complexity server_workspace/results/complexity_context_v2_20260815/native_complexity_v2/complexity_features.tsv \
      --out-dir "$entex_out/rna_analysis"
    ;;
  followups)
    for assay in ctcf h3k27ac; do
      python -m tasks.entex.measurement_followups --assay "$assay" \
        --probe-root "results/entex/v1/p2/$assay" --out-dir "$entex_out/followups/$assay"
    done
    python -m tasks.entex.followup_report --root "$entex_out/followups" \
      --out-dir "$entex_out/report"
    ;;
  *) printf '%s\n' 'Usage: bash scripts/server/run_entex_meeting_20260929.sh {rna-smoke|rna-matrix|followups}' >&2; exit 2 ;;
esac
