#!/usr/bin/env bash
set -euo pipefail

# Reproduce one training seed for the leakage-audited, locus-matched benchmark.
# Outputs are versioned by resolve_run_dir, so an existing run is not overwritten.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${GRAPHGENOMEFM_PYTHON:-python}"
device="${GRAPHGENOMEFM_DEVICE:-cpu}"
seed="${1:-42}"
requested_variant="${2:-all}"

export PYTHONPATH="${project_root}/src"

manifest="${project_root}/data/hprc/benchmark_matched_nonoverlap_v2_review_20260804/manifest.csv"
segments="${project_root}/data/hprc/full_segments.csv"
result_root="${project_root}/results/review_20260804/matched_training_v2"

if [[ ! -s "${manifest}" || ! -s "${segments}" ]]; then
  echo "Missing benchmark manifest or full segment table." >&2
  exit 2
fi

run_variant() {
  local variant="$1"
  local command=(
    "${python_bin}" -m training.pretrain
    --manifest "${manifest}"
    --full_segments "${segments}"
    --out_dir "${result_root}/${variant}_seed${seed}"
    --closures strict 1hop
    --dual_stream
    --multiscale_rope --n_rope_scales 3
    --orientation_rope
    --adaptive_window --adaptive_window_base 32 --adaptive_window_alpha 4
    --focal_loss --focal_gamma 2
    --drop_edge --drop_edge_rate 0.1
    --mask_query_edges
    --save_predictions
    --epochs 100 --patience 20
    --seed "${seed}"
    --split_seed 20260804
    --device "${device}"
    --test_chrs
      "GRCh38#0#chr1" "GRCh38#0#chr8" "GRCh38#0#chr19" "GRCh38#0#chrY"
    --val_chrs "GRCh38#0#chr16"
  )
  case "${variant}" in
    full) ;;
    graph) command+=(--stream_mode graph) ;;
    coordinate) command+=(--stream_mode coordinate) ;;
    nogate) command+=(--no_fusion_gate) ;;
    *) echo "Unknown variant: ${variant}" >&2; exit 2 ;;
  esac

  "${command[@]}"
}

if [[ "${requested_variant}" == "all" ]]; then
  for variant in full graph coordinate nogate; do
    run_variant "${variant}"
  done
else
  run_variant "${requested_variant}"
fi
