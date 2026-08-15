#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-20260815}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"

CONFIG="${CONFIG:-configs/server_full_multicohort_20260806.json}"
RESULTS_ROOT="${RESULTS_ROOT:-server_workspace/results/full_multicohort_server_20260806}"
BASELINE_ROOT="${BASELINE_ROOT:-server_workspace/results/rotating_link_baselines_20260809}"
MANIFEST="${MANIFEST:-server_workspace/data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv}"
OUT_ROOT="${OUT_ROOT:-server_workspace/results/complexity_context_v2_${RUN_TAG}}"
COMPLEXITY_OUT="$OUT_ROOT/native_complexity_v2"
ANALYSIS_OUT="$OUT_ROOT/exact_analysis"

BASELINES=(
  topology_preferential_attachment
  topology_degree_sum
  coordinate_sgd
  sequence_composition_sgd
)
CONTEXTS=(strict 1hop)

audit_is_complete() {
  local audit_path="$1"
  "$PYTHON_BIN" - "$audit_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    status = json.loads(path.read_text()).get("status")
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if status in {"PASS", "complete"} else 1)
PY
}

preserve_incomplete_output() {
  local out_dir="$1"
  if [[ -d "$out_dir" ]]; then
    local archive="${out_dir}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$out_dir" "$archive"
    echo "PRESERVED_INCOMPLETE_OUTPUT=$archive"
  fi
}

if [[ "$MAX_CONCURRENT" -lt 1 || "$MAX_CONCURRENT" -gt 4 ]]; then
  echo "MAX_CONCURRENT must be between 1 and 4" >&2
  exit 2
fi

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT_ROOT/logs"

for required in "$CONFIG" "$MANIFEST"; do
  test -s "$required" || { echo "Missing required input: $required" >&2; exit 2; }
done
test -d "$RESULTS_ROOT" || { echo "Missing results root: $RESULTS_ROOT" >&2; exit 2; }
test -d "$BASELINE_ROOT" || { echo "Missing baseline root: $BASELINE_ROOT" >&2; exit 2; }

if ! audit_is_complete "$COMPLEXITY_OUT/complexity_audit.json"; then
  preserve_incomplete_output "$COMPLEXITY_OUT"
  "$PYTHON_BIN" scripts/extract_graph_complexity.py \
    --manifest "$MANIFEST" \
    --dataset hprc_r2_native_5mb \
    --contexts strict \
    --config configs/complexity_definition_v2.yaml \
    --out-dir "$COMPLEXITY_OUT" \
    2>&1 | tee "$OUT_ROOT/logs/complexity.log"
fi

run_score() {
  local baseline="$1"
  local context="$2"
  local out_dir="$OUT_ROOT/scores/${baseline}/${context}"
  local policy="error"
  if [[ "$context" == "1hop" ]]; then
    policy="exclude"
  fi
  if audit_is_complete "$out_dir/audit.json"; then
    echo "SKIP complete score: $baseline $context"
    return 0
  fi
  preserve_incomplete_output "$out_dir"
  "$PYTHON_BIN" scripts/server/prepare_dense_region_scores.py \
    --config "$CONFIG" \
    --results-root "$RESULTS_ROOT" \
    --baseline-root "$BASELINE_ROOT" \
    --out-dir "$out_dir" \
    --regime hprc_r2 \
    --dataset hprc_r2 \
    --closure "$context" \
    --baseline "$baseline" \
    --canonical-conflict-policy "$policy" \
    --minimum-test-candidates 10 \
    > "$OUT_ROOT/logs/${baseline}.${context}.log" 2>&1
}

for context in "${CONTEXTS[@]}"; do
  pids=()
  names=()
  for baseline in "${BASELINES[@]}"; do
    run_score "$baseline" "$context" &
    pids+=("$!")
    names+=("$baseline.$context")
    if [[ "${#pids[@]}" -ge "$MAX_CONCURRENT" ]]; then
      for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
          echo "FAILED: ${names[$index]}" >&2
          exit 1
        fi
        echo "COMPLETE: ${names[$index]}"
      done
      pids=()
      names=()
    fi
  done
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "FAILED: ${names[$index]}" >&2
      exit 1
    fi
    echo "COMPLETE: ${names[$index]}"
  done
done

score_args=()
for baseline in "${BASELINES[@]}"; do
  for context in "${CONTEXTS[@]}"; do
    score_args+=(
      --score
      "$baseline=$OUT_ROOT/scores/$baseline/$context/region_scores.csv"
    )
  done
done

if ! audit_is_complete "$ANALYSIS_OUT/audit.json"; then
  preserve_incomplete_output "$ANALYSIS_OUT"
  "$PYTHON_BIN" scripts/server/analyze_complexity_context_performance.py \
    --complexity "$COMPLEXITY_OUT/complexity_features.tsv" \
    "${score_args[@]}" \
    --require-baselines "${BASELINES[@]}" \
    --out-dir "$ANALYSIS_OUT" \
    --n-bootstrap 2000 \
    --n-permutations 10000 \
    --seed 20260815 \
    2>&1 | tee "$OUT_ROOT/logs/analysis.log"
fi

(
  cd "$OUT_ROOT"
  find native_complexity_v2 scores exact_analysis -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
)

echo "COMPLEXITY_CONTEXT_V2_PIPELINE_COMPLETE"
echo "Output: $OUT_ROOT"
