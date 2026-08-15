#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
RUN_TAG="${RUN_TAG:-20260815}"
BENCHMARK_DIR="${BENCHMARK_DIR:-data/hprc/benchmark_matched_nonoverlap_v3_canonical_${RUN_TAG}}"
COMPLEXITY_DIR="${COMPLEXITY_DIR:-results/complexity/graph_window_complexity_v2_local_${RUN_TAG}}"

cd "$REPO_ROOT"
export PYTHONPATH="src:.${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m pytest -q \
  tests/test_canonical_negative_sampling.py \
  tests/test_training_recovery.py \
  tests/test_comparative_benchmark.py \
  tests/test_complexity.py \
  tests/test_dense_region_scores.py \
  tests/test_complexity_context_performance.py

if [[ ! -f "$BENCHMARK_DIR/manifest.csv" ]]; then
  "$PYTHON_BIN" -m graphgenomefm make-benchmark \
    --data-dir data/hprc \
    --out-dir "$BENCHMARK_DIR" \
    --seed 20260804 \
    --window-bp 50000 \
    --n-windows 10 \
    --targets all \
    --target-prefix 'GRCh38#0' \
    --closures strict 1hop \
    --negative-sampler distance_matched \
    --negative-tol-bp 1000 \
    --negative-tol-frac 0.10 \
    --non-overlapping-windows \
    --matched-closure-windows \
    --no-network-analysis \
    --no-viz
fi

"$PYTHON_BIN" scripts/audit_canonical_candidates.py \
  --manifest "$BENCHMARK_DIR/manifest.csv" \
  --out-dir "$BENCHMARK_DIR/audit" \
  --overwrite

"$PYTHON_BIN" scripts/extract_graph_complexity.py \
  --manifest "$BENCHMARK_DIR/manifest.csv" \
  --dataset hprc_local_canonical_50kb_v3 \
  --config configs/complexity_definition_v2.yaml \
  --out-dir "$COMPLEXITY_DIR" \
  --overwrite

echo "LOCAL_CANONICAL_COMPLEXITY_PIPELINE_COMPLETE"
echo "Benchmark: $BENCHMARK_DIR"
echo "Complexity: $COMPLEXITY_DIR"
