#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate pangenomefm-server

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PANGENOMEFM_RESULTS_ROOT=${PANGENOMEFM_RESULTS_ROOT:-server_workspace/results}
export PANGENOMEFM_DATA_ROOT=${PANGENOMEFM_DATA_ROOT:-server_workspace/data}

RESULTS_ROOT=$PANGENOMEFM_RESULTS_ROOT
TAG=${PANGENOMEFM_MANUSCRIPT_GAP_TAG:-20260830}
GAP_ROOT="$RESULTS_ROOT/manuscript_gap_fill_$TAG"
LOG_ROOT="$GAP_ROOT/logs"
STATE_ROOT="$GAP_ROOT/state"
FIGURE_ROOT="$GAP_ROOT/figures"
TABLE_ROOT="$GAP_ROOT/tables"
PROVENANCE_ROOT="$GAP_ROOT/provenance"
SOURCE_DIR=output/pdf/PangenomeFM_Overleaf_20260827/source_data

MAIN_RESULTS="$RESULTS_ROOT/full_multicohort_server_20260806"
CCRE_RESULTS="$RESULTS_ROOT/ccre_sequence_fm_factorial_20260815"
SV_RESULTS="$RESULTS_ROOT/hgsvc3_sv_sequence_fm_factorial_20260815"
BASELINE_RESULTS="$RESULTS_ROOT/rotating_link_baselines_20260809"
COMPLEXITY_RESULTS="$RESULTS_ROOT/complexity_context_v2_20260815/native_complexity_v2"
CAPACITY_RESULTS="$RESULTS_ROOT/capacity_principal_h48_l2_20260830"

CCRE_LABELS=server_workspace/data/downstream/ccre/hprc_r2_screen_v4/node_labels.csv.gz
MANIFEST=server_workspace/data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv
FULL_SEGMENTS=server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz
EXPERIMENT_METRICS="$MAIN_RESULTS/paper_source_data/experiment_seed_metrics.csv"
CCRE_FOLD_METRICS="$CCRE_RESULTS/paper_source_data_v2_20260817/ccre_fold_metrics.csv"
SV_FOLD_METRICS="$SV_RESULTS/paper_source_data_v2_20260817/sv_fold_metrics.csv"

mkdir -p "$LOG_ROOT" "$STATE_ROOT" "$FIGURE_ROOT" "$TABLE_ROOT" "$PROVENANCE_ROOT"
export MPLCONFIGDIR="$GAP_ROOT/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "[$(timestamp)] MISSING_OR_EMPTY: $1" >&2
    exit 1
  fi
}

run_step() {
  local name=$1
  shift
  local done_file="$STATE_ROOT/$name.done"
  local log_file="$LOG_ROOT/$name.log"
  if [[ -s "$done_file" ]]; then
    echo "[$(timestamp)] SKIP complete step: $name"
    return 0
  fi
  echo "[$(timestamp)] START: $name"
  if /usr/bin/time -v "$@" >"$log_file" 2>&1; then
    printf '%s\n' "$(timestamp)" >"$done_file"
    echo "[$(timestamp)] COMPLETE: $name"
  else
    local code=$?
    echo "[$(timestamp)] FAILED: $name (exit $code; log $log_file)" >&2
    tail -n 80 "$log_file" >&2 || true
    return "$code"
  fi
}

echo "[$(timestamp)] === manuscript gap-fill workflow ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "RESULTS_ROOT=$RESULTS_ROOT"
echo "GAP_ROOT=$GAP_ROOT"
echo "GIT_COMMIT=$(git rev-parse HEAD)"

for path in \
  "$CCRE_LABELS" \
  "$MANIFEST" \
  "$FULL_SEGMENTS" \
  "$EXPERIMENT_METRICS" \
  "$CCRE_FOLD_METRICS" \
  "$SV_FOLD_METRICS" \
  "$COMPLEXITY_RESULTS/complexity_features.tsv" \
  "$COMPLEXITY_RESULTS/complexity_thresholds.json" \
  "$BASELINE_RESULTS/paper_source_data/baseline_summary.csv" \
  "$SOURCE_DIR/figure3_reconstruction.csv" \
  "$SOURCE_DIR/figure3_transfer.csv" \
  "$SOURCE_DIR/figure4_capacity_runs.csv" \
  "$SOURCE_DIR/figure5_absolute.csv" \
  "$SOURCE_DIR/figure5_contributions.csv" \
  "$SOURCE_DIR/figure5_sv_strata.csv"
do
  require_file "$path"
done

if [[ $(find "$CCRE_RESULTS" -path '*/test_predictions.csv.gz' -type f | wc -l) -ne 30 ]]; then
  echo "Expected exactly 30 cCRE sequence-FM prediction files below $CCRE_RESULTS" >&2
  exit 1
fi

run_step tests \
  python -m pytest -q \
    tests/test_modality_factorial.py \
    tests/test_ccre_subtype_complexity.py \
    tests/test_manuscript_gap_outputs.py

run_step link_prevalence \
  python scripts/server/summarize_link_prediction_prevalence.py \
    --experiment-seed-metrics "$EXPERIMENT_METRICS" \
    --out-dir "$GAP_ROOT/link_prevalence"

CCRE_DONE="$STATE_ROOT/ccre_stratification.done"
CCRE_PID=""
if [[ -s "$CCRE_DONE" ]]; then
  echo "[$(timestamp)] SKIP complete step: ccre_stratification"
else
  echo "[$(timestamp)] START background: ccre_stratification"
  (
    if /usr/bin/time -v python scripts/server/analyze_ccre_subtype_complexity.py \
      --probe-root "$CCRE_RESULTS" \
      --node-labels "$CCRE_LABELS" \
      --complexity-features "$COMPLEXITY_RESULTS/complexity_features.tsv" \
      --out-dir "$GAP_ROOT/ccre_stratification" \
      --n-bootstrap 10000 \
      --seed 20260830 \
      --expected-files 30 \
      >"$LOG_ROOT/ccre_stratification.log" 2>&1
    then
      printf '%s\n' "$(timestamp)" >"$CCRE_DONE"
    else
      exit 1
    fi
  ) &
  CCRE_PID=$!
fi

run_step principal_capacity_gpu \
  python scripts/server/run_gpu_matrix.py \
    --phase folds \
    --config configs/server_capacity_principal_h48_l2_20260830.json \
    --gpus 0,1,2,3 \
    --execute

if [[ -n "$CCRE_PID" ]]; then
  echo "[$(timestamp)] WAIT: ccre_stratification pid=$CCRE_PID"
  if ! wait "$CCRE_PID"; then
    echo "cCRE stratification failed; inspect $LOG_ROOT/ccre_stratification.log" >&2
    tail -n 100 "$LOG_ROOT/ccre_stratification.log" >&2 || true
    exit 1
  fi
  if [[ ! -s "$CCRE_DONE" ]]; then
    echo "cCRE stratification ended without completion sentinel" >&2
    tail -n 100 "$LOG_ROOT/ccre_stratification.log" >&2 || true
    exit 1
  fi
  echo "[$(timestamp)] COMPLETE: ccre_stratification"
fi

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PANGENOMEFM_RESULTS_ROOT"])
summary = json.loads(
    (root / "capacity_principal_h48_l2_20260830/gpu_queue_folds_summary.json").read_text()
)
assert summary["jobs_requested"] == 15, summary
assert summary["jobs_recorded"] == 15, summary
assert summary["failures"] == 0, summary
print("PRINCIPAL_CAPACITY_GPU_COMPLETE")
PY

run_step principal_capacity_aggregate \
  python scripts/server/aggregate_server_results.py \
    --results-root "$CAPACITY_RESULTS" \
    --out-dir "$CAPACITY_RESULTS/paper_source_data" \
    --n-boot 10000 \
    --seed 20260830

run_step visible_graph_baseline_audit \
  python scripts/server/audit_visible_graph_baselines.py \
    --baseline-root "$BASELINE_RESULTS" \
    --manifest "$MANIFEST" \
    --full-segments "$FULL_SEGMENTS" \
    --out-dir "$GAP_ROOT/visible_graph_baseline_audit" \
    --seed 42

run_step figures \
  python scripts/server/build_manuscript_gap_figures.py \
    --source-dir "$SOURCE_DIR" \
    --ccre-strata "$GAP_ROOT/ccre_stratification/ccre_stratum_topology_gains.csv" \
    --prevalence-summary "$GAP_ROOT/link_prevalence/link_prediction_prevalence_summary.csv" \
    --principal-capacity-metrics "$CAPACITY_RESULTS/paper_source_data/experiment_seed_metrics.csv" \
    --output-dir "$FIGURE_ROOT" \
    --require-ccre-strata

run_step tables \
  python scripts/server/build_manuscript_gap_tables.py \
    --ccre-strata "$GAP_ROOT/ccre_stratification/ccre_stratum_topology_gains.csv" \
    --prevalence-summary "$GAP_ROOT/link_prevalence/link_prediction_prevalence_summary.csv" \
    --complexity-thresholds "$COMPLEXITY_RESULTS/complexity_thresholds.json" \
    --complexity-features "$COMPLEXITY_RESULTS/complexity_features.tsv" \
    --ccre-fold-metrics "$CCRE_FOLD_METRICS" \
    --sv-fold-metrics "$SV_FOLD_METRICS" \
    --old-capacity-runs "$SOURCE_DIR/figure4_capacity_runs.csv" \
    --principal-capacity-metrics "$CAPACITY_RESULTS/paper_source_data/experiment_seed_metrics.csv" \
    --visible-graph-audit "$GAP_ROOT/visible_graph_baseline_audit/audit.json" \
    --out-dir "$TABLE_ROOT"

echo "[$(timestamp)] === final audit ==="
python - <<'PY'
import json
import os
from pathlib import Path

results = Path(os.environ["PANGENOMEFM_RESULTS_ROOT"])
tag = os.environ.get("PANGENOMEFM_MANUSCRIPT_GAP_TAG", "20260830")
root = results / f"manuscript_gap_fill_{tag}"
checks = {
    "ccre": root / "ccre_stratification/audit.json",
    "prevalence": root / "link_prevalence/audit.json",
    "visible_graph": root / "visible_graph_baseline_audit/audit.json",
    "figures": root / "figures/audit.json",
    "tables": root / "tables/audit.json",
}
for name, path in checks.items():
    audit = json.loads(path.read_text())
    assert audit["status"] in {"complete", "pass"}, (name, audit)
for stem in [
    "figure3_ccre",
    "figure4_sv",
    "figure5_generalization_transfer",
    "supplementary_figure1_controls_capacity",
]:
    for suffix in ["pdf", "png"]:
        path = root / "figures" / f"{stem}.{suffix}"
        assert path.stat().st_size > 10_000, path
print("MANUSCRIPT_GAP_OUTPUTS_VERIFIED")
PY

git rev-parse HEAD >"$PROVENANCE_ROOT/git_commit.txt"
git status --short >"$PROVENANCE_ROOT/git_status.txt"
conda list --explicit >"$PROVENANCE_ROOT/conda_explicit.txt"
python -m pip freeze | sort >"$PROVENANCE_ROOT/python_packages.txt"
cp configs/server_capacity_principal_h48_l2_20260830.json "$PROVENANCE_ROOT/"

find "$PROVENANCE_ROOT" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  >"$PROVENANCE_ROOT/SHA256SUMS"
sha256sum -c "$PROVENANCE_ROOT/SHA256SUMS"

PACKAGE="$RESULTS_ROOT/manuscript_gap_fill_${TAG}.tar.zst"
MANIFEST_FILE="$RESULTS_ROOT/manuscript_gap_fill_${TAG}.included.txt"
find "$GAP_ROOT" -type f ! -path '*/.matplotlib/*' -print \
  | sort \
  >"$MANIFEST_FILE"
find "$CAPACITY_RESULTS/paper_source_data" -maxdepth 1 -type f -print \
  | sort \
  >>"$MANIFEST_FILE"
printf '%s\n' configs/server_capacity_principal_h48_l2_20260830.json >>"$MANIFEST_FILE"
sort -u "$MANIFEST_FILE" -o "$MANIFEST_FILE"

tar --use-compress-program='zstd -T0 -10' -cf "$PACKAGE" -T "$MANIFEST_FILE"
(
  cd "$(dirname "$PACKAGE")"
  sha256sum "$(basename "$PACKAGE")" >"$(basename "$PACKAGE").sha256"
  sha256sum -c "$(basename "$PACKAGE").sha256"
)
zstd -t "$PACKAGE"

cat >"$GAP_ROOT/final_status.json" <<EOF
{
  "schema_version": 1,
  "status": "complete",
  "git_commit": "$(git rev-parse HEAD)",
  "output_root": "$GAP_ROOT",
  "capacity_root": "$CAPACITY_RESULTS",
  "package": "$PACKAGE",
  "package_sha256": "$(sha256sum "$PACKAGE" | awk '{print $1}')"
}
EOF

echo "[$(timestamp)] MANUSCRIPT_GAP_FILL_COMPLETE"
cat "$GAP_ROOT/final_status.json"
