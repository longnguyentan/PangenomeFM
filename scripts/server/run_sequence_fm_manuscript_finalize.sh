#!/usr/bin/env bash
# Finalize the sequence-FM manuscript evidence in one restart-safe server run.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

TAG=${PANGENOMEFM_FINALIZE_TAG:-20260817}
RESULTS_ROOT=${PANGENOMEFM_RESULTS_ROOT:-server_workspace/results}
CCRE_ROOT=${PANGENOMEFM_CCRE_SEQUENCE_ROOT:-$RESULTS_ROOT/ccre_sequence_fm_factorial_20260815}
SV_ROOT=${PANGENOMEFM_SV_SEQUENCE_ROOT:-$RESULTS_ROOT/hgsvc3_sv_sequence_fm_factorial_20260815}
SEQUENCE_ROOT=${PANGENOMEFM_SEQUENCE_CACHE_ROOT:-$RESULTS_ROOT/frozen_sequence_fm_cache_20260815}
CANONICAL_AUDIT=${PANGENOMEFM_CANONICAL_AUDIT:-$RESULTS_ROOT/complexity_context_v2_20260815/canonical_affected_locus_sensitivity/audit.json}
CCRE_AGGREGATE="$CCRE_ROOT/paper_source_data_v2_${TAG}"
SV_AGGREGATE="$SV_ROOT/paper_source_data_v2_${TAG}"
DECISION_AUDIT="$RESULTS_ROOT/manuscript_sequence_fm_decision_audit_v2_${TAG}.json"
MANUSCRIPT_OUTPUTS="$RESULTS_ROOT/manuscript_sequence_fm_outputs_${TAG}"
RUN_ROOT="$RESULTS_ROOT/manuscript_sequence_fm_finalize_${TAG}"
PROVENANCE="$RUN_ROOT/provenance"
PACKAGE="$RESULTS_ROOT/manuscript_sequence_fm_evidence_v2_${TAG}.tar.zst"
LOCK_DIR="$RUN_ROOT/.lock"

mkdir -p "$RUN_ROOT" "$PROVENANCE"
if [[ ${PANGENOMEFM_FINALIZE_LOG_ACTIVE:-0} != 1 ]]; then
  set +e
  PANGENOMEFM_FINALIZE_LOG_ACTIVE=1 bash "$0" "$@" \
    2>&1 | tee -a "$RUN_ROOT/console.log"
  driver_status=${PIPESTATUS[0]}
  exit "$driver_status"
fi

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%S%z
}

stage() {
  echo
  echo "[$(timestamp)] === $* ==="
}

fail() {
  echo "[$(timestamp)] ERROR: $*" >&2
  exit 1
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  fail "another finalization run holds $LOCK_DIR"
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

if [[ ${PANGENOMEFM_SKIP_CONDA:-0} != 1 ]]; then
  CONDA_SH=${PANGENOMEFM_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
  [[ -f "$CONDA_SH" ]] || fail "conda initialization is missing: $CONDA_SH"
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "${PANGENOMEFM_CONDA_ENV:-pangenomefm-server}"
fi

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$RUN_ROOT/matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"

TIME_COMMAND=(/usr/bin/time -p)
if /usr/bin/time --version >/dev/null 2>&1; then
  TIME_COMMAND=(/usr/bin/time -v)
fi

required_paths=(
  "$CCRE_ROOT/matrix_summary.json"
  "$SV_ROOT/matrix_summary.json"
  "$SEQUENCE_ROOT/hprc_target_union_sequence_fm.npz"
  "$SEQUENCE_ROOT/hprc_target_union_sequence_fm.npz.audit.json"
  "$CANONICAL_AUDIT"
)
for path in "${required_paths[@]}"; do
  [[ -s "$path" ]] || fail "required input is missing or empty: $path"
done

CURRENT_COMMIT=$(git rev-parse HEAD)
echo "$CURRENT_COMMIT" > "$PROVENANCE/git_commit.txt"
git status --short > "$PROVENANCE/git_status.txt"

stage "focused code verification"
TEST_MARKER="$RUN_ROOT/tests_git_commit.txt"
if [[ -s "$TEST_MARKER" ]] && grep -Fxq "$CURRENT_COMMIT" "$TEST_MARKER"; then
  echo "Focused tests already passed for $CURRENT_COMMIT; skipping."
else
  python -m pytest -q \
    tests/test_modality_factorial.py \
    tests/test_manuscript_decision_audit.py \
    tests/test_sequence_fm_cache.py
  python -m compileall -q \
    scripts/server/aggregate_ccre_frozen_probes.py \
    scripts/server/aggregate_sv_frozen_probes.py \
    scripts/server/audit_manuscript_decision_experiments.py \
    scripts/server/build_sequence_fm_manuscript_outputs.py
  printf '%s\n' "$CURRENT_COMMIT" > "$TEST_MARKER"
fi

aggregate_valid() {
  local directory=$1
  python - "$directory" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
path = directory / "audit.json"
if not path.exists():
    raise SystemExit(1)
audit = json.loads(path.read_text())
required = "topology_given_coordinate_and_frozen_sequence_fm"
passed = all([
    audit.get("status") == "complete",
    audit.get("files") == 30,
    audit.get("n_bootstrap") == 10_000,
    required in audit.get("modality_contrasts", []),
    audit.get("exact_feature_universe", {}).get("status") == "pass",
])
raise SystemExit(0 if passed else 1)
PY
}

preserve_incomplete() {
  local directory=$1
  if [[ -d "$directory" ]]; then
    local archived="${directory}_incomplete_$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$directory" "$archived"
    echo "Preserved incomplete output: $archived"
  fi
}

run_ccre_aggregate() {
  set -o pipefail
  "${TIME_COMMAND[@]}" python scripts/server/aggregate_ccre_frozen_probes.py \
    --probe-root "$CCRE_ROOT" \
    --out-dir "$CCRE_AGGREGATE" \
    --n-bootstrap 10000 \
    --seed 20260806 \
    2>&1 | tee "$RUN_ROOT/ccre_aggregate.log"
}

run_sv_aggregate() {
  set -o pipefail
  "${TIME_COMMAND[@]}" python scripts/server/aggregate_sv_frozen_probes.py \
    --probe-root "$SV_ROOT" \
    --out-dir "$SV_AGGREGATE" \
    --n-bootstrap 10000 \
    --seed 20260806 \
    2>&1 | tee "$RUN_ROOT/sv_aggregate.log"
}

stage "concurrent versioned aggregation"
ccre_pid=""
sv_pid=""
if aggregate_valid "$CCRE_AGGREGATE"; then
  echo "Valid cCRE V2 aggregate already exists; skipping."
else
  preserve_incomplete "$CCRE_AGGREGATE"
  run_ccre_aggregate &
  ccre_pid=$!
fi
if aggregate_valid "$SV_AGGREGATE"; then
  echo "Valid SV V2 aggregate already exists; skipping."
else
  preserve_incomplete "$SV_AGGREGATE"
  run_sv_aggregate &
  sv_pid=$!
fi

aggregate_failure=0
if [[ -n "$ccre_pid" ]]; then
  if ! wait "$ccre_pid"; then
    echo "cCRE aggregation failed; inspect $RUN_ROOT/ccre_aggregate.log" >&2
    aggregate_failure=1
  fi
fi
if [[ -n "$sv_pid" ]]; then
  if ! wait "$sv_pid"; then
    echo "SV aggregation failed; inspect $RUN_ROOT/sv_aggregate.log" >&2
    aggregate_failure=1
  fi
fi
[[ "$aggregate_failure" -eq 0 ]] || fail "one or more aggregate jobs failed"
aggregate_valid "$CCRE_AGGREGATE" || fail "cCRE V2 aggregate failed validation"
aggregate_valid "$SV_AGGREGATE" || fail "SV V2 aggregate failed validation"

stage "strengthened manuscript decision audit"
python scripts/server/audit_manuscript_decision_experiments.py \
  --canonical-sensitivity-audit "$CANONICAL_AUDIT" \
  --ccre-root "$CCRE_ROOT" \
  --ccre-aggregate "$CCRE_AGGREGATE" \
  --sv-root "$SV_ROOT" \
  --sv-aggregate "$SV_AGGREGATE" \
  --sequence-cache "$SEQUENCE_ROOT/hprc_target_union_sequence_fm.npz" \
  --expected-bootstrap 10000 \
  --output "$DECISION_AUDIT"

python - "$DECISION_AUDIT" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text())
assert audit["status"] == "pass", audit
print("STRENGTHENED_MANUSCRIPT_DECISION_GATE_PASSED")
PY

stage "manuscript-ready tables, figure, report, and notebook"
shard_audits=()
for shard in 0 1 2 3; do
  path="$SEQUENCE_ROOT/shard_${shard}.npz.audit.json"
  [[ -s "$path" ]] || fail "sequence shard audit is missing: $path"
  shard_audits+=("$path")
done

if [[ -d "$MANUSCRIPT_OUTPUTS" ]]; then
  if python - "$MANUSCRIPT_OUTPUTS/audit.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
raise SystemExit(
    0 if path.exists() and json.loads(path.read_text()).get("status") == "complete" else 1
)
PY
  then
    archived="${MANUSCRIPT_OUTPUTS}_previous_$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$MANUSCRIPT_OUTPUTS" "$archived"
    echo "Preserved previous manuscript outputs: $archived"
  else
    preserve_incomplete "$MANUSCRIPT_OUTPUTS"
  fi
fi

python scripts/server/build_sequence_fm_manuscript_outputs.py \
  --ccre-aggregate "$CCRE_AGGREGATE" \
  --sv-aggregate "$SV_AGGREGATE" \
  --sequence-cache-audit "$SEQUENCE_ROOT/hprc_target_union_sequence_fm.npz.audit.json" \
  --sequence-shard-audit "${shard_audits[@]}" \
  --out-dir "$MANUSCRIPT_OUTPUTS"

stage "provenance capture"
conda list --explicit > "$PROVENANCE/conda_explicit.txt"
python -m pip freeze | sort > "$PROVENANCE/python_packages.txt"
cp "$SEQUENCE_ROOT/model_revision.txt" "$PROVENANCE/model_revision.txt"
cp "$DECISION_AUDIT" "$PROVENANCE/"
cp configs/server_full_multicohort_20260806.json "$PROVENANCE/"
(
  cd "$RESULTS_ROOT"
  provenance_relative=${PROVENANCE#"$RESULTS_ROOT"/}
  find "$provenance_relative" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > "$provenance_relative/SHA256SUMS"
  sha256sum -c "$provenance_relative/SHA256SUMS"
)

stage "compact standalone evidence package"
INCLUDED="$RUN_ROOT/included.txt"
{
  printf '%s\n' "${CCRE_AGGREGATE#$RESULTS_ROOT/}"
  printf '%s\n' "${SV_AGGREGATE#$RESULTS_ROOT/}"
  printf '%s\n' "${DECISION_AUDIT#$RESULTS_ROOT/}"
  printf '%s\n' "${MANUSCRIPT_OUTPUTS#$RESULTS_ROOT/}"
  printf '%s\n' "${PROVENANCE#$RESULTS_ROOT/}"
  printf '%s\n' "${CCRE_ROOT#$RESULTS_ROOT/}/matrix_summary.json"
  printf '%s\n' "${SV_ROOT#$RESULTS_ROOT/}/matrix_summary.json"
  printf '%s\n' "${SEQUENCE_ROOT#$RESULTS_ROOT/}/hprc_target_union_sequence_fm.npz.audit.json"
  printf '%s\n' "${SEQUENCE_ROOT#$RESULTS_ROOT/}/model_revision.txt"
  for shard in 0 1 2 3; do
    printf '%s\n' "${SEQUENCE_ROOT#$RESULTS_ROOT/}/shard_${shard}.npz.audit.json"
  done
} > "$INCLUDED"
printf '%s\n' "${INCLUDED#$RESULTS_ROOT/}" >> "$INCLUDED"

if [[ -e "$PACKAGE" || -e "$PACKAGE.sha256" ]]; then
  archive_suffix=$(date -u +%Y%m%dT%H%M%SZ)
  [[ ! -e "$PACKAGE" ]] || mv "$PACKAGE" "${PACKAGE}_previous_${archive_suffix}"
  [[ ! -e "$PACKAGE.sha256" ]] || mv "$PACKAGE.sha256" "${PACKAGE}.sha256_previous_${archive_suffix}"
fi

PACKAGE_PART="$PACKAGE.part"
rm -f "$PACKAGE_PART"
(
  cd "$RESULTS_ROOT"
  tar \
    --use-compress-program='zstd -T0 -10' \
    --exclude='*.npz' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.ckpt' \
    -cf "$(basename "$PACKAGE_PART")" \
    -T "${INCLUDED#$RESULTS_ROOT/}"
)
mv "$PACKAGE_PART" "$PACKAGE"
(
  cd "$(dirname "$PACKAGE")"
  package_name=$(basename "$PACKAGE")
  sha256sum "$package_name" > "$package_name.sha256"
  sha256sum -c "$package_name.sha256"
)
zstd -t "$PACKAGE"

stage "complete"
PACKAGE_BYTES=$(wc -c < "$PACKAGE" | tr -d '[:space:]')
cat > "$RUN_ROOT/final_status.json" <<EOF
{
  "schema_version": 1,
  "status": "complete",
  "git_commit": "$CURRENT_COMMIT",
  "ccre_aggregate": "$CCRE_AGGREGATE",
  "sv_aggregate": "$SV_AGGREGATE",
  "decision_audit": "$DECISION_AUDIT",
  "manuscript_outputs": "$MANUSCRIPT_OUTPUTS",
  "package": "$PACKAGE",
  "package_bytes": $PACKAGE_BYTES,
  "package_sha256": "$(sha256sum "$PACKAGE" | awk '{print $1}')"
}
EOF
cat "$RUN_ROOT/final_status.json"
echo "SEQUENCE_FM_MANUSCRIPT_FINALIZATION_COMPLETE"
