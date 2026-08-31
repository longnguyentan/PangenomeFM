#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
RESULTS_ROOT=${PANGENOMEFM_RESULTS_ROOT:-server_workspace/results}
GAP_ROOT="$RESULTS_ROOT/manuscript_gap_fill_20260830"
CAPACITY_ROOT="$RESULTS_ROOT/capacity_principal_h48_l2_20260830"

date -u
echo "=== active processes ==="
pgrep -af '[r]un_gpu_matrix.py|[r]un_full_multicohort_all_chromosomes.py|[a]nalyze_ccre_subtype_complexity.py|[a]udit_visible_graph_baselines.py' \
  || echo "No active gap-fill process"

echo
echo "=== completed workflow steps ==="
find "$GAP_ROOT/state" -maxdepth 1 -name '*.done' -type f -printf '%f\n' 2>/dev/null | sort || true

echo
echo "=== capacity jobs ==="
python - "$CAPACITY_ROOT/gpu_queue_folds_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("queue summary not written yet")
else:
    data = json.loads(path.read_text())
    print({key: data.get(key) for key in ["jobs_requested", "jobs_recorded", "failures"]})
    for record in data.get("records", []):
        if record.get("returncode") not in (None, 0):
            print(
                "FAILED:",
                record.get("name"),
                f"gpu={record.get('gpu_id')}",
                f"exit={record.get('returncode')}",
                f"log={record.get('log')}",
            )
PY

echo
echo "=== latest logs ==="
for path in \
  "$GAP_ROOT/logs/ccre_stratification.log" \
  "$GAP_ROOT/logs/principal_capacity_gpu.log" \
  "$GAP_ROOT/logs/visible_graph_baseline_audit.log"
do
  echo "--- $path"
  tail -n 3 "$path" 2>/dev/null || echo "not started"
done

echo
echo "=== GPU processes ==="
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null || true
