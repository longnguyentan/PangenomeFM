#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
RESULTS_ROOT=${PANGENOMEFM_RESULTS_ROOT:-server_workspace/results}
GAP_ROOT="$RESULTS_ROOT/manuscript_gap_fill_20260830"
PRINCIPAL_METRICS="$RESULTS_ROOT/full_multicohort_server_20260806/paper_source_data/experiment_seed_metrics.csv"

date -u
echo "=== active processes ==="
pgrep -af '[a]nalyze_ccre_subtype_complexity.py|[a]udit_visible_graph_baselines.py' \
  || echo "No active gap-fill process"

echo
echo "=== completed workflow steps ==="
find "$GAP_ROOT/state" -maxdepth 1 -name '*.done' -type f -printf '%f\n' 2>/dev/null | sort || true

echo
echo "=== archived principal capacity evidence ==="
python - "$PRINCIPAL_METRICS" <<'PY'
import sys
from pathlib import Path

import pandas as pd

path = Path(sys.argv[1])
if not path.is_file():
    print("principal metrics are missing")
else:
    data = pd.read_csv(path)
    rows = data.loc[
        data["regime"].eq("hprc_r2")
        & data["closure"].eq("strict")
        & data["metric_scope"].eq("split")
        & data["split"].eq("heldout_chr_test")
    ]
    print(
        {
            "runs": int(len(rows)),
            "targets": int(rows["n_targets"].sum()),
            "mean_auprc": float(rows["auprc"].mean()),
            "sd_auprc": float(rows["auprc"].std(ddof=1)),
        }
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
