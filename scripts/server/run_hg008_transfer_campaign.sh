#!/usr/bin/env bash
# Run from the repository in the existing pangenomefm-server environment.
set -euo pipefail
export PYTHONPATH="src:.${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
root=results/downstream_v2/v1
python -m pytest -q tests/test_transfer.py tests/test_entex.py tests/test_sv_frozen_probe.py tests/test_prepare_sv_breakpoint_examples.py > "$root/server_tests.log" 2>&1
python - <<'PY'
import json
from pathlib import Path
audit = json.loads(Path('results/downstream_v2/v1/hg008_smoke/fold_a/seed_42/strict/audit.json').read_text())
assert audit['status'] == 'complete' and not audit['hg008_training_or_calibration']
PY
pids=()
for batch in 0 1 2 3; do
  python scripts/server/run_foundation_model_roadmap.py \
    --config configs/hg008_transfer_jobs_v1.json --execute \
    --stage "batch_$batch" --result-root "$root/execution/batch_$batch" \
    > "$root/batch_$batch.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if ((failed)); then
  echo "Incomplete campaign: inspect batch logs and original-probe regression checks." >&2
  exit 1
fi
python -m tasks.transfer.summarize --probe-root "$root/hg008_full" --out-dir "$root/analysis/hg008"
