#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${TMUX_SESSION:-pangenomefm-full}"
: "${PANGENOMEFM_DATA_ROOT:?Export PANGENOMEFM_DATA_ROOT first.}"
: "${PANGENOMEFM_RESULTS_ROOT:?Export PANGENOMEFM_RESULTS_ROOT first.}"
RESULT_DIR="${PANGENOMEFM_RESULTS_ROOT}/full_multicohort_server_20260806"
mkdir -p "${RESULT_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session ${SESSION} already exists. Attach with: tmux attach -t ${SESSION}" >&2
  exit 2
fi

printf -v PIPELINE_CMD 'cd %q && bash scripts/server/run_full_server_pipeline.sh 2>&1 | tee -a %q; exec bash' \
  "${REPO_ROOT}" "${RESULT_DIR}/master_pipeline.log"
tmux new-session -d -s "${SESSION}" -n pipeline "${PIPELINE_CMD}"

if command -v nvidia-smi >/dev/null 2>&1; then
  tmux new-window -t "${SESSION}" -n gpu \
    "watch -n 10 nvidia-smi; exec bash"
fi
printf -v STATUS_CMD 'while true; do date; find %q -name "*summary.json" -o -name "execution_state.json" | sort | tail -30; sleep 30; clear; done' \
  "${RESULT_DIR}"
tmux new-window -t "${SESSION}" -n status "${STATUS_CMD}"
tmux select-window -t "${SESSION}:pipeline"

echo "Started tmux session: ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Detach: Ctrl-b d"
echo "Master log: ${RESULT_DIR}/master_pipeline.log"
