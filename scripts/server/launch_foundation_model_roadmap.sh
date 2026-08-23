#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${TMUX_SESSION:-pangenomefm-foundation-v2}"
RESULTS_BASE="${PANGENOMEFM_RESULTS_ROOT:-${REPO_ROOT}/server_workspace/results}"
RESULT_DIR="${RESULTS_BASE}/foundation_model_v2_roadmap_20260823"
mkdir -p "${RESULT_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session ${SESSION} already exists." >&2
  echo "Attach with: tmux attach -t ${SESSION}" >&2
  exit 2
fi

RUN_ARGS=(--execute --stage 00_preflight)
if [[ "${RUN_READY_STAGES:-0}" == "1" ]]; then
  RUN_ARGS=(--execute)
fi
if [[ "${ALLOW_GPU:-0}" == "1" ]]; then
  RUN_ARGS+=(--allow-gpu)
fi
if [[ "${ALLOW_EXPENSIVE:-0}" == "1" ]]; then
  RUN_ARGS+=(--allow-expensive)
fi
if [[ "${ALLOW_EXTERNAL:-0}" == "1" ]]; then
  RUN_ARGS+=(--allow-external)
fi

printf -v COMMAND 'cd %q && PYTHONPATH=%q:%q python scripts/server/run_foundation_model_roadmap.py %s 2>&1 | tee -a %q; exec bash' \
  "${REPO_ROOT}" "${REPO_ROOT}/src" "${REPO_ROOT}" "${RUN_ARGS[*]}" "${RESULT_DIR}/roadmap.log"
tmux new-session -d -s "${SESSION}" -n roadmap "${COMMAND}"

if command -v nvidia-smi >/dev/null 2>&1; then
  tmux new-window -t "${SESSION}" -n gpu "watch -n 10 nvidia-smi; exec bash"
fi
printf -v STATUS_COMMAND 'while true; do date; test -f %q && cat %q; sleep 30; clear; done' \
  "${RESULT_DIR}/final_status.json" "${RESULT_DIR}/final_status.json"
tmux new-window -t "${SESSION}" -n status "${STATUS_COMMAND}"
tmux select-window -t "${SESSION}:roadmap"

echo "Started: ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log: ${RESULT_DIR}/roadmap.log"

