#!/usr/bin/env bash
# Launch the complete sequence-FM manuscript finalization in one tmux session.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
SESSION=${PANGENOMEFM_FINALIZE_SESSION:-pangenomefm-sequence-fm-finalize-20260817}
TAG=${PANGENOMEFM_FINALIZE_TAG:-20260817}
RUN_ROOT=${PANGENOMEFM_RESULTS_ROOT:-server_workspace/results}/manuscript_sequence_fm_finalize_${TAG}

command -v tmux >/dev/null || {
  echo "tmux is required" >&2
  exit 1
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"
else
  tmux new-session -d -s "$SESSION" \
    "bash -lc 'cd \"$PROJECT_ROOT\"; bash scripts/server/run_sequence_fm_manuscript_finalize.sh; rc=\$?; echo FINALIZE_EXIT_CODE=\$rc; exec bash'"
  echo "Started session: $SESSION"
fi

echo "Attach:  tmux attach -t $SESSION"
echo "Monitor: tail -f $PROJECT_ROOT/$RUN_ROOT/console.log"
echo "Status:  cat $PROJECT_ROOT/$RUN_ROOT/final_status.json"
