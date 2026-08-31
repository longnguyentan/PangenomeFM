#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SESSION=${1:-pangenomefm-manuscript-gap-20260830}
RESULTS_ROOT=${PANGENOMEFM_RESULTS_ROOT:-server_workspace/results}
CONSOLE_LOG="$REPO_ROOT/$RESULTS_ROOT/manuscript_gap_fill_20260830.console.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

mkdir -p "$(dirname "$CONSOLE_LOG")"
tmux new-session -d -s "$SESSION" \
  "bash -lc 'cd \"$REPO_ROOT\" && bash scripts/server/run_manuscript_gap_experiments.sh 2>&1 | tee \"$CONSOLE_LOG\"; exec bash'"

echo "Started: $SESSION"
echo "Console log: $CONSOLE_LOG"
echo "Attach: tmux attach -t $SESSION"
echo "Monitor: bash scripts/server/monitor_manuscript_gap_experiments.sh"
