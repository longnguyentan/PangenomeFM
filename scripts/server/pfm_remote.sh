#!/usr/bin/env bash
# Run from your Mac Terminal, inside the repository.  Passwords/2FA are typed by
# you into ssh itself; nothing is stored.  One authenticated connection is kept
# open (OpenSSH ControlMaster) so later subcommands do not prompt again.
#
#   bash scripts/server/pfm_remote.sh connect      # prompts for both logins once
#   bash scripts/server/pfm_remote.sh status       # read-only report -> server_reports/
#   bash scripts/server/pfm_remote.sh push         # push the v2 branch to GitHub
#   bash scripts/server/pfm_remote.sh sync         # separate v2 worktree on the server
#   bash scripts/server/pfm_remote.sh job <name>   # start a v2 job in tmux (see pfm_v2_jobs.sh)
#   bash scripts/server/pfm_remote.sh fetch        # copy compact v2 outputs back
#   bash scripts/server/pfm_remote.sh disconnect
#
# Override hosts with PFM_JUMP / PFM_TARGET, e.g. PFM_TARGET=tuv43532@cis-chen.
set -uo pipefail
JUMP="${PFM_JUMP:-tuv43532@cis-linux2.temple.edu}"
TARGET="${PFM_TARGET:-nguyen@100.107.98.119}"
BRANCH="${PFM_BRANCH:-v2/shortcut-free-pretraining-20260926}"
SOCK="/tmp/pfm-ssh-%C"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPORTS="$REPO_ROOT/server_reports"
STAMP="$(date +%Y%m%d_%H%M%S)"
SSH_OPTS=(-o ControlPath="$SOCK" -o ServerAliveInterval=15 -o ServerAliveCountMax=120 -o ProxyJump="$JUMP")
mkdir -p "$REPORTS"

remote() { ssh "${SSH_OPTS[@]}" "$TARGET" "$@"; }
need_master() {
  if ! ssh "${SSH_OPTS[@]}" -O check "$TARGET" >/dev/null 2>&1; then
    echo "No open connection. Run: bash scripts/server/pfm_remote.sh connect" >&2; exit 2
  fi
}

case "${1:-}" in
  connect)
    echo "Opening one shared connection: $JUMP -> $TARGET (enter each password/2FA when asked)"
    ssh "${SSH_OPTS[@]}" -o ControlMaster=yes -o ControlPersist=10h -fN "$TARGET" \
      && echo "Connected. Keep this Mac awake; later subcommands reuse this login."
    ;;
  status)
    need_master
    out="$REPORTS/status_${STAMP}.txt"
    remote 'bash -s' < "$REPO_ROOT/scripts/server/pfm_remote_status.sh" > "$out" 2>&1
    echo "Wrote $out"; tail -n 25 "$out"
    ;;
  push)
    git -C "$REPO_ROOT" push -u origin "$BRANCH"
    ;;
  sync)
    need_master
    out="$REPORTS/sync_${STAMP}.txt"
    remote "PFM_BRANCH='$BRANCH' bash -s" < "$REPO_ROOT/scripts/server/pfm_v2_jobs.sh" sync > "$out" 2>&1
    echo "Wrote $out"; tail -n 25 "$out"
    ;;
  job)
    need_master
    name="${2:?job name required (audit|diag|topo|randinit|pilot|pilot-kmer|pilot-large|probes|list)}"
    out="$REPORTS/job_${name}_${STAMP}.txt"
    remote "PFM_BRANCH='$BRANCH' bash -s -- $name" < "$REPO_ROOT/scripts/server/pfm_v2_jobs.sh" > "$out" 2>&1
    echo "Wrote $out"; tail -n 25 "$out"
    ;;
  fetch)
    need_master
    dest="$REPO_ROOT/server_imports/v2_${STAMP}"
    mkdir -p "$dest"
    remote 'bash -s -- pack' < "$REPO_ROOT/scripts/server/pfm_v2_jobs.sh" > "$REPORTS/pack_${STAMP}.txt" 2>&1
    tarball="$(grep '^PACKED ' "$REPORTS/pack_${STAMP}.txt" | tail -1 | cut -d' ' -f2)"
    if [ -n "$tarball" ]; then
      scp "${SSH_OPTS[@]}" "$TARGET:$tarball" "$dest/" && tar -xzf "$dest/$(basename "$tarball")" -C "$dest" \
        && echo "Fetched into $dest"
    else
      echo "Nothing packed; see $REPORTS/pack_${STAMP}.txt"
    fi
    ;;
  disconnect)
    ssh "${SSH_OPTS[@]}" -O exit "$TARGET" 2>/dev/null; echo "Closed."
    ;;
  *)
    sed -n '2,15p' "$0"; exit 1 ;;
esac
