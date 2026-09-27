#!/usr/bin/env bash
# Read-only server status for PangenomeFM.  Prints no secrets or environment.
set -uo pipefail
echo "== host: $(hostname)  user: $(whoami)  date: $(date -Is)"
uptime
echo "== GPUs"; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
echo "== GPU processes"; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null | head -20
echo "== memory"; free -g 2>/dev/null | head -3
echo "== tmux"; tmux ls 2>/dev/null || echo "no tmux sessions"
echo "== my python processes (truncated)"
ps -u "$(whoami)" -o pid,etime,pcpu,pmem,args 2>/dev/null | grep -E "python|finalize|roadmap" | grep -v grep | cut -c1-220 | head -40
echo "== conda"; (command -v conda || ls -d ~/miniconda3 ~/anaconda3 /opt/conda /opt/anaconda3 2>/dev/null) | head -3
for d in "$HOME/PangenomeFM" /home/tuv43532/PangenomeFM /data/shilab/graphgenome-fm "$HOME/PangenomeFM_v2"; do
  [ -d "$d/.git" ] || [ -f "$d/.git" ] || continue
  echo "== repo: $d"; df -h "$d" 2>/dev/null | tail -1
  git -C "$d" rev-parse --abbrev-ref HEAD 2>/dev/null; git -C "$d" log --oneline -3 2>/dev/null
  git -C "$d" status --short 2>/dev/null | head -10
  C="$d/results/foundation_campaign/20260924"
  if [ -d "$C" ]; then
    echo "-- scaling finalization status:"; cat "$C/scaling_finalization_status.json" 2>/dev/null | head -30
    echo "-- scaling probe audits complete: $(find "$C/scaling_probes" -name audit.json 2>/dev/null | wc -l) / 360"
    echo "-- scaling bio summary files:"; ls "$C/scaling_bio_summary" 2>/dev/null | head
    for s in "$C"/scaling_probe_execution/batch_*; do
      [ -d "$s" ] && echo "   $(basename "$s"): $(grep -rhoE '"status": *"[a-z_]+"' "$s" 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
    done
  fi
  E="$d/results/entex/meeting_20260929"
  if [ -d "$E" ]; then
    echo "-- EN-TEx RNA runs with audit.json: $(find "$E/rna" -name audit.json 2>/dev/null | wc -l) / 30"
    echo "-- EN-TEx follow-up summaries:"; ls "$E/followups"/*/summary.csv 2>/dev/null
    ls "$E/rna_analysis" 2>/dev/null | head
  fi
  H="$d/results/downstream_v2/v1"
  [ -d "$H" ] && echo "-- HG008 external runs: $(find "$H" -maxdepth 3 -name audit.json 2>/dev/null | wc -l)"
  echo "-- principal fold_a seed_42 checkpoints:"
  find "$d" -path "*rotating_folds/hprc_r2/fold_a/seed_42/*" -name "ckpt_*.pt" 2>/dev/null | head -6
  echo "-- benchmark manifest:"; ls "$d"/server_workspace/data/benchmarks/*/manifest.csv 2>/dev/null | head
done
echo "== done"
