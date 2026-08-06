#!/usr/bin/env bash
set -euo pipefail

# Overnight runner for the masked-training follow-up analyses.
#
# Default behavior:
#   - uses the latest completed masked-trained full/ablation runs already on disk
#   - bootstraps HPRC held-out CIs for the full model
#   - runs HGSVC3 strict and 1-hop external transfer from the full checkpoints
#   - bootstraps HGSVC3 transfer CIs
#   - runs topology-survival analyses for HPRC strict, HPRC 1-hop, and HGSVC 1-hop
#   - writes a compact masked ablation summary table
#
# Useful switches:
#   RUN_CORE=1                 rerun the four masked training jobs first
#   RUN_ABLATION_BOOTSTRAP=1   bootstrap all four HPRC ablations, not only the full model
#   RUN_HGSVC_STRICT_TOPOLOGY=1 also run topology-survival on HGSVC strict transfer
#   FORCE=1                    rerun steps even when their summaries already exist
#   N_BOOT=200                 faster dry-run CIs; default is 1000
#
# Example:
#   bash scripts/run_masked_full_overnight.sh

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
N_BOOT="${N_BOOT:-1000}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
RUN_CORE="${RUN_CORE:-0}"
RUN_ABLATION_BOOTSTRAP="${RUN_ABLATION_BOOTSTRAP:-0}"
RUN_HGSVC_STRICT_TOPOLOGY="${RUN_HGSVC_STRICT_TOPOLOGY:-0}"

HPRC_DATA_DIR="${HPRC_DATA_DIR:-data/hprc}"
HPRC_BENCHMARK_DIR="${HPRC_BENCHMARK_DIR:-data/hprc/benchmark_paper_hardneg}"
HPRC_OUT_ROOT="${HPRC_OUT_ROOT:-results/hprc}"

HGSVC_DATA_DIR="${HGSVC_DATA_DIR:-data/hgsvc3_expanded}"
HGSVC_BENCHMARK_DIR="${HGSVC_BENCHMARK_DIR:-data/hgsvc3_expanded/benchmark_chr19_chr21_chr22_chrY_hardneg}"

ANALYSIS_ROOT="${ANALYSIS_ROOT:-results/analysis}"
LOG_DIR="${LOG_DIR:-$ANALYSIS_ROOT/logs}"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/masked_full_overnight_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

first_file() {
  local dir="$1"
  local pattern="$2"
  find "$dir" -maxdepth 1 -type f -name "$pattern" | sort | head -n 1
}

latest_run_with_file() {
  local parent="$1"
  local pattern="$2"
  local run

  if [[ ! -d "$parent" ]]; then
    return 1
  fi

  while IFS= read -r run; do
    if [[ -n "$(first_file "$run" "$pattern")" ]]; then
      printf '%s\n' "$run"
      return 0
    fi
  done < <(find "$parent" -maxdepth 1 -type d -name 'run_*' | sort -r)

  return 1
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ -z "$path" || ! -f "$path" ]]; then
    printf 'Missing %s: %s\n' "$label" "$path" >&2
    exit 1
  fi
}

require_run() {
  local run="$1"
  local label="$2"
  if [[ -z "$run" || ! -d "$run" ]]; then
    printf 'Missing %s run directory: %s\n' "$label" "$run" >&2
    exit 1
  fi
}

run_python_step() {
  local marker="$1"
  shift
  if [[ "$FORCE" != "1" && -f "$marker" ]]; then
    log "Skipping existing step: $marker"
    return 0
  fi
  "$PYTHON_BIN" "$@"
}

if [[ "$RUN_CORE" == "1" ]]; then
  log "Running the four core masked-training jobs first"
  DEVICE="$DEVICE" PYTHON_BIN="$PYTHON_BIN" bash scripts/run_masked_core_experiments.sh
fi

log "Discovering latest completed masked-training runs"
FULL_RUN="$(latest_run_with_file "$HPRC_OUT_ROOT/pretrain_masked_train_full" 'ckpt_1hop__*.pt')"
COORD_RUN="$(latest_run_with_file "$HPRC_OUT_ROOT/pretrain_masked_train_coordinate" '1hop_results__*.csv')"
GRAPH_RUN="$(latest_run_with_file "$HPRC_OUT_ROOT/pretrain_masked_train_graph" '1hop_results__*.csv')"
NOGATE_RUN="$(latest_run_with_file "$HPRC_OUT_ROOT/pretrain_masked_train_nogate" '1hop_results__*.csv')"

require_run "$FULL_RUN" "full"
require_run "$COORD_RUN" "coordinate-only"
require_run "$GRAPH_RUN" "graph-only"
require_run "$NOGATE_RUN" "no-gate"

STRICT_PRED="$(first_file "$FULL_RUN" 'strict_pooled_predictions__*.csv.gz')"
HOP_PRED="$(first_file "$FULL_RUN" '1hop_pooled_predictions__*.csv.gz')"
STRICT_CKPT="$(first_file "$FULL_RUN" 'ckpt_strict__*.pt')"
HOP_CKPT="$(first_file "$FULL_RUN" 'ckpt_1hop__*.pt')"

require_file "$STRICT_PRED" "HPRC strict pooled predictions"
require_file "$HOP_PRED" "HPRC 1-hop pooled predictions"
require_file "$STRICT_CKPT" "strict checkpoint"
require_file "$HOP_CKPT" "1-hop checkpoint"

log "Using full run: $FULL_RUN"
log "Using coordinate-only run: $COORD_RUN"
log "Using graph-only run: $GRAPH_RUN"
log "Using no-gate run: $NOGATE_RUN"

ABLATION_SUMMARY_OUT="$ANALYSIS_ROOT/maskedtrain_full_ablation_summary.csv"
if [[ "$FORCE" == "1" || ! -f "$ABLATION_SUMMARY_OUT" ]]; then
  log "Writing masked ablation summary: $ABLATION_SUMMARY_OUT"
  FULL_RUN="$FULL_RUN" \
  COORD_RUN="$COORD_RUN" \
  GRAPH_RUN="$GRAPH_RUN" \
  NOGATE_RUN="$NOGATE_RUN" \
  ABLATION_SUMMARY_OUT="$ABLATION_SUMMARY_OUT" \
  "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import pandas as pd

runs = [
    ("full", Path(os.environ["FULL_RUN"])),
    ("coordinate_only", Path(os.environ["COORD_RUN"])),
    ("graph_only", Path(os.environ["GRAPH_RUN"])),
    ("no_gate", Path(os.environ["NOGATE_RUN"])),
]

rows = []
for model_label, run_dir in runs:
    for closure in ["strict", "1hop"]:
        matches = sorted(run_dir.glob(f"{closure}_results__*.csv"))
        if not matches:
            raise FileNotFoundError(f"No {closure} results file in {run_dir}")
        path = matches[0]
        df = pd.read_csv(path)
        if "skipped" in df.columns:
            df = df[~df["skipped"].astype(bool)].copy()
        if "split" in df.columns:
            groups = list(df.groupby("split", sort=False))
        else:
            groups = [("all", df)]
        for split, sub in groups:
            rows.append(
                {
                    "model": model_label,
                    "run_dir": str(run_dir),
                    "closure": closure,
                    "split": split,
                    "n_slices": int(len(sub)),
                    "mean_test_auc": float(sub["test_auc"].mean()),
                    "median_test_auc": float(sub["test_auc"].median()),
                    "mean_best_val_auc": float(sub["best_val_auc"].mean())
                    if "best_val_auc" in sub.columns
                    else float("nan"),
                    "mean_epochs_run": float(sub["epochs_run"].mean())
                    if "epochs_run" in sub.columns
                    else float("nan"),
                    "results_file": str(path),
                }
            )

out = pd.DataFrame(rows)
out_path = Path(os.environ["ABLATION_SUMMARY_OUT"])
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False)
print(out.to_string(index=False))
PY
else
  log "Skipping existing ablation summary: $ABLATION_SUMMARY_OUT"
fi

log "Bootstrapping HPRC held-out CIs for full masked-trained model"
run_python_step "$ANALYSIS_ROOT/bootstrap_ci_maskedtrain_full_hprc_heldout/summary.csv" \
  scripts/bootstrap_link_prediction_ci.py \
  --inputs \
    "hprc_strict=$STRICT_PRED" \
    "hprc_1hop=$HOP_PRED" \
  --out-dir "$ANALYSIS_ROOT/bootstrap_ci_maskedtrain_full_hprc_heldout" \
  --n-boot "$N_BOOT" \
  --seed "$SEED" \
  --filter-column split \
  --filter-values heldout_chr_test

if [[ "$RUN_ABLATION_BOOTSTRAP" == "1" ]]; then
  log "Bootstrapping HPRC held-out CIs for all four masked ablations"
  COORD_STRICT_PRED="$(first_file "$COORD_RUN" 'strict_pooled_predictions__*.csv.gz')"
  COORD_HOP_PRED="$(first_file "$COORD_RUN" '1hop_pooled_predictions__*.csv.gz')"
  GRAPH_STRICT_PRED="$(first_file "$GRAPH_RUN" 'strict_pooled_predictions__*.csv.gz')"
  GRAPH_HOP_PRED="$(first_file "$GRAPH_RUN" '1hop_pooled_predictions__*.csv.gz')"
  NOGATE_STRICT_PRED="$(first_file "$NOGATE_RUN" 'strict_pooled_predictions__*.csv.gz')"
  NOGATE_HOP_PRED="$(first_file "$NOGATE_RUN" '1hop_pooled_predictions__*.csv.gz')"

  require_file "$COORD_STRICT_PRED" "coordinate strict pooled predictions"
  require_file "$COORD_HOP_PRED" "coordinate 1-hop pooled predictions"
  require_file "$GRAPH_STRICT_PRED" "graph strict pooled predictions"
  require_file "$GRAPH_HOP_PRED" "graph 1-hop pooled predictions"
  require_file "$NOGATE_STRICT_PRED" "no-gate strict pooled predictions"
  require_file "$NOGATE_HOP_PRED" "no-gate 1-hop pooled predictions"

  run_python_step "$ANALYSIS_ROOT/bootstrap_ci_maskedtrain_ablation_hprc_heldout/summary.csv" \
    scripts/bootstrap_link_prediction_ci.py \
    --inputs \
      "full_strict=$STRICT_PRED" \
      "full_1hop=$HOP_PRED" \
      "coordinate_strict=$COORD_STRICT_PRED" \
      "coordinate_1hop=$COORD_HOP_PRED" \
      "graph_strict=$GRAPH_STRICT_PRED" \
      "graph_1hop=$GRAPH_HOP_PRED" \
      "nogate_strict=$NOGATE_STRICT_PRED" \
      "nogate_1hop=$NOGATE_HOP_PRED" \
    --out-dir "$ANALYSIS_ROOT/bootstrap_ci_maskedtrain_ablation_hprc_heldout" \
    --n-boot "$N_BOOT" \
    --seed "$SEED" \
    --filter-column split \
    --filter-values heldout_chr_test
fi

log "Running or reusing HGSVC3 strict external transfer"
if [[ "$FORCE" == "1" || -z "$(latest_run_with_file results/hgsvc3/external_eval_maskedtrain_full_strict 'pooled_predictions.csv.gz' || true)" ]]; then
  "$PYTHON_BIN" -m graphgenomefm eval-external \
    --data-dir "$HGSVC_DATA_DIR" \
    --benchmark-dir "$HGSVC_BENCHMARK_DIR" \
    --checkpoint "$STRICT_CKPT" \
    --out-dir results/hgsvc3/external_eval_maskedtrain_full_strict \
    --closure strict \
    --split all \
    --device "$DEVICE" \
    --mask-query-edges
else
  log "Skipping existing HGSVC3 strict transfer output"
fi

log "Running or reusing HGSVC3 1-hop external transfer"
if [[ "$FORCE" == "1" || -z "$(latest_run_with_file results/hgsvc3/external_eval_maskedtrain_full_1hop 'pooled_predictions.csv.gz' || true)" ]]; then
  "$PYTHON_BIN" -m graphgenomefm eval-external \
    --data-dir "$HGSVC_DATA_DIR" \
    --benchmark-dir "$HGSVC_BENCHMARK_DIR" \
    --checkpoint "$HOP_CKPT" \
    --out-dir results/hgsvc3/external_eval_maskedtrain_full_1hop \
    --closure 1hop \
    --split all \
    --device "$DEVICE" \
    --mask-query-edges
else
  log "Skipping existing HGSVC3 1-hop transfer output"
fi

HGSVC_STRICT_RUN="$(latest_run_with_file results/hgsvc3/external_eval_maskedtrain_full_strict 'pooled_predictions.csv.gz')"
HGSVC_HOP_RUN="$(latest_run_with_file results/hgsvc3/external_eval_maskedtrain_full_1hop 'pooled_predictions.csv.gz')"
HGSVC_STRICT_PRED="$HGSVC_STRICT_RUN/pooled_predictions.csv.gz"
HGSVC_HOP_PRED="$HGSVC_HOP_RUN/pooled_predictions.csv.gz"

require_file "$HGSVC_STRICT_PRED" "HGSVC strict pooled predictions"
require_file "$HGSVC_HOP_PRED" "HGSVC 1-hop pooled predictions"

log "Bootstrapping HGSVC3 transfer CIs"
run_python_step "$ANALYSIS_ROOT/bootstrap_ci_maskedtrain_full_hgsvc/summary.csv" \
  scripts/bootstrap_link_prediction_ci.py \
  --inputs \
    "hgsvc_strict=$HGSVC_STRICT_PRED" \
    "hgsvc_1hop=$HGSVC_HOP_PRED" \
  --out-dir "$ANALYSIS_ROOT/bootstrap_ci_maskedtrain_full_hgsvc" \
  --n-boot "$N_BOOT" \
  --seed "$SEED"

log "Running HPRC strict topology-survival analysis"
run_python_step "$ANALYSIS_ROOT/topology_survival_maskedtrain_full_hprc_strict_heldout/summary.json" \
  scripts/topology_survival_analysis.py \
  --manifest "$HPRC_BENCHMARK_DIR/manifest.csv" \
  --full-segments "$HPRC_DATA_DIR/full_segments.csv" \
  --predictions "$STRICT_PRED" \
  --out-dir "$ANALYSIS_ROOT/topology_survival_maskedtrain_full_hprc_strict_heldout" \
  --filter-column split \
  --filter-values heldout_chr_test

log "Running HPRC 1-hop topology-survival analysis"
run_python_step "$ANALYSIS_ROOT/topology_survival_maskedtrain_full_hprc_1hop_heldout/summary.json" \
  scripts/topology_survival_analysis.py \
  --manifest "$HPRC_BENCHMARK_DIR/manifest.csv" \
  --full-segments "$HPRC_DATA_DIR/full_segments.csv" \
  --predictions "$HOP_PRED" \
  --out-dir "$ANALYSIS_ROOT/topology_survival_maskedtrain_full_hprc_1hop_heldout" \
  --filter-column split \
  --filter-values heldout_chr_test

log "Running HGSVC3 1-hop topology-survival analysis"
run_python_step "$ANALYSIS_ROOT/topology_survival_maskedtrain_full_hgsvc_1hop/summary.json" \
  scripts/topology_survival_analysis.py \
  --manifest "$HGSVC_BENCHMARK_DIR/manifest.csv" \
  --full-segments "$HGSVC_DATA_DIR/full_segments.csv.gz" \
  --predictions "$HGSVC_HOP_PRED" \
  --out-dir "$ANALYSIS_ROOT/topology_survival_maskedtrain_full_hgsvc_1hop"

if [[ "$RUN_HGSVC_STRICT_TOPOLOGY" == "1" ]]; then
  log "Running HGSVC3 strict topology-survival analysis"
  run_python_step "$ANALYSIS_ROOT/topology_survival_maskedtrain_full_hgsvc_strict/summary.json" \
    scripts/topology_survival_analysis.py \
    --manifest "$HGSVC_BENCHMARK_DIR/manifest.csv" \
    --full-segments "$HGSVC_DATA_DIR/full_segments.csv.gz" \
    --predictions "$HGSVC_STRICT_PRED" \
    --out-dir "$ANALYSIS_ROOT/topology_survival_maskedtrain_full_hgsvc_strict"
fi

log "Overnight masked-analysis run complete"
log "Log file: $LOG_FILE"
log "Key outputs:"
printf '  %s\n' \
  "$ABLATION_SUMMARY_OUT" \
  "$ANALYSIS_ROOT/bootstrap_ci_maskedtrain_full_hprc_heldout/summary.csv" \
  "$ANALYSIS_ROOT/bootstrap_ci_maskedtrain_full_hgsvc/summary.csv" \
  "$ANALYSIS_ROOT/topology_survival_maskedtrain_full_hprc_strict_heldout/summary.json" \
  "$ANALYSIS_ROOT/topology_survival_maskedtrain_full_hprc_1hop_heldout/summary.json" \
  "$ANALYSIS_ROOT/topology_survival_maskedtrain_full_hgsvc_1hop/summary.json"
