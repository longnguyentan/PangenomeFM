#!/usr/bin/env bash
# Server-side v2 job launcher.  Normally piped by scripts/server/pfm_remote.sh:
#   ssh host 'bash -s -- <job>' < scripts/server/pfm_v2_jobs.sh
# Jobs run inside tmux in a separate git worktree (PangenomeFM_v2) so the
# running v1 campaign checkout is never modified.  Completed outputs are never
# overwritten (each Python entry point refuses existing output directories).
set -uo pipefail
JOB="${1:-list}"
BRANCH="${PFM_BRANCH:-v2/shortcut-free-pretraining-20260926}"

MAIN=""
for d in "${PFM_MAIN:-}" "$HOME/PangenomeFM" /home/tuv43532/PangenomeFM /data/shilab/graphgenome-fm; do
  [ -n "$d" ] && [ -d "$d/.git" ] && [ -d "$d/server_workspace/data/benchmarks" ] && { MAIN="$d"; break; }
done
[ -z "$MAIN" ] && { echo "Could not find the main PangenomeFM checkout; set PFM_MAIN"; exit 2; }
WT="$(dirname "$MAIN")/PangenomeFM_v2"
SW="$MAIN/server_workspace"
GRAPH="$SW/data/processed/hprc_r2_sv/full_segments.csv.gz"
LINKS="$SW/data/processed/hprc_r2_sv/full_links.csv.gz"
BENCH="$SW/data/benchmarks/hprc_r2_pretrain_5mb_paired"
NT="$SW/results/frozen_sequence_fm_cache_20260815/hprc_target_union_sequence_fm.npz"
V1ROOT="$SW/results/full_multicohort_server_20260806"
OUT="$WT/results/v2"
FOLD_A_TEST="chr1 chr6 chr11 chr16 chr21"
FOLD_A_VAL="chr2 chr7 chr12 chr17 chr22"
echo "main=$MAIN worktree=$WT job=$JOB"

run_tmux() {  # name, command
  local name="$1"; shift
  if tmux has-session -t "$name" 2>/dev/null; then echo "tmux session $name already running"; return; fi
  mkdir -p "$OUT/logs"
  tmux new-session -d -s "$name" "cd '$WT' && source .pfm_env.sh && ( $* ) > '$OUT/logs/$name.log' 2>&1; echo EXIT=\$? >> '$OUT/logs/$name.log'"
  echo "started tmux session $name -> $OUT/logs/$name.log"
}

pretrain_cmd() {  # variant ctx gpu extra-flags...
  local variant="$1" ctx="$2" gpu="$3"; shift 3
  echo "CUDA_VISIBLE_DEVICES=$gpu python -m training.pretrain --manifest $BENCH/manifest.csv --full_segments $GRAPH \
 --out_dir $OUT/pretrain/$variant/rotating_folds/hprc_r2/fold_a/seed_42/$ctx --primary_dataset_name hprc_r2 --closures $ctx \
 --epochs 100 --patience 20 --seed 42 --split_seed 20260806 --device cuda --warmup_epochs 5 --mask_query_edges \
 --save_predictions --recovery_every 1 --batch_size 512 --lazy_tensorize --dual_stream --adaptive_window \
 --adaptive_window_base 32 --adaptive_window_alpha 4.0 --multiscale_rope --n_rope_scales 3 --orientation_rope \
 --focal_loss --focal_gamma 2.0 --drop_edge --drop_edge_rate 0.1 --test_chrs $FOLD_A_TEST --val_chrs $FOLD_A_VAL \
 --canonical_conflict_policy exclude --objective junction_repair --node_structure_source visible --pair_geometry $*"
}

case "$JOB" in
  sync)
    git -C "$MAIN" fetch origin "$BRANCH" || exit 3
    if [ ! -e "$WT" ]; then
      git -C "$MAIN" worktree add -B "$BRANCH" "$WT" "origin/$BRANCH" || exit 4
    else
      git -C "$WT" pull --ff-only origin "$BRANCH" || exit 5
    fi
    mkdir -p "$WT/server_workspace" "$OUT"
    [ -e "$WT/server_workspace/data" ] || ln -s "$SW/data" "$WT/server_workspace/data"
    cat > "$WT/.pfm_env.sh" <<ENV
for base in "\$HOME/miniconda3" "\$HOME/anaconda3" /opt/conda /opt/anaconda3; do
  [ -f "\$base/etc/profile.d/conda.sh" ] && . "\$base/etc/profile.d/conda.sh" && break
done
command -v conda >/dev/null && conda activate pangenomefm-server 2>/dev/null
export PYTHONPATH=src:. OMP_NUM_THREADS=\${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=\${OPENBLAS_NUM_THREADS:-4}
ENV
    cd "$WT" && . ./.pfm_env.sh && git log --oneline -2 && python -c "import torch,sys;print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.cuda.is_available())" \
      && python -m pytest -q -p no:cacheprovider tests/test_junction_repair.py tests/test_topology_controls.py 2>&1 | tail -2
    ;;
  audit)
    run_tmux pfm_v2_audit "python scripts/audit_masking_degree_shortcut.py --slices '$BENCH/*_strict_segments.csv.gz' --max-slices 200 --out-dir $OUT/audit_masking_hprc_r2_strict_200; \
python scripts/audit_masking_degree_shortcut.py --slices '$BENCH/*_1hop_segments.csv.gz' --max-slices 200 --out-dir $OUT/audit_masking_hprc_r2_1hop_200"
    ;;
  diag)
    cmd=""
    for ctx in strict 1hop; do
      cmd="$cmd python scripts/diagnose_degree_feature_reliance.py --checkpoint '$V1ROOT/rotating_folds/hprc_r2/fold_a/seed_42/$ctx/run_*/ckpt_${ctx}__*.pt' --manifest $BENCH/manifest.csv --full-segments $GRAPH --test-chrs $FOLD_A_TEST --closure $ctx --out-dir $OUT/diag_degree_reliance/fold_a_seed42_$ctx;"
    done
    run_tmux pfm_v2_diag "$cmd"
    ;;
  topo)
    run_tmux pfm_v2_topo "python -m tasks.transfer.topology_controls --full-segments $GRAPH --full-links $LINKS --target-cache $NT --out $OUT/topology_control/hprc_r2_topology_control.npz"
    ;;
  randinit)
    cmd=""
    for ck in "$V1ROOT"/rotating_folds/hprc_r2/fold_*/seed_*/*/run_*/ckpt_*.pt; do
      rel="${ck#$V1ROOT/}"; dir="$OUT/random_init/$(dirname "$(dirname "$rel")")/run_001"
      cmd="$cmd mkdir -p $dir && python scripts/make_random_init_checkpoint.py --checkpoint '$ck' --out '$dir/$(basename "$ck")' --seed 7;"
    done
    run_tmux pfm_v2_randinit "$cmd"
    ;;
  pilot)       run_tmux pfm_v2_pilot_strict "$(pretrain_cmd base strict 0)"; run_tmux pfm_v2_pilot_1hop "$(pretrain_cmd base 1hop 1)" ;;
  pilot-kmer)  run_tmux pfm_v2_kmer_strict "$(pretrain_cmd kmer strict 2 --node_extra_features kmer)"; run_tmux pfm_v2_kmer_1hop "$(pretrain_cmd kmer 1hop 3 --node_extra_features kmer)" ;;
  pilot-large) run_tmux pfm_v2_large_strict "$(pretrain_cmd large96x4 strict 0 --hidden_dim 96 --n_layers 4)"; run_tmux pfm_v2_large_1hop "$(pretrain_cmd large96x4 1hop 1 --hidden_dim 96 --n_layers 4)" ;;
  probes)
    TOPO="$OUT/topology_control/hprc_r2_topology_control.npz"
    cmd=""
    for model in v1:"$V1ROOT" random_init:"$OUT/random_init" v2_base:"$OUT/pretrain/base" v2_kmer:"$OUT/pretrain/kmer"; do
      name="${model%%:*}"; root="${model#*:}"
      for ctx in strict 1hop; do
        ck=$(ls "$root"/rotating_folds/hprc_r2/fold_a/seed_42/$ctx/run_*/ckpt_${ctx}__*.pt 2>/dev/null | head -1)
        [ -z "$ck" ] && { echo "skip $name $ctx (no checkpoint yet)"; continue; }
        common="--checkpoint '$ck' --manifest $BENCH/manifest.csv --full-segments $GRAPH --fold fold_a --test-chrs $FOLD_A_TEST --val-chrs $FOLD_A_VAL --closure $ctx --device cuda --seed 42 --external-sequence-cache $NT --canonical-conflict-policy exclude --topology-control-cache $TOPO"
        cmd="$cmd python scripts/server/run_sv_frozen_probe_fold.py $common --examples $SW/data/processed/hgsvc3_sv_breakpoint_examples_20260809/sv_breakpoint_examples.csv.gz --feature-cache $SW/data/processed/hprc_r2_hgsvc3_sv_features_20260809.npz --out-dir $OUT/probes/$name/sv/fold_a/seed_42/$ctx;"
        cmd="$cmd python scripts/server/run_ccre_frozen_probe_fold.py $common --node-labels $SW/data/downstream/ccre/hprc_r2_screen_v4/node_labels.csv.gz --feature-cache $SW/data/processed/hprc_r2_ccre_screen_v4_features.npz --out-dir $OUT/probes/$name/ccre/fold_a/seed_42/$ctx;"
      done
    done
    run_tmux pfm_v2_probes "$cmd"
    ;;
  list)
    tmux ls 2>/dev/null | grep pfm_v2 || echo "no v2 tmux sessions"
    for f in "$OUT"/logs/*.log; do [ -f "$f" ] && echo "--- $(basename "$f")" && tail -n 4 "$f"; done
    ;;
  pack)
    [ -d "$OUT" ] || { echo "no v2 outputs"; exit 0; }
    tarball="/tmp/pfm_v2_pack_$(date +%Y%m%d_%H%M%S).tar.gz"
    (cd "$OUT" && find . \( -name "*.csv" -o -name "*.json" -o -name "*.log" \) ! -name "*prediction*" \
        ! -path "./pretrain/*/recovery*" -size -20M -print0 \
        | tar -czf "$tarball" --null -T -) && echo "PACKED $tarball"
    ;;
  *) echo "unknown job $JOB"; exit 1 ;;
esac
