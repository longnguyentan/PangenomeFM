# Manuscript-decision experiments: server runbook

This runbook executes the experiments that should be resolved before the next
manuscript rewrite. It does not rerun upstream pretraining. Every learned
downstream probe uses the existing chromosome-held-out checkpoints and keeps
training, validation, and test nodes in one checkpoint-specific latent space.

## What is implemented

1. **Canonical affected-locus sensitivity.** Remove both contexts at all loci
   that contained a reverse-equivalent duplicate or a canonical label
   conflict, then recompute chromosome-block Figure 4 summaries.
2. **Complete C/S/T modality factorial for cCRE and SV.** Evaluate coordinate
   (C), sequence k-mer composition (S), frozen PangenomeFM topology (T), all
   three pairs, and C+S+T on identical examples. The aggregate reports paired
   added-modality gains with fold-then-seed bootstrap intervals.
3. **Frozen sequence-foundation-model adapter.** Create an audited,
   label-blind node cache from a pinned Nucleotide Transformer revision and
   repeat the same-task probes. This is a real frozen sequence-FM comparator;
   it is **not** described as an official DeepGene reproduction.
4. **Path branch-choice candidate builder.** Convert observed HGSVC path
   transitions into focal candidate groups with same-source graph alternatives.
   The candidate task is valid and donor-disjoint. A PangenomeFM learned probe
   remains gated until GBZ-native node identities have an audited mapping to a
   frozen embedding universe.

The minimum evidence needed for the manuscript decision is Waves 0--2. Wave 3
is the strongest sequence comparator. Wave 4 is optional and should not delay
the cCRE/SV manuscript decision.

## Timing and concurrency summary

| Wave | Work | Resources | Expected wall time | Concurrency |
|---|---|---:|---:|---|
| 0 | checkout, preflight, focused tests | CPU | 5--10 min | first, alone |
| 1 | canonical affected-locus sensitivity | CPU | <2 min | after Wave 0; may overlap pilot |
| 2a | one cCRE + one SV factorial pilot | GPUs 0 and 2 | 4--10 min | run together |
| 2b | 30 cCRE + 30 SV factorial jobs | 4 GPUs | 25--50 min | cCRE on 0,1; SV on 2,3 |
| 2c | two factorial aggregates | CPU | 2--8 min | run together after 2b |
| 3a | 10,000-node sequence-FM timing pilot | 1 GPU | 10--45 min | after Wave 2 |
| 3b | four sequence-FM shards | 4 GPUs | pilot-calibrated; usually 1--6 h | four shards together |
| 3c | merge cache | CPU/RAM | 5--25 min | after all shards |
| 3d | cCRE + SV sequence-FM matrices | 4 GPUs | 30--70 min | split GPUs 0,1 and 2,3 |
| 4a | 10,000-transition branch pilot | CPU/I/O | 20--60 min | do not overlap 3b |
| 4b | full HGSVC branch candidates | CPU/I/O | 30--90 min | optional; after pilot |

The old server medians were approximately 3.1 minutes per cCRE job and 2.0
minutes per SV job. The ranges above allow for the additional feature sets,
shared filesystem contention, and final compression. Sequence-FM time must be
re-estimated from the pinned-model pilot; do not rely on a generic estimate.

Expected additional disk use is roughly 1--5 GiB for probe outputs, 3--10 GiB
while sequence-FM shards plus the merged cache coexist, and below 1 GiB for
branch candidates. Check before launch.

## Wave 0 -- checkout and preflight

Run first and do not start GPUs until the tests pass.

```bash
cd ~/PangenomeFM
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server

git fetch origin
git switch codex/canonical-complexity-v2-20260815
git pull --ff-only origin codex/canonical-complexity-v2-20260815

export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python -m pytest -q \
  tests/test_modality_factorial.py \
  tests/test_ccre_frozen_probe.py \
  tests/test_sv_frozen_probe.py \
  tests/test_ccre_probe_matrix.py \
  tests/test_sv_probe_matrix.py \
  tests/test_sequence_fm_cache.py \
  tests/test_canonical_affected_slice_sensitivity.py \
  tests/test_path_branch_choice_examples.py

python -m compileall -q \
  src/evaluation/modality_factorial.py \
  scripts/server/prepare_node_sequence_fm_cache.py \
  scripts/server/merge_node_sequence_fm_caches.py \
  scripts/server/analyze_canonical_affected_slice_sensitivity.py \
  scripts/server/build_path_branch_choice_examples.py

df -h "$PWD"
free -h
nvidia-smi
```

Expected gate: 17 focused tests pass. Record the exact commit:

```bash
git rev-parse HEAD | tee server_workspace/results/manuscript_decision_git_commit_20260815.txt
```

Define server inputs once:

```bash
export CONFIG="configs/server_full_multicohort_20260806.json"
export MAIN_RESULTS="server_workspace/results/full_multicohort_server_20260806"
export CCRE_LABELS="server_workspace/data/downstream/ccre/hprc_r2_screen_v4/node_labels.csv.gz"
export CCRE_CACHE="server_workspace/data/processed/hprc_r2_ccre_screen_v4_features.npz"
export SV_EXAMPLES="server_workspace/data/processed/hgsvc3_sv_breakpoint_examples_20260809/sv_breakpoint_examples.csv.gz"
export SV_CACHE="server_workspace/data/processed/hprc_r2_hgsvc3_sv_features_20260809.npz"
export HPRC_SEGMENTS="server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz"

for required in \
  "$CONFIG" "$MAIN_RESULTS" "$CCRE_LABELS" "$CCRE_CACHE" \
  "$SV_EXAMPLES" "$SV_CACHE" "$HPRC_SEGMENTS"
do
  test -e "$required" || { echo "MISSING: $required"; exit 1; }
done
```

Use fresh output names. The programs refuse to overwrite completed artifacts.

```bash
export CCRE_FACTORIAL="server_workspace/results/ccre_modality_factorial_20260815"
export SV_FACTORIAL="server_workspace/results/hgsvc3_sv_modality_factorial_20260815"
export CANONICAL_SENSITIVITY="server_workspace/results/complexity_context_v2_20260815/canonical_affected_locus_sensitivity"

for output in "$CCRE_FACTORIAL" "$SV_FACTORIAL" "$CANONICAL_SENSITIVITY"
do
  if [ -e "$output" ]; then
    echo "STOP: output already exists; audit or choose a new RUN_TAG: $output"
    exit 1
  fi
done
```

## Wave 1 -- canonical affected-locus sensitivity

This is a conservative deletion sensitivity, not a silent relabeling. It drops
both strict and 1-hop rows at every genomic locus affected in either context.

```bash
cd ~/PangenomeFM

/usr/bin/time -v python \
  scripts/server/analyze_canonical_affected_slice_sensitivity.py \
  --region-level \
    server_workspace/results/complexity_context_v2_20260815/exact_analysis/region_level_exact_comparisons.csv.gz \
  --complexity-features \
    server_workspace/results/complexity_context_v2_20260815/native_complexity_v2/complexity_features.tsv \
  --out-dir "$CANONICAL_SENSITIVITY" \
  --primary-baseline sequence_composition_sgd \
  --max-auprc-shift 0.01 \
  --max-auroc-shift 0.01 \
  --max-model-score-shift 0.01 \
  --n-bootstrap 10000 \
  --seed 20260806 \
  --fail-on-gate \
  2>&1 | tee \
    server_workspace/results/complexity_context_v2_20260815/canonical_sensitivity.console.log
```

Validate:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path(
    "server_workspace/results/complexity_context_v2_20260815/"
    "canonical_affected_locus_sensitivity/audit.json"
)
x = json.loads(p.read_text())
print(json.dumps(x, indent=2))
assert x["status"] == "complete"
assert x["sensitivity_gate"] == "pass"
assert x["canonical_identity_affected_genomic_loci"] == 8
assert x["reverse_equivalent_candidate_duplicates"] == 12
assert x["orientation_equivalent_candidate_label_conflicts"] == 9
print("CANONICAL_SENSITIVITY_GATE_PASSED")
PY
```

The imported local result already passes this gate. Its largest primary shift
was about 0.0018, well inside the prespecified 0.01 tolerance. The server run
is still required to place the audit beside the authoritative outputs.

## Wave 2 -- complete cCRE and SV modality factorial

The default fold runners now include C, S, T, C+S, C+T, S+T, C+S+T, graph and
structural baselines, prevalence, and seeded random scores. All feature sets in
a fold use identical rows and identical chromosome splits.

### Wave 2a -- simultaneous one-job pilots

```bash
tmux new-session -d -s pangenomefm-ccre-factorial-pilot-20260815 "bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server
cd ~/PangenomeFM
export PYTHONPATH=\"$PWD:$PWD/src\"
export PYTHONUNBUFFERED=1
set -o pipefail
python scripts/server/run_ccre_frozen_probe_matrix.py \
  --config $CONFIG \
  --results-root $MAIN_RESULTS \
  --node-labels $CCRE_LABELS \
  --feature-cache $CCRE_CACHE \
  --out-root $CCRE_FACTORIAL \
  --gpus 0 --seeds 42 --contexts strict --max-jobs 1 --execute \
  2>&1 | tee ${CCRE_FACTORIAL}.pilot.console.log
echo CCRE_FACTORIAL_PILOT_EXIT_CODE=\${PIPESTATUS[0]}
exec bash
'"

tmux new-session -d -s pangenomefm-sv-factorial-pilot-20260815 "bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server
cd ~/PangenomeFM
export PYTHONPATH=\"$PWD:$PWD/src\"
export PYTHONUNBUFFERED=1
set -o pipefail
python scripts/server/run_sv_frozen_probe_matrix.py \
  --config $CONFIG \
  --results-root $MAIN_RESULTS \
  --examples $SV_EXAMPLES \
  --feature-cache $SV_CACHE \
  --out-root $SV_FACTORIAL \
  --gpus 2 --seeds 42 --contexts strict --max-jobs 1 --execute \
  2>&1 | tee ${SV_FACTORIAL}.pilot.console.log
echo SV_FACTORIAL_PILOT_EXIT_CODE=\${PIPESTATUS[0]}
exec bash
'"
```

Monitor without attaching:

```bash
watch -n 30 '
date
tail -n 8 server_workspace/results/ccre_modality_factorial_20260815.pilot.console.log 2>/dev/null || true
tail -n 8 server_workspace/results/hgsvc3_sv_modality_factorial_20260815.pilot.console.log 2>/dev/null || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
'
```

Do not start the full matrix unless both pilot summaries have one recorded job,
zero failures, and the fold audit lists all seven factorial features.

### Wave 2b -- simultaneous full matrices

The completed pilot job is verified and skipped; the remaining jobs run.

```bash
tmux new-session -d -s pangenomefm-ccre-factorial-full-20260815 "bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server
cd ~/PangenomeFM
export PYTHONPATH=\"$PWD:$PWD/src\"
export PYTHONUNBUFFERED=1
set -o pipefail
/usr/bin/time -v python scripts/server/run_ccre_frozen_probe_matrix.py \
  --config $CONFIG \
  --results-root $MAIN_RESULTS \
  --node-labels $CCRE_LABELS \
  --feature-cache $CCRE_CACHE \
  --out-root $CCRE_FACTORIAL \
  --gpus 0,1 --execute \
  2>&1 | tee ${CCRE_FACTORIAL}.full.console.log
echo CCRE_FACTORIAL_FULL_EXIT_CODE=\${PIPESTATUS[0]}
exec bash
'"

tmux new-session -d -s pangenomefm-sv-factorial-full-20260815 "bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server
cd ~/PangenomeFM
export PYTHONPATH=\"$PWD:$PWD/src\"
export PYTHONUNBUFFERED=1
set -o pipefail
/usr/bin/time -v python scripts/server/run_sv_frozen_probe_matrix.py \
  --config $CONFIG \
  --results-root $MAIN_RESULTS \
  --examples $SV_EXAMPLES \
  --feature-cache $SV_CACHE \
  --out-root $SV_FACTORIAL \
  --gpus 2,3 --execute \
  2>&1 | tee ${SV_FACTORIAL}.full.console.log
echo SV_FACTORIAL_FULL_EXIT_CODE=\${PIPESTATUS[0]}
exec bash
'"
```

Gate both matrices:

```bash
python - <<'PY'
import json
from pathlib import Path

for name in [
    "ccre_modality_factorial_20260815",
    "hgsvc3_sv_modality_factorial_20260815",
]:
    root = Path("server_workspace/results") / name
    x = json.loads((root / "matrix_summary.json").read_text())
    audits = list(root.glob("fold_*/seed_*/*/audit.json"))
    complete = sum(json.loads(p.read_text()).get("status") == "complete" for p in audits)
    print(name, x["jobs_requested"], x["jobs_recorded"], x["failures"], complete)
    assert x["jobs_requested"] == 30
    assert x["jobs_recorded"] == 30
    assert x["failures"] == 0
    assert complete == 30
print("CORE_FACTORIAL_MATRICES_COMPLETE")
PY
```

### Wave 2c -- concurrent aggregation

```bash
tmux new-session -d -s pangenomefm-factorial-aggregate-20260815 "bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server
cd ~/PangenomeFM
export PYTHONPATH=\"$PWD:$PWD/src\"
python scripts/server/aggregate_ccre_frozen_probes.py \
  --probe-root $CCRE_FACTORIAL \
  --out-dir $CCRE_FACTORIAL/paper_source_data \
  --n-bootstrap 10000 --seed 20260806 \
  > ${CCRE_FACTORIAL}.aggregate.log 2>&1 &
p1=\$!
python scripts/server/aggregate_sv_frozen_probes.py \
  --probe-root $SV_FACTORIAL \
  --out-dir $SV_FACTORIAL/paper_source_data \
  --n-bootstrap 10000 --seed 20260806 \
  > ${SV_FACTORIAL}.aggregate.log 2>&1 &
p2=\$!
wait \$p1
rc1=\$?
wait \$p2
rc2=\$?
echo FACTORIAL_AGGREGATE_EXIT_CODES=ccre:\$rc1,sv:\$rc2
exec bash
'"
```

Primary manuscript table inputs:

```text
ccre_modality_factorial_20260815/paper_source_data/ccre_summary.csv
ccre_modality_factorial_20260815/paper_source_data/ccre_modality_contributions.csv
hgsvc3_sv_modality_factorial_20260815/paper_source_data/sv_summary.csv
hgsvc3_sv_modality_factorial_20260815/paper_source_data/sv_modality_contributions.csv
```

Do not claim complementarity merely because C+S+T has the largest point
estimate. The key rows are the paired gains and their confidence intervals,
especially `topology_given_coordinate_and_sequence`.

## Wave 3 -- pinned frozen sequence-FM comparison

Run after Wave 2 so the four GPUs are free.

### Wave 3a -- model pin and timing pilot

```bash
cd ~/PangenomeFM
python -c 'import transformers, huggingface_hub; print(transformers.__version__)'

export SEQ_MODEL="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
export SEQ_MODEL_REVISION="$(python - <<'PY'
from huggingface_hub import HfApi
print(HfApi().model_info(
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
).sha)
PY
)"
test -n "$SEQ_MODEL_REVISION"
echo "PINNED_SEQUENCE_MODEL=$SEQ_MODEL@$SEQ_MODEL_REVISION"

export SEQ_ROOT="server_workspace/results/frozen_sequence_fm_cache_20260815"
mkdir -p "$SEQ_ROOT"

CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v python \
  scripts/server/prepare_node_sequence_fm_cache.py \
  --full-segments "$HPRC_SEGMENTS" \
  --target-cache "$CCRE_CACHE" "$SV_CACHE" \
  --output "$SEQ_ROOT/pilot_10000.npz" \
  --model-name "$SEQ_MODEL" \
  --revision "$SEQ_MODEL_REVISION" \
  --batch-size 32 \
  --device cuda \
  --max-length 1000 \
  --trust-remote-code \
  --max-nodes 10000 \
  2>&1 | tee "$SEQ_ROOT/pilot_10000.log"
```

The pilot deliberately uses a deterministic, ID-dispersed node sample and
therefore scans the segment source rather than timing only early node IDs.
Use its audit `wall_seconds`, observed GPU memory, and nodes/second to update
the full estimate. A rough four-shard compute estimate is:

```bash
python - <<'PY'
import json
from pathlib import Path
x = json.loads(Path(
    "server_workspace/results/frozen_sequence_fm_cache_20260815/"
    "pilot_10000.npz.audit.json"
).read_text())
union_nodes = x["target_union_nodes"]
estimate_hours = x["wall_seconds"] * union_nodes / x["embedded_nodes"] / 4 / 3600
print("Naive four-GPU estimate, hours:", round(estimate_hours, 2))
print("Use a 1.5x safety factor for four simultaneous source scans and compression.")
PY
```

### Wave 3b -- four complete shards

```bash
for shard in 0 1 2 3
do
  tmux new-session -d \
    -s "pangenomefm-sequence-fm-shard-${shard}-20260815" \
    "bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server
cd ~/PangenomeFM
export PYTHONPATH=\"$PWD:$PWD/src\"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=$shard
/usr/bin/time -v python scripts/server/prepare_node_sequence_fm_cache.py \
  --full-segments $HPRC_SEGMENTS \
  --target-cache $CCRE_CACHE $SV_CACHE \
  --output $SEQ_ROOT/shard_${shard}.npz \
  --model-name $SEQ_MODEL \
  --revision $SEQ_MODEL_REVISION \
  --batch-size 32 \
  --device cuda \
  --max-length 1000 \
  --trust-remote-code \
  --num-shards 4 \
  --shard-index $shard \
  2>&1 | tee $SEQ_ROOT/shard_${shard}.log
exec bash
'"
done
```

Four simultaneous scans are appropriate on the NVMe server but should not be
combined with the path-branch GFA scan. If disk throughput collapses, run two
shards at a time; scientific output is unchanged.

### Wave 3c -- merge and validate

```bash
cd ~/PangenomeFM

python scripts/server/merge_node_sequence_fm_caches.py \
  --shard \
    "$SEQ_ROOT/shard_0.npz" \
    "$SEQ_ROOT/shard_1.npz" \
    "$SEQ_ROOT/shard_2.npz" \
    "$SEQ_ROOT/shard_3.npz" \
  --target-cache "$CCRE_CACHE" "$SV_CACHE" \
  --output "$SEQ_ROOT/hprc_target_union_sequence_fm.npz" \
  2>&1 | tee "$SEQ_ROOT/merge.log"

python - <<'PY'
import json
from pathlib import Path
p = Path(
    "server_workspace/results/frozen_sequence_fm_cache_20260815/"
    "hprc_target_union_sequence_fm.npz.audit.json"
)
x = json.loads(p.read_text())
print(json.dumps(x, indent=2))
assert x["status"] == "complete"
assert x["coverage_fraction"] == 1.0
assert x["downstream_label_access"] == "none"
print("SEQUENCE_FM_CACHE_COMPLETE")
PY
```

### Wave 3d -- same-task sequence-FM matrices

Use new roots; these runs filter to the external-cache universe before fitting
every comparator, so exact fairness remains auditable.

```bash
export SEQ_CACHE="$SEQ_ROOT/hprc_target_union_sequence_fm.npz"
export CCRE_SEQ="server_workspace/results/ccre_sequence_fm_factorial_20260815"
export SV_SEQ="server_workspace/results/hgsvc3_sv_sequence_fm_factorial_20260815"

tmux new-session -d -s pangenomefm-ccre-sequence-fm-20260815 "bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server
cd ~/PangenomeFM
export PYTHONPATH=\"$PWD:$PWD/src\"
python scripts/server/run_ccre_frozen_probe_matrix.py \
  --config $CONFIG --results-root $MAIN_RESULTS \
  --node-labels $CCRE_LABELS --feature-cache $CCRE_CACHE \
  --external-sequence-cache $SEQ_CACHE --minimum-external-coverage 0.999 \
  --out-root $CCRE_SEQ --gpus 0,1 --execute \
  2>&1 | tee ${CCRE_SEQ}.console.log
exec bash
'"

tmux new-session -d -s pangenomefm-sv-sequence-fm-20260815 "bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server
cd ~/PangenomeFM
export PYTHONPATH=\"$PWD:$PWD/src\"
python scripts/server/run_sv_frozen_probe_matrix.py \
  --config $CONFIG --results-root $MAIN_RESULTS \
  --examples $SV_EXAMPLES --feature-cache $SV_CACHE \
  --external-sequence-cache $SEQ_CACHE --minimum-external-coverage 0.999 \
  --out-root $SV_SEQ --gpus 2,3 --execute \
  2>&1 | tee ${SV_SEQ}.console.log
exec bash
'"
```

Aggregate with the Wave 2c commands, substituting `CCRE_SEQ` and `SV_SEQ`.
The contribution tables will additionally contain:

- `frozen_sequence_fm_vs_kmer_composition`
- `frozen_sequence_fm_vs_kmer_given_topology`
- `frozen_sequence_fm_vs_kmer_given_coordinate_and_topology`

These are paired comparisons, not claims that either pretrained model is an
official DeepGene checkpoint.

## Wave 4 -- optional HGSVC focal branch-choice candidates

This wave is CPU/I/O heavy and uses the 2.6-GiB GBZ-native no-paths GFA only as
an aggregate adjacency source. The observed paths still come from the audited
GBZ path materialization.

```bash
export HGSVC_POSITIVE="server_workspace/results/path_examples_20260809/hgsvc3_all/positive_path_edges.csv.gz"
export HGSVC_NATIVE_GFA="server_workspace/data/derived/hgsvc3_gbz_native/hgsvc3-2024-02-23-mc-chm13.gbz-native.no-paths.gfa.gz"
export HGSVC_NATIVE_AUDIT="server_workspace/data/processed/hgsvc3_gbz_native/segment_index_audit.json"
export HGSVC_BRANCH="server_workspace/results/hgsvc3_path_branch_choice_20260815"

tmux new-session -d -s pangenomefm-hgsvc-branch-pilot-20260815 "bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server
cd ~/PangenomeFM
export PYTHONPATH=\"$PWD:$PWD/src\"
/usr/bin/time -v python scripts/server/build_path_branch_choice_examples.py \
  --positive-edges $HGSVC_POSITIVE \
  --gfa $HGSVC_NATIVE_GFA \
  --graph-audit $HGSVC_NATIVE_AUDIT \
  --out-dir ${HGSVC_BRANCH}_pilot_10000 \
  --alternatives-per-positive 5 \
  --max-positive-examples 10000 \
  2>&1 | tee ${HGSVC_BRANCH}_pilot_10000.log
exec bash
'"
```

Pilot gates: `status=complete`, candidate groups greater than zero, exactly one
positive per group, no observed transition absent from the audited GFA, and no
donor assigned to multiple splits. Then run full:

```bash
tmux new-session -d -s pangenomefm-hgsvc-branch-full-20260815 "bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server
cd ~/PangenomeFM
export PYTHONPATH=\"$PWD:$PWD/src\"
/usr/bin/time -v python scripts/server/build_path_branch_choice_examples.py \
  --positive-edges $HGSVC_POSITIVE \
  --gfa $HGSVC_NATIVE_GFA \
  --graph-audit $HGSVC_NATIVE_AUDIT \
  --out-dir $HGSVC_BRANCH \
  --alternatives-per-positive 5 \
  2>&1 | tee ${HGSVC_BRANCH}.log
exec bash
'"
```

The negative label means “not the observed next handle at this focal path
occurrence.” It does not mean that the alternative edge is globally absent.
Do not train a PangenomeFM branch probe until its node-identity/embedding gate
is resolved and audited.

## Final audit before manuscript rewrite

For the minimum decision package:

```bash
python scripts/server/audit_manuscript_decision_experiments.py \
  --canonical-sensitivity-audit \
    "$CANONICAL_SENSITIVITY/audit.json" \
  --ccre-root "$CCRE_FACTORIAL" \
  --ccre-aggregate "$CCRE_FACTORIAL/paper_source_data" \
  --sv-root "$SV_FACTORIAL" \
  --sv-aggregate "$SV_FACTORIAL/paper_source_data" \
  --output \
    server_workspace/results/manuscript_decision_audit_20260815.json
```

For the extended sequence/path package, substitute `CCRE_SEQ` and `SV_SEQ` and
add:

```bash
  --sequence-cache "$SEQ_CACHE" \
  --path-branch-audit "$HGSVC_BRANCH/audit.json"
```

Only rewrite the quantitative Results after the minimum audit reports
`status=pass`. Interpret outcomes using this order:

1. canonical sensitivity determines whether Figure 4 is robust;
2. topology-given-C+S determines the defensible “topology adds information”
   claim for each downstream task;
3. sequence-FM comparisons determine whether the result survives a stronger
   sequence representation;
4. the path branch dataset is a future topology-native task, not current
   performance evidence until its embedding gate is resolved.
