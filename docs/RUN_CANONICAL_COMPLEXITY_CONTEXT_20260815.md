# Canonical candidate and complexity-by-context runbook

This runbook covers the complete remaining implementation pipeline: local
validation, GitHub handoff, server execution, verification, and result return.
It does not rerun completed submission matrices and does not attempt blocked
Section 11.

## Order and expected duration

| Wave | Location | Dependency | Concurrency | Expected time |
|---|---|---|---|---:|
| 1. Focused and full tests | local | code checkout | sequential | 1–3 min |
| 2. Canonical 50-kb benchmark | local | Wave 1 | sequential | 1–5 min observed; allow 20–90 min on slower storage |
| 3. Local complexity-v2 validation | local | Wave 2 | sequential | 1–5 min |
| 4. Push implementation branch | local | Waves 1–3 | sequential | 1–10 min |
| 5. Native strict complexity-v2 | server | server checkout | sequential | 5–20 min |
| 6. Eight exact dense-score jobs | server | server checkout; may overlap Wave 5 only if using a separate I/O volume | two at once | 5–25 min |
| 7. Block-based statistical analysis and Figure 4 | server | Waves 5–6 | sequential | 2–10 min |
| 8. Verify and copy results home | server/local | Wave 7 | sequential | 5–30 min |

The server wrapper runs Waves 5–7 in dependency order and uses two concurrent
dense-score processes. The expected wall time is approximately 15–45 minutes;
allow up to 90 minutes under heavy shared-storage contention. These jobs are
CPU- and I/O-bound and do not require a GPU.

The wrapper is restart-safe. A completed audit is reused. If an earlier attempt
left an absent, invalid, or non-complete audit, its output directory is moved to
an `.incomplete.<UTC timestamp>` sibling before a clean retry; it is never
silently overwritten.

## Wave 1–3: local validation

From the repository root:

```bash
cd "/Users/longnguyentan/Downloads/Temple Uni/PangenomeFM"

RUN_TAG=20260815 \
PYTHON_BIN=/opt/anaconda3/bin/python \
bash scripts/run_canonical_complexity_local.sh
```

The wrapper is idempotent: it reuses an existing benchmark manifest, then
reruns the canonical audit and complexity extraction. For a genuinely new
benchmark, choose a new `RUN_TAG`; never overwrite the old benchmark directory.

Run the full repository suite before committing:

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python -m pytest -q
```

Required benchmark audit values:

```text
status = PASS
exact_duplicate_rows = 0
reverse_equivalent_duplicate_rows = 0
orientation_equivalent_label_conflicts = 0
invalid_label_rows = 0
```

## Wave 4: commit and push

The implementation branch is:

```text
codex/canonical-complexity-v2-20260815
```

Inspect the staged set carefully because the repository contained unrelated
working-tree edits before this implementation:

```bash
git status --short
git diff --check
git diff --cached --stat
```

Commit and push only the implementation files described in this runbook:

```bash
git commit -m "Add canonical candidate and complexity-context pipeline"
git push -u origin codex/canonical-complexity-v2-20260815
```

## Wave 5–7: server execution

Update the server checkout without modifying the completed result directories:

```bash
cd ~/PangenomeFM
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server

git fetch origin
git switch codex/canonical-complexity-v2-20260815
git pull --ff-only origin codex/canonical-complexity-v2-20260815

export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q \
  tests/test_canonical_negative_sampling.py \
  tests/test_training_recovery.py \
  tests/test_complexity.py \
  tests/test_dense_region_scores.py \
  tests/test_complexity_context_performance.py
```

Start the full server pipeline in one tmux session:

```bash
tmux new-session -d \
  -s pangenomefm-complexity-context-v2-20260815 \
  "bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server
cd ~/PangenomeFM
export PYTHONPATH=\"$PWD:$PWD/src\"
export PYTHONUNBUFFERED=1
set -o pipefail

RUN_TAG=20260815 \
MAX_CONCURRENT=2 \
bash scripts/server/run_complexity_context_pipeline.sh \
  2>&1 | tee server_workspace/results/complexity_context_v2_20260815.console.log

rc=\${PIPESTATUS[0]}
echo COMPLEXITY_CONTEXT_PIPELINE_EXIT_CODE=\$rc
exec bash
'"
```

Two score jobs at once is recommended. Four concurrent jobs repeatedly rebuild
the slice/node cache and can turn this into a storage-contention benchmark.

Monitor without attaching:

```bash
watch -n 60 '
date
pgrep -af "[p]repare_dense_region_scores|[e]xtract_graph_complexity|[a]nalyze_complexity_context" || true
tail -n 25 server_workspace/results/complexity_context_v2_20260815.console.log 2>/dev/null || true
free -h
'
```

The complexity stage intentionally reads only strict candidates. Strict graph
structure freezes the locus categories. Expanded score runs explicitly exclude
the few audited legacy canonical-label conflicts; they do not choose a label.

## Wave 7 verification

```bash
cd ~/PangenomeFM

python - <<'PY'
import json
from pathlib import Path

root = Path("server_workspace/results/complexity_context_v2_20260815")

complexity = json.loads(
    (root / "native_complexity_v2/complexity_audit.json").read_text()
)
assert complexity["status"] == "PASS", complexity
assert complexity["performance_columns_read"] == [], complexity

baselines = [
    "topology_preferential_attachment",
    "topology_degree_sum",
    "coordinate_sgd",
    "sequence_composition_sgd",
]

for baseline in baselines:
    for context in ["strict", "1hop"]:
        path = root / "scores" / baseline / context / "audit.json"
        audit = json.loads(path.read_text())
        assert audit["status"] == "complete", path
        assert audit["baseline"] == baseline, path
        assert audit["closure"] == context, path
        runs = audit["candidate_coverage_by_run"]
        assert all(row["common_slice_exact_fraction"] == 1 for row in runs), path
        if context == "strict":
            assert all(
                row["canonical_label_conflict_identities"] == 0
                for row in runs
            ), path

analysis = json.loads((root / "exact_analysis/audit.json").read_text())
assert analysis["status"] == "complete", analysis
assert analysis["inference_unit"] == "chromosome", analysis
assert set(analysis["baselines"]) == set(baselines), analysis
assert set(analysis["contexts"]) == {"strict", "1hop"}, analysis

print("COMPLEXITY_CONTEXT_V2_VERIFIED")
print("Eligible score rows:", analysis["eligible_region_rows"])
print("Paired context rows:", analysis["paired_context_rows"])
PY

cd ~/PangenomeFM/server_workspace/results/complexity_context_v2_20260815
sha256sum -c SHA256SUMS
```

Inspect the primary result table:

```bash
python - <<'PY'
import pandas as pd

path = (
    "server_workspace/results/complexity_context_v2_20260815/"
    "exact_analysis/complexity_stratum_summary.csv"
)
frame = pd.read_csv(path)
primary = frame.loc[
    frame["metric"].isin(["auprc_advantage", "model_score"])
]
print(primary.to_string(index=False))
PY
```

Positive values mean PangenomeFM outperforms the named baseline. Interpret a
stratum as supported only when the chromosome-block interval is compatible with
the claim. Report negative and null strata as primary results as well.

## Wave 8: copy server results to the local repository

Run locally:

```bash
cd "/Users/longnguyentan/Downloads/Temple Uni/PangenomeFM"
mkdir -p server_imports/complexity_context_v2_20260815

rsync -avP \
  tuv43532@cis-chen:~/PangenomeFM/server_workspace/results/complexity_context_v2_20260815/ \
  server_imports/complexity_context_v2_20260815/

cd server_imports/complexity_context_v2_20260815
sha256sum -c SHA256SUMS
```

The manuscript-ready figure will be:

```text
server_imports/complexity_context_v2_20260815/
  exact_analysis/figure4_complexity_context.pdf
```

After the group reviews the complete table, copy the accepted figure into
`paper/figures/next_stage/` and update the discussion notebook/LaTeX from the
audited CSV files. Do not manually transcribe plotted values.

## Work that remains deliberately out of scope

- Section 11 requires a path-bearing donor-filtered graph and remains blocked.
- DeepGene can be added later as an approximate downstream sequence-task
  comparator, not as an exact native edge predictor.
- PangenomeX requires a separate shallow-WGS CNV benchmark.
- A complete rerun of the historic 5-Mb matrix on regenerated canonical
  candidates should be considered only if the small audited expanded exclusion
  materially changes the conclusions.
