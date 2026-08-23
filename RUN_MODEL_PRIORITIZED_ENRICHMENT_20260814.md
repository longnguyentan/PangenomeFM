# Model-prioritized matched enrichment: implementation and server runbook

This runbook closes the missing implementation between the completed rotating
chromosome experiments and Section 13 of `RUN_SUBMISSION_TIER_EXPERIMENTS.md`.
It is intentionally separate from the PSB manuscript. The purpose is to leave
one reproducible record of what is scored, what is held out, which external
annotations are used, how the cases are selected, and which claims the result
can or cannot support.

Copy only the fenced command blocks into the shell. Explanatory text and time
estimates are not shell commands.

## 1. Scientific definition fixed before the enrichment

The native rotating benchmark manifest has 608 deterministic, non-overlapping 5-Mb
HPRC R2 tiles on chromosomes 1--22, X, and Y. The held-out prediction files
contain about 90 test query edges per tile. A nominal 50-kb subdivision would
contain roughly one held-out query edge on average and therefore cannot satisfy
the earlier proposed ten-candidate gate. It would create apparent genomic
resolution without new model evidence.

The implemented analysis therefore uses the native 5-Mb tiles.

For every fold and seed:

1. Temperature calibration is fit only on `val_chr_test` predictions.
2. The calibrated model is evaluated only on `heldout_chr_test` candidates.
3. The comparator is the prespecified `sequence_composition_sgd` baseline,
   whose Platt calibration was already fit only on validation chromosomes.
4. Neural local endpoints are mapped back to stable oriented graph IDs using
   the original slice link tables. Neural and baseline candidates are joined
   by exact `(slice, u_oid, v_oid, occurrence)` identity, never by row order.
   Exact candidate identity is required within every slice produced by both
   methods. A whole slice absent from either method is excluded and recorded;
   label disagreement or partial candidate disagreement within a common slice
   remains a hard error.
5. The tile score is

   `baseline held-out NLL - PangenomeFM held-out NLL`.

6. Scores are averaged across seeds 42, 314159, and 20260806.
7. A tile requires at least ten held-out candidates and both target classes.
8. Priority is assigned within chromosome: top 10% by descending mean score,
   at least one tile, with `region_id` as the deterministic tie-break.
9. QTL and GWAS files are not accepted by either score or priority program.

Positive score means lower held-out log loss for PangenomeFM than for the
sequence-composition baseline. It is not a biological effect size.

### Recorded coverage incident and resolution

The first server run stopped at fold C with 12,059 rows present in both
outputs and three rows present only in the baseline output. This was an input
support mismatch, not a training, calibration, GPU, or download failure. The
first implementation aligned candidates by their order within each slice, so
it correctly refused to continue once the totals differed, but it could not
safely identify which edges were common.

The corrected implementation reconstructs the neural evaluator's exact
local-to-global oriented-node map from `full_segments.csv.gz` and every native
slice's `links_path`. It then joins exact oriented edges. The manifest contains
608 strict tiles while the neural result files cover 606; tiles absent from a
method remain in `region_scores.csv` with an audited ineligibility reason. The
program writes `candidate_coverage_summary.csv` and
`candidate_coverage_exclusions.csv.gz`, and still stops if the two methods
disagree on even one candidate inside a slice that both produced.

## 2. New programs and tests

- `scripts/server/prepare_dense_region_scores.py`
- `scripts/server/prepare_model_prioritized_regions.py`
- `scripts/server/run_matched_enrichment_matrix.py`
- `tests/test_dense_region_scores.py`
- `tests/test_model_prioritized_regions.py`
- `tests/test_matched_enrichment_matrix.py`

The first program is signal-blind and uses completed model/baseline results.
The second streams or chunks the downloaded covariates and creates
`regions.csv`. The third runs four independent enrichment jobs concurrently and
writes a matrix summary.

## 3. Dependencies and concurrency

| Stage | Prerequisite | Main resource | Can run concurrently? | Expected time |
|---|---|---|---|---|
| Code tests | Updated checkout | CPU | Yes | under 2 minutes |
| chr22 score pilot | Rotating neural and baseline predictions | CPU/read I/O | Yes, including with GPU jobs | 2--10 minutes |
| Full 608-tile score | Pilot passes | CPU/read I/O | Yes, but avoid another result-bundle checksum pass | 10--45 minutes |
| Genomic covariates and priority | Full score and four verified downloads | CPU, gzip, disk I/O, 4--8 GB RAM | Run alone from other heavy disk scans | 1--6 hours |
| Four enrichment jobs | Valid `regions.csv` | Four CPU workers | Run all four together | 10--60 minutes |
| Packaging | All desired analyses audited | disk I/O | Last | 10--45 minutes |

The 1--6 hour covariate estimate is deliberately broad. The 719-MB compressed
Umap track and 42,140,988 normalized HPRC records dominate it. The command uses
vectorized million-row chunks and reports progress. `/usr/bin/time -v` records
the actual wall time and peak memory for the final report.

The native benchmark uses fixed 5-Mb bins, so the terminal bin on most
chromosomes extends beyond the canonical GRCh38 chromosome length. The region
builder preserves the original interval in `benchmark_start` and
`benchmark_end`, clips the analysis `end` to the FASTA chromosome boundary,
recomputes `region_length`, and records every change in
`reference_boundary_clipping.csv`. All covariates, signal overlaps, matching,
and enrichment use only the clipped interval.

Section 11 remains independent and blocked until an exact path-bearing HPRC
graph is rebuilt without the four shared donors. It is not a prerequisite for
this enrichment analysis, and these enrichment outputs do not repair that
graph-level donor-leakage limitation.

## 4. Update the server checkout and test the code

These commands assume the implementation has been committed and pushed to
`origin/dev-exp`.

```bash
cd ~/PangenomeFM
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server

git pull --ff-only origin dev-exp

export PANGENOMEFM_DATA_ROOT="$PWD/server_workspace/data"
export PANGENOMEFM_RESULTS_ROOT="$PWD/server_workspace/results"
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python -m pytest -q \
  tests/test_dense_region_scores.py \
  tests/test_model_prioritized_regions.py \
  tests/test_matched_locus_enrichment.py \
  tests/test_matched_enrichment_matrix.py
```

Expected: 15 tests pass. Confirm that the programs are present:

```bash
for script in \
  scripts/server/prepare_dense_region_scores.py \
  scripts/server/prepare_model_prioritized_regions.py \
  scripts/server/run_matched_enrichment_matrix.py
do
  test -s "$script" || {
    echo "STOP: missing $script"
    exit 1
  }
done

echo "REGION_PIPELINE_CODE_PRESENT"
```

## 5. One preflight for every required input

```bash
cd ~/PangenomeFM

CONFIG="configs/server_full_multicohort_20260806.json"
MAIN_RESULTS="server_workspace/results/full_multicohort_server_20260806"
BASELINE_RESULTS="server_workspace/results/rotating_link_baselines_20260809"
COVARIATE_ROOT="server_workspace/data/downstream/region_covariates_20260814"
HPRC_VARIANTS="server_workspace/results/sv_truth_audit_20260809/hprc_r2_wave/normalized_variants.csv.gz"

for path in \
  "$CONFIG" \
  "$MAIN_RESULTS" \
  "$BASELINE_RESULTS" \
  "$COVARIATE_ROOT/hg38.fa.gz" \
  "$COVARIATE_ROOT/k100.umap.bedgraph.gz" \
  "$COVARIATE_ROOT/gencode.v50.annotation.gtf.gz" \
  "$HPRC_VARIANTS" \
  server_workspace/results/qtl_signals_20260809/eqtl_chr8/signals.csv.gz \
  server_workspace/results/qtl_signals_20260809/sqtl_chr8/signals.csv.gz \
  server_workspace/results/gwas_catalog_signals_20260809/all_traits/signals.csv.gz \
  server_workspace/results/gwas_catalog_signals_20260809/parkinson/signals.csv.gz
do
  test -s "$path" || {
    echo "STOP: missing required input: $path"
    exit 1
  }
done

sha256sum -c "$COVARIATE_ROOT/SHA256SUMS"

python - <<'PY'
import json
from pathlib import Path

checks = [
    (
        Path("server_workspace/results/rotating_link_baselines_20260809/matrix_summary.json"),
        30,
    ),
]
for path, expected in checks:
    x = json.loads(path.read_text())
    assert x["jobs_requested"] == expected
    assert x["jobs_recorded"] == expected
    assert x["failures"] == 0
print("REGION_PIPELINE_INPUTS_VERIFIED")
PY
```

## 6. Run the bounded chr22 score pilot

This pilot reads one fold, one seed, and nine chr22 tiles. It does not use a
GPU and does not read the multi-gigabyte genomic covariates.

```bash
cd ~/PangenomeFM

PILOT="server_workspace/results/dense_region_scores_chr22_pilot_20260814"

test ! -e "$PILOT" || {
  echo "STOP: pilot output already exists: $PILOT"
  exit 1
}

/usr/bin/time -v \
python scripts/server/prepare_dense_region_scores.py \
  --config configs/server_full_multicohort_20260806.json \
  --results-root server_workspace/results/full_multicohort_server_20260806 \
  --baseline-root server_workspace/results/rotating_link_baselines_20260809 \
  --out-dir "$PILOT" \
  --regime hprc_r2 \
  --dataset hprc_r2 \
  --closure strict \
  --baseline sequence_composition_sgd \
  --seeds 42 \
  --chromosomes chr22 \
  --minimum-test-candidates 10 \
  2>&1 | tee server_workspace/results/dense_region_scores_chr22_pilot_20260814.log
```

Validate the pilot:

```bash
cd ~/PangenomeFM

python - <<'PY'
import json
import pandas as pd
from pathlib import Path

root = Path("server_workspace/results/dense_region_scores_chr22_pilot_20260814")
audit = json.loads((root / "audit.json").read_text())
regions = pd.read_csv(root / "region_scores.csv")

print({
    "status": audit["status"],
    "tiles": audit["tiles"],
    "eligible_tiles": audit["eligible_tiles"],
    "folds": audit["folds"],
    "seeds": audit["seeds"],
    "seconds": audit["wall_seconds"],
})

assert audit["status"] == "complete"
assert audit["tiles"] == 9
assert audit["folds"] == ["fold_b"]
assert audit["seeds"] == [42]
assert audit["eligible_tiles"] > 0
assert all(
    row["common_slice_exact_fraction"] == 1
    for row in audit["candidate_coverage_by_run"]
)
assert regions["chromosome"].eq("chr22").all()
assert audit["downstream_signal_access"].startswith("none")
assert "excluded that tile's chromosome" in audit["leakage_control"]

print("CHR22_DENSE_SCORE_PILOT_COMPLETE")
PY

(cd server_workspace/results/dense_region_scores_chr22_pilot_20260814 && \
 sha256sum -c SHA256SUMS)
```

If the pilot fails, do not launch the full score. Preserve the output and log,
then inspect the exception. Do not delete or silently overwrite it.

## 7. Run all folds and three seeds

Expected time: 10--45 minutes. No GPU is required. The exact-identity version
also reads the 608 slice link tables once to reconstruct the local-to-global
endpoint mapping used by the neural evaluator.

```bash
cd ~/PangenomeFM

FULL_SCORE="server_workspace/results/dense_region_scores_20260814"
FULL_LOG="server_workspace/results/dense_region_scores_20260814.console.log"

test ! -e "$FULL_SCORE" || {
  echo "STOP: full score output already exists: $FULL_SCORE"
  exit 1
}

tmux new-session -d \
  -s pangenomefm-dense-region-scores-20260814 \
  "bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server
cd ~/PangenomeFM
export PANGENOMEFM_DATA_ROOT=\"$PWD/server_workspace/data\"
export PANGENOMEFM_RESULTS_ROOT=\"$PWD/server_workspace/results\"
export PYTHONPATH=\"$PWD:$PWD/src\"
export PYTHONUNBUFFERED=1
set -o pipefail

/usr/bin/time -v \
python scripts/server/prepare_dense_region_scores.py \
  --config configs/server_full_multicohort_20260806.json \
  --results-root server_workspace/results/full_multicohort_server_20260806 \
  --baseline-root server_workspace/results/rotating_link_baselines_20260809 \
  --out-dir server_workspace/results/dense_region_scores_20260814 \
  --regime hprc_r2 \
  --dataset hprc_r2 \
  --closure strict \
  --baseline sequence_composition_sgd \
  --minimum-test-candidates 10 \
  2>&1 | tee server_workspace/results/dense_region_scores_20260814.console.log

rc=\${PIPESTATUS[0]}
echo DENSE_REGION_SCORE_EXIT_CODE=\$rc
exec bash
'"
```

Monitor without attaching:

```bash
tmux capture-pane \
  -p \
  -t pangenomefm-dense-region-scores-20260814 \
  -S -120 | tail -n 120

tail -n 120 \
  server_workspace/results/dense_region_scores_20260814.console.log
```

Full-score gate:

```bash
cd ~/PangenomeFM

python - <<'PY'
import json
import pandas as pd
from pathlib import Path

root = Path("server_workspace/results/dense_region_scores_20260814")
audit = json.loads((root / "audit.json").read_text())
regions = pd.read_csv(root / "region_scores.csv")
coverage = pd.read_csv(root / "candidate_coverage_summary.csv")
coverage_exclusions = pd.read_csv(
    root / "candidate_coverage_exclusions.csv.gz"
)

print({
    "status": audit["status"],
    "tiles": audit["tiles"],
    "eligible_tiles": audit["eligible_tiles"],
    "folds": audit["folds"],
    "seeds": audit["seeds"],
    "exclusions": audit["exclusion_reason_counts"],
    "seconds": audit["wall_seconds"],
})

assert audit["status"] == "complete"
assert audit["tiles"] == 608
assert audit["folds"] == ["fold_a", "fold_b", "fold_c", "fold_d", "fold_e"]
assert audit["seeds"] == [42, 314159, 20260806]
assert len(audit["input_prediction_files"]) == 30
assert audit["eligible_tiles"] > 0
assert regions["chromosome"].nunique() == 24
assert regions["region_id"].is_unique
assert len(coverage) == 15
assert coverage["common_slice_exact_fraction"].eq(1).all()
assert (
    coverage["common_candidates"]
    + coverage["neural_only_candidates"]
    == coverage["neural_candidates"]
).all()
assert (
    coverage["common_candidates"]
    + coverage["baseline_only_candidates"]
    == coverage["baseline_candidates"]
).all()
assert len(coverage_exclusions) == audit["candidate_coverage_exclusions"]

print("Candidate exclusions by side:")
print(coverage_exclusions["_merge"].value_counts().to_string())

print("FULL_DENSE_REGION_SCORES_COMPLETE")
PY

(cd server_workspace/results/dense_region_scores_20260814 && \
 sha256sum -c SHA256SUMS)
```

## 8. Build the covariates and fixed region universe

Run this I/O-heavy stage after the full-score gate. Avoid running another large
GBZ/GFA scan at the same time. GPU-only work may run concurrently if it does
not saturate the same disk.

Expected time: 1--6 hours. Expected peak RAM: approximately 4--8 GB, primarily
from the decompressed GRCh38 FASTA buffer.

```bash
cd ~/PangenomeFM

MODEL_REGIONS="server_workspace/results/model_prioritized_regions_20260809"
MODEL_LOG="server_workspace/results/model_prioritized_regions_20260809.console.log"

test ! -e "$MODEL_REGIONS" || {
  echo "STOP: model-region output already exists: $MODEL_REGIONS"
  exit 1
}

tmux new-session -d \
  -s pangenomefm-model-regions-20260814 \
  "bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server
cd ~/PangenomeFM
export PYTHONPATH=\"$PWD:$PWD/src\"
export PYTHONUNBUFFERED=1
set -o pipefail

/usr/bin/time -v \
python scripts/server/prepare_model_prioritized_regions.py \
  --dense-scores server_workspace/results/dense_region_scores_20260814/region_scores.csv \
  --reference-fasta server_workspace/data/downstream/region_covariates_20260814/hg38.fa.gz \
  --mappability-bedgraph server_workspace/data/downstream/region_covariates_20260814/k100.umap.bedgraph.gz \
  --gene-annotation server_workspace/data/downstream/region_covariates_20260814/gencode.v50.annotation.gtf.gz \
  --normalized-variants server_workspace/results/sv_truth_audit_20260809/hprc_r2_wave/normalized_variants.csv.gz \
  --out-dir server_workspace/results/model_prioritized_regions_20260809 \
  --priority-fraction 0.10 \
  --controls-per-case 5 \
  --mappability-chunksize 1000000 \
  --variant-chunksize 1000000 \
  2>&1 | tee server_workspace/results/model_prioritized_regions_20260809.console.log

rc=\${PIPESTATUS[0]}
echo MODEL_REGION_BUILD_EXIT_CODE=\$rc
exec bash
'"
```

Progress is printed for every ten million mappability or variant rows and at
each major stage. During the final checksum pass the log explicitly says that
multi-gigabyte inputs are being hashed.

```bash
tmux capture-pane \
  -p \
  -t pangenomefm-model-regions-20260814 \
  -S -160 | tail -n 160

tail -n 160 \
  server_workspace/results/model_prioritized_regions_20260809.console.log
```

Validate the resulting region universe:

```bash
cd ~/PangenomeFM

python - <<'PY'
import json
import numpy as np
import pandas as pd
from pathlib import Path

root = Path("server_workspace/results/model_prioritized_regions_20260809")
audit = json.loads((root / "audit.json").read_text())
frame = pd.read_csv(root / "regions.csv")

required = {
    "region_id", "chromosome", "start", "end", "is_prioritized",
    "region_length", "gc_content", "mappability", "graph_complexity",
    "variant_density", "distance_to_gene", "model_score", "fold",
    "benchmark_start", "benchmark_end", "reference_chromosome_length",
    "reference_clipped", "reference_clipped_bp",
}
missing = required - set(frame)
assert not missing, sorted(missing)
assert audit["status"] == "complete"
assert frame["region_id"].is_unique
assert frame["is_prioritized"].astype(bool).any()
assert (~frame["is_prioritized"].astype(bool)).any()
assert frame["chromosome"].nunique() == 24
assert (frame["region_length"] == frame["end"] - frame["start"]).all()

numeric = [
    "start", "end", "region_length", "gc_content", "mappability",
    "graph_complexity", "variant_density", "distance_to_gene", "model_score",
]
assert np.isfinite(frame[numeric].to_numpy(float)).all()
assert frame["gc_content"].between(0, 1).all()
assert frame["mappability"].between(0, 1).all()
assert audit["reference_clipped_regions"] > 0
assert audit["reference_clipped_bases"] > 0
assert (
    frame["end"]
    <= frame["reference_chromosome_length"]
).all()
assert (
    frame["reference_clipped_bp"]
    == frame["benchmark_end"] - frame["end"]
).all()
clipping = pd.read_csv(root / "reference_boundary_clipping.csv")
assert len(clipping) == audit["reference_clipped_regions"]

for chromosome, group in frame.groupby("chromosome"):
    cases = int(group["is_prioritized"].astype(bool).sum())
    controls = len(group) - cases
    assert cases >= 1, chromosome
    assert controls >= 5, chromosome

print({
    "input_regions": audit["input_regions"],
    "retained_regions": audit["retained_regions"],
    "prioritized_regions": audit["prioritized_regions"],
    "control_regions": audit["control_regions"],
    "variant_rows": audit["variant_rows_processed"],
    "seconds": audit["wall_seconds"],
})
print("MODEL_PRIORITIZED_REGION_UNIVERSE_COMPLETE")
PY

(cd server_workspace/results/model_prioritized_regions_20260809 && \
 sha256sum -c SHA256SUMS)
```

If all 608 tiles pass score eligibility, the 10% within-chromosome rule yields
49 prioritized tiles. The code does not assert 49 because missing candidates,
single-class tiles, or incomplete seeds are valid audited exclusions.

## 9. Run four enrichment analyses concurrently

The four signal sets are independent once `regions.csv` is fixed. The matrix
runner uses four CPU workers, preserves separate logs, verifies matching
checksums on resume, and refuses to overwrite an incomplete output directory.

```bash
cd ~/PangenomeFM

ENRICH_ROOT="server_workspace/results/matched_enrichment_20260809"

test ! -e "$ENRICH_ROOT" || {
  echo "STOP: enrichment root already exists: $ENRICH_ROOT"
  echo "Verify a completed matrix or archive an incomplete directory first."
  exit 1
}

tmux new-session -d \
  -s pangenomefm-matched-enrichment-20260814 \
  "bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pangenomefm-server
cd ~/PangenomeFM
export PYTHONPATH=\"$PWD:$PWD/src\"
export PYTHONUNBUFFERED=1
set -o pipefail

/usr/bin/time -v \
python scripts/server/run_matched_enrichment_matrix.py \
  --regions server_workspace/results/model_prioritized_regions_20260809/regions.csv \
  --signal eqtl_chr8=server_workspace/results/qtl_signals_20260809/eqtl_chr8/signals.csv.gz \
  --signal sqtl_chr8=server_workspace/results/qtl_signals_20260809/sqtl_chr8/signals.csv.gz \
  --signal gwas_all_traits=server_workspace/results/gwas_catalog_signals_20260809/all_traits/signals.csv.gz \
  --signal gwas_parkinson=server_workspace/results/gwas_catalog_signals_20260809/parkinson/signals.csv.gz \
  --out-root server_workspace/results/matched_enrichment_20260809 \
  --jobs 4 \
  --controls-per-case 5 \
  --relative-caliper 0.25 \
  --n-permutations 10000 \
  --n-bootstrap 2000 \
  --seed 20260806 \
  --execute \
  2>&1 | tee server_workspace/results/matched_enrichment_20260809.console.log

rc=\${PIPESTATUS[0]}
echo MATCHED_ENRICHMENT_MATRIX_EXIT_CODE=\$rc
exec bash
'"
```

Monitor:

```bash
tmux capture-pane \
  -p \
  -t pangenomefm-matched-enrichment-20260814 \
  -S -160 | tail -n 160

tail -n 80 \
  server_workspace/results/matched_enrichment_20260809/logs/*.log
```

Final matrix gate:

```bash
cd ~/PangenomeFM

python - <<'PY'
import json
import pandas as pd
from pathlib import Path

root = Path("server_workspace/results/matched_enrichment_20260809")
matrix = json.loads((root / "matrix_summary.json").read_text())

print({
    "requested": matrix["jobs_requested"],
    "recorded": matrix["jobs_recorded"],
    "failures": matrix["failures"],
})
assert matrix["jobs_requested"] == 4
assert matrix["jobs_recorded"] == 4
assert matrix["failures"] == 0

for name in ["eqtl_chr8", "sqtl_chr8", "gwas_all_traits", "gwas_parkinson"]:
    audit_path = root / name / "audit.json"
    summary_path = root / name / "enrichment_summary.csv"
    assert audit_path.is_file(), audit_path
    assert summary_path.is_file(), summary_path
    audit = json.loads(audit_path.read_text())
    summary = pd.read_csv(summary_path)
    print(name, {
        "signals": audit["signals"],
        "direct_overlaps": audit["direct_overlaps"],
        "prioritized_regions": audit["prioritized_regions"],
        "complete_matched_sets": audit["complete_matched_sets"],
        "excluded_cases": audit["excluded_cases"],
        "matched_risk_difference": summary.loc[0, "matched_risk_difference"],
        "permutation_p": summary.loc[0, "permutation_p_greater"],
    })

print("SECTION_13_MATCHED_ENRICHMENT_COMPLETE")
PY
```

## 10. Interpretation gates

The primary test is complete matched-set permutation. Pooled Fisher statistics
are secondary. `direct_overlaps.csv.gz` is descriptive.

Before reporting any enrichment:

- report the number of prioritized regions, complete matched sets, and excluded
  cases;
- report overlap saturation in cases and controls;
- retain a null or negative result;
- do not describe association overlap as causality, colocalization, functional
  validation, or fine-mapping;
- disclose that the region score is derived from topology-link generalization,
  not sequence pretraining or a biological phenotype;
- keep the four analyses separate rather than choosing the most favorable one.

The chr8 eQTL and sQTL analyses have only the prioritized chr8 tiles as
informative cases. The all-trait Catalog signal is extremely dense. These runs
may therefore be underpowered or overlap-saturated despite being technically
valid. If that happens, the correct result is an audited limitation, not a
claim. A strong journal QTL analysis still needs independent loci or credible
sets, matched backgrounds, and a finer independently supported model-scoring
unit.

## 11. Package after the matrix gate

```bash
cd ~/PangenomeFM/server_workspace/results

tar --use-compress-program='zstd -T0 -10' \
  -cf submission_followups_complete_20260814.tar.zst \
  rotating_ablation_hprc_20260809 \
  rotating_link_baselines_20260809 \
  ccre_crossfit_20260809 \
  hgsvc3_sv_crossfit_20260809 \
  sv_truth_audit_20260809 \
  path_sample_overlap_20260809 \
  path_resolved_splits_20260809 \
  path_examples_20260809/hprc_r2_all \
  qtl_signals_20260809 \
  gwas_catalog_signals_20260809 \
  dense_region_scores_20260814 \
  model_prioritized_regions_20260809 \
  matched_enrichment_20260809 \
  capacity_scaling_hprc_20260809 \
  full_multicohort_server_20260806/recovered_benchmark_audits

sha256sum submission_followups_complete_20260814.tar.zst \
  > submission_followups_complete_20260814.tar.zst.sha256

sha256sum -c submission_followups_complete_20260814.tar.zst.sha256
```

Do not add the failed/path-free Section 11 graph output to the package. Add the
shared-donor-excluded transfer only after a valid retained-assembly graph is
built, audited, and its rotating matrix reports zero failures.
