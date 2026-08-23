# Next-stage command reference

Run commands from the repository root. `PYTHONPATH=src:.` is shown explicitly so scripts work without an editable install. Commands that require server-only inputs use task-scoped environment variables and must be filled with verified paths.

## 1. Environment setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
```

The audited local alternative is `/opt/anaconda3/bin/python` with `PYTHONPATH=src:.`.

## 2. Tests

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python -m pytest -q
```

Focused next-stage tests:

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python -m pytest -q \
  tests/test_training_recovery.py \
  tests/test_comparative_benchmark.py \
  tests/test_complexity.py
```

## 3. Graph/data validation

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python -m cli check-data --data-dir data/hprc
PYTHONPATH=src:. /opt/anaconda3/bin/python -m cli check-data --data-dir data/hgsvc3
```

Repository-wide inventory:

```bash
/opt/anaconda3/bin/python scripts/build_repository_audit.py \
  --root . --out-dir results/repository_audit_20260814
```

## 4. Benchmark manifest generation and fairness audit

Complexity is generated first so the pilot manifest can join frozen locus labels:

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python scripts/extract_graph_complexity.py
PYTHONPATH=src:. /opt/anaconda3/bin/python scripts/build_comparative_benchmark_pilot.py
```

The explicit equality gate should currently fail with exit code 3:

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python \
  scripts/build_comparative_benchmark_pilot.py --overwrite --require-exact-all
```

## 5. PangenomeFM pilot

The full fold-b run is currently blocked: the whole-source audit found 28 orientation-equivalent pairs with conflicting labels in 12 non-pilot slices. Regenerate those candidate files before cross-chromosome execution; do not choose labels silently.

The following one-epoch chr22-only command is a **loader/code-path smoke test**, not held-out evidence:

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python -m training.pretrain \
  --manifest results/comparative_benchmark/pilot/source_slice_manifest.csv \
  --full_segments data/hprc/full_segments.csv \
  --out_dir results/comparative_benchmark/pilot/pangenomefm_smoke \
  --closures strict \
  --hidden_dim 48 --n_heads 4 --n_layers 2 \
  --dual_stream --multiscale_rope --orientation_rope --adaptive_window \
  --focal_loss --drop_edge \
  --epochs 1 --patience 1 \
  --mask_query_edges --save_predictions \
  --seed 42 --device cpu
```

Do not promote this one-epoch inner-split smoke score. A manuscript pilot must use the repaired source candidates, the frozen chromosome fold/training budget and recorded GPU/RAM/runtime.

## 6. DeepGene pilot

There is currently **no scientifically valid exact DeepGene edge-scoring command**. The official code has no candidate-edge objective and its checkpoint is not local. The only prepared operation is the approximately matched endpoint-sequence inventory:

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python \
  scripts/build_comparative_benchmark_pilot.py --overwrite
```

Any future DeepGene execution must pin official commit `486343e5212361d6cd7ed03c624f430ed3d5f02e`, checksum the checkpoint, freeze the new supervised adapter and label the result `approximately matched`.

## 7. PangenomeX pilot

There is currently **no endpoint-edge conversion command**. PangenomeX remains contextual. A task-faithful run requires the same samples' shallow-WGS BAMs, CNV truth, reference/length files and phylogeny; it must be evaluated as a separate large-CNV benchmark.

## 8. Complexity extraction

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python scripts/extract_graph_complexity.py \
  --manifest data/hprc/benchmark_matched_nonoverlap_v2_review_20260804/manifest.csv \
  --config configs/complexity_definition_v1.yaml \
  --out-dir results/complexity/graph_window_complexity_v1
```

Frozen thresholds are protected from overwrite; use `--overwrite` only when intentionally creating a new audited artifact.

## 9. Curated-region integration

Collaborators edit:

```text
data/annotations/complex_regions_v1.tsv
```

Coordinates with an unverified source/build stay blank. A join command will be added after the first verified catalogue row exists; no guessed coordinate is accepted.

## 10. Performance stratification

Blocked until at least two methods have paired predictions keyed by `example_id`. Join predictions to `benchmark_manifest.parquet` and use chromosome/fold-aware resampling; do not bootstrap individual edges as independent replicates.

## 11. Result aggregation

```bash
/opt/anaconda3/bin/python scripts/build_canonical_primary_results.py
```

Regenerate the existing full-server analysis bundle when needed:

```bash
PYTHONPATH=src:. MPLCONFIGDIR=/private/tmp/pangenomefm-mpl \
  /opt/anaconda3/bin/python scripts/server/analyze_downloaded_server_results.py --help
```

## 12. SV probe

With verified server paths:

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python scripts/server/run_sv_frozen_probe_matrix.py \
  --config "$PANGENOMEFM_SERVER_CONFIG" \
  --results-root "$PANGENOMEFM_SERVER_RESULTS" \
  --examples "$PANGENOMEFM_SV_EXAMPLES" \
  --feature-cache "$PANGENOMEFM_SV_FEATURE_CACHE" \
  --out-root "$PANGENOMEFM_SV_OUT" \
  --gpus 0 --max-jobs 1
```

Omit `--execute` for the smoke/queue audit. Add it only after inspecting the generated job plan. Aggregate after all expected folds/seeds/contexts exist:

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python scripts/server/aggregate_sv_frozen_probes.py \
  --probe-root "$PANGENOMEFM_SV_OUT" \
  --out-dir "$PANGENOMEFM_SV_AGGREGATE"
```

## 13. cCRE probe

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python scripts/server/run_ccre_frozen_probe_matrix.py \
  --config "$PANGENOMEFM_SERVER_CONFIG" \
  --results-root "$PANGENOMEFM_SERVER_RESULTS" \
  --node-labels "$PANGENOMEFM_CCRE_LABELS" \
  --feature-cache "$PANGENOMEFM_CCRE_FEATURE_CACHE" \
  --out-root "$PANGENOMEFM_CCRE_OUT" \
  --gpus 0 --max-jobs 1
```

Aggregate only after the matrix audit:

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python scripts/server/aggregate_ccre_frozen_probes.py \
  --probe-root "$PANGENOMEFM_CCRE_OUT" \
  --out-dir "$PANGENOMEFM_CCRE_AGGREGATE"
```

## 14–15. Figures 1 and 2

```bash
MPLCONFIGDIR=/private/tmp/pangenomefm-mpl \
  /opt/anaconda3/bin/python scripts/make_next_stage_figures.py
```

## 16. Figure 3

Source tables are ready; the combined next-stage layout is pending. Existing verified component plots can be regenerated by `scripts/server/analyze_downloaded_server_results.py` after reviewing its required input arguments.

## 17. Figure 4

The complexity-definition panel is available now:

```bash
MPLCONFIGDIR=/private/tmp/pangenomefm-mpl \
  /opt/anaconda3/bin/python scripts/make_complexity_pilot_figure.py
```

Performance panels remain blocked by the comparator task mismatch.

## 18. Figure 5

Blocked until copied SV/cCRE aggregate outputs pass seed/fold/context and checksum audits. Do not plot terminal-ledger metrics.

## 19. Supplementary figures

Existing paper and review figures can be regenerated with:

```bash
MPLCONFIGDIR=/private/tmp/pangenomefm-mpl \
  /opt/anaconda3/bin/python scripts/make_paper_figures.py
MPLCONFIGDIR=/private/tmp/pangenomefm-mpl \
  /opt/anaconda3/bin/python scripts/make_review_figures.py
```

## 20. Full benchmark job matrix

Do not generate or execute the full matrix yet. The gate is an approved exact/approximate protocol plus a completed single-chromosome runtime/memory pilot. Once approved, clone the existing manifest-driven server queue pattern rather than launching ad hoc jobs.
