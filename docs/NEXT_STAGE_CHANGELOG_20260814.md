# Next-stage implementation changelog — 2026-08-14

Pre-existing user changes in `docs/RUN_PROF_SUGGESTED_EXPERIMENTS.md`, `paper/psb2026_graphgenomefm_prof_revision.tex`, and `RUN_MODEL_PRIORITIZED_ENRICHMENT_20260814.md` were not edited by this implementation.

## Modified tracked files

- `.gitignore`: narrowly unignore the new audit, paper-support, pilot, complexity and canonical-result artifacts while keeping bulk historical data/results ignored.
- `requirements.txt`: add `PyYAML>=6.0` for versioned benchmark/complexity configs.
- `src/training/pretrain.py`: seed Python/NumPy/PyTorch deterministically; deduplicate exact and reverse-complement-equivalent candidates; reject orientation-equivalent label conflicts; preserve oriented node IDs through tensorization; hide both query traversals during message passing.
- `tests/test_training_recovery.py`: add seeding, reverse-complement masking, duplicate collapse, label-conflict and node-OID regression tests.

## New source/configuration files

- `src/evaluation/comparative_benchmark.py`: stable example IDs, candidate partitions, orientation equivalence, manifest validation and method fairness rows.
- `src/analysis/complexity.py`: deterministic graph-window metrics, robust composite fitting and frozen-definition application.
- `scripts/build_comparative_benchmark_pilot.py`: chr22 example manifest, whole-source candidate audit, DeepGene approximate inventory, per-method fairness audit and equality gate.
- `scripts/extract_graph_complexity.py`: extract 480 context rows, fit core-window thresholds, join locus categories and emit definitions/audit.
- `scripts/build_canonical_primary_results.py`: validate and combine rotating/transfer source tables into TSV/Parquet.
- `scripts/make_next_stage_figures.py`: publication-ready conceptual Figures 1 and 2 in SVG/PDF/PNG.
- `scripts/make_complexity_pilot_figure.py`: frozen-score distribution, representative windows and source TSV.
- `configs/comparative_benchmark/comparators_v1.yaml`: exact DeepGene/PangenomeX identities, pinned audited commits, tasks, I/O and compatibility tiers.
- `configs/complexity_definition_v1.yaml`: four-feature, robust-normalized, equal-weight, core-window tertile definition.
- `data/annotations/complex_regions_v1.tsv`: editable catalogue with intentionally blank, source-unverified TODO intervals.
- `tests/test_comparative_benchmark.py`: stable-ID, partition, orientation and conversion-loss tests.
- `tests/test_complexity.py`: graph-metric determinism and performance-independence tests.

## New audit and command documentation

- `docs/NEXT_STAGE_PHASE0_AUDIT_20260814.md`: twelve-category repository/results audit, comparator resolution, pilot/complexity/figure/manuscript plan and evidence statuses.
- `docs/NEXT_STAGE_COMMANDS.md`: environment, validation, pilot, downstream, figures, aggregation and execution-gate commands.
- `docs/NEXT_STAGE_CHANGELOG_20260814.md`: this file.
- `experiments/comparative_benchmark/pilot/README.md`: pilot scope and passing/blocking criteria.

## Generated primary and pilot outputs

- `results/canonical_primary_results.tsv`
- `results/canonical_primary_results.parquet`
- `results/canonical_primary_results_audit.json`
- `results/comparative_benchmark/pilot/benchmark_manifest.tsv`
- `results/comparative_benchmark/pilot/benchmark_manifest.parquet`
- `results/comparative_benchmark/pilot/benchmark_audit.tsv`
- `results/comparative_benchmark/pilot/benchmark_audit.json`
- `results/comparative_benchmark/pilot/conversion_failures.tsv`
- `results/comparative_benchmark/pilot/deepgene_approximate_adapter_manifest.tsv`
- `results/comparative_benchmark/pilot/pangenomefm_examples.tsv`
- `results/comparative_benchmark/pilot/pilot_method_status.tsv`
- `results/comparative_benchmark/pilot/source_slice_manifest.csv`
- `results/comparative_benchmark/pilot/source_benchmark_candidate_audit.tsv`
- `results/comparative_benchmark/pilot/pangenomefm_smoke_summary.json`
- `results/comparative_benchmark/pilot/run_manifest.json`

Local raw smoke outputs/checkpoints were written under `results/comparative_benchmark/pilot/pangenomefm_smoke/` and intentionally remain ignored. `run_003` is the post-fix deterministic smoke; earlier runs exposed the missing-PyTorch-seed defect and are not evidence.

## Generated complexity outputs

- `results/complexity/graph_window_complexity_v1/complexity_features.tsv`
- `results/complexity/graph_window_complexity_v1/complexity_features.parquet`
- `results/complexity/graph_window_complexity_v1/complexity_feature_definitions.tsv`
- `results/complexity/graph_window_complexity_v1/complexity_thresholds.json`
- `results/complexity/graph_window_complexity_v1/complexity_audit.json`
- `results/complexity/graph_window_complexity_v1/representative_complexity_windows.tsv`

## New paper/supplementary support

- `paper/next_stage/README.md`: Results 1–6 readiness and manuscript guardrails/placeholders.
- `paper/next_stage/results_manifest.json`: panel-level inputs/scripts/configs/outputs/status.
- `paper/figures/next_stage/figure1_three_paradigms.{svg,pdf,png}`
- `paper/figures/next_stage/figure2_pretraining_and_reuse.{svg,pdf,png}`
- `paper/figures/next_stage/complexity_framework_draft.{svg,pdf,png}`
- `supplementary/README.md`
- `supplementary/figures/README.md`
- `supplementary/tables/README.md`
- `supplementary/text/README.md`

## Verification

- Full suite: 165 passed; 11 pre-existing single-class metric warnings.
- New/modified source: compileall and Ruff checks pass (Ruff scoped to new files; the legacy trainer has unrelated existing lint findings).
- Complexity: 480/480 context rows processed; graph-integrity audit passes; 240 core windows freeze to 81 low, 79 medium and 80 high.
- Pilot: 1,438 source rows become 1,311 unique chr22 relations after 127 duplicate removals; no chr22 label disagreements or missing endpoints/coordinates.
- Whole source: 69,254 rows, 3,311 exact repeats, 3,368 reverse-equivalent repeats and 28 conflicting labels across 12 slices; full fold execution is blocked pending candidate regeneration.
- Determinism: two identical post-fix smoke commands produced identical 113-row prediction tables.
- PDF QA: Figure 1 and Figure 2 PDFs are one page each and were rendered with Poppler for visual inspection; no clipping was found.

