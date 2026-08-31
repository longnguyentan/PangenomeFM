# Manuscript gap-fill source tables

This directory contains the compact, static source tables required by
`scripts/server/run_manuscript_gap_experiments.sh` to regenerate the manuscript
figures and capacity comparison. They are tracked here because generated
Overleaf builds under `output/` are intentionally excluded from version
control and are not guaranteed to exist on the compute server.

Files:

- `figure3_reconstruction.csv`: within-resource reconstruction summaries;
- `figure3_transfer.csv`: cross-resource and cross-release transfer summaries;
- `figure4_ablation_runs.csv`: component-ablation run metrics;
- `figure4_baselines.csv`: shortcut-baseline summaries;
- `figure4_capacity_runs.csv`: original diagnostic capacity-run metrics;
- `figure4_complexity.csv`: context effects by native-region complexity;
- `figure5_absolute.csv`: downstream absolute-performance summaries;
- `figure5_contributions.csv`: paired information-source contributions;
- `figure5_sv_strata.csv`: structural-variant stratified contributions.

`source_manifest.json` records the upstream analysis files from which these
compact tables were derived.

`SHA256SUMS` records the exact imported bytes. The server workflow verifies the
manifest before starting any long-running analysis.
