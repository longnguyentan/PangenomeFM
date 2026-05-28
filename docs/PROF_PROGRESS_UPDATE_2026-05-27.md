# Progress Update for Prof

Date: 2026-05-27  
Project: GraphGenome-FM / PangenomeFM PSB 2026

## Revised Paper Story

The paper is now centered on a cleaner HPRC + HGSVC story:

- GraphGenome-FM learns graph-native pangenome topology from HPRC slices using
  self-supervised link prediction.
- The model generalizes to held-out HPRC chromosomes, including chr1, chr8,
  chr19, and chrY.
- A distance-matched hard-negative rerun preserves high AUROC, which reduces
  the risk that the result is only due to easy random negatives.
- The HPRC-trained checkpoint transfers to an independent HGSVC/HGSVC3 graph
  across CHM13 chr19, chr21, chr22, and chrY.
- cCRE prediction is framed as a downstream stress test and next improvement
  area, not as a solved result.

## Updated Manuscript

Updated `paper/psb2026_graphgenomefm_prof_revision.tex`:

- Shortened and generalized the abstract.
- Organized Results into HPRC held-out pretraining, robustness/ablation,
  external HGSVC transfer, comparative cCRE prediction, HGSVC graph
  enhancement, haplotype-specific functional annotation, and draft figures.
- Added per-held-out-chromosome HPRC distance-matched AUROC table.
- Added HGSVC pooled and per-target transfer tables.
- Added explicit cCRE caution: current GAT and logistic evaluations are not
  aligned.
- Added HGSVC imputation/graph enhancement and haplotype-specific functional
  annotation as application/future-work sections.

## New Audit Result

I added a cCRE window-coverage audit:

- Script: `scripts/check_ccre_window_coverage.py`
- Output: `results/hprc/ccre_window_coverage/coverage_audit.md`
- Key result: current GAT benchmark windows cover 1,330 unique labeled test
  nodes across chr8/chr19/chr22, while the all-node logistic test set contains
  31,560 labeled nodes.
- Coverage: 4.21% of the logistic test universe.

Interpretation: the cCRE GAT comparison should not be overclaimed. Next we
should either build cCRE-specific windows with broad labeled-node coverage or
restrict all baselines to the same window-covered node universe.

## cCRE Metadata Check

The local ENCODE cCRE source files are:

- `data/encode/GRCh38-human-cCREs.bed`
- `data/encode/GRCh38-human-cCREs-chr22.bed`

The full BED has six fields per row: chromosome, start, end, cCRE ID,
ENCODE element ID, and cCRE class. It does not include local signal,
confidence, DNase, CTCF, H3K4me3, H3K27ac, or Z-score columns. For the current
draft, reduced cCRE categories are the practical next step unless we add richer
SCREEN/ENCODE metadata.

Proposed reduced groups:

- background
- promoter-like: PLS, CA-H3K4me3
- enhancer-like: pELS, dELS
- CTCF/TF-associated: CA-CTCF, CA-TF, TF
- open chromatin: CA

## Draft Figures and Tables

Generated draft figure assets:

- `results/figures/hprc_link_prediction_auroc.png`
- `results/figures/hgsvc_external_transfer_per_chromosome.png`
- `results/figures/ccre_binary_current_comparison.png`
- `results/figures/ccre_class_distribution.png`

Draft manuscript tables now cover:

- HPRC held-out random vs distance-matched AUROC.
- HPRC hard-negative per-heldout-chromosome AUROC.
- HGSVC pooled external transfer.
- HGSVC per-target external transfer.
- Current cCRE comparison.
- cCRE benchmark-window coverage audit.
- Planned HGSVC imputation pilot.

## Next Experiments

Immediate next experiments:

- Fair binary cCRE GAT: align GAT and logistic node universes.
- Reduced-category cCRE task using the five groups above.
- Linear-reference/no-graph baseline.
- Coordinate-only and topology-only baselines.
- DeepGene or DeepGene-style linearized graph comparison if feasible.
- HGSVC imputation/enhancement pilot: score candidate missing links and
  compare against a newer graph build if Tomoya can generate one.

## Validation

Local checks completed:

- `python scripts/summarize_experiment_status.py`
- `PYTHONPATH=src pytest -q`
- `bash -n scripts/run_next_wave_experiments.sh`
- `PYTHONPATH=src python -m graphgenomefm --help`
- `python -m py_compile scripts/check_ccre_window_coverage.py scripts/make_draft_figures.py scripts/summarize_experiment_status.py`

Note: plain `python -m graphgenomefm --help` does not work unless the package is
installed or `PYTHONPATH=src` is set.
