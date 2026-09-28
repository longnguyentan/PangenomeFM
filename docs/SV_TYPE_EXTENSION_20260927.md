# SV type extension: completed matched frozen experiment

## Data and coordinate audit

The two existing HGSVC3 GRCh38 annotation tables contain 176,231 INS/DEL events and 300 inversions. These are SV-Pop BED+6 tables. The parser validates integer half-open intervals, class-specific interval lengths, unique event IDs, and type/length agreement. The publisher documents this table format and versioned IDs in [SV-Pop](https://github.com/EichlerLab/svpop) and its [variant utilities](https://github.com/EichlerLab/svpop/blob/main/svpoplib/variant.py).

A strict initial ID-position check exposed seven `.1` ID suffixes and 181 retained INS/DEL IDs whose encoded position equals BED POS rather than POS+1. The full independent audit confirms all 176,231 INS/DEL events against the existing padded-allele VCF: coordinates, class, length and allele padding agree. The parser retains these IDs, uses the coordinate columns, and counts discrepancies; it never changes a coordinate to fit an ID. This audit passed before fitting; the original symbolic IDs remain unchanged.

| Population | INS | DEL | INV | Total |
|---|---:|---:|---:|---:|
| Original annotation rows | 111,803 | 64,428 | 300 | 176,531 |
| Original chromosome-fold universe | 110,623 | 63,346 | 298 | 174,267 |
| Fixed chromosome/length-bin common support | 223 | 223 | 223 | 669 |

The 2,264 non-primary-contig rows remain in `excluded_events.parquet` with their reason. No new graph release or labels are downloaded. All 298 primary-chromosome inversions remain in `natural_events.parquet`; 75 do not enter the matched sensitivity because their chromosome/length bin lacks equal support across the three classes. No bins or inclusion criteria depend on model performance.

The natural population has 147,429 distinct anchors, including 2,588 anchors with more than one class. The matched population has 667 anchors, with no conflicting class labels at identical anchors. Distinct event IDs are preserved even when their input vectors will be identical. These limitations must not be disguised as allele-aware modeling.

## Predeclared task

[Protocol](../configs/sv_type_matched_20260927.json) records exact prepared-table hashes before fitting. The first completed experiment targets the **matched common-support population**, with prevalence 1/3 for each one-versus-rest task. It is not a result at natural prevalence, and it is not an evaluation of every inversion.

Each event uses the graph segment containing its first affected base, consistently across INS, DEL and INV. This avoids directly revealing class-specific endpoint span conventions through feature assembly. Existing C/S/T definitions and graph/checkpoint versions remain unchanged. Explicit log event length L is reported as its own control; it is never silently added to C. This differs from the historical two-breakpoint INS/DEL task and must not replace its reported numbers.

The experiment uses the original chromosome folds, three seeds and both contexts. Three separate native balanced logistic probes (C=1, uniform 4,000-iteration ceiling) predict DEL, INS and INV versus the other two types. Encoders remain frozen; scaling uses training chromosomes, calibration and thresholds use validation chromosomes. The original H control is included. All 13 declared feature arms are reported, producing 1,170 class/feature evaluations in 30 runs.

Primary reports are per-type AUPRC and macro AUPRC, followed by AUROC, normalized AP and native binary metrics. Macro binary metrics are averages over the three one-versus-rest probes, not joint three-class accuracy. Four paired topology contrasts are evaluated with the existing hierarchical bootstrap, exact fold sign-flip tests and BH correction within metric across all classes/macro, contexts and contrasts.

## Validity and limits

- No relabeling inversions as deletions; the historical binary probe remains unchanged.
- No random row split, donor-based duplication across partitions or nearest-segment fallback.
- No source IDs, donor counts, variant type metadata or endpoint distances in learned features.
- Complete mapped feature coverage required for every declared event.
- Save every prediction, test identity and convergence record; replay metrics independently.
- Known-event classification is distinct from SV discovery, genotyping, pathogenicity or breakpoint detection.
- No DUP or complex-SV labels are supplied by these tables. Those classes remain a separate unresolved extension.
- The graph-donor overlap and historical v1 shortcut caveats remain. This experiment cannot establish superiority over random encoders.

## Reproduce

```bash
PYTHONPATH=src:. python -m tasks.transfer.sv_types \
  --annotations <original-insdel-annotation.tsv.gz> <original-inversion-annotation.tsv.gz> \
  --out-dir <fresh-preparation-dir>

PYTHONPATH=src:. python -m tasks.entex.mapping \
  --loci <preparation-dir>/length_matched_loci.parquet \
  --full-segments <exact-manuscript-graph>/full_segments.csv.gz \
  --out-dir <fresh-mapping-dir>

PYTHONPATH=src:. python -m tasks.transfer.sv_type_probe --help
PYTHONPATH=src:. python -m tasks.transfer.sv_type_report \
  --root <completed-probe-root> \
  --examples <preparation-dir>/length_matched_events.parquet \
  --out-dir <fresh-report-dir>
```

[Preparation QC](../results/foundation_evidence_20260927/sv_type_preparation/qc.json) · [Split support](../results/foundation_evidence_20260927/sv_type_preparation/fold_support.csv) · [All matching strata](../results/foundation_evidence_20260927/sv_type_preparation/matching_strata.csv).

Execution status: **all 30 runs and 1,170 class/feature evaluations complete** at native commit `98a1935`. All probes converge. All 667 anchors map and all 669 declared events have complete frozen features. Independent local prediction replay passes, and all four server/local numerical result tables match exactly.

## Actual results

Class prevalence is 1/3 in every split by construction; these AUPRC values must not be compared numerically with the natural-prevalence historical binary task.

| Class | C+S | C+S+T strict | Strict ΔT, 95% CI | C+S+T one-hop | One-hop ΔT, 95% CI |
|---|---:|---:|---|---:|---|
| DEL | 0.397108 | 0.425800 | +0.028692 [0.000098, 0.056152] | 0.443225 | +0.046117 [0.013368, 0.077370] |
| INS | 0.448878 | 0.460745 | +0.011867 [-0.009647, 0.033599] | 0.464845 | +0.015967 [-0.012622, 0.049527] |
| INV | 0.396448 | 0.426821 | +0.030374 [0.011325, 0.054520] | 0.414511 | +0.018063 [0.005267, 0.031607] |
| Macro | 0.414145 | 0.437789 | +0.023644 [0.015802, 0.031968] | 0.440860 | +0.026716 [0.004101, 0.050809] |

The length-adjusted macro comparison retains gains of +0.023226 strict and +0.027166 one-hop with positive pointwise bootstrap intervals. Fixed coarse matching does not make exact length identical: length alone still achieves macro AUPRC 0.367129. This is why the explicit L control is retained.

**Handcrafted graph statistics are stronger than T alone in this task.** C+S+H reaches macro AUPRC 0.473760. Adding T after H gives +0.002470 strict (CI [-0.005189, 0.011544]) and -0.001892 one-hop (CI [-0.011793, 0.008245]); all class-level T-after-H intervals also cross zero. The result supports graph information beyond the C+S baseline in this population, but does not establish superior learned topology representations.

No AP contrast survives the supplied BH correction on exact fold sign-flip p-values. The strict macro primary contrast has p=0.0625 and q=0.285714; one-hop p=0.1250 and q=0.363636. Five folds limit the resolution of these tests. Positive pointwise bootstrap intervals should be described alongside these results, not as unqualified confirmatory significance.

## Auditable outputs

- [Full-source VCF coordinate/allele audit](../results/foundation_evidence_20260927/sv_type_driver/coordinate_audit.json)
- [Exact graph mapping](../results/foundation_evidence_20260927/sv_type_mapping/mapping_qc.json)
- [Native command receipt](../results/foundation_evidence_20260927/sv_type_driver/launch.json)
- [All per-run metrics](../results/foundation_evidence_20260927/sv_type_full_analysis/per_run.csv)
- [All absolute metrics](../results/foundation_evidence_20260927/sv_type_full_analysis/absolute.csv)
- [All paired contrasts, intervals and multiplicity checks](../results/foundation_evidence_20260927/sv_type_full_analysis/contrasts.csv)
- [Figure PDF](../results/foundation_evidence_20260927/sv_type_full_analysis/sv_type_topology_gains.pdf)
- [Independent local replay audit](../results/foundation_evidence_20260927/sv_type_local_replay/audit.json)

Saved predictions remain in local/server `results/foundation_evidence_20260927/sv_type_full/`. The natural 174,267-event population is prepared but **not fitted**; DUP and complex-SV classes remain unresolved. No natural-population, all-inversion or best-model claim is made.
