# Measured genotyping quality: completed frozen downstream experiment

## Completed checklist

- [x] Locate published, sample-excluded genotyping outcomes with genomic coordinates.
- [x] Pin the public source revision and download only the small outcome/coordinate files.
- [x] Implement and test reusable preparation with duplicate, identity, coordinate and quality checks.
- [x] Prepare 14,838 unique sample–locus outcomes and 265 locus summaries.
- [x] Map all 265 loci to the exact manuscript HPRC R2 graph with the existing mapper.
- [x] Check complete C/K/S/H and all 30 frozen T cache coverage: 100% of loci.
- [x] Commit the continuous-target protocol before fitting, including length/H controls.
- [x] Complete all 30 fold/seed/context runs: 900 model/reference evaluations.
- [x] Independently replay all predictions and validation penalty choices locally.
- [x] Verify exact equality of all four server/local summary tables.
- [x] Retain all null/negative results: this experiment does not support a topology benefit.

## What the data measures

COSIGT reports alignment-based quality of predicted haplotypes against assembled truth. Its SV leave-all-out experiment uses 62 HGSVC3 short-read samples against an HPRC Year 1 genotyping panel after removing overlapping individuals. It releases results for 265 SV-containing regions. This provides measured locus-level quality, distinct from individual-variant PanGenie genotype concordance. [Primary methods and release links](https://link.springer.com/article/10.1186/s13059-026-04242-4).

The source commit is `3d84bf8490bbff5ec33cf72706e713d1a41730ed` in [the author repository](https://github.com/davidebolo1993/cosigt_paper/tree/3d84bf8490bbff5ec33cf72706e713d1a41730ed). The table contains observed and best-available haplotype quality, error rates, sample IDs and GRCh38 region IDs. Coordinates join exactly to the supplied BED. The prepared targets are continuous; no binary label threshold is selected from model results.

## Observed QC

| Item | Result |
|---|---:|
| Raw rows | 14,862 |
| Exact duplicate rows | 24 |
| Conflicting sample–locus measurements | 0 |
| Unique measured sample–locus pairs | 14,838 |
| Samples / loci | 62 / 265 |
| Cartesian sample–locus combinations | 16,430 |
| Combinations without a published outcome | 1,592 (9.69%) |
| Observed donors per locus | 28–62 |
| Missing required fields in observed rows | 0 |
| Overlapping target interval pairs | 0 |
| Mapped loci / target segments | 265 / 11,070 |
| Complete C/K/S/H/T locus coverage | 265/265 |

The 24 duplicates are identical data records across CDK20, NUTM2G and SPATA31D4 regions on chromosome 9 (region identities are retained in the raw source). Preparation removes exact duplicate rows before aggregation so they cannot reweight a locus. Conflicting duplicate outcomes cause an error. Targets and model choices were committed before fitting; no target was selected for a favorable association.

Eighteen observed quality ratios slightly exceed one; the largest excess in summed QV is 0.0005. These small source rounding differences are retained and counted, never clipped. A larger violation of the supplied best-match quality causes an error. Summed per-haplotype QVs replay for all measured rows.

Missing combinations are **unmeasured**, not failed genotypes or negatives. The Cartesian count describes support among released samples and loci; it is not an independently verified count of attempted or callable analyses. The source evaluates samples with suitable assembled ground-truth haplotypes. This means a model could estimate quality conditional on that evaluable population, not general callability.

## Prepared targets and scientific limits

`measured_outcomes.parquet` preserves the published columns and adds the QV fraction, mean per-haplotype QV and canonical locus coordinates. `loci.parquet` contains one row per interval, observed-donor count, mean/median QV fraction, mean predicted QV and mean predicted error rate. `unobserved_pairs.parquet` explicitly preserves unsupported sample–locus combinations without labels.

The completed analysis uses loci as units and holds out chromosomes, keeping every donor occurrence of a locus in the same partition. The 14,838 rows do not provide 14,838 independent genomic examples: there are only 265 distinct intervals. Aggregate continuous regression is a cleaner initial endpoint than inventing a binary cutoff. The completed analysis includes training/validation-only regularization selection, matched C/S/H controls and a length baseline.

The publisher's genotyping panel excludes the measured samples, but our unchanged HPRC R2 representation may contain overlapping donors. Thus this does not demonstrate donor-excluded PangenomeFM generalization. PangenomeFM's graph also differs from the publisher's locus graphs. Keep those facts explicit and do not claim improved genotype calling from predicting quality alone.

## Reproduce

Use the pinned files recorded in `cosigt_quality_preparation/qc.json` under `results/foundation_evidence_20260927/`. Each includes its URL, SHA-256 and source commit. Raw downloads live under `data/external/cosigt_readiness_20260927/` and are not added to Git.

```bash
PYTHONPATH=src:. python -m tasks.transfer.genotyping_quality \
  --outcomes <pinned-cosigt-svs-outcomes.tsv.gz> \
  --regions <pinned-svs_refined.hgsvcv3.bed> \
  --source-receipt <download_receipts.json> \
  --out-dir <fresh-preparation-directory>

PYTHONPATH=src:. python -m tasks.entex.mapping \
  --loci <preparation-directory>/loci.parquet \
  --full-segments <original-HPRC-R2>/full_segments.csv.gz \
  --out-dir <fresh-mapping-directory>
```

The graph checksum remains `e0d832a820969403797662f9af267440599898f068ba9df4069b114d669a3347`. Every target overlaps multiple original graph segments; complete-interval feature coverage is required. No partial-locus pooling or exclusion is needed for the audited original caches.

[Preparation QC](../results/foundation_evidence_20260927/cosigt_quality_preparation/qc.json) · [Locus summaries](../results/foundation_evidence_20260927/cosigt_quality_preparation/loci.csv) · [Mapping QC](../results/foundation_evidence_20260927/cosigt_quality_mapping/mapping_qc.json) · [Cache coverage](../results/foundation_evidence_20260927/cosigt_quality_mapping/feature_coverage.json).

## Completed experiment and actual results

Native execution pin: `ce75c6f`. The [protocol](../configs/cosigt_quality_20260927.json) was committed before fitting. Both continuous outcomes use the original five chromosome folds, three frozen-checkpoint seeds and two contexts. StandardScaler is fitted on training chromosomes only. Validation MAE selects ridge alpha from 0.001 through 1,000 (seven values), with stronger alpha breaking exact ties. No validation refit, outcome clipping or feature-based example removal occurs.

All 265 loci are retained. The 15 model/reference arms include unchanged C/K/S/T combinations, the 14-statistic H control, explicit log locus length L, and the training-median reference. There are 840 selected ridge models, 5,880 candidate ridge fits, and 60 training-median evaluations. All predictions and all selected penalties replay; the four local/server summary tables match exactly. The primary outcome is mean QV fraction; mean predicted QV is secondary.

| Primary QV-fraction MAE | Strict | One-hop |
|---|---:|---:|
| Training median | 0.046169 | 0.046169 |
| C+S | 0.053351 | 0.053351 |
| C+S+T | 0.052699 | 0.053824 |
| C+S+H | 0.051984 | 0.051984 |
| C+S+H+T | 0.052211 | 0.053384 |

Lower MAE is better. The training-median predictor has lower average error than every fitted primary-outcome feature arm; this is a descriptive comparison, not an additional prespecified hypothesis test. A small strict T increment over C+S is insufficient evidence when a simple constant predictor remains stronger.

| Context | Primary contrast | MAE reduction with T | 95% CI |
|---|---|---:|---|
| 1hop | T_given_CS | -0.000473 | [-0.001633, +0.000676] |
| 1hop | T_given_CSH | -0.001401 | [-0.004623, +0.000692] |
| 1hop | T_given_CSL | -0.000481 | [-0.001593, +0.000627] |
| 1hop | T_given_CSLH | -0.000032 | [-0.001249, +0.001134] |
| strict | T_given_CS | +0.000651 | [-0.000202, +0.001641] |
| strict | T_given_CSH | -0.000227 | [-0.001796, +0.000896] |
| strict | T_given_CSL | +0.000613 | [-0.000256, +0.001625] |
| strict | T_given_CSLH | +0.001079 | [-0.001508, +0.004154] |

**Every primary and secondary MAE topology interval crosses zero.** This is a completed null/inconclusive transfer result, not improved genotype calling. Only five chromosome groups support inference; checkpoint seeds do not create additional biological samples. The small locus count and conditional outcome population remain material limits.

The first smoke attempt exposed a historical H-sidecar schema without an output hash; the reader now requires the independently pinned cache checksum. The next exposed float32 ridge conditioning warnings. Those failed/diagnostic outputs remain on server. The accepted smoke and full matrix use double precision and fail on ill-conditioned solver warnings. These were numerical/provenance repairs, with unchanged targets and model-selection grid.

## Exact rerun and artifacts

[Command receipt](../results/foundation_evidence_20260927/cosigt_quality_launch.json) and [launcher](../results/foundation_evidence_20260927/cosigt_quality_full.sh) record every original server path. Use a fresh output root; completed directories are never overwritten.

```bash
PYTHONPATH=src:. python -m tasks.transfer.genotyping_probe --help
PYTHONPATH=src:. MPLCONFIGDIR=/tmp/pfm-mpl python -m tasks.transfer.genotyping_report \
  --root results/foundation_evidence_20260927/cosigt_quality_full \
  --loci results/foundation_evidence_20260927/cosigt_quality_preparation/loci.parquet \
  --out-dir <fresh-local-replay-directory>
```

[All per-run metrics](../results/foundation_evidence_20260927/cosigt_quality_full_analysis/per_run.csv) · [All absolute metrics](../results/foundation_evidence_20260927/cosigt_quality_full_analysis/absolute.csv) · [Paired gains/intervals](../results/foundation_evidence_20260927/cosigt_quality_full_analysis/contrasts.csv) · [Figure PDF](../results/foundation_evidence_20260927/cosigt_quality_full_analysis/genotyping_quality_gains.pdf) · [Independent replay audit](../results/foundation_evidence_20260927/cosigt_quality_local_replay/audit.json).

Saved per-locus predictions remain in the complete local/server `cosigt_quality_full/` result tree. They are not bulk-added to Git. The exact resource identities and native command are retained in its `status.json`.


## Completed validation-only fallback sensitivity

After the original primary-outcome models underperformed the constant reference, a separate [protocol](../configs/cosigt_fallback_20260927.json) was committed at `cde3c8e` before the new selection. For every ridge arm, target, fold, seed and context, validation MAE chooses either its original ridge predictions or the training-median predictions. Exact ties choose the median. The procedure never reads test outcomes to choose an arm and does not refit any model. This is an exploratory follow-up on the same previously inspected test loci, not independent confirmation.

All original predictions and penalty choices replay first. The follow-up adds 840 derived evaluations, retaining all 900 original evaluations and every one of the 265 loci. Validation selects the median for 308/840 comparisons overall, including 295/420 primary-outcome comparisons. Both outcomes and every original feature arm are reported. Constant predictions have undefined Spearman correlation, so paired inference for this follow-up uses MAE, RMSE and R²; per-run correlations remain explicit.

| Primary QV-fraction MAE | Original strict | Fallback strict | Original one-hop | Fallback one-hop |
|---|---:|---:|---:|---:|
| C+S | 0.053351 | 0.048161 | 0.053351 | 0.048161 |
| C+S+T | 0.052699 | 0.047911 | 0.053824 | 0.047193 |
| C+S+H | 0.051984 | 0.047492 | 0.051984 | 0.047492 |
| C+S+H+T | 0.052211 | 0.047346 | 0.053384 | 0.047714 |
| Training median | 0.046169 | 0.046169 | 0.046169 | 0.046169 |

The fallback reduces observed mean error, but the paired C+S+T reductions have intervals crossing zero: strict +0.004788 [-0.005178, 0.015260], one-hop +0.006632 [-0.003417, 0.016864]. The constant reference remains stronger than these principal arms. After fallback, ΔT in MAE reduction is +0.000250 strict [0, 0.000662] and +0.000969 one-hop [-0.001362, 0.004661]; neither establishes a topology effect. All retained fold-test/BH results are supplied. No superiority or better genotype-calling claim follows.

```bash
PYTHONPATH=src:. python -m tasks.transfer.genotyping_fallback \
  --config configs/cosigt_fallback_20260927.json \
  --root results/foundation_evidence_20260927/cosigt_quality_full \
  --loci results/foundation_evidence_20260927/cosigt_quality_preparation/loci.parquet \
  --out-dir <fresh-fallback-output>
```

[Complete fallback results](../results/foundation_evidence_20260927/cosigt_fallback_full/absolute.csv) · [All paired comparisons](../results/foundation_evidence_20260927/cosigt_fallback_full/contrasts.csv) · [Every validation decision](../results/foundation_evidence_20260927/cosigt_fallback_full/selection.csv) · [Figure](../results/foundation_evidence_20260927/cosigt_fallback_full/genotyping_fallback.pdf) · [Source/replay audit](../results/foundation_evidence_20260927/cosigt_fallback_full/audit.json).
