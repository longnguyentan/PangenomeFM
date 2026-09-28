# Measured genotyping quality: a new downstream starting point

## Completed checklist

- [x] Locate published, sample-excluded genotyping outcomes with genomic coordinates.
- [x] Pin the public source revision and download only the small outcome/coordinate files.
- [x] Implement and test reusable preparation with duplicate, identity, coordinate and quality checks.
- [x] Prepare 14,838 unique sample–locus outcomes and 265 locus summaries.
- [x] Map all 265 loci to the exact manuscript HPRC R2 graph with the existing mapper.
- [x] Check complete C/K/S/H and all 30 frozen T cache coverage: 100% of loci.
- [ ] Specify and run a frozen prediction experiment. **No PangenomeFM performance has been measured on these outcomes.**

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

The 24 duplicates are identical data records across CDK20, NUTM2G and SPATA31D4 regions on chromosome 9 (region identities are retained in the raw source). Preparation removes exact duplicate rows before aggregation so they cannot reweight a locus. Conflicting duplicate outcomes cause an error. The target has not been fitted or selected for a favorable association.

Eighteen observed quality ratios slightly exceed one; the largest excess in summed QV is 0.0005. These small source rounding differences are retained and counted, never clipped. A larger violation of the supplied best-match quality causes an error. Summed per-haplotype QVs replay for all measured rows.

Missing combinations are **unmeasured**, not failed genotypes or negatives. The Cartesian count describes support among released samples and loci; it is not an independently verified count of attempted or callable analyses. The source evaluates samples with suitable assembled ground-truth haplotypes. This means a model could estimate quality conditional on that evaluable population, not general callability.

## Prepared targets and scientific limits

`measured_outcomes.parquet` preserves the published columns and adds the QV fraction, mean per-haplotype QV and canonical locus coordinates. `loci.parquet` contains one row per interval, observed-donor count, mean/median QV fraction, mean predicted QV and mean predicted error rate. `unobserved_pairs.parquet` explicitly preserves unsupported sample–locus combinations without labels.

A prospective analysis should use loci as independent units and hold out chromosomes, including every donor occurrence of a locus in the same partition. The 14,838 rows do not provide 14,838 independent genomic examples: there are only 265 distinct intervals. Aggregate continuous regression is a cleaner initial endpoint than inventing a binary cutoff. Training/validation-only regularization selection, matched C/S/H controls, and a length baseline are needed before interpreting any gain.

The publisher's genotyping panel excludes the measured samples, but our unchanged HPRC R2 representation may contain overlapping donors. Thus this would not demonstrate donor-excluded PangenomeFM generalization. PangenomeFM's graph also differs from the publisher's locus graphs. Keep those facts explicit and do not claim improved genotype calling from predicting quality alone.

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
