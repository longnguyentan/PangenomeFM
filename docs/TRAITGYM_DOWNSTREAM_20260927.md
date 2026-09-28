# TraitGym downstream experiments — team report

Verified 27 September 2026. Code, data quality and measured performance; no manuscript edits.

## Status checklist

- [x] Pin both official matched datasets and verify every reference allele against the manuscript HPRC R2 graph.
- [x] Verify all 30 original frozen topology caches and C/K/S/H coverage; retain all 14,780 original variants.
- [x] Complete 60 locus-prior runs (two datasets × five chromosome folds × three seeds × two contexts): 540 logistic fits, all converged; maximum 542 iterations.
- [x] Replay all saved predictions on server and laptop; all 540 per-run rows, 288 summaries, 720 paired rows and 48 contrasts agree exactly.
- [x] Audit whether segment sampling retains the variant itself.
- [x] Verify pinned author NT-2.5B allele-likelihood score files and their original row-order contract.
- [x] Implement and pass the two-task allele-score smoke test: 16 fits, all converged.
- [x] Complete and replay the separate 60-run allele-score sensitivity: 480 converged fits; retain the failed improvement.

## Biological question and protocol

Can an unchanged frozen pangenome segment representation help identify supplied causal regulatory variants among the original matched controls? This is a locus-prior adaptation. The graph embedding does not encode which alternate allele was supplied. The original NT-v2 50M baseline also describes the reference segment, not a variant effect.

Complex traits: 11,400 variants (1,140 positives); Mendelian traits: 3,380 variants (338 positives; 3,354 distinct positions). Each original group has one positive and nine controls on one chromosome. All groups, alleles and chromosome-X examples are retained. No biological annotation, PIP, OMIM, MAF, LD, consequence, TSS distance or match-group ID enters a feature matrix.

The five manuscript test/validation chromosome folds and three seeds are reused. Both frozen encoder checkpoints and downstream probes respect the held-out chromosomes. This differs from the official leave-one-chromosome-out TraitGym protocol; these are not leaderboard numbers. The standard class-balanced logistic probe, validation calibration and validation threshold selection are unchanged, with a uniform 4,000-iteration ceiling. [Official benchmark](https://github.com/songlab-cal/TraitGym).

## Completed locus-prior results

Mean fold AUPRC; random baseline is 0.10. Intervals are the existing hierarchical fold/seed bootstrap.

| Dataset | Context | C+S AP | C+S+T AP | ΔT [95% CI] | ΔT after H [95% CI] |
|---|---|---:|---:|---|---|
| complex_traits | strict | 0.117272 | 0.115248 | -0.002025 [-0.004411, -0.000102] | -0.001735 [-0.003590, +0.000052] |
| complex_traits | 1hop | 0.117272 | 0.116285 | -0.000987 [-0.003610, +0.001404] | -0.000608 [-0.002669, +0.001216] |
| mendelian_traits | strict | 0.120840 | 0.115771 | -0.005069 [-0.017554, +0.007397] | +0.000458 [-0.015901, +0.022022] |
| mendelian_traits | 1hop | 0.120840 | 0.124794 | +0.003954 [-0.016815, +0.021650] | +0.003523 [-0.015026, +0.022994] |

No positive topology interval is established. Strict complex-trait ΔT has a slightly negative pointwise bootstrap interval. The five-fold exact sign-flip test has limited resolution; no topology contrast passes the reported within-metric BH correction. These negative/inconclusive results remain part of the experiment.

All feature sets, chromosome-weighted AP, AUROC, normalized AP, balanced accuracy, F1, precision and recall are in the machine-readable tables. The chromosome-weighted metric here still uses the adapted five-fold probes. It is not an official-protocol reproduction.

## Why the original representation is weak for this task

The source graph segments are long: median 42,671 bp for complex-trait variants and 42,443.5 bp for Mendelian variants. The manuscript NT policy retains at most 6,000 raw bases from the two segment ends, then applies tokenizer truncation.

| Dataset | Tested bases removed before tokenization | Raw-base retention | Positive retention | Negative retention |
|---|---:|---:|---:|---:|
| complex_traits | 8,784/11,400 | 22.95% | 23.68% | 22.87% |
| mendelian_traits | 2,442/3,380 | 27.75% | 50.89% | 25.18% |

These are exact raw-sampling counts, not a test of causation and not proof that every retained base reaches NT after tokenization. They show why a reference-segment average should not be treated as a competitive allele-effect representation. Differential retention by Mendelian label is also retained in the audit.

Separately, 1,136 complex-trait variants and 155 Mendelian variants occur in segments containing both labels. All such variants share the same static segment inputs. They are not excluded to improve performance.

## Prespecified allele-score sensitivity

The separate comparison adds V = signed and absolute masked-allele log-likelihood ratio from the authors’ published NT-2.5B model scores. The two feature transforms are fixed before fitting. Primary comparison: C+S+V+T versus C+S+V; the H-controlled comparison is C+S+V+H+T versus C+S+V+H. C+S is retained to quantify the contribution of V on exactly the same examples.

This follow-up was motivated by the observed original results and input audit, so it is exploratory. Neither PangenomeFM nor either NT model is trained on these labels. Only the existing logistic probe is fitted. There is no score-directed subset selection or alteration of original C/K/S/T meanings.

The public score files are pinned to dataset revision `1fde19555fe8c0a55b1382bdf4c6f7082209f566` and verified against the published LFS SHA-256. The original code at `4d80fe889415ffc02d45c7cb5446a326288819bd` reads the corresponding `test.parquet`, predicts in order and writes a positional score column. We verify the original row/index contract, then join prepared examples by variant identity. The scores themselves have no genomic ID; underlying model-weight revision/runtime are not recorded by the authors. This is reuse of published allele scores, not locally rerunning NT-2.5B weights or reproducing its full embedding benchmark.

## Completed allele-score sensitivity

All 60 runs and 480 fits completed; every fit converges (maximum 560 iterations). The four summary tables replay exactly from predictions on the laptop. In addition, all 120 C+S/C+T comparisons have exactly identical original and follow-up predictions, thresholds, examples and labels.

| Dataset | Context | C+S+V AP | C+S+V+T AP | ΔT [95% CI] | ΔV beyond C+S [95% CI] |
|---|---|---:|---:|---|---|
| complex_traits | strict | 0.117765 | 0.115593 | -0.002173 [-0.004831, +0.000003] | +0.000493 [-0.000429, +0.001362] |
| complex_traits | 1hop | 0.117765 | 0.116712 | -0.001054 [-0.004027, +0.001481] | +0.000493 [-0.000429, +0.001362] |
| mendelian_traits | strict | 0.102110 | 0.101008 | -0.001102 [-0.012369, +0.011257] | -0.018730 [-0.028974, -0.008670] |
| mendelian_traits | 1hop | 0.102110 | 0.107100 | +0.004990 [-0.010788, +0.021374] | -0.018730 [-0.028974, -0.008670] |

**This attempted improvement does not rescue the full representation.** V alone attains mean Mendelian AP 0.160370, but C+S+V is 0.102110 versus 0.120840 for C+S. Adding coordinates alone to V also reduces AP to 0.099046. Thus an informative allele score can be harmed by combination with the current locus features/probe. This is measured behavior, not an inferred cache or optimizer failure: source hashes, feature alignment, invariant baselines and convergence all pass.

Complex-trait ΔV is +0.000493 with an interval crossing zero. No topology contrast, including after H, has a positive 95% interval in this extension. Do not present the larger model as a successful improvement to PangenomeFM or select the favorable Mendelian one-hop point estimate alone.

A future probe-capacity/regularization study must use train/validation selection and retain an untouched evaluation endpoint. This result does not identify whether regularization, chromosome distribution shift, or input information is the dominant cause. The primary experiment remains the fixed manuscript-style probe.

[All allele-score results](../results/foundation_evidence_20260927/traitgym_allele_full_analysis/per_run.csv) · [paired intervals](../results/foundation_evidence_20260927/traitgym_allele_full_analysis/contrasts.csv) · [baseline identity audit](../results/foundation_evidence_20260927/traitgym_cross_experiment_baseline_audit.json) · [feature figure](../results/foundation_evidence_20260927/traitgym_allele_full_analysis/traitgym_feature_ap.pdf).

## Reproduce

The saved launch receipts contain every absolute server path and the exact commands. From the pinned server environment, set `PYTHONPATH=src:.` and use fresh output directories.

- Original matrix: `traitgym_full_launch.json` below; its command invokes `python -m tasks.transfer.traitgym`.
- Original report: `python -m tasks.transfer.traitgym_report --root <traitgym_full> --out-dir <fresh-report>`.
- Allele-score sensitivity: `traitgym_allele_driver/launch.json` records the committed driver copies, file hashes, separate config and both smoke/full commands.
- Sequence visibility: `python scripts/server/audit_traitgym_sequence_visibility.py --mapping-dir <traitgym_reference_audit> --full-segments <original-full_segments.csv.gz> --out-dir <fresh-audit>`.

## Files and execution notes

- [Frozen protocol](../configs/traitgym_locus_prior_20260927.json) · [allele-score protocol](../configs/traitgym_allele_score_20260927.json).
- [Completed per-run results](../results/foundation_evidence_20260927/traitgym_full_analysis/per_run.csv) · [paired contrasts](../results/foundation_evidence_20260927/traitgym_full_analysis/contrasts.csv).
- [Feature AP figure](../results/foundation_evidence_20260927/traitgym_full_analysis/traitgym_feature_ap.pdf) · [topology-gain figure](../results/foundation_evidence_20260927/traitgym_full_analysis/traitgym_topology_gains.pdf).
- [Sampling audit](../results/foundation_evidence_20260927/traitgym_sequence_visibility/visibility.csv) · [full original launch](../results/foundation_evidence_20260927/traitgym_full_launch.json).
- [Allele-score launch](../results/foundation_evidence_20260927/traitgym_allele_driver/launch.json).

Resolved smoke issues: merged NT provenance resides in original shard receipts; several chromosomes have no source Mendelian examples; the server lacks optional `tabulate`. The reader now validates shard provenance, replay uses the observed source chromosome support, and Markdown output has no optional dependency. No original examples or scores were changed to clear these engineering checks.

The original native runner was pinned at `48ca40a`; the allele-score driver is `d30521e` with recorded immutable copies. Raw predictions remain on the server and in the ignored local import; compact reproducible tables and figures are versioned.
