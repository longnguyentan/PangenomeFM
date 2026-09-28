# PangenomeFM: evidence review, 28 September

Takeaway: SV-type analysis is complete; masked-feature learning passes development controls and awaits chromosome replication.

The current experiments test whether frozen graph representations add useful information to sequence and coordinate features. The natural-frequency structural-variant study completed all 30 runs, while the new sequence-conditioned encoder completed its three-seed development comparison. Topology improves macro SV-type AUPRC beyond the measured length and graph-statistic controls, and the new encoder beats its matched random version on both development tasks at every seed. These findings support further evaluation, but the inversion-specific adjusted gains remain uncertain and development performance is not an independent chromosome result. The original null EN-TEx, TraitGym and genotyping-quality contrasts remain part of the evidence. The next fixed chromosome experiment is launched and waiting for the separate NT cache job to release the GPUs.

## Scientific context and question

The original frozen topology representation T and the new sequence-conditioned
representation E are distinct. C denotes the manuscript coordinate features,
S the unchanged frozen NT-v2 50M representation, H the 14 graph statistics,
and L measured SV event log length. Biological labels train only downstream
logistic probes. The exact existing HPRC R2 graph and chromosome folds remain
unchanged; no replacement graph or biological fine-tuning is used.

## Lifecycle and endpoints

| Study | Completed execution | Endpoint / population |
|---|---|---|
| Natural-frequency SV type | 30 runs / 1,170 feature-class evaluations | One-versus-rest AP; 110,623 INS, 63,346 DEL, 298 INV, all retained |
| Masked-feature development | 12 pretraining runs / 24 probes / 96 evaluations | Fold-A biological validation; four arms, three seeds, one-hop |
| Fixed chromosome replication | Launched, resource-waiting | 120 probes / 480 evaluations planned; no chromosome result yet |
| Whole-graph NT completion | Running on four GPUs | 271,760 missing segments; infrastructure, not a biological result |

These are native repository/tmux experiments, not registered Workbench runs.
The completed metrics were independently replayed from saved predictions; native
fitting, frozen inputs and matched sample identities were checked before review.

## Key findings

| Comparison | Strict | One-hop |
|---|---:|---:|
| SV-type macro AP, C+S | 0.406955 | 0.406955 |
| SV-type macro AP, C+S+T | 0.474012 | 0.458608 |
| ΔT given C+S+L+H | +0.047664 | +0.014988 |
| Pointwise bootstrap 95% interval | [0.039885, 0.055252] | [0.007719, 0.021421] |

The inversion class has only 298 events. After C+S+L+H, its strict gain is
+0.015190 [-0.004323, +0.036473], and one-hop gain is
-0.001538 [-0.019657, +0.011729]. Event length is a particularly strong inversion
predictor; the positive macro result must not hide this unresolved class.
The macro fold sign-flip p is 0.0625, BH q=0.0741; pointwise bootstrap intervals
do not establish multiplicity-controlled significance.

| Development mean over three seeds | INS/DEL | cCRE |
|---|---:|---:|
| Trained E minus matched random E, after C+S+H | +0.005609 | +0.001836 |
| Full trained E minus coordinate-only trained E, after C+S+H | +0.006145 | +0.002372 |
| E minus same-width junction Q, after C+S+H | +0.001877 | +0.000099 |
| E minus Q, after C+S | +0.006154 | -0.000145 |

Every seed is retained. The first two rows are positive at every seed/task,
passing the predefined development gate. There are no chromosome confidence
intervals for these three initializations. F1 and balanced accuracy are not
uniformly better in every seed. The Q comparison also has different epoch
budgets, so it is not a controlled estimate of the objective alone.

## Interpretation

The SV extension establishes a reproducible conditional classification result
on the ascertained HGSVC3 callset. It is not SV discovery, genotyping accuracy,
unseen-donor validation, or evidence that learned weights beat random weights;
this extension has no random arm. Its larger gain than the matched-cohort study
reflects a different evaluation population, not improved v1 weights.

The new objective gives a clearer learned-versus-random development result than
the failed junction candidate. It adapts standard GraphMAE-style masked-feature
learning to the existing backbone, with paired random and coordinate controls.
Neither masked autoencoding nor sequence concatenation is claimed as novel.
Removing the graph stream also changes capacity, and the small cCRE advantage
over Q warrants restraint. A final architecture or best-model claim is premature.

## Results, QC, provenance and synthesis

- [Natural-SV per-run metrics](../results/foundation_evidence_20260928/sv_type_natural_full_analysis/per_run.csv), [paired intervals](../results/foundation_evidence_20260928/sv_type_natural_full_analysis/contrasts.csv), [figure](../results/foundation_evidence_20260928/sv_type_natural_full_analysis/sv_type_topology_gains.pdf), [QC audit](../results/foundation_evidence_20260928/sv_type_natural_full_analysis/audit.json).
- [Masked-feature metrics](../results/foundation_evidence_20260928/masked_feature_full_analysis/audited_per_run.csv), [paired controls](../results/foundation_evidence_20260928/masked_feature_full_analysis/paired_differences.csv), [figure](../results/foundation_evidence_20260928/masked_feature_full_analysis/masked_feature_controls.pdf), [gate audit](../results/foundation_evidence_20260928/masked_feature_full_analysis/audit.json).
- [All same-width Q comparisons](../results/foundation_evidence_20260928/masked_junction_full_reference/paired_per_seed.csv).
- [Immutable replication protocol](../configs/masked_nt_chromosome_replication_20260928.json), [explicit launch review](../results/foundation_evidence_20260928/masked_chromosome_replication_driver/development_review.json).
- [Execution details and reproduction](EXECUTION_20260928.md), [complete historical checklist and null results](MODEL_AND_DOWNSTREAM_STATUS_20260927.md).

The development audit fingerprinted the driver's status while the report ran;
the driver then changed `running` to `complete`. Its exact report-time bytes
were recovered and verified against the original SHA-256, in
[the receipt snapshot](../results/foundation_evidence_20260928/masked_feature_full_analysis/execution_receipt_at_report.json).
The gate audit and scores are unchanged. Future reporters save immutable input
receipts directly, avoiding this live-status provenance ambiguity.

## Remaining blockers and next action

1. Finish the queued fixed chromosome matrix. Its primary summary uses folds
   A/C/D/E; fold B is separately reported because its test chromosomes were used
   for fold-A development validation. Historical v1 label exposure remains a
   limitation even for the four-fold summary. Save and verify all fitted probes.
2. Finish the frozen whole-graph NT cache and verify every original row remains
   byte-identical. This enables broader-context work but creates no paths or
   verified directed bubbles by itself.
3. DART-Eval still requires an authorized copy of the official processed table;
   anonymous Synapse access returned 403. No task rows or labels were invented.
4. DUP/complex SV classes, per-variant genotyping concordance with an all-callable
   denominator, all-tested QTL universes and verified haplotype correspondence
   remain unresolved input dependencies. The public 1,218-genome release has
   INS/DEL genotype/frequency fields and does not supply these missing labels.
5. Official graph-SSL architecture comparisons, stronger full sequence embeddings
   and new-E biological transfer remain outstanding. No additional assays or
   architecture changes will be chosen from the pending chromosome-test scores.
