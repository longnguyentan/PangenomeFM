# TraitGym probe repair study

## Scope and prospective choices

The previous two frozen TraitGym studies are complete and retained. The published NT-2.5B allele score alone gave Mendelian AUPRC 0.160370, while concatenating it with manuscript C+S gave 0.102110 (C+S alone: 0.120840). This motivates a separate downstream-classifier study. It does not justify tuning the encoder on these test labels or claiming an independent confirmation.

[Machine-readable protocol](../configs/traitgym_probe_study_20260927.json) is committed before the first fit. All 14,780 original variants, matched groups, five chromosome folds, three seeds and both contexts remain unchanged. All encoders stay frozen. No labels or feature definitions change.

- Nine linear input combinations: V, C+S, C+S+T, C+S+H, C+S+H+T, C+S+V, C+S+V+T, C+S+V+H, C+S+V+H+T.
- Regularization C selected by validation AUPRC from 0.0001, 0.001, 0.01, 0.1, 1, 10. Exact ties prefer smaller C. Scaling fits training chromosomes only; no train-plus-validation refit. Every candidate must converge at the uniform 4,000-iteration ceiling.
- Six fixed C=1 arms replay the manuscript classifier. The same six inputs receive a fixed shallow histogram-gradient-boosting comparison: 150 iterations, seven leaves, minimum 20 examples per leaf, learning rate 0.05, L2=1, class-balanced training weights. No automatic random validation split or early stopping. Completing 150 iterations is not convergence.
- Four late-fusion arms mix the selected linear V probability with selected linear CS/CST/CSH/CSHT probabilities. Validation chooses the weight on V from 0, 0.25, 0.5, 0.75, 1; ties prefer the V-only endpoint. This retains both unimproved component predictors as options.
- All 25 outputs are reported: 1,500 evaluations in 60 runs. There are 3,240 linear candidate fits and 360 boosting fits; fixed arms and fusion reuse fitted models.
- Temperature scaling and F1 thresholds use validation chromosomes. Test identities, labels, metrics, calibration and selected parameters are replayed from saved predictions. Selection sweeps and validation predictions are retained.

The study includes all planned topology contrasts within probe families and after H, all six tuned/nonlinear-versus-fixed comparisons, and every fusion-versus-concatenation/V comparison. Existing hierarchical bootstrap intervals and paired fold sign-flip/BH checks are retained. No favorable arm will be relabeled as a prespecified confirmatory test. A probe improvement alone does not show that learned topology weights beat random weights.

## Numerical and regression tests

Synthetic tests establish bitwise fixed-C prediction equivalence to the original evaluator, selection invariance when test labels change, tie-breaking and grid guards, explicit boosting completion semantics, and validation/test replay with tamper detection. Histogram boosting uses one OpenMP thread: the local mixed Torch/sklearn runtime crashed in binning with two; the bounded implementation passes that environment. This changes execution parallelism, not model settings or examples.

## Execution

Use the original arguments and resource paths from [the previous allele-score launch](../results/foundation_evidence_20260927/traitgym_allele_driver/launch.json), replacing its frozen driver with `python -m tasks.transfer.traitgym`, its config with `configs/traitgym_probe_study_20260927.json`, and its output root with a fresh directory. Run a fold-A/seed-42/strict smoke for both datasets, replay it, and require fixed-C prediction equivalence before the full matrix.

```bash
PYTHONPATH=src:. python -m tasks.transfer.traitgym --help
PYTHONPATH=src:. python -m tasks.transfer.traitgym_report \
  --root <completed-study-root> --out-dir <fresh-report-root>
```

Status: **all 60 runs / 1,500 evaluations / 3,600 classifier fits are complete**. All linear candidates converged and all boosting fits completed their fixed 150-iteration budget. Independent local replay matches all four server numerical tables exactly. All 360 fixed-C comparisons reproduce the previous allele-score study bit-for-bit. The smoke has 50 evaluations, all 12 fixed-C real-data comparisons are bitwise identical to the prior experiment, and all four numerical summary tables replay exactly on the laptop. No smoke performance was used to change model choices.

The first real smoke exposed mixed float32/float64 fusion arithmetic (maximum discrepancy about 3e-8). Its outputs are retained but not used for the accepted full run. Commit `ee466cb` promotes native probabilities losslessly to float64 before blending. Another independent replay issue was platform-specific optimizer drift: refitted calibration temperatures differ by up to 1.04e-7, while stored probabilities replay within 2.3e-16 and validation NLL differs by at most 1.2e-16. Replay now validates saved probabilities/thresholds and optimal validation NLL, recording parameter drift rather than requiring an optimizer's final bits to match across library versions. Models and scores are unchanged by this reporting repair.

[Exact launch protocol](../results/foundation_evidence_20260927/traitgym_probe_float64_driver/launch.json) · [Parallel runner and audited merge](../results/foundation_evidence_20260927/traitgym_probe_float64_driver/parallel_run.py) · [Fixed-C equality gate](../results/foundation_evidence_20260927/traitgym_probe_float64_driver/smoke_gate.json).

Native fitting uses commit `ee466cb`; new SV modules were added subsequently without modifying the running TraitGym implementation. Shards retain exact source, runtime, checkpoint and feature identities. Full aggregation refuses incomplete, duplicate or inconsistent shards.


## Final merge repair and verification

The five fitting processes all completed. The initial native merge compared JSON lists of chromosomes with native dataclass tuples and rejected an otherwise exact 30-job union. All values were independently checked before repair. `tasks.transfer.traitgym_shards` now normalizes containers and still requires every field, fold, seed, context and chromosome to match; tests reject changed holdouts, duplicates, differing sources and overwrites. Fitting remains pinned to `ee466cb`; aggregation/replay recovery uses `f2d09c1`. The original failed `parallel_status.json` is preserved; `recovery_status.json` records successful completion. No outputs were selected or discarded by performance.

The native historical `parallel_run.py` is an execution record with that known merge error. For a new sharded execution, use the fixed merge command below followed by the reporter; do not rerun the archived driver into completed directories.

```bash
PYTHONPATH=src:. python -m tasks.transfer.traitgym_shards \
  --shards <completed-fold-shard-parent> --out-dir <fresh-merged-root>
PYTHONPATH=src:. python -m tasks.transfer.traitgym_report \
  --root <fresh-merged-root> --out-dir <fresh-report-root>
python results/foundation_evidence_20260927/traitgym_probe_float64_driver/full_gate.py \
  --base results/foundation_evidence_20260927 --out <fresh-full-regression-audit.json>
```

Local replay verifies validation and test fusion composition, unchanged variant/match-group support, all metrics, every declared selection grid, and calibration. The largest cross-platform temperature-refit drift is 2.14e-7, with validation NLL excess at most 2.23e-16; saved calibrated probabilities/thresholds replay under the strict declared tolerances. This numerical equivalence does not change any performance value.

## Results: classifier changes help some endpoints, topology remains unsupported here

All tables retain the 10% positive prevalence, five chromosome folds and three checkpoint seeds. These are exploratory adaptations on already inspected datasets, not the official TraitGym leaderboard. Strict/one-hop copies of T-free baselines are identical and must not be counted as independent replications.

| T-free arm | Complex-trait AUPRC | Mendelian-trait AUPRC |
|---|---:|---:|
| V_fixed | 0.108112 | 0.160370 |
| V_linear | 0.108462 | 0.159686 |
| V_histgb | 0.106299 | 0.123217 |
| CS_fixed | 0.117272 | 0.120840 |
| CS_linear | 0.117199 | 0.131402 |
| CS_histgb | 0.112509 | 0.191960 |
| CSV_fixed | 0.117765 | 0.102110 |
| CSV_linear | 0.119812 | 0.112921 |
| CSV_histgb | 0.116722 | 0.171056 |
| CSVH_fixed | 0.115414 | 0.106295 |
| CSVH_linear | 0.117082 | 0.109583 |
| CSVH_histgb | 0.113691 | 0.166664 |
| V_CS_fusion | 0.118257 | 0.132091 |
| V_CSH_fusion | 0.116416 | 0.136413 |

The fixed nonlinear C+S+V probe improves Mendelian AUPRC by +0.068946 [0.017423, 0.145067] over its fixed linear counterpart. C+S boosting rises from 0.120840 to 0.191960, but its gain interval is [-0.000316, 0.150165]. In both comparisons four of five fold means improve; fold E worsens. All folds remain included. Complex-trait gains are small or absent, and late fusion does not consistently improve the stronger component. Validation selection is not a guarantee of better test performance.

No predeclared topology AUPRC comparison has a wholly positive interval. In the selected linear C+S comparison:

| Dataset | Context | C+S AP | C+S+T AP | ΔT, 95% CI |
|---|---|---:|---:|---|
| complex_traits | strict | 0.117199 | 0.115293 | -0.001907 [-0.004723, +0.001020] |
| complex_traits | 1hop | 0.117199 | 0.117952 | +0.000753 [-0.002269, +0.004003] |
| mendelian_traits | strict | 0.131402 | 0.123751 | -0.007651 [-0.020335, +0.003413] |
| mendelian_traits | 1hop | 0.131402 | 0.126760 | -0.004642 [-0.019640, +0.006289] |

For Mendelian C+S+V boosting, adding T changes AP by -0.017290 strict and -0.000218 one-hop; both intervals cross zero. T after H is likewise unsupported. Several strict complex-trait contrasts have negative pointwise intervals and remain in the results. No AP comparison passes BH correction on the exact fold sign-flip tests; with five folds their minimum two-sided p-value is 0.0625. Report this limited resolution alongside the bootstrap intervals.

**Decision:** retain the nonlinear probe as an implemented, completed sensitivity, but do not label it a superior graph encoder or select it retrospectively as a confirmatory primary analysis. The results support investigating classifier choice and genuine allele/context representations. They do not support expanding the current T-only regulatory claim or promoting the v2 architecture whose learned-versus-random gate remains mixed.

## Complete outputs

- [All 1,500 per-run evaluations](../results/foundation_evidence_20260927/traitgym_probe_float64_full_analysis/per_run.csv)
- [Every feature/metric mean, SD and CI](../results/foundation_evidence_20260927/traitgym_probe_float64_full_analysis/absolute.csv)
- [All paired comparisons](../results/foundation_evidence_20260927/traitgym_probe_float64_full_analysis/contrasts.csv)
- [Feature-set figure](../results/foundation_evidence_20260927/traitgym_probe_float64_full_analysis/traitgym_feature_ap.pdf)
- [Topology-gain figure](../results/foundation_evidence_20260927/traitgym_probe_float64_full_analysis/traitgym_topology_gains.pdf)
- [All 360 unchanged-baseline comparisons](../results/foundation_evidence_20260927/traitgym_probe_float64_driver/full_gate.json)
- [Recovery commands](../results/foundation_evidence_20260927/traitgym_probe_float64_driver/recovery_launch.json)
- [Successful recovery status](../results/foundation_evidence_20260927/traitgym_probe_float64_driver/recovery_status.json)
- [Independent local replay](../results/foundation_evidence_20260927/traitgym_probe_float64_local_replay/audit.json)
- [Server/local table equality](../results/foundation_evidence_20260927/completed_new_task_replay.json)

Raw validation/test predictions and selection sweeps remain under local/server `results/foundation_evidence_20260927/traitgym_probe_float64_full/`; these large files are excluded from Git. Frozen resource and checkpoint hashes, runtime details and the exact fitting implementation are retained in its `status.json` and per-run audits.
