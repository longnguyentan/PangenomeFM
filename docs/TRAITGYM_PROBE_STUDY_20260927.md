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

Status: implemented and tested locally; server smoke/full execution pending. This line will be updated from verified output evidence.
