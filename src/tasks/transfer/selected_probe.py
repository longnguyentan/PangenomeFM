"""Predeclared validation-selected probes; biological encoders remain frozen."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from evaluation.calibration import apply_temperature, fit_temperature
from scripts.server.run_ccre_frozen_probe_fold import binary_metrics
from tasks.ccre.aligned_baselines import _fit_model
from tasks.ccre.binary import _choose_threshold


def select_logistic(x_train, y_train, x_val, y_val, seed, penalties, max_iter):
    """Fit every declared C on training data; select by validation AP only."""
    if not penalties or 1.0 not in penalties or any(c <= 0 or not np.isfinite(c) for c in penalties):
        raise ValueError("Require positive finite C grid containing the manuscript C=1")
    if len(set(penalties)) != len(penalties):
        raise ValueError("Duplicate regularization candidates")
    models, sweep = {}, []
    for c in sorted(penalties):
        model = _fit_model("logistic", seed)
        model.set_params(logisticregression__C=c, logisticregression__max_iter=max_iter,
                         logisticregression__n_jobs=None)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x_train, y_train)
        score = model.predict_proba(x_val)[:, 1]
        sweep.append(dict(parameter="C", value=c, validation_auprc=float(average_precision_score(y_val, score))))
        models[c] = model
    selected = min(sweep, key=lambda row: (-row["validation_auprc"], row["value"]))
    return models, selected, sweep


def choose_mixture(y_val, variant_probability, locus_probability, weights):
    if (not weights or len(set(weights)) != len(weights) or not {0., 1.} <= set(weights)
            or any(not np.isfinite(w) or w < 0 or w > 1 for w in weights)):
        raise ValueError("Convex mixture grid must contain both unmodified endpoints")
    sweep = [dict(parameter="weight_on_V", value=w, validation_auprc=float(average_precision_score(
        y_val, w*variant_probability+(1-w)*locus_probability))) for w in weights]
    selected = min(sweep, key=lambda row: (-row["validation_auprc"], -row["value"]))
    return selected, sweep


def evaluate_selected(*, segids, chromosomes, labels, features, test_chrs, val_chrs, seed, plan):
    """Native metrics/calibration, with a separately declared probe study."""
    labels, chromosomes = np.asarray(labels), np.asarray(chromosomes)
    test, val = np.isin(chromosomes, list(test_chrs)), np.isin(chromosomes, list(val_chrs))
    train = ~(test | val)
    if test_chrs & val_chrs or any(len(np.unique(labels[m])) != 2 for m in [train, val, test]):
        raise ValueError("Require disjoint chromosome partitions with both classes")
    policy = plan["selected_probe"]
    cached, selection = {}, []
    # Each input model is fit once and can serve its fixed-C, tuned-C and fusion arms.
    for name in policy["linear_inputs"]:
        x = features[name]
        if len(x) != len(labels) or not np.isfinite(x).all():
            raise ValueError("Misaligned or nonfinite frozen features")
        models, chosen, sweep = select_logistic(x[train], labels[train], x[val], labels[val], seed,
                                               policy["C_grid"], plan["probe_max_iter"])
        selection.extend(dict(input_feature=name, **row, selected=row["value"] == chosen["value"]) for row in sweep)
        for tag, c in [("fixed", 1.), ("linear", chosen["value"])]:
            model = models[c]
            cached[(name, tag)] = dict(val=model.predict_proba(x[val])[:, 1], test=model.predict_proba(x[test])[:, 1],
                selected_parameter=c, probe_solver="lbfgs", probe_max_iter=plan["probe_max_iter"],
                probe_iterations=int(max(model[-1].n_iter_)), probe_converged=True, probe_numerical_pass=True,
                probe_completion="converged", probe_convergence_messages="")
    for name in policy["histgb_inputs"]:
        x = features[name]
        if len(x) != len(labels) or not np.isfinite(x).all():
            raise ValueError("Misaligned or nonfinite frozen features")
        model = HistGradientBoostingClassifier(**policy["histgb"], random_state=seed, early_stopping=False)
        # Bound OpenMP independently of BLAS; mixed Torch/sklearn runtimes can
        # otherwise crash in histogram binning on the supported macOS runtime.
        with threadpool_limits(limits=1, user_api="openmp"):
            model.fit(x[train], labels[train], sample_weight=compute_sample_weight("balanced", labels[train]))
            val_probability = model.predict_proba(x[val])[:, 1]
            test_probability = model.predict_proba(x[test])[:, 1]
        cached[(name, "histgb")] = dict(val=val_probability, test=test_probability,
            selected_parameter=np.nan, probe_solver="hist_gradient_boosting", probe_max_iter=policy["histgb"]["max_iter"],
            probe_iterations=int(model.n_iter_), probe_converged=np.nan, probe_numerical_pass=True,
            probe_completion="fixed_budget_completed; no convergence criterion", probe_convergence_messages="")
    metrics, chromosome_metrics, predictions, validation = [], [], [], []
    for output, spec in policy["outputs"].items():
        if spec["estimator"] == "fusion":
            v = cached[(spec["variant_input"], "linear")]
            b = cached[(spec["locus_input"], "linear")]
            chosen, sweep = choose_mixture(labels[val], v["val"], b["val"], policy["mixture_weights"])
            selection.extend(dict(input_feature=output, **row, selected=row["value"] == chosen["value"]) for row in sweep)
            weight = chosen["value"]
            result = dict(b, val=weight*v["val"]+(1-weight)*b["val"], test=weight*v["test"]+(1-weight)*b["test"],
                selected_parameter=weight, probe_solver="validation_convex_probability_blend",
                probe_iterations=max(v["probe_iterations"], b["probe_iterations"]),
                probe_completion="blend_of_converged_linear_probes")
        else:
            result = cached[(spec["input"], spec["estimator"])]
        raw_val, raw_test = result["val"], result["test"]
        if not np.isfinite(raw_val).all() or not np.isfinite(raw_test).all():
            raise ValueError("Nonfinite probe prediction")
        temperature = fit_temperature(labels[val], raw_val)
        cal_val, cal_test = apply_temperature(raw_val, temperature), apply_temperature(raw_test, temperature)
        threshold = _choose_threshold(labels[val], cal_val)
        common = {k: v for k, v in result.items() if k not in ["val", "test"]}
        common.update(feature_set=output, feature_access=str(spec), temperature=temperature,
                      validation_f1_threshold=threshold, n_train=int(train.sum()), n_validation=int(val.sum()), n_test=int(test.sum()))
        metrics.append(dict(**common, scope="all_test_chromosomes", chromosome="all",
                            **binary_metrics(labels[test], cal_test, threshold)))
        for chrom in sorted(set(chromosomes[test])):
            mask = chromosomes[test] == chrom
            chromosome_metrics.append(dict(**common, scope="test_chromosome", chromosome=chrom,
                **binary_metrics(labels[test][mask], cal_test[mask], threshold)))
        for mask, raw, calibrated, dest in [(test, raw_test, cal_test, predictions), (val, raw_val, cal_val, validation)]:
            dest.append(pd.DataFrame(dict(segid=np.asarray(segids)[mask], chromosome=chromosomes[mask], y_true=labels[mask],
                feature_set=output, p_raw=raw, p_calibrated=calibrated, threshold=threshold,
                y_pred=(calibrated >= threshold).astype(np.int8))))
    return (pd.DataFrame(metrics), pd.DataFrame(chromosome_metrics), pd.concat(predictions, ignore_index=True),
            pd.DataFrame(selection), pd.concat(validation, ignore_index=True))


def completion_summary(metrics: pd.DataFrame) -> dict:
    """Do not describe fixed-budget boosting as numerical convergence."""
    if "probe_numerical_pass" not in metrics:
        return dict(all_probes_converged=bool(metrics.probe_converged.all()))
    linear = metrics.loc[metrics.probe_solver.ne("hist_gradient_boosting")]
    boosting = metrics.loc[metrics.probe_solver.eq("hist_gradient_boosting")]
    passed = bool(metrics.probe_numerical_pass.all() and linear.probe_converged.eq(True).all()
                  and boosting.probe_iterations.eq(boosting.probe_max_iter).all())
    return dict(all_probes_converged=None if len(boosting) else passed,
                all_linear_probes_converged=bool(linear.probe_converged.eq(True).all()),
                fixed_budget_estimators_present=bool(len(boosting)), all_probes_completed=passed)


def validate_selection(sweep: pd.DataFrame, validation: pd.DataFrame, metrics: pd.DataFrame, plan: dict) -> None:
    """Replay the predeclared selection rule, including all mixture candidates."""
    policy = plan["selected_probe"]
    expected = {(name, "C") for name in policy["linear_inputs"]} | {
        (name, "weight_on_V") for name, spec in policy["outputs"].items() if spec["estimator"] == "fusion"}
    if set(sweep[["input_feature", "parameter"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("Missing validation candidate groups")
    chosen = {}
    for (name, parameter), part in sweep.groupby(["input_feature", "parameter"]):
        grid = policy["C_grid" if parameter == "C" else "mixture_weights"]
        if (part.value.duplicated().any() or set(part.value) != set(grid)
                or not np.isfinite(part.validation_auprc).all() or not part.validation_auprc.between(0, 1).all()):
            raise ValueError("Incomplete or invalid validation candidate grid")
        row = part.sort_values(["validation_auprc", "value"], ascending=[False, parameter == "C"]).iloc[0]
        if not part.selected.eq(part.value.eq(row.value)).all():
            raise ValueError("Selection differs from validation rule")
        chosen[name] = row
    indexed = metrics.set_index("feature_set")
    def probabilities(output):
        return validation.loc[validation.feature_set.eq(output)].sort_values("example_id")
    linear_output = {spec["input"]: name for name, spec in policy["outputs"].items() if spec["estimator"] == "linear"}
    identity = None
    for output, spec in policy["outputs"].items():
        data = probabilities(output)
        current = data[["example_id", "chromosome", "y_true"]].reset_index(drop=True)
        if current.example_id.duplicated().any() or not len(current):
            raise ValueError("Invalid validation identities")
        if identity is not None:
            pd.testing.assert_frame_equal(identity, current)
        identity = current
        score = average_precision_score(data.y_true, data.p_raw)
        estimator = spec["estimator"]
        expected_value = None
        if estimator in ["linear", "fusion"]:
            row = chosen[output if estimator == "fusion" else spec["input"]]
            expected_value = row.value
            if not np.isclose(score, row.validation_auprc, atol=1e-12, rtol=0):
                raise ValueError("Validation prediction AP differs from selected candidate")
        elif estimator == "fixed":
            expected_value = 1.
        if expected_value is not None and indexed.loc[output, "selected_parameter"] != expected_value:
            raise ValueError("Selected parameter differs from validation rule")
        if estimator == "fusion":
            v = probabilities(linear_output[spec["variant_input"]]).p_raw.to_numpy()
            b = probabilities(linear_output[spec["locus_input"]]).p_raw.to_numpy()
            _, replay = choose_mixture(data.y_true, v, b, policy["mixture_weights"])
            recorded = sweep.loc[sweep.input_feature.eq(output)].set_index("value")
            for candidate in replay:
                if not np.isclose(recorded.loc[candidate["value"], "validation_auprc"], candidate["validation_auprc"], atol=1e-12, rtol=0):
                    raise ValueError("Mixture candidate AP differs from validation prediction replay")
            if not np.allclose(data.p_raw, expected_value*v+(1-expected_value)*b, rtol=0, atol=1e-12):
                raise ValueError("Selected mixture predictions do not replay")
