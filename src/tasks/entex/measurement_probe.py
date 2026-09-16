"""The manuscript logistic probe with count-equivalent repeated-measurement fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.calibration import apply_temperature, fit_temperature
from scripts.server.run_ccre_frozen_probe_fold import FEATURE_ACCESS, binary_metrics
from tasks.ccre.aligned_baselines import _fit_model
from tasks.ccre.binary import _choose_threshold


def fit_count_equivalent(matrix, row_positions, labels, seed):
    """Collapse identical locus/label copies without changing the weighted objective.

    Scaler moments use occurrence counts. Explicit class weights use original row
    class totals, not compressed class totals. Logistic C/solver/tolerance remain
    those of the manuscript. Validation/test remain individual measurements.
    """
    counts = (
        pd.DataFrame({"position": row_positions, "label": labels})
        .value_counts(sort=False)
        .rename("count")
        .reset_index()
    )
    model = _fit_model("logistic", seed)
    totals = np.bincount(labels.astype(int), minlength=2)
    if (totals == 0).any():
        raise ValueError("Both training classes required")
    model.set_params(
        logisticregression__class_weight={
            i: len(labels) / (2 * totals[i]) for i in [0, 1]
        }
    )
    model.fit(
        matrix[counts.position.to_numpy()],
        counts.label.to_numpy(),
        standardscaler__sample_weight=counts["count"].to_numpy(),
        logisticregression__sample_weight=counts["count"].to_numpy(),
    )
    return model


def evaluate_measurements(*, loci, measurements, features, test_chrs, val_chrs, seed):
    if measurements.measurement_id.duplicated().any():
        raise ValueError("Duplicate measurement IDs")
    selected = measurements.loc[measurements.locus_id.isin(loci.locus_id)].copy()
    index = pd.Series(np.arange(len(loci)), index=loci.locus_id)
    selected["position"] = selected.locus_id.map(index)
    selected["example_id"] = selected.position.map(
        pd.Series(loci.example_id.to_numpy())
    )
    positions = selected.position.to_numpy(int)
    labels = selected.label.to_numpy(int)
    chroms = selected.chrom.to_numpy()
    test = np.isin(chroms, list(test_chrs))
    val = np.isin(chroms, list(val_chrs))
    train = ~(test | val)
    if set(test_chrs) & set(val_chrs):
        raise ValueError("Overlapping chromosomes")
    if any(
        not mask.any() or len(np.unique(labels[mask])) != 2
        for mask in [train, val, test]
    ):
        raise ValueError("Every measurement partition must have both classes")
    rows, predictions = [], []
    for name, matrix in features.items():
        model = fit_count_equivalent(matrix, positions[train], labels[train], seed)
        unique_scores = model.predict_proba(matrix)[:, 1]
        raw_val = unique_scores[positions[val]]
        raw_test = unique_scores[positions[test]]
        temperature = fit_temperature(labels[val], raw_val)
        val_scores = apply_temperature(raw_val, temperature)
        scores = apply_temperature(raw_test, temperature)
        threshold = _choose_threshold(labels[val], val_scores)
        rows.append(
            dict(
                feature_set=name,
                feature_access=FEATURE_ACCESS[name],
                temperature=float(temperature),
                validation_f1_threshold=float(threshold),
                n_train=int(train.sum()),
                n_validation=int(val.sum()),
                n_test=int(test.sum()),
                scope="all_test_chromosomes",
                chromosome="all",
                **binary_metrics(labels[test], scores, threshold),
            )
        )
        frame = selected.loc[
            test,
            [
                "measurement_id",
                "locus_id",
                "experiment_accession",
                "donor",
                "tissue",
                "assay",
                "example_id",
            ],
        ].copy()
        # The shared caller joins genomic metadata using example_id.
        frame = frame.drop(columns="locus_id").rename(columns={"example_id": "segid"})
        frame["feature_set"] = name
        frame["y_true"] = labels[test]
        frame["p_raw"] = raw_test
        frame["p_calibrated"] = scores
        frame["threshold"] = threshold
        frame["y_pred"] = (scores >= threshold).astype(np.int8)
        predictions.append(frame)
    return pd.DataFrame(rows), pd.DataFrame(), pd.concat(predictions, ignore_index=True)
