#!/usr/bin/env python3
"""Calibrate rotating-fold predictions without touching held-out chromosomes.

For every rotating chromosome run, this script fits temperature scaling and a
binary decision threshold exclusively on ``val_chr_test`` predictions.  It then
evaluates the untouched ``heldout_chr_test`` predictions.  Ranking metrics are
reported for completeness but are not expected to change after temperature
scaling.  The output is deliberately compact: source prediction files remain
the authoritative edge-level artifact and this script writes only metrics,
calibration parameters, and aggregate confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from evaluation.calibration import apply_temperature, expected_calibration_error, fit_temperature


REQUIRED_COLUMNS = {"dataset", "split", "y_true", "p_edge"}


def choose_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Choose an F1-maximizing threshold using validation predictions only."""

    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(probability):
        raise ValueError("Labels and probabilities must have equal nonzero length.")
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if len(thresholds) == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = np.flatnonzero(np.isclose(f1, np.nanmax(f1), rtol=0.0, atol=1e-12))
    # A deterministic conservative tie break avoids depending on row ordering.
    return float(np.max(thresholds[best]))


def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    n_bins: int,
) -> dict[str, float | int]:
    """Return ranking, calibration, and threshold-dependent binary metrics."""

    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = probability >= float(threshold)
    two_classes = len(np.unique(y_true)) == 2
    return {
        "n_targets": int(len(y_true)),
        "positive_fraction": float(y_true.mean()),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, probability)) if two_classes else np.nan,
        "auprc": float(average_precision_score(y_true, probability)) if two_classes else np.nan,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "nll": float(log_loss(y_true, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece": float(expected_calibration_error(y_true, probability, n_bins=n_bins)),
    }


def parse_rotating_path(path: Path, rotating_root: Path) -> dict[str, object]:
    """Parse regime, fold, seed, and context from the canonical result layout."""

    relative = path.relative_to(rotating_root)
    parts = relative.parts
    if len(parts) < 6:
        raise ValueError(f"Unexpected rotating-fold path: {relative}")
    seed_match = re.fullmatch(r"seed_(\d+)", parts[2])
    if seed_match is None:
        raise ValueError(f"Could not parse seed from {relative}")
    return {
        "regime": parts[0],
        "fold": parts[1],
        "seed": int(seed_match.group(1)),
        "closure": parts[3],
        "prediction_file": str(path.resolve()),
    }


def calibrate_prediction_file(
    path: Path,
    *,
    rotating_root: Path,
    calibration_split: str,
    evaluation_split: str,
    n_bins: int,
) -> list[dict[str, object]]:
    """Fit on one validation split and score one held-out split."""

    frame = pd.read_csv(
        path,
        compression="infer",
        usecols=lambda column: column in REQUIRED_COLUMNS,
    )
    missing = REQUIRED_COLUMNS - set(frame)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    validation = frame.loc[frame["split"].astype(str).eq(calibration_split)].copy()
    heldout = frame.loc[frame["split"].astype(str).eq(evaluation_split)].copy()
    if validation.empty or heldout.empty:
        raise ValueError(
            f"{path} has validation={len(validation)} and heldout={len(heldout)} rows"
        )

    temperature = fit_temperature(
        validation["y_true"].to_numpy(int), validation["p_edge"].to_numpy(float)
    )
    validation_probability = apply_temperature(
        validation["p_edge"].to_numpy(float), temperature
    )
    validation_threshold = choose_f1_threshold(
        validation["y_true"].to_numpy(int), validation_probability
    )
    heldout_probability = apply_temperature(heldout["p_edge"].to_numpy(float), temperature)
    metadata = parse_rotating_path(path, rotating_root)

    rows: list[dict[str, object]] = []
    dataset_groups: Iterable[tuple[str, pd.DataFrame]] = [
        ("all", heldout),
        *((str(name), group) for name, group in heldout.groupby("dataset", sort=True)),
    ]
    for dataset, group in dataset_groups:
        indices = group.index.to_numpy()
        # heldout retains the original RangeIndex, so align calibrated probabilities
        # by an explicit position map rather than assuming a contiguous subgroup.
        positions = heldout.index.get_indexer(indices)
        y_true = group["y_true"].to_numpy(int)
        raw_probability = group["p_edge"].to_numpy(float)
        calibrated_probability = heldout_probability[positions]
        common = {
            **metadata,
            "dataset": dataset,
            "calibration_split": calibration_split,
            "evaluation_split": evaluation_split,
            "n_calibration": int(len(validation)),
            "temperature": float(temperature),
            "validation_f1_threshold": float(validation_threshold),
        }
        for evaluation, probability, threshold in (
            ("uncalibrated_threshold_0.5", raw_probability, 0.5),
            ("temperature_threshold_0.5", calibrated_probability, 0.5),
            ("temperature_validation_f1_threshold", calibrated_probability, validation_threshold),
        ):
            rows.append(
                {
                    **common,
                    "evaluation": evaluation,
                    **classification_metrics(
                        y_true, probability, threshold=threshold, n_bins=n_bins
                    ),
                }
            )
    return rows


def hierarchical_bootstrap_summary(
    metrics: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize runs and bootstrap folds, then seeds within sampled folds."""

    metric_names = [
        "auroc",
        "auprc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "nll",
        "brier",
        "ece",
    ]
    group_columns = ["regime", "closure", "dataset", "evaluation"]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for key, group in metrics.groupby(group_columns, sort=True, dropna=False):
        folds = sorted(group["fold"].unique())
        for metric in metric_names:
            values = group[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            values_by_fold = [
                group.loc[group["fold"].eq(fold_name), metric]
                .dropna()
                .to_numpy(float)
                for fold_name in folds
            ]
            seed_counts = {len(fold_values) for fold_values in values_by_fold}
            if len(seed_counts) == 1:
                # The production matrix has three seeds per fold.  Indexing the
                # whole bootstrap tensor at once is orders of magnitude faster
                # than repeatedly filtering pandas frames inside the loop.
                fold_matrix = np.stack(values_by_fold)
                n_folds, n_seeds = fold_matrix.shape
                sampled_folds = rng.integers(
                    0, n_folds, size=(n_bootstrap, n_folds)
                )
                sampled_seeds = rng.integers(
                    0, n_seeds, size=(n_bootstrap, n_folds, n_seeds)
                )
                sampled_values = fold_matrix[
                    sampled_folds[:, :, None], sampled_seeds
                ]
                draws = sampled_values.mean(axis=(1, 2))
            else:
                # Defensive fallback for incomplete development matrices.
                draws = np.empty(n_bootstrap, dtype=float)
                for draw_index in range(n_bootstrap):
                    sampled_folds = rng.integers(0, len(folds), size=len(folds))
                    sampled_values = [
                        rng.choice(values_by_fold[fold_index])
                        for fold_index in sampled_folds
                        if len(values_by_fold[fold_index])
                    ]
                    draws[draw_index] = float(np.mean(sampled_values))
            rows.append(
                {
                    **dict(zip(group_columns, key)),
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "std_across_runs": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "n_runs": int(len(values)),
                    "n_folds": int(group["fold"].nunique()),
                    "n_seeds": int(group["seed"].nunique()),
                    "resampling_unit": "chromosome_fold_then_seed",
                    "bootstrap_replicates": int(n_bootstrap),
                    "bootstrap_seed": int(seed),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration-split", default="val_chr_test")
    parser.add_argument("--evaluation-split", default="heldout_chr_test")
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    rotating_root = args.results_root.resolve() / "rotating_folds"
    prediction_files = sorted(rotating_root.glob("**/*pooled_predictions*.csv.gz"))
    if not prediction_files:
        raise FileNotFoundError(f"No rotating-fold predictions under {rotating_root}")

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for path in prediction_files:
        try:
            rows.extend(
                calibrate_prediction_file(
                    path,
                    rotating_root=rotating_root,
                    calibration_split=args.calibration_split,
                    evaluation_split=args.evaluation_split,
                    n_bins=args.n_bins,
                )
            )
        except Exception as error:  # preserve a complete machine-readable audit
            failures.append({"prediction_file": str(path), "error": str(error)})

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError(f"No predictions calibrated; failures={failures[:3]}")
    summary = hierarchical_bootstrap_summary(
        metrics, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "rotating_fold_calibration_metrics.csv", index=False)
    summary.to_csv(out_dir / "rotating_fold_calibration_summary.csv", index=False)
    audit = {
        "schema_version": 1,
        "results_root": str(args.results_root.resolve()),
        "prediction_files_discovered": len(prediction_files),
        "prediction_files_completed": int(metrics["prediction_file"].nunique()),
        "failures": failures,
        "calibration_split": args.calibration_split,
        "evaluation_split": args.evaluation_split,
        "calibration_policy": "temperature and F1 threshold fit on validation chromosomes only",
        "bootstrap_policy": "resample chromosome folds, then seeds within sampled folds",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "status": "completed" if not failures else "completed_with_failures",
    }
    (out_dir / "calibration_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
