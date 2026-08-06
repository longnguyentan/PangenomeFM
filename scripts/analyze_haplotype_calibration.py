#!/usr/bin/env python3
"""Calibration and held-out-donor consistency audit for haplotype models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


def _slope_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(y_pred)), y_pred])
    intercept, slope = np.linalg.lstsq(design, y_true, rcond=None)[0]
    return float(slope), float(intercept)


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    slope, intercept = _slope_intercept(y_true, y_pred)
    return {
        "n": int(len(frame)),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "direction_accuracy": float(np.mean((y_true > 0) == (y_pred > 0))),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "prediction_sd": float(np.std(y_pred)),
        "target_sd": float(np.std(y_true)),
    }


def _bootstrap_mean(
    values: np.ndarray, *, seed: int, replicates: int
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        values, size=(replicates, len(values)), replace=True
    ).mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper)


def analyze(
    *,
    predictions_path: str | Path,
    out_dir: str | Path,
    models: list[str],
    labels: list[str] | None = None,
    seed: int = 42,
    bootstrap_replicates: int = 10_000,
) -> dict[str, object]:
    if labels is None:
        labels = models
    if len(labels) != len(models):
        raise ValueError("--labels and --models must have the same length.")
    label_map = dict(zip(models, labels))
    predictions = pd.read_csv(predictions_path, compression="infer")
    predictions = predictions.loc[predictions["model"].isin(models)].copy()
    missing = set(models) - set(predictions["model"])
    if missing:
        raise ValueError(f"Predictions lack required models: {sorted(missing)}")

    donor_rows = []
    for (model, donor), frame in predictions.groupby(["model", "donor_id"]):
        donor_rows.append(
            {
                "model": model,
                "label": label_map[model],
                "donor_id": donor,
                **_metrics(frame),
            }
        )
    donor_metrics = pd.DataFrame(donor_rows)

    macro_rows = []
    metric_names = [
        "spearman",
        "mae",
        "r2",
        "direction_accuracy",
        "calibration_slope",
        "calibration_intercept",
        "prediction_sd",
        "target_sd",
    ]
    for model_index, model in enumerate(models):
        frame = donor_metrics.loc[donor_metrics["model"] == model]
        row: dict[str, object] = {
            "model": model,
            "label": label_map[model],
            "n_donors": int(frame["donor_id"].nunique()),
        }
        for metric_index, metric in enumerate(metric_names):
            estimate, lower, upper = _bootstrap_mean(
                frame[metric].to_numpy(),
                seed=seed + 100 * model_index + metric_index,
                replicates=bootstrap_replicates,
            )
            row[metric] = estimate
            row[f"{metric}_ci95_lower"] = lower
            row[f"{metric}_ci95_upper"] = upper
        macro_rows.append(row)
    macro = pd.DataFrame(macro_rows)

    calibration_rows = []
    for model, frame in predictions.groupby("model"):
        quantiles = min(20, int(frame["y_pred"].nunique()))
        frame = frame.copy()
        frame["prediction_bin"] = pd.qcut(
            frame["y_pred"], q=quantiles, labels=False, duplicates="drop"
        )
        for prediction_bin, group in frame.groupby("prediction_bin"):
            donor_means = group.groupby("donor_id")[["y_true", "y_pred"]].mean()
            observed, observed_low, observed_high = _bootstrap_mean(
                donor_means["y_true"].to_numpy(),
                seed=seed + int(prediction_bin),
                replicates=bootstrap_replicates,
            )
            calibration_rows.append(
                {
                    "model": model,
                    "label": label_map[model],
                    "prediction_bin": int(prediction_bin),
                    "n": int(len(group)),
                    "mean_predicted": float(group["y_pred"].mean()),
                    "mean_observed": observed,
                    "observed_ci95_lower": observed_low,
                    "observed_ci95_upper": observed_high,
                }
            )
    calibration = pd.DataFrame(calibration_rows)

    colors = ["#4B5563", "#D97706", "#1D4ED8", "#7C3AED"]
    figure, axes = plt.subplots(
        1, 3, figsize=(11.2, 3.4), constrained_layout=True
    )
    ax = axes[0]
    limits = [
        min(calibration["mean_predicted"].min(), calibration["mean_observed"].min()),
        max(calibration["mean_predicted"].max(), calibration["mean_observed"].max()),
    ]
    ax.plot(limits, limits, linestyle="--", color="#94A3B8", linewidth=1)
    for color, model in zip(colors, models):
        frame = calibration.loc[calibration["model"] == model]
        ax.errorbar(
            frame["mean_predicted"],
            frame["mean_observed"],
            yerr=np.vstack(
                [
                    frame["mean_observed"] - frame["observed_ci95_lower"],
                    frame["observed_ci95_upper"] - frame["mean_observed"],
                ]
            ),
            marker="o",
            markersize=3,
            linewidth=1.2,
            capsize=1.5,
            color=color,
            label=label_map[model],
        )
    ax.set_xlabel("Mean predicted H1-H2 methylation")
    ax.set_ylabel("Mean observed H1-H2 methylation")
    ax.set_title("A  Donor-bootstrap calibration", loc="left")
    ax.legend(frameon=False, fontsize=7)

    for panel, metric, title in [
        (axes[1], "spearman", "B  Held-out donor rank correlation"),
        (axes[2], "r2", r"C  Held-out donor $R^2$"),
    ]:
        pivot = donor_metrics.pivot(
            index="donor_id", columns="model", values=metric
        )
        donors = pivot.index.tolist()
        y = np.arange(len(donors))
        for offset, (color, model) in zip(
            np.linspace(-0.18, 0.18, len(models)), zip(colors, models)
        ):
            panel.scatter(
                pivot[model],
                y + offset,
                s=18,
                color=color,
                label=label_map[model],
            )
        panel.axvline(0, color="#94A3B8", linestyle="--", linewidth=0.8)
        panel.set_yticks(y, donors)
        panel.set_xlabel(metric.replace("_", " ").title())
        panel.set_title(title, loc="left")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / "haplotype_calibration_and_donor_consistency.png"
    figure.savefig(figure_path, dpi=350, bbox_inches="tight")
    plt.close(figure)

    donor_metrics.to_csv(output / "per_donor_metrics.csv", index=False)
    macro.to_csv(output / "donor_macro_calibration.csv", index=False)
    calibration.to_csv(output / "calibration_bins.csv", index=False)
    summary = {
        "task": "haplotype_calibration_and_donor_consistency",
        "predictions_path": str(predictions_path),
        "models": label_map,
        "bootstrap_unit": "heldout_donor",
        "bootstrap_replicates": bootstrap_replicates,
        "donor_macro": macro.to_dict("records"),
        "figure": str(figure_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    analyze(
        predictions_path=args.predictions,
        out_dir=args.out_dir,
        models=args.models,
        labels=args.labels,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )


if __name__ == "__main__":
    main()
