"""Validation-only probability calibration for graph prediction outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


EPSILON = 1e-7


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), EPSILON, 1 - EPSILON)
    return np.log(probability) - np.log1p(-probability)


def apply_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    """Scale binary logits by a positive temperature."""

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    scaled = _logit(probability) / float(temperature)
    return 1.0 / (1.0 + np.exp(-np.clip(scaled, -50, 50)))


def fit_temperature(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Fit one temperature by minimizing validation negative log likelihood."""

    y_true = np.asarray(y_true, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(probability):
        raise ValueError("Labels and probabilities must have equal nonzero length.")

    def objective(log_temperature: float) -> float:
        calibrated = apply_temperature(probability, float(np.exp(log_temperature)))
        return float(log_loss(y_true, calibrated, labels=[0.0, 1.0]))

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, *, n_bins: int = 15
) -> float:
    """Equal-width expected calibration error."""

    y_true = np.asarray(y_true, dtype=float)
    probability = np.asarray(probability, dtype=float)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.clip(
        np.digitize(probability, boundaries, right=False) - 1, 0, n_bins - 1
    )
    error = 0.0
    for bin_index in range(n_bins):
        selected = assignments == bin_index
        if selected.any():
            error += float(selected.mean()) * abs(
                float(probability[selected].mean()) - float(y_true[selected].mean())
            )
    return float(error)


def binary_calibration_metrics(
    y_true: np.ndarray, probability: np.ndarray, *, n_bins: int = 15
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    probability = np.asarray(probability, dtype=float)
    return {
        "auroc": float(roc_auc_score(y_true, probability)),
        "nll": float(log_loss(y_true, probability, labels=[0.0, 1.0])),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece": expected_calibration_error(y_true, probability, n_bins=n_bins),
    }


def _cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    n_bins: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    cluster_column = "slice" if "slice" in frame else None
    if cluster_column is None:
        clusters = np.arange(len(frame)).astype(str)
    else:
        clusters = frame[cluster_column].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    indices_by_cluster = {
        cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters
    }
    draws: dict[str, list[float]] = {
        "nll_delta": [],
        "brier_delta": [],
        "ece_delta": [],
    }
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        indices = np.concatenate([indices_by_cluster[cluster] for cluster in sampled])
        sampled_frame = frame.iloc[indices]
        y = sampled_frame["y_true"].to_numpy(float)
        before = binary_calibration_metrics(
            y, sampled_frame["p_edge"].to_numpy(float), n_bins=n_bins
        )
        after = binary_calibration_metrics(
            y, sampled_frame["p_edge_calibrated"].to_numpy(float), n_bins=n_bins
        )
        for metric in ("nll", "brier", "ece"):
            draws[f"{metric}_delta"].append(after[metric] - before[metric])
    return draws


def calibrate_frame(
    frame: pd.DataFrame,
    *,
    temperature: float,
    n_bins: int = 15,
    n_bootstrap: int = 1_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"y_true", "p_edge"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    output = frame.copy()
    output["p_edge_calibrated"] = apply_temperature(
        output["p_edge"].to_numpy(float), temperature
    )
    y = output["y_true"].to_numpy(float)
    before = binary_calibration_metrics(
        y, output["p_edge"].to_numpy(float), n_bins=n_bins
    )
    after = binary_calibration_metrics(
        y, output["p_edge_calibrated"].to_numpy(float), n_bins=n_bins
    )
    draws = _cluster_bootstrap(
        output,
        n_bootstrap=n_bootstrap,
        n_bins=n_bins,
        seed=seed,
    )
    intervals = {
        name: {
            "estimate": float(after[name.removesuffix("_delta")] - before[name.removesuffix("_delta")]),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for name, values in draws.items()
    }
    return output, {
        "n": int(len(output)),
        "n_clusters": int(output["slice"].nunique()) if "slice" in output else int(len(output)),
        "temperature": float(temperature),
        "before": before,
        "after": after,
        "after_minus_before": intervals,
    }


def _parse_labeled_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(f"Expected LABEL=PATH, received {specification!r}.")
    label, path = specification.split("=", 1)
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-predictions", required=True)
    parser.add_argument("--calibration-split")
    parser.add_argument("--inputs", nargs="+", required=True, help="LABEL=PATH")
    parser.add_argument("--evaluation-split")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-bootstrap", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    calibration = pd.read_csv(args.calibration_predictions, compression="infer")
    if args.calibration_split:
        if "split" not in calibration:
            raise ValueError("Calibration split requested but no split column exists.")
        calibration = calibration[
            calibration["split"].astype(str).eq(args.calibration_split)
        ]
    temperature = fit_temperature(
        calibration["y_true"].to_numpy(float),
        calibration["p_edge"].to_numpy(float),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {
        "calibration_source": str(args.calibration_predictions),
        "calibration_split": args.calibration_split,
        "n_calibration": int(len(calibration)),
        "temperature": temperature,
        "evaluations": {},
    }
    for specification in args.inputs:
        label, path = _parse_labeled_path(specification)
        frame = pd.read_csv(path, compression="infer")
        if args.evaluation_split and "split" in frame:
            frame = frame[frame["split"].astype(str).eq(args.evaluation_split)]
        calibrated, summary = calibrate_frame(
            frame,
            temperature=temperature,
            n_bins=args.n_bins,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        calibrated.to_csv(
            out_dir / f"{label}_calibrated_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
        summaries["evaluations"][label] = {"path": str(path), **summary}
    (out_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
