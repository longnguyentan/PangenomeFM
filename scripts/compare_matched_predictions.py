#!/usr/bin/env python
"""Paired model comparison on one exact evaluation universe.

The script intersects prediction IDs, verifies labels, averages duplicate
predictions for the same ID, reports paired DeLong AUROC tests, and computes
paired bootstrap confidence intervals for AUROC/AUPRC/Brier differences.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}")
    label, raw_path = value.split("=", 1)
    if not label:
        raise ValueError(f"Prediction label is empty in {value!r}")
    return label, Path(raw_path)


def _probability_column(frame: pd.DataFrame) -> str:
    for column in ("probability", "p_positive", "p_ccre", "p_edge"):
        if column in frame:
            return column
    raise ValueError(f"No probability column in {list(frame.columns)}")


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[start : end + 1] = 0.5 * (start + end) + 1.0
        start = end + 1
    out = np.empty(len(values), dtype=float)
    out[order] = ranks
    return out


def _delong_covariance(predictions: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positives = labels == 1
    m = int(positives.sum())
    n = int((~positives).sum())
    if m < 2 or n < 2:
        raise ValueError("DeLong comparison requires at least two positives and negatives.")
    ordered = np.concatenate([np.where(positives)[0], np.where(~positives)[0]])
    scores = predictions[:, ordered]
    k = scores.shape[0]
    positive_ranks = np.empty((k, m), dtype=float)
    negative_ranks = np.empty((k, n), dtype=float)
    all_ranks = np.empty((k, m + n), dtype=float)
    for idx in range(k):
        positive_ranks[idx] = _midrank(scores[idx, :m])
        negative_ranks[idx] = _midrank(scores[idx, m:])
        all_ranks[idx] = _midrank(scores[idx])
    aucs = all_ranks[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (all_ranks[:, :m] - positive_ranks) / n
    v10 = 1.0 - (all_ranks[:, m:] - negative_ranks) / m
    covariance = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, covariance


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def _bootstrap_pair(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    clusters: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    attempts = 0
    cluster_indices: dict[str, np.ndarray] | None = None
    unique_clusters: np.ndarray | None = None
    if clusters is not None:
        clusters = np.asarray(clusters).astype(str)
        unique_clusters = np.unique(clusters)
        cluster_indices = {
            cluster: np.flatnonzero(clusters == cluster)
            for cluster in unique_clusters
        }
    while len(rows) < n_boot and attempts < n_boot * 5:
        attempts += 1
        if cluster_indices is None or unique_clusters is None:
            indices = rng.integers(0, len(y), size=len(y))
        else:
            sampled_clusters = rng.choice(
                unique_clusters, size=len(unique_clusters), replace=True
            )
            indices = np.concatenate(
                [cluster_indices[cluster] for cluster in sampled_clusters]
            )
        sampled_y = y[indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        ma = _metrics(sampled_y, a[indices])
        mb = _metrics(sampled_y, b[indices])
        rows.append({metric: ma[metric] - mb[metric] for metric in ma})
    if not rows:
        raise ValueError("No valid paired bootstrap resamples.")
    frame = pd.DataFrame(rows)
    return {
        metric: {
            "mean": float(frame[metric].mean()),
            "lower": float(frame[metric].quantile(0.025)),
            "upper": float(frame[metric].quantile(0.975)),
            "p_two_sided": float(
                min(1.0, 2.0 * min((frame[metric] <= 0).mean(), (frame[metric] >= 0).mean()))
            ),
        }
        for metric in frame
    }


def _load_prediction(label: str, path: Path, id_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="infer")
    missing = set(id_columns + ["y_true"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: {path} missing columns {sorted(missing)}")
    probability = _probability_column(frame)
    selected = frame[id_columns + ["y_true", probability]].copy()
    selected["y_true"] = selected["y_true"].astype(int)
    selected[probability] = selected[probability].astype(float)
    grouped = (
        selected.groupby(id_columns, as_index=False)
        .agg(y_true=("y_true", "first"), probability=(probability, "mean"))
        .rename(columns={"probability": label})
    )
    label_counts = selected.groupby(id_columns)["y_true"].nunique()
    if int((label_counts > 1).sum()):
        raise ValueError(f"{label}: duplicate IDs have inconsistent labels.")
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--id-columns", nargs="+", default=["segid"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--segments",
        help=(
            "Optional full_segments table. With a single integer segid ID column, "
            "paired bootstrap resamples chromosome-coordinate blocks."
        ),
    )
    parser.add_argument("--block-bp", type=int, default=1_000_000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed = [_spec(value) for value in args.inputs]
    labels = [label for label, _ in parsed]
    if len(labels) != len(set(labels)):
        raise ValueError("Prediction labels must be unique.")

    merged: pd.DataFrame | None = None
    input_counts: dict[str, int] = {}
    for label, path in parsed:
        current = _load_prediction(label, path, args.id_columns)
        input_counts[label] = int(len(current))
        if merged is None:
            merged = current
        else:
            merged = merged.merge(
                current.drop(columns=["y_true"]),
                on=args.id_columns,
                how="inner",
                validate="one_to_one",
            )
            truth = current[args.id_columns + ["y_true"]]
            check = merged[args.id_columns + ["y_true"]].merge(
                truth,
                on=args.id_columns,
                how="left",
                suffixes=("_left", "_right"),
            )
            if not (check["y_true_left"] == check["y_true_right"]).all():
                raise ValueError(f"{label}: labels disagree on the shared universe.")
    assert merged is not None
    if merged.empty:
        raise ValueError("Prediction files have no shared evaluation IDs.")
    cluster_values: np.ndarray | None = None
    resampling_unit = "matched node"
    if args.segments:
        if args.id_columns != ["segid"]:
            raise ValueError("--segments currently requires --id-columns segid.")
        segments = pd.read_csv(
            args.segments,
            compression="infer",
            usecols=["SN", "SO"],
        )
        segids = merged["segid"].astype(int).to_numpy()
        if segids.min() < 0 or segids.max() >= len(segments):
            raise ValueError("segid values fall outside the supplied segment table.")
        selected_segments = segments.iloc[segids].reset_index(drop=True)
        merged["genomic_block"] = (
            selected_segments["SN"].astype(str)
            + ":"
            + (
                pd.to_numeric(selected_segments["SO"], errors="coerce")
                .fillna(-1)
                .astype(int)
                // args.block_bp
            ).astype(str)
        )
        cluster_values = merged["genomic_block"].to_numpy(str)
        resampling_unit = f"{args.block_bp}-bp chromosome-coordinate block"
    merged.to_csv(out_dir / "matched_predictions.csv.gz", index=False, compression="gzip")

    y = merged["y_true"].to_numpy(int)
    metric_rows = [
        {"model": label, "n": int(len(merged)), **_metrics(y, merged[label].to_numpy(float))}
        for label in labels
    ]
    pd.DataFrame(metric_rows).to_csv(out_dir / "matched_metrics.csv", index=False)

    comparison_rows: list[dict[str, object]] = []
    comparison_details: list[dict[str, object]] = []
    for pair_idx, (label_a, label_b) in enumerate(itertools.combinations(labels, 2)):
        a = merged[label_a].to_numpy(float)
        b = merged[label_b].to_numpy(float)
        aucs, covariance = _delong_covariance(np.vstack([a, b]), y)
        contrast_variance = float(
            covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1]
        )
        z = float((aucs[0] - aucs[1]) / np.sqrt(max(contrast_variance, 1e-15)))
        delong_p = float(2.0 * norm.sf(abs(z)))
        bootstrap = _bootstrap_pair(
            y,
            a,
            b,
            n_boot=args.n_boot,
            seed=args.seed + pair_idx,
            clusters=cluster_values,
        )
        row: dict[str, object] = {
            "model_a": label_a,
            "model_b": label_b,
            "n": int(len(y)),
            "auroc_a": float(aucs[0]),
            "auroc_b": float(aucs[1]),
            "auroc_delta_a_minus_b": float(aucs[0] - aucs[1]),
            "delong_z": z,
            "delong_p_two_sided": delong_p,
        }
        for metric, values in bootstrap.items():
            row[f"{metric}_delta_ci_lower"] = values["lower"]
            row[f"{metric}_delta_ci_upper"] = values["upper"]
            row[f"{metric}_bootstrap_p_two_sided"] = values["p_two_sided"]
        comparison_rows.append(row)
        comparison_details.append({**row, "paired_bootstrap": bootstrap})
    pd.DataFrame(comparison_rows).to_csv(out_dir / "paired_comparisons.csv", index=False)

    summary = {
        "evaluation_design": "exact ID intersection; duplicate predictions averaged by ID",
        "id_columns": args.id_columns,
        "input_unique_id_counts": input_counts,
        "matched_n": int(len(merged)),
        "positive_fraction": float(y.mean()),
        "bootstrap_resampling_unit": resampling_unit,
        "n_bootstrap_clusters": (
            int(len(np.unique(cluster_values))) if cluster_values is not None else int(len(y))
        ),
        "metrics": metric_rows,
        "comparisons": comparison_details,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
