"""Held-out-donor allele-specific expression from paired path embeddings.

The predictor operates on ``z_h1 - z_h2``. This makes the representation
antisymmetric by construction: swapping the two haplotypes negates the input
and, for the no-intercept linear models used here, negates the regression
prediction and complements the direction probability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def paired_delta_features(
    h1_ids: list[str],
    h2_ids: list[str],
    embedding_ids: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    lookup = {str(key): i for i, key in enumerate(embedding_ids)}
    missing = sorted(
        {
            key
            for key in [*map(str, h1_ids), *map(str, h2_ids)]
            if key not in lookup
        }
    )
    if missing:
        raise KeyError(f"{len(missing)} embedding IDs are missing; examples: {missing[:5]}")
    h1 = np.stack([embeddings[lookup[str(key)]] for key in h1_ids])
    h2 = np.stack([embeddings[lookup[str(key)]] for key in h2_ids])
    return (h1 - h2).astype(np.float32)


def _continuous_label(pairs: pd.DataFrame) -> np.ndarray:
    if "ase_value" in pairs:
        return pairs["ase_value"].to_numpy(float)
    if {"h1_count", "h2_count"}.issubset(pairs.columns):
        return np.log2(
            (pairs["h1_count"].to_numpy(float) + 0.5)
            / (pairs["h2_count"].to_numpy(float) + 0.5)
        )
    raise ValueError("Pairs require ase_value or both h1_count and h2_count.")


def _fold_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    direction = (y > 0).astype(int)
    direction_pred = (prob >= 0.5).astype(int)
    rho = spearmanr(y, pred).statistic if len(np.unique(y)) > 1 else float("nan")
    out = {
        "spearman": float(rho),
        "mae": float(mean_absolute_error(y, pred)),
        "direction_accuracy": float(np.mean(direction == direction_pred)),
        "direction_balanced_accuracy": float(
            balanced_accuracy_score(direction, direction_pred)
        ),
    }
    if len(np.unique(direction)) == 2:
        out["direction_auroc"] = float(roc_auc_score(direction, prob))
        out["direction_auprc"] = float(average_precision_score(direction, prob))
    return out


def run_ase_benchmark(
    *,
    pairs_path: str | Path,
    embeddings_path: str | Path,
    out_dir: str | Path,
    donor_column: str = "donor_id",
    h1_column: str = "h1_embedding_id",
    h2_column: str = "h2_embedding_id",
    n_splits: int = 5,
    ridge_alpha: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    pairs = pd.read_csv(pairs_path)
    required = {donor_column, h1_column, h2_column}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Pairs table missing columns: {sorted(missing)}")
    packed = np.load(embeddings_path, allow_pickle=False)
    if not {"ids", "embeddings"}.issubset(packed.files):
        raise ValueError("Embedding NPZ requires arrays named 'ids' and 'embeddings'.")

    X = paired_delta_features(
        pairs[h1_column].astype(str).tolist(),
        pairs[h2_column].astype(str).tolist(),
        packed["ids"],
        packed["embeddings"],
    )
    y = _continuous_label(pairs)
    groups = pairs[donor_column].astype(str).to_numpy()
    unique_donors = np.unique(groups)
    if len(unique_donors) < 2:
        raise ValueError("Held-out-donor evaluation requires at least two donors.")
    folds = min(n_splits, len(unique_donors))
    splitter = GroupKFold(n_splits=folds)

    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        regressor = Ridge(alpha=ridge_alpha, fit_intercept=False).fit(
            X_train, y[train_idx]
        )
        pred = regressor.predict(X_test)
        y_direction = (y[train_idx] > 0).astype(int)
        if len(np.unique(y_direction)) == 2:
            classifier = LogisticRegression(
                penalty="l2",
                C=1.0,
                fit_intercept=False,
                class_weight="balanced",
                random_state=seed + fold,
                max_iter=1000,
            ).fit(X_train, y_direction)
            prob = classifier.predict_proba(X_test)[:, 1]
        else:
            prob = np.full(len(test_idx), float(y_direction[0]))

        metrics = _fold_metrics(y[test_idx], pred, prob)
        fold_rows.append(
            {
                "fold": fold,
                "heldout_donors": ",".join(sorted(np.unique(groups[test_idx]))),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                **metrics,
            }
        )
        for idx, prediction, probability in zip(test_idx, pred, prob):
            prediction_rows.append(
                {
                    "row_index": int(idx),
                    "fold": fold,
                    "donor_id": groups[idx],
                    "y_true": float(y[idx]),
                    "y_pred": float(prediction),
                    "p_h1_higher": float(probability),
                }
            )

    predictions = pd.DataFrame(prediction_rows).sort_values("row_index")
    pooled = _fold_metrics(
        predictions["y_true"].to_numpy(),
        predictions["y_pred"].to_numpy(),
        predictions["p_h1_higher"].to_numpy(),
    )
    zero_prob = np.full(len(y), 0.5)
    baseline = _fold_metrics(y, np.zeros(len(y)), zero_prob)
    antisymmetry_max_error = 0.0
    for train_idx, _ in splitter.split(X, y, groups):
        scaler = StandardScaler(with_mean=False).fit(X[train_idx])
        regressor = Ridge(alpha=ridge_alpha, fit_intercept=False).fit(
            scaler.transform(X[train_idx]), y[train_idx]
        )
        probe = scaler.transform(X[: min(100, len(X))])
        error = np.max(np.abs(regressor.predict(probe) + regressor.predict(-probe)))
        antisymmetry_max_error = max(antisymmetry_max_error, float(error))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(out_dir / "heldout_donor_predictions.csv.gz", index=False, compression="gzip")
    summary = {
        "task": "haplotype_specific_ase",
        "representation": "paired_path_embedding_difference",
        "split": "held_out_donor_group_kfold",
        "n_examples": int(len(pairs)),
        "n_donors": int(len(unique_donors)),
        "n_splits": int(folds),
        "embedding_dim": int(X.shape[1]),
        "pooled_metrics": pooled,
        "zero_baseline_metrics": baseline,
        "antisymmetry_max_abs_error": antisymmetry_max_error,
        "folds": fold_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--donor-column", default="donor_id")
    parser.add_argument("--h1-column", default="h1_embedding_id")
    parser.add_argument("--h2-column", default="h2_embedding_id")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_ase_benchmark(
        pairs_path=args.pairs,
        embeddings_path=args.embeddings,
        out_dir=args.out_dir,
        donor_column=args.donor_column,
        h1_column=args.h1_column,
        h2_column=args.h2_column,
        n_splits=args.n_splits,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
