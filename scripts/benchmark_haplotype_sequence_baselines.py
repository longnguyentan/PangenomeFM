#!/usr/bin/env python3
"""Held-out-donor sequence controls for haplotype methylation prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def _lookup(path: str | Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {
        str(key): value.astype(np.float32)
        for key, value in zip(payload["ids"], payload["embeddings"])
    }


def _deltas(frame: pd.DataFrame, path: str | Path) -> np.ndarray:
    lookup = _lookup(path)
    return np.stack(
        [
            lookup[str(left)] - lookup[str(right)]
            for left, right in zip(
                frame["h1_embedding_id"], frame["h2_embedding_id"]
            )
        ]
    )


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(y, prediction).statistic),
        "mae": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
        "direction_accuracy": float(np.mean((y > 0) == (prediction > 0))),
    }


def _nested_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    val_scaled = scaler.transform(val_x)
    test_scaled = scaler.transform(test_x)
    sweep = []
    models = {}
    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr")
        model.fit(train_scaled, train_y)
        sweep.append(
            {
                "alpha": float(alpha),
                "validation_mae": float(
                    mean_absolute_error(val_y, model.predict(val_scaled))
                ),
            }
        )
        models[alpha] = model
    selected = min(sweep, key=lambda row: (row["validation_mae"], row["alpha"]))
    alpha = float(selected["alpha"])
    return models[alpha].predict(test_scaled), alpha, sweep


def _antisymmetric_histogram_boost(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
) -> np.ndarray:
    # Sign-flip augmentation plus antisymmetrized inference gives
    # f(-x)=-f(x) exactly while allowing nonlinear full-window sequence effects.
    augmented_x = np.concatenate([train_x, -train_x], axis=0)
    augmented_y = np.concatenate([train_y, -train_y], axis=0)
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(augmented_x, augmented_y)
    return 0.5 * (model.predict(test_x) - model.predict(-test_x))


def benchmark(
    *,
    pairs_path: str | Path,
    sequence_embeddings: str | Path,
    out_dir: str | Path,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    seed: int = 42,
) -> dict[str, object]:
    header = pd.read_csv(pairs_path, compression="infer", nrows=0).columns
    sequence_columns = [
        "delta_gc_fraction",
        "delta_cpg_density",
        "delta_base_entropy",
        *sorted(column for column in header if column.startswith("delta_kmer3_")),
    ]
    columns = [
        "donor_id",
        "h1_embedding_id",
        "h2_embedding_id",
        "methylation_delta",
        *sequence_columns,
    ]
    pairs = pd.read_csv(
        pairs_path,
        compression="infer",
        low_memory=False,
        usecols=columns,
    )
    whole_window = pairs[sequence_columns].fillna(0).to_numpy(np.float32)
    nucleotide_transformer = _deltas(pairs, sequence_embeddings)
    combined = np.concatenate([nucleotide_transformer, whole_window], axis=1)
    target = pairs["methylation_delta"].to_numpy(np.float32)
    groups = pairs["donor_id"].astype(str).to_numpy()
    unique_groups = sorted(set(groups))
    prediction_rows: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []

    for fold, heldout in enumerate(unique_groups):
        print(
            f"[sequence-baseline] fold={fold + 1}/{len(unique_groups)} "
            f"heldout={heldout}",
            flush=True,
        )
        training_groups = [group for group in unique_groups if group != heldout]
        validation = training_groups[fold % len(training_groups)]
        train_idx = np.flatnonzero(
            (groups != heldout) & (groups != validation)
        )
        val_idx = np.flatnonzero(groups == validation)
        test_idx = np.flatnonzero(groups == heldout)
        predictions: dict[str, np.ndarray] = {}
        for name, features in (
            ("whole_window_nested_ridge", whole_window),
            ("nucleotide_transformer_nested_ridge", nucleotide_transformer),
            ("nt_plus_whole_window_nested_ridge", combined),
        ):
            prediction, alpha, sweep = _nested_ridge(
                features[train_idx],
                target[train_idx],
                features[val_idx],
                target[val_idx],
                features[test_idx],
                alphas,
            )
            predictions[name] = prediction
            selections.append(
                {
                    "fold": fold,
                    "heldout_group": heldout,
                    "validation_group": validation,
                    "model": name,
                    "selected_alpha": alpha,
                    "validation_sweep": sweep,
                }
            )
        predictions["whole_window_antisymmetric_histgb"] = (
            _antisymmetric_histogram_boost(
                whole_window[train_idx],
                target[train_idx],
                whole_window[test_idx],
                seed + fold,
            )
        )
        sequence_ablation_indices = {
            "cpg_density_only_antisymmetric_histgb": [1],
            "kmer3_only_antisymmetric_histgb": list(
                range(3, whole_window.shape[1])
            ),
            "whole_window_without_cpg_antisymmetric_histgb": [
                0,
                2,
                *range(3, whole_window.shape[1]),
            ],
        }
        for offset, (name, indices) in enumerate(
            sequence_ablation_indices.items(), start=1
        ):
            predictions[name] = _antisymmetric_histogram_boost(
                whole_window[train_idx][:, indices],
                target[train_idx],
                whole_window[test_idx][:, indices],
                seed + fold + 100 * offset,
            )
        for local, row_index in enumerate(test_idx):
            for model, prediction in predictions.items():
                prediction_rows.append(
                    {
                        "row_index": int(row_index),
                        "heldout_group": heldout,
                        "model": model,
                        "y_true": float(target[row_index]),
                        "y_pred": float(prediction[local]),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    per_group_rows = []
    for (model, group), frame in predictions.groupby(
        ["model", "heldout_group"], sort=False
    ):
        per_group_rows.append(
            {
                "model": model,
                "heldout_group": group,
                "n": int(len(frame)),
                **_metrics(
                    frame["y_true"].to_numpy(),
                    frame["y_pred"].to_numpy(),
                ),
            }
        )
    per_group = pd.DataFrame(per_group_rows)
    macro = per_group.groupby("model", as_index=False)[
        ["spearman", "mae", "r2", "direction_accuracy"]
    ].mean()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        out_dir / "heldout_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    per_group.to_csv(out_dir / "per_group_metrics.csv", index=False)
    macro.to_csv(out_dir / "macro_metrics.csv", index=False)
    summary = {
        "task": "heldout_donor_sequence_baselines",
        "pairs_path": str(pairs_path),
        "sequence_embeddings": str(sequence_embeddings),
        "n_pairs": int(len(pairs)),
        "n_donors": int(len(unique_groups)),
        "whole_window_features": sequence_columns,
        "ridge_alpha_grid": list(alphas),
        "nested_validation": True,
        "histgb_antisymmetry": (
            "sign-flip training augmentation and (f(x)-f(-x))/2 inference"
        ),
        "histgb_feature_ablations": {
            "cpg_density_only_antisymmetric_histgb": ["delta_cpg_density"],
            "kmer3_only_antisymmetric_histgb": [
                column
                for column in sequence_columns
                if column.startswith("delta_kmer3_")
            ],
            "whole_window_without_cpg_antisymmetric_histgb": [
                column
                for column in sequence_columns
                if column != "delta_cpg_density"
            ],
        },
        "macro_metrics": macro.to_dict("records"),
        "ridge_selection": selections,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--sequence-embeddings", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    benchmark(
        pairs_path=args.pairs,
        sequence_embeddings=args.sequence_embeddings,
        out_dir=args.out_dir,
        alphas=tuple(args.alphas),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
