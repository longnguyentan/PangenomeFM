#!/usr/bin/env python3
"""Nonlinear conditional test of graph embeddings beyond sequence predictions.

The sequence prediction is locked and cross-fitted. Residual models are trained
only on other donors, use a nested validation donor for shrinkage, and enforce
exact H1/H2 swap antisymmetry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


ANNOTATION_CANDIDATES = [
    "delta_repeat_fraction",
    "delta_hmmflagger_fraction",
    "delta_hmmflagger_dup_fraction",
    "delta_hmmflagger_col_fraction",
    "delta_hmmflagger_err_fraction",
    "delta_hmmflagger_nnn_fraction",
    "delta_copy_number_proxy",
    "delta_graph_mappability_proxy",
    "sv_context_score",
    "sv_length_delta_bp",
    "node_jaccard",
]


def _embedding_lookup(path: str | Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {
        str(identifier): embedding.astype(np.float32)
        for identifier, embedding in zip(payload["ids"], payload["embeddings"])
    }


def _embedding_deltas(
    pairs: pd.DataFrame, path: str | Path
) -> np.ndarray:
    lookup = _embedding_lookup(path)
    return np.stack(
        [
            lookup[str(left)] - lookup[str(right)]
            for left, right in zip(
                pairs["h1_embedding_id"], pairs["h2_embedding_id"]
            )
        ]
    )


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(y, prediction).statistic),
        "mae": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
        "direction_accuracy": float(
            np.mean((y > 0) == (prediction > 0))
        ),
    }


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    *,
    swap_signs: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    swapped_train = train_x * swap_signs
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=31,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        np.concatenate([train_x, swapped_train], axis=0),
        np.concatenate([train_y, -train_y], axis=0),
    )

    def antisymmetric(values: np.ndarray) -> np.ndarray:
        swapped = values * swap_signs
        return 0.5 * (model.predict(values) - model.predict(swapped))

    return antisymmetric(val_x), antisymmetric(test_x)


def _select_shrinkage(
    y: np.ndarray,
    baseline: np.ndarray,
    residual: np.ndarray,
    grid: tuple[float, ...],
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for alpha in grid:
        prediction = baseline + alpha * residual
        rows.append(
            {
                "alpha": alpha,
                "mae": float(mean_absolute_error(y, prediction)),
                "r2": float(r2_score(y, prediction)),
                "spearman": float(spearmanr(y, prediction).statistic),
            }
        )
    selected = min(rows, key=lambda row: (row["mae"], row["alpha"]))
    return float(selected["alpha"]), rows


def _paired_bootstrap(
    per_group: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    seed: int,
    replicates: int,
) -> list[dict[str, object]]:
    rows = []
    rng = np.random.default_rng(seed)
    for metric in ("spearman", "mae", "r2", "direction_accuracy"):
        pivot = per_group.pivot(
            index="heldout_group", columns="model", values=metric
        )[[baseline, candidate]].dropna()
        if metric == "mae":
            difference = (
                pivot[baseline] - pivot[candidate]
            ).to_numpy(dtype=float)
            orientation = "baseline_minus_candidate"
        else:
            difference = (
                pivot[candidate] - pivot[baseline]
            ).to_numpy(dtype=float)
            orientation = "candidate_minus_baseline"
        draws = rng.choice(
            difference,
            size=(replicates, len(difference)),
            replace=True,
        ).mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "baseline": baseline,
                "candidate": candidate,
                "n_groups": int(len(difference)),
                "difference": float(difference.mean()),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "positive_groups": int((difference > 0).sum()),
                "orientation": orientation,
            }
        )
    return rows


def benchmark(
    *,
    pairs_path: str | Path,
    predictions_path: str | Path,
    graph_embeddings: str | Path,
    out_dir: str | Path,
    sequence_model: str = "whole_window_antisymmetric_histgb",
    shrinkage_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    seed: int = 42,
    bootstrap_replicates: int = 10_000,
) -> dict[str, object]:
    header = pd.read_csv(pairs_path, compression="infer", nrows=0).columns
    annotation_columns = [
        column for column in ANNOTATION_CANDIDATES if column in header
    ]
    pairs = pd.read_csv(
        pairs_path,
        compression="infer",
        usecols=[
            "donor_id",
            "h1_embedding_id",
            "h2_embedding_id",
            "methylation_delta",
            *annotation_columns,
        ],
        low_memory=False,
    )
    locked = pd.read_csv(predictions_path, compression="infer")
    locked = (
        locked.loc[
            locked["model"] == sequence_model,
            ["row_index", "donor_id", "y_true", "y_pred"],
        ]
        .sort_values("row_index")
        .reset_index(drop=True)
    )
    expected_indices = np.arange(len(pairs))
    if not np.array_equal(locked["row_index"].to_numpy(), expected_indices):
        raise ValueError("Locked sequence predictions do not cover each pair once.")
    if not np.array_equal(
        locked["donor_id"].astype(str).to_numpy(),
        pairs["donor_id"].astype(str).to_numpy(),
    ):
        raise ValueError("Prediction and pair donor order differs.")
    target = pairs["methylation_delta"].to_numpy(np.float32)
    if not np.allclose(target, locked["y_true"].to_numpy(), atol=1e-6):
        raise ValueError("Prediction targets differ from the pair table.")
    baseline = locked["y_pred"].to_numpy(np.float32)

    annotation_missing = (
        pairs[annotation_columns].isna().astype(np.float32).to_numpy()
    )
    annotation = (
        pairs[annotation_columns].fillna(0).to_numpy(np.float32)
    )
    annotation_x = np.concatenate([annotation, annotation_missing], axis=1)
    annotation_signs = np.concatenate(
        [
            np.asarray(
                [
                    -1.0 if column.startswith("delta_") else 1.0
                    for column in annotation_columns
                ],
                dtype=np.float32,
            ),
            np.ones(len(annotation_columns), dtype=np.float32),
        ]
    )
    graph_x = _embedding_deltas(pairs, graph_embeddings)
    feature_sets = {
        "nonlinear_annotation_residual": (
            annotation_x,
            annotation_signs,
        ),
        "nonlinear_graph_residual": (
            graph_x,
            -np.ones(graph_x.shape[1], dtype=np.float32),
        ),
        "nonlinear_graph_annotation_residual": (
            np.concatenate([graph_x, annotation_x], axis=1),
            np.concatenate(
                [
                    -np.ones(graph_x.shape[1], dtype=np.float32),
                    annotation_signs,
                ]
            ),
        ),
    }
    groups = pairs["donor_id"].astype(str).to_numpy()
    unique_groups = sorted(set(groups))
    prediction_rows = []
    shrinkage_rows = []
    for fold, heldout in enumerate(unique_groups):
        print(
            f"[incremental-residual] fold={fold + 1}/{len(unique_groups)} "
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
        fold_predictions = {sequence_model: baseline[test_idx]}
        for model_offset, (name, (features, signs)) in enumerate(
            feature_sets.items()
        ):
            residual_val, residual_test = _fit_predict(
                features[train_idx],
                target[train_idx] - baseline[train_idx],
                features[val_idx],
                features[test_idx],
                swap_signs=signs,
                seed=seed + 100 * model_offset + fold,
            )
            alpha, sweep = _select_shrinkage(
                target[val_idx],
                baseline[val_idx],
                residual_val,
                shrinkage_grid,
            )
            output_name = f"{name}_shrunk"
            fold_predictions[output_name] = (
                baseline[test_idx] + alpha * residual_test
            )
            shrinkage_rows.append(
                {
                    "heldout_group": heldout,
                    "validation_group": validation,
                    "model": name,
                    "selected_alpha": alpha,
                    "validation_sweep": sweep,
                }
            )
        for local, row_index in enumerate(test_idx):
            for model, prediction in fold_predictions.items():
                prediction_rows.append(
                    {
                        "row_index": int(row_index),
                        "heldout_group": heldout,
                        "donor_id": groups[row_index],
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
    comparisons = []
    for candidate_index, candidate in enumerate(
        name for name in macro["model"] if name != sequence_model
    ):
        comparisons.extend(
            _paired_bootstrap(
                per_group,
                baseline=sequence_model,
                candidate=candidate,
                seed=seed + candidate_index,
                replicates=bootstrap_replicates,
            )
        )

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        output / "heldout_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    per_group.to_csv(output / "per_group_metrics.csv", index=False)
    macro.to_csv(output / "macro_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        output / "paired_donor_comparisons.csv", index=False
    )
    (output / "validation_shrinkage.json").write_text(
        json.dumps(shrinkage_rows, indent=2), encoding="utf-8"
    )
    summary = {
        "task": "nonlinear_incremental_graph_residual",
        "n_pairs": int(len(pairs)),
        "n_donors": len(unique_groups),
        "sequence_model": sequence_model,
        "graph_embeddings": str(graph_embeddings),
        "annotation_columns": annotation_columns,
        "missingness_indicators": True,
        "exact_swap_antisymmetry": True,
        "nested_validation_shrinkage": list(shrinkage_grid),
        "macro_metrics": macro.to_dict("records"),
        "paired_donor_comparisons": comparisons,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--graph-embeddings", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--sequence-model", default="whole_window_antisymmetric_histgb"
    )
    parser.add_argument(
        "--shrinkage-grid",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    benchmark(
        pairs_path=args.pairs,
        predictions_path=args.predictions,
        graph_embeddings=args.graph_embeddings,
        out_dir=args.out_dir,
        sequence_model=args.sequence_model,
        shrinkage_grid=tuple(args.shrinkage_grid),
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )


if __name__ == "__main__":
    main()
