#!/usr/bin/env python3
"""Exploratory rank-only residual heads for haplotype methylation.

This analysis is intentionally labeled exploratory because it was motivated by
post-hoc review. It must be replicated on an independent chromosome before it
can support a manuscript claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from benchmark_incremental_graph_residual import (
    ANNOTATION_CANDIDATES,
    _embedding_deltas,
)


def _signed_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    magnitude_rank = rankdata(np.abs(values), method="average") / (
        len(values) + 1.0
    )
    return np.sign(values) * magnitude_rank


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    *,
    swap_signs: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=31,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        np.concatenate([train_x, train_x * swap_signs]),
        np.concatenate([train_y, -train_y]),
    )

    def antisymmetric(values: np.ndarray) -> np.ndarray:
        return 0.5 * (
            model.predict(values) - model.predict(values * swap_signs)
        )

    return antisymmetric(val_x), antisymmetric(test_x)


def _metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(y, score).statistic),
        "direction_accuracy": float(np.mean((y > 0) == (score > 0))),
    }


def _bootstrap(
    per_group: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    seed: int,
    replicates: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows = []
    for metric in ("spearman", "direction_accuracy"):
        pivot = per_group.pivot(
            index="heldout_group", columns="model", values=metric
        )[[baseline, candidate]].dropna()
        difference = (
            pivot[candidate] - pivot[baseline]
        ).to_numpy(dtype=float)
        draws = rng.choice(
            difference, (replicates, len(difference)), replace=True
        ).mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "baseline": baseline,
                "candidate": candidate,
                "n_donors": int(len(difference)),
                "difference": float(difference.mean()),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "positive_donors": int((difference > 0).sum()),
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
    alpha_grid: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0),
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
    if not np.array_equal(
        locked["row_index"].to_numpy(), np.arange(len(pairs))
    ):
        raise ValueError("Locked sequence predictions do not cover all pairs.")
    target = pairs["methylation_delta"].to_numpy(np.float32)
    baseline = locked["y_pred"].to_numpy(np.float32)
    groups = pairs["donor_id"].astype(str).to_numpy()

    annotation_missing = (
        pairs[annotation_columns].isna().astype(np.float32).to_numpy()
    )
    annotation = pairs[annotation_columns].fillna(0).to_numpy(np.float32)
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
        "rank_annotation_residual": (
            annotation_x,
            annotation_signs,
        ),
        "rank_graph_residual": (
            graph_x,
            -np.ones(graph_x.shape[1], dtype=np.float32),
        ),
        "rank_graph_annotation_residual": (
            np.concatenate([graph_x, annotation_x], axis=1),
            np.concatenate(
                [
                    -np.ones(graph_x.shape[1], dtype=np.float32),
                    annotation_signs,
                ]
            ),
        ),
    }

    target_rank = np.empty(len(pairs), dtype=np.float32)
    baseline_rank = np.empty(len(pairs), dtype=np.float32)
    for donor in sorted(set(groups)):
        donor_idx = np.flatnonzero(groups == donor)
        target_rank[donor_idx] = _signed_rank(target[donor_idx])
        baseline_rank[donor_idx] = _signed_rank(baseline[donor_idx])
    rank_residual = target_rank - baseline_rank

    unique_groups = sorted(set(groups))
    prediction_rows = []
    selection_rows = []
    baseline_name = f"{sequence_model}_rank_score"
    for fold, heldout in enumerate(unique_groups):
        print(
            f"[rank-residual] fold={fold + 1}/{len(unique_groups)} "
            f"heldout={heldout}",
            flush=True,
        )
        available = [group for group in unique_groups if group != heldout]
        validation = available[fold % len(available)]
        train_idx = np.flatnonzero(
            (groups != heldout) & (groups != validation)
        )
        val_idx = np.flatnonzero(groups == validation)
        test_idx = np.flatnonzero(groups == heldout)
        fold_scores = {baseline_name: baseline_rank[test_idx]}
        for model_index, (name, (features, signs)) in enumerate(
            feature_sets.items()
        ):
            residual_val, residual_test = _fit_predict(
                features[train_idx],
                rank_residual[train_idx],
                features[val_idx],
                features[test_idx],
                swap_signs=signs,
                seed=seed + 100 * model_index + fold,
            )
            sweep = []
            for alpha in alpha_grid:
                score = baseline_rank[val_idx] + alpha * residual_val
                sweep.append(
                    {
                        "alpha": alpha,
                        "validation_spearman": float(
                            spearmanr(target[val_idx], score).statistic
                        ),
                    }
                )
            selected = max(
                sweep,
                key=lambda row: (
                    row["validation_spearman"],
                    -row["alpha"],
                ),
            )
            alpha = float(selected["alpha"])
            output_name = f"{name}_selected"
            fold_scores[output_name] = (
                baseline_rank[test_idx] + alpha * residual_test
            )
            selection_rows.append(
                {
                    "heldout_group": heldout,
                    "validation_group": validation,
                    "model": name,
                    "selected_alpha": alpha,
                    "validation_sweep": sweep,
                }
            )
        for local, row_index in enumerate(test_idx):
            for model, score in fold_scores.items():
                prediction_rows.append(
                    {
                        "row_index": int(row_index),
                        "heldout_group": heldout,
                        "donor_id": groups[row_index],
                        "model": model,
                        "y_true": float(target[row_index]),
                        "rank_score": float(score[local]),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    per_group_rows = []
    for (model, group), frame in predictions.groupby(
        ["model", "heldout_group"]
    ):
        per_group_rows.append(
            {
                "model": model,
                "heldout_group": group,
                "n": int(len(frame)),
                **_metrics(
                    frame["y_true"].to_numpy(),
                    frame["rank_score"].to_numpy(),
                ),
            }
        )
    per_group = pd.DataFrame(per_group_rows)
    macro = per_group.groupby("model", as_index=False)[
        ["spearman", "direction_accuracy"]
    ].mean()
    comparisons = []
    for model_index, candidate in enumerate(
        model for model in macro["model"] if model != baseline_name
    ):
        comparisons.extend(
            _bootstrap(
                per_group,
                baseline=baseline_name,
                candidate=candidate,
                seed=seed + model_index,
                replicates=bootstrap_replicates,
            )
        )
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        output / "heldout_rank_scores.csv.gz",
        index=False,
        compression="gzip",
    )
    per_group.to_csv(output / "per_group_metrics.csv", index=False)
    macro.to_csv(output / "macro_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        output / "paired_donor_comparisons.csv", index=False
    )
    (output / "validation_selection.json").write_text(
        json.dumps(selection_rows, indent=2), encoding="utf-8"
    )
    summary = {
        "task": "exploratory_rank_only_haplotype_residual",
        "post_hoc": True,
        "independent_replication_required": True,
        "n_pairs": len(pairs),
        "n_donors": len(unique_groups),
        "sequence_baseline": baseline_name,
        "nested_validation_metric": "heldout_validation_donor_spearman",
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
        "--alpha-grid",
        type=float,
        nargs="+",
        default=(0.0, 0.1, 0.25, 0.5, 1.0),
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
        alpha_grid=tuple(args.alpha_grid),
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )


if __name__ == "__main__":
    main()
