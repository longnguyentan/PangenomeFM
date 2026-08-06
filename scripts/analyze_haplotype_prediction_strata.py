#!/usr/bin/env python3
"""Evaluate locked haplotype predictions in label-independent biological strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(frame["y_true"], frame["y_pred"]).statistic),
        "mae": float(mean_absolute_error(frame["y_true"], frame["y_pred"])),
        "r2": float(r2_score(frame["y_true"], frame["y_pred"])),
        "direction_accuracy": float(
            np.mean((frame["y_true"] > 0) == (frame["y_pred"] > 0))
        ),
    }


def _bootstrap(
    values: np.ndarray, *, seed: int, replicates: int
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (replicates, len(values)), replace=True).mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper)


def analyze(
    *,
    pairs_path: str | Path,
    predictions_path: str | Path,
    out_dir: str | Path,
    sequence_model: str = "nt_plus_whole_window_sequence_ridge",
    context_model: str = "hierarchical_annotation_only_residual",
    combined_model: str = "hierarchical_graph_context_residual",
    seed: int = 42,
    bootstrap_replicates: int = 10_000,
) -> dict[str, object]:
    columns = [
        "donor_id",
        "h1_cpgs",
        "h2_cpgs",
        "h1_depth",
        "h2_depth",
        "h1_repeat_fraction",
        "h2_repeat_fraction",
        "node_jaccard",
        "delta_path_bp",
    ]
    pairs = pd.read_csv(
        pairs_path,
        compression="infer",
        usecols=columns,
        low_memory=False,
    )
    pairs["minimum_cpgs"] = pairs[["h1_cpgs", "h2_cpgs"]].min(axis=1)
    pairs["minimum_depth"] = pairs[["h1_depth", "h2_depth"]].min(axis=1)
    pairs["mean_repeat_fraction"] = pairs[
        ["h1_repeat_fraction", "h2_repeat_fraction"]
    ].mean(axis=1)
    pairs["graph_divergence"] = 1.0 - pairs["node_jaccard"]
    repeat_median = pairs.groupby("donor_id")["mean_repeat_fraction"].transform(
        "median"
    )
    divergence_q25 = pairs.groupby("donor_id")["graph_divergence"].transform(
        lambda values: values.quantile(0.25)
    )
    divergence_q75 = pairs.groupby("donor_id")["graph_divergence"].transform(
        lambda values: values.quantile(0.75)
    )
    strata = {
        "all": np.ones(len(pairs), dtype=bool),
        "high_coverage": (
            (pairs["minimum_cpgs"] >= 50)
            & (pairs["minimum_depth"] >= 1000)
        ).to_numpy(),
        "low_repeat": (pairs["mean_repeat_fraction"] <= repeat_median)
        .fillna(False)
        .to_numpy(),
        "high_repeat": (pairs["mean_repeat_fraction"] > repeat_median)
        .fillna(False)
        .to_numpy(),
        "graph_conserved_quartile": (
            pairs["graph_divergence"] <= divergence_q25
        ).to_numpy(),
        "graph_divergent_quartile": (
            pairs["graph_divergence"] >= divergence_q75
        ).to_numpy(),
        "path_length_imbalanced": pairs["delta_path_bp"].ne(0).to_numpy(),
    }
    predictions = pd.read_csv(predictions_path, compression="infer")
    selected_models = [sequence_model, context_model, combined_model]
    predictions = predictions[predictions["model"].isin(selected_models)].copy()
    predictions = predictions.drop(
        columns=["donor_id"], errors="ignore"
    ).merge(
        pairs.reset_index(names="row_index"),
        on="row_index",
        how="left",
        validate="many_to_one",
    )

    rows: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for stratum_index, (stratum, mask) in enumerate(strata.items()):
        row_indices = set(np.flatnonzero(mask).tolist())
        selected = predictions[predictions["row_index"].isin(row_indices)]
        for (model, donor), frame in selected.groupby(["model", "donor_id"]):
            rows.append(
                {
                    "stratum": stratum,
                    "model": model,
                    "donor_id": donor,
                    "n": int(len(frame)),
                    **_metrics(frame),
                }
            )
        per_donor = pd.DataFrame(
            row for row in rows if row["stratum"] == stratum
        )
        for comparison_index, baseline in enumerate(
            (sequence_model, context_model)
        ):
            for metric in ("spearman", "direction_accuracy", "mae", "r2"):
                pivot = per_donor.pivot(
                    index="donor_id", columns="model", values=metric
                )[[baseline, combined_model]].dropna()
                if metric == "mae":
                    differences = (
                        pivot[baseline] - pivot[combined_model]
                    ).to_numpy()
                    orientation = "baseline_minus_combined"
                else:
                    differences = (
                        pivot[combined_model] - pivot[baseline]
                    ).to_numpy()
                    orientation = "combined_minus_baseline"
                estimate, lower, upper = _bootstrap(
                    differences,
                    seed=seed
                    + 100 * stratum_index
                    + 10 * comparison_index,
                    replicates=bootstrap_replicates,
                )
                comparisons.append(
                    {
                        "stratum": stratum,
                        "metric": metric,
                        "baseline": baseline,
                        "candidate": combined_model,
                        "n_donors": int(len(differences)),
                        "difference": estimate,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "positive_donors": int((differences > 0).sum()),
                        "orientation": orientation,
                    }
                )

    per_donor = pd.DataFrame(rows)
    macro = (
        per_donor.groupby(["stratum", "model"], as_index=False)[
            ["spearman", "direction_accuracy", "mae", "r2"]
        ]
        .mean()
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_donor.to_csv(out_dir / "per_donor_metrics.csv", index=False)
    macro.to_csv(out_dir / "macro_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        out_dir / "paired_donor_comparisons.csv", index=False
    )
    summary = {
        "task": "label_independent_prediction_strata",
        "pairs_path": str(pairs_path),
        "predictions_path": str(predictions_path),
        "models": selected_models,
        "stratum_sizes": {
            name: int(np.sum(mask)) for name, mask in strata.items()
        },
        "macro_metrics": macro.to_dict("records"),
        "paired_donor_comparisons": comparisons,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--sequence-model",
        default="nt_plus_whole_window_sequence_ridge",
    )
    parser.add_argument(
        "--context-model",
        default="hierarchical_annotation_only_residual",
    )
    parser.add_argument(
        "--combined-model",
        default="hierarchical_graph_context_residual",
    )
    args = parser.parse_args()
    analyze(
        pairs_path=args.pairs,
        predictions_path=args.predictions,
        out_dir=args.out_dir,
        sequence_model=args.sequence_model,
        context_model=args.context_model,
        combined_model=args.combined_model,
    )


if __name__ == "__main__":
    main()
