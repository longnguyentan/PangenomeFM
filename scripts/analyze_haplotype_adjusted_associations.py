#!/usr/bin/env python3
"""Label-independent biological and QC audit for haplotype methylation pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


CONTINUOUS_COVARIATES = [
    "mean_cpg_density",
    "mean_repeat_fraction",
    "mean_depth_log1p",
    "mean_base_entropy",
    "absolute_gc_delta",
    "absolute_path_bp_delta_log1p",
]


def _bootstrap_mean(
    values: np.ndarray, *, seed: int, replicates: int
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (replicates, len(values)), replace=True).mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper)


def _derived(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["absolute_methylation_delta"] = result["methylation_delta"].abs()
    result["graph_divergence"] = 1.0 - result["node_jaccard"]
    result["mean_cpg_density"] = result[
        ["h1_cpg_density", "h2_cpg_density"]
    ].mean(axis=1)
    result["mean_repeat_fraction"] = result[
        ["h1_repeat_fraction", "h2_repeat_fraction"]
    ].mean(axis=1)
    result["mean_depth_log1p"] = np.log1p(
        result[["h1_depth", "h2_depth"]].mean(axis=1)
    )
    result["mean_base_entropy"] = result[
        ["h1_base_entropy", "h2_base_entropy"]
    ].mean(axis=1)
    result["absolute_gc_delta"] = result["delta_gc_fraction"].abs()
    result["absolute_path_bp_delta_log1p"] = np.log1p(
        result["delta_path_bp"].abs()
    )
    result["minimum_cpgs"] = result[["h1_cpgs", "h2_cpgs"]].min(axis=1)
    result["minimum_depth"] = result[["h1_depth", "h2_depth"]].min(axis=1)
    return result


def _adjusted_donor_coefficients(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    features = ["graph_divergence", *CONTINUOUS_COVARIATES]
    global_medians = frame[features].median(numeric_only=True)
    for donor, group in frame.groupby("donor_id", sort=True):
        complete = group.dropna(subset=["absolute_methylation_delta"]).copy()
        donor_medians = complete[features].median(numeric_only=True)
        feature_frame = (
            complete[features]
            .fillna(donor_medians)
            .fillna(global_medians)
            .fillna(0.0)
        )
        x = StandardScaler().fit_transform(feature_frame)
        y = complete["absolute_methylation_delta"].to_numpy()
        model = Ridge(alpha=10.0).fit(x, y)
        rows.append(
            {
                "donor_id": str(donor),
                "n": int(len(complete)),
                "imputed_covariate_fraction": float(
                    group[features].isna().to_numpy().mean()
                ),
                "standardized_graph_divergence_coefficient": float(model.coef_[0]),
                "partial_residual_spearman": float(
                    spearmanr(
                        complete["graph_divergence"],
                        y
                        - Ridge(alpha=10.0)
                        .fit(x[:, 1:], y)
                        .predict(x[:, 1:]),
                    ).statistic
                ),
            }
        )
    return pd.DataFrame(rows)


def _strata(frame: pd.DataFrame) -> pd.DataFrame:
    definitions: dict[str, pd.Series] = {
        "all": pd.Series(True, index=frame.index),
        "high_coverage": (frame["minimum_cpgs"] >= 50)
        & (frame["minimum_depth"] >= 1000),
        "low_repeat": frame["mean_repeat_fraction"] <= frame.groupby("donor_id")[
            "mean_repeat_fraction"
        ].transform("median"),
        "high_repeat": frame["mean_repeat_fraction"] > frame.groupby("donor_id")[
            "mean_repeat_fraction"
        ].transform("median"),
        "structurally_conserved": frame["graph_divergence"]
        <= frame.groupby("donor_id")["graph_divergence"].transform(
            lambda values: values.quantile(0.25)
        ),
        "structurally_divergent": frame["graph_divergence"]
        >= frame.groupby("donor_id")["graph_divergence"].transform(
            lambda values: values.quantile(0.75)
        ),
    }
    rows: list[dict[str, float | int | str]] = []
    for name, mask in definitions.items():
        selected = frame.loc[mask.fillna(False)]
        for donor, group in selected.groupby("donor_id"):
            rows.append(
                {
                    "stratum": name,
                    "donor_id": str(donor),
                    "n": int(len(group)),
                    "mean_absolute_methylation_delta": float(
                        group["absolute_methylation_delta"].mean()
                    ),
                    "spearman_abs_delta_vs_graph_divergence": float(
                        spearmanr(
                            group["absolute_methylation_delta"],
                            group["graph_divergence"],
                        ).statistic
                    ),
                }
            )
    return pd.DataFrame(rows)


def analyze(
    *,
    pairs_path: str | Path,
    out_dir: str | Path,
    seed: int = 42,
    bootstrap_replicates: int = 10_000,
) -> dict[str, object]:
    columns = [
        "donor_id",
        "h1_contig",
        "h2_contig",
        "h1_window_start",
        "h2_window_start",
        "h1_embedding_id",
        "h2_embedding_id",
        "methylation_delta",
        "node_jaccard",
        "h1_cpgs",
        "h2_cpgs",
        "h1_depth",
        "h2_depth",
        "h1_cpg_density",
        "h2_cpg_density",
        "h1_repeat_fraction",
        "h2_repeat_fraction",
        "h1_base_entropy",
        "h2_base_entropy",
        "delta_gc_fraction",
        "delta_path_bp",
        "sv_context_score",
    ]
    frame = _derived(
        pd.read_csv(
            pairs_path,
            compression="infer",
            usecols=columns,
            low_memory=False,
        )
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coefficients = _adjusted_donor_coefficients(frame)
    coefficients.to_csv(out_dir / "adjusted_donor_coefficients.csv", index=False)
    strata = _strata(frame)
    strata.to_csv(out_dir / "label_independent_strata.csv", index=False)

    coefficient, coefficient_low, coefficient_high = _bootstrap_mean(
        coefficients["standardized_graph_divergence_coefficient"].to_numpy(),
        seed=seed,
        replicates=bootstrap_replicates,
    )
    partial, partial_low, partial_high = _bootstrap_mean(
        coefficients["partial_residual_spearman"].to_numpy(),
        seed=seed + 1,
        replicates=bootstrap_replicates,
    )

    candidate_columns = [
        "donor_id",
        "h1_contig",
        "h2_contig",
        "h1_window_start",
        "h2_window_start",
        "h1_embedding_id",
        "h2_embedding_id",
        "methylation_delta",
        "absolute_methylation_delta",
        "graph_divergence",
        "mean_repeat_fraction",
        "minimum_cpgs",
        "minimum_depth",
        "sv_context_score",
    ]
    candidates = (
        frame.sort_values(
            ["absolute_methylation_delta", "graph_divergence"],
            ascending=False,
        )
        .groupby("donor_id", sort=True)
        .head(20)[candidate_columns]
    )
    candidates.to_csv(out_dir / "high_effect_candidate_loci.csv", index=False)

    high = strata[strata["stratum"].eq("structurally_divergent")].set_index(
        "donor_id"
    )
    low = strata[strata["stratum"].eq("structurally_conserved")].set_index(
        "donor_id"
    )
    paired_effects = (
        high["mean_absolute_methylation_delta"]
        - low["mean_absolute_methylation_delta"]
    )
    structural_difference = _bootstrap_mean(
        paired_effects.to_numpy(),
        seed=seed + 2,
        replicates=bootstrap_replicates,
    )
    summary: dict[str, object] = {
        "task": "label_independent_haplotype_biological_audit",
        "pairs_path": str(pairs_path),
        "n_pairs": int(len(frame)),
        "n_donors": int(frame["donor_id"].nunique()),
        "adjustment_covariates": CONTINUOUS_COVARIATES,
        "adjustment_model": "within-donor standardized ridge(alpha=10)",
        "standardized_graph_divergence_coefficient": {
            "donor_macro_mean": coefficient,
            "ci95_lower": coefficient_low,
            "ci95_upper": coefficient_high,
            "positive_donors": int(
                (coefficients["standardized_graph_divergence_coefficient"] > 0).sum()
            ),
        },
        "partial_residual_spearman": {
            "donor_macro_mean": partial,
            "ci95_lower": partial_low,
            "ci95_upper": partial_high,
            "positive_donors": int(
                (coefficients["partial_residual_spearman"] > 0).sum()
            ),
        },
        "divergent_minus_conserved_absolute_delta": {
            "donor_macro_mean": structural_difference[0],
            "ci95_lower": structural_difference[1],
            "ci95_upper": structural_difference[2],
            "positive_donors": int((paired_effects > 0).sum()),
        },
        "methylation_qc": {
            "median_minimum_cpgs": float(frame["minimum_cpgs"].median()),
            "median_minimum_depth": float(frame["minimum_depth"].median()),
            "high_coverage_fraction": float(
                (
                    (frame["minimum_cpgs"] >= 50)
                    & (frame["minimum_depth"] >= 1000)
                ).mean()
            ),
        },
        "candidate_loci_output": str(out_dir / "high_effect_candidate_loci.csv"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    result = analyze(
        pairs_path=args.pairs,
        out_dir=args.out_dir,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
