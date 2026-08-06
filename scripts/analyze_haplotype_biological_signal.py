#!/usr/bin/env python3
"""Audit held-out haplotype methylation signal with donor-paired statistics.

This is a descriptive, post-hoc biological audit of locked predictions.  The
effect-size strata use the observed target only to summarize where an already
trained model is informative; they are never used for training or model
selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_SEQUENCE_MODEL = "frozen_sequence_ridge"
CONTEXT_MODEL = "hierarchical_annotation_only_residual"
COMBINED_MODEL = "hierarchical_graph_context_residual"


def _bootstrap_mean(
    values: np.ndarray, *, seed: int, replicates: int
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        values, size=(replicates, len(values)), replace=True
    ).mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper)


def _direction_accuracy(frame: pd.DataFrame) -> float:
    return float(
        np.mean(
            (frame["methylation_delta"].to_numpy() > 0)
            == (frame["y_pred"].to_numpy() > 0)
        )
    )


def _spearman(frame: pd.DataFrame) -> float:
    if len(frame) < 2:
        return float("nan")
    return float(
        spearmanr(frame["methylation_delta"], frame["y_pred"]).statistic
    )


def _per_donor_model_metrics(
    frame: pd.DataFrame, stratum: str
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for (donor, model), group in frame.groupby(["donor_id", "model"]):
        rows.append(
            {
                "stratum": stratum,
                "donor_id": str(donor),
                "model": str(model),
                "n": int(len(group)),
                "direction_accuracy": _direction_accuracy(group),
                "spearman": _spearman(group),
            }
        )
    return rows


def _paired_model_bootstrap(
    per_donor: pd.DataFrame,
    *,
    stratum: str,
    metric: str,
    baseline: str,
    candidate: str,
    seed: int,
    replicates: int,
) -> dict[str, float | int | str]:
    pivot = (
        per_donor.loc[per_donor["stratum"] == stratum]
        .pivot(index="donor_id", columns="model", values=metric)
        [[baseline, candidate]]
        .dropna()
    )
    differences = (
        pivot[candidate].to_numpy() - pivot[baseline].to_numpy()
    )
    observed, lower, upper = _bootstrap_mean(
        differences, seed=seed, replicates=replicates
    )
    return {
        "stratum": stratum,
        "metric": metric,
        "baseline": baseline,
        "candidate": candidate,
        "n_donors": int(len(differences)),
        "difference": observed,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "positive_donors": int(np.sum(differences > 0)),
        "orientation": "candidate_minus_baseline",
        "bootstrap_replicates": int(replicates),
    }


def _define_strata(pairs: pd.DataFrame) -> dict[str, np.ndarray]:
    absolute = pairs["methylation_delta"].abs()
    within_donor_cutoff = pairs.groupby("donor_id")[
        "methylation_delta"
    ].transform(lambda values: values.abs().quantile(0.9))
    graph_divergence_cutoff = pairs.groupby("donor_id")[
        "node_jaccard"
    ].transform(lambda values: values.quantile(0.25))
    return {
        "All pairs": np.ones(len(pairs), dtype=bool),
        r"$|\Delta m|\geq0.02$": (absolute >= 0.02).to_numpy(),
        r"$|\Delta m|\geq0.05$": (absolute >= 0.05).to_numpy(),
        r"$|\Delta m|\geq0.10$": (absolute >= 0.10).to_numpy(),
        "Within-donor top decile": (
            absolute >= within_donor_cutoff
        ).to_numpy(),
        "Graph-divergent quartile": (
            pairs["node_jaccard"] <= graph_divergence_cutoff
        ).to_numpy(),
    }


def _structural_association(
    pairs: pd.DataFrame, *, seed: int, replicates: int
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    work = pairs[["donor_id", "methylation_delta", "node_jaccard"]].copy()
    work["graph_divergence"] = 1.0 - work["node_jaccard"]
    work["absolute_delta"] = work["methylation_delta"].abs()
    work["divergence_percentile"] = work.groupby("donor_id")[
        "graph_divergence"
    ].rank(pct=True, method="average")
    work["divergence_quartile"] = np.minimum(
        (work["divergence_percentile"] * 4).astype(int), 3
    )

    donor_rows = []
    for donor, group in work.groupby("donor_id"):
        rho = spearmanr(
            group["graph_divergence"], group["absolute_delta"]
        ).statistic
        high = group.loc[
            group["divergence_quartile"] == 3, "absolute_delta"
        ].mean()
        low = group.loc[
            group["divergence_quartile"] == 0, "absolute_delta"
        ].mean()
        donor_rows.append(
            {
                "donor_id": donor,
                "spearman_abs_delta_vs_graph_divergence": float(rho),
                "divergent_minus_conserved_abs_delta": float(high - low),
            }
        )
    donor_frame = pd.DataFrame(donor_rows)
    rho, rho_low, rho_high = _bootstrap_mean(
        donor_frame["spearman_abs_delta_vs_graph_divergence"].to_numpy(),
        seed=seed,
        replicates=replicates,
    )
    delta, delta_low, delta_high = _bootstrap_mean(
        donor_frame["divergent_minus_conserved_abs_delta"].to_numpy(),
        seed=seed + 1,
        replicates=replicates,
    )
    summary: dict[str, float | int] = {
        "n_donors": int(len(donor_frame)),
        "donor_macro_spearman_abs_delta_vs_graph_divergence": rho,
        "spearman_ci95_lower": rho_low,
        "spearman_ci95_upper": rho_high,
        "positive_spearman_donors": int(
            (
                donor_frame[
                    "spearman_abs_delta_vs_graph_divergence"
                ]
                > 0
            ).sum()
        ),
        "divergent_minus_conserved_abs_delta": delta,
        "delta_ci95_lower": delta_low,
        "delta_ci95_upper": delta_high,
        "positive_delta_donors": int(
            (
                donor_frame["divergent_minus_conserved_abs_delta"] > 0
            ).sum()
        ),
    }
    quartile_rows = []
    for (donor, quartile), group in work.groupby(
        ["donor_id", "divergence_quartile"]
    ):
        quartile_rows.append(
            {
                "donor_id": donor,
                "divergence_quartile": int(quartile),
                "mean_absolute_methylation_delta": float(
                    group["absolute_delta"].mean()
                ),
            }
        )
    return pd.DataFrame(quartile_rows), {
        **summary,
        "per_donor": donor_rows,
    }


def _macro_model_rows(
    per_donor: pd.DataFrame, *, seed: int, replicates: int
) -> pd.DataFrame:
    rows = []
    for row_number, ((stratum, model), group) in enumerate(
        per_donor.groupby(["stratum", "model"], sort=False)
    ):
        for metric in ("direction_accuracy", "spearman"):
            estimate, lower, upper = _bootstrap_mean(
                group[metric].to_numpy(),
                seed=seed + row_number,
                replicates=replicates,
            )
            rows.append(
                {
                    "stratum": stratum,
                    "model": model,
                    "metric": metric,
                    "n_donors": int(group["donor_id"].nunique()),
                    "estimate": estimate,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def _plot(
    *,
    macro: pd.DataFrame,
    per_donor: pd.DataFrame,
    quartiles: pd.DataFrame,
    sequence_model: str,
    context_model: str,
    combined_model: str,
    model_labels: dict[str, str],
    out_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "font.family": "DejaVu Sans",
        }
    )
    colors = {
        sequence_model: "#6B7280",
        context_model: "#D97706",
        combined_model: "#1D4ED8",
    }
    figure, axes = plt.subplots(
        1, 3, figsize=(10.7, 3.05), constrained_layout=True
    )

    # A: observed functional difference increases with graph divergence.
    ax = axes[0]
    donor_pivot = quartiles.pivot(
        index="donor_id",
        columns="divergence_quartile",
        values="mean_absolute_methylation_delta",
    )
    for _, values in donor_pivot.iterrows():
        ax.plot(
            np.arange(4),
            values.to_numpy(),
            color="#CBD5E1",
            linewidth=0.7,
            alpha=0.8,
        )
    ax.plot(
        np.arange(4),
        donor_pivot.mean(axis=0).to_numpy(),
        color="#7C3AED",
        marker="o",
        linewidth=2.0,
        label="Donor macro-mean",
    )
    ax.set_xticks(np.arange(4), ["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("Within-donor graph-divergence quartile")
    ax.set_ylabel(r"Mean $|\Delta$ methylation$|$")
    ax.set_title("A  Functional divergence follows graph divergence", loc="left")
    ax.legend(frameon=False, loc="upper left")

    # B: direction accuracy as observed allelic effect grows.
    ax = axes[1]
    effect_strata = [
        "All pairs",
        r"$|\Delta m|\geq0.02$",
        r"$|\Delta m|\geq0.05$",
        r"$|\Delta m|\geq0.10$",
        "Within-donor top decile",
    ]
    offsets = (-0.20, 0.0, 0.20)
    for offset, model in zip(offsets, model_labels):
        frame = macro.loc[
            (macro["metric"] == "direction_accuracy")
            & (macro["model"] == model)
        ].set_index("stratum").loc[effect_strata]
        x = np.arange(len(effect_strata)) + offset
        ax.errorbar(
            x,
            frame["estimate"],
            yerr=np.vstack(
                [
                    frame["estimate"] - frame["ci95_lower"],
                    frame["ci95_upper"] - frame["estimate"],
                ]
            ),
            color=colors[model],
            marker="o",
            linewidth=1.4,
            capsize=2,
            label=model_labels[model],
        )
    ax.axhline(0.5, color="#94A3B8", linestyle="--", linewidth=0.8)
    ax.set_xticks(
        np.arange(len(effect_strata)),
        ["All", r"$\geq.02$", r"$\geq.05$", r"$\geq.10$", "Top 10%"],
    )
    ax.set_ylim(0.48, 0.64)
    ax.set_xlabel(r"Observed $|\Delta$ methylation$|$ stratum")
    ax.set_ylabel("Allelic-direction accuracy")
    ax.set_title("B  Signal strengthens with effect size", loc="left")
    ax.legend(frameon=False, loc="upper left")

    # C: donor-level marginal lift from the pretrained graph embedding.
    ax = axes[2]
    top = per_donor.loc[
        per_donor["stratum"] == "Within-donor top decile"
    ].pivot(
        index="donor_id", columns="model", values="direction_accuracy"
    )
    lift = (
        top[combined_model]
        - top[context_model]
    ).sort_values()
    y = np.arange(len(lift))
    bar_colors = np.where(lift.to_numpy() > 0, "#1D4ED8", "#CBD5E1")
    ax.barh(y, lift.to_numpy(), color=bar_colors, height=0.68)
    ax.axvline(0, color="#334155", linewidth=0.8)
    ax.set_yticks(y, lift.index)
    ax.set_xlabel("Graph + context minus context")
    ax.set_title("C  Marginal graph lift by held-out donor", loc="left")

    figure.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(figure)


def analyze(
    *,
    pairs_path: str | Path,
    predictions_path: str | Path,
    out_dir: str | Path,
    sequence_model: str = DEFAULT_SEQUENCE_MODEL,
    context_model: str = CONTEXT_MODEL,
    combined_model: str = COMBINED_MODEL,
    seed: int = 42,
    bootstrap_replicates: int = 20_000,
) -> dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_columns = [
        "donor_id",
        "methylation_delta",
        "node_jaccard",
    ]
    pairs = pd.read_csv(
        pairs_path,
        compression="infer",
        usecols=pair_columns,
        low_memory=False,
    ).reset_index(names="row_index")
    predictions = pd.read_csv(predictions_path, compression="infer")
    model_labels = {
        sequence_model: (
            "Sequence"
            if sequence_model == DEFAULT_SEQUENCE_MODEL
            else "NT + whole-window sequence"
        ),
        context_model: "Sequence + annotation",
        combined_model: "Sequence + graph + annotation",
    }
    missing_models = set(model_labels) - set(predictions["model"])
    if missing_models:
        raise ValueError(f"Predictions lack required models: {missing_models}")
    predictions = predictions.loc[
        predictions["model"].isin(model_labels),
        ["row_index", "donor_id", "model", "y_true", "y_pred"],
    ]
    merged = predictions.merge(
        pairs,
        on=["row_index", "donor_id"],
        how="inner",
        validate="many_to_one",
    )
    if len(merged) != len(predictions):
        raise ValueError("Predictions and pair rows do not join exactly.")
    if not np.allclose(
        merged["y_true"], merged["methylation_delta"], atol=1e-6
    ):
        raise ValueError("Prediction targets do not match the pair table.")

    strata = _define_strata(pairs)
    per_donor_rows: list[dict[str, float | int | str]] = []
    for stratum, mask in strata.items():
        selected_rows = set(pairs.loc[mask, "row_index"].astype(int))
        frame = merged.loc[merged["row_index"].isin(selected_rows)]
        per_donor_rows.extend(_per_donor_model_metrics(frame, stratum))
    per_donor = pd.DataFrame(per_donor_rows)
    macro = _macro_model_rows(
        per_donor, seed=seed, replicates=bootstrap_replicates
    )

    comparison_rows = []
    for stratum in strata:
        for metric in ("direction_accuracy", "spearman"):
            for baseline in (
                sequence_model,
                context_model,
            ):
                comparison_rows.append(
                    _paired_model_bootstrap(
                        per_donor,
                        stratum=stratum,
                        metric=metric,
                        baseline=baseline,
                        candidate=combined_model,
                        seed=seed + len(comparison_rows),
                        replicates=bootstrap_replicates,
                    )
                )
    comparisons = pd.DataFrame(comparison_rows)
    quartiles, structural = _structural_association(
        pairs, seed=seed, replicates=bootstrap_replicates
    )

    per_donor.to_csv(out_dir / "per_donor_stratified_metrics.csv", index=False)
    macro.to_csv(out_dir / "macro_stratified_metrics.csv", index=False)
    comparisons.to_csv(
        out_dir / "paired_donor_bootstrap_differences.csv", index=False
    )
    quartiles.to_csv(
        out_dir / "graph_divergence_quartile_metrics.csv", index=False
    )
    (out_dir / "structural_biological_association.json").write_text(
        json.dumps(structural, indent=2), encoding="utf-8"
    )
    plot_path = out_dir / "haplotype_biological_evidence.png"
    _plot(
        macro=macro,
        per_donor=per_donor,
        quartiles=quartiles,
        sequence_model=sequence_model,
        context_model=context_model,
        combined_model=combined_model,
        model_labels=model_labels,
        out_path=plot_path,
    )

    summary: dict[str, object] = {
        "analysis": "locked_prediction_haplotype_biological_audit",
        "post_hoc_descriptive": True,
        "no_retraining_or_model_selection_from_effect_strata": True,
        "n_pairs": int(len(pairs)),
        "n_donors": int(pairs["donor_id"].nunique()),
        "models": model_labels,
        "bootstrap_unit": "held-out donor",
        "bootstrap_replicates": int(bootstrap_replicates),
        "structural_biological_association": structural,
        "macro_stratified_metrics": macro.to_dict(orient="records"),
        "paired_donor_comparisons": comparisons.to_dict(orient="records"),
        "figure": str(plot_path),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--sequence-model", default=DEFAULT_SEQUENCE_MODEL
    )
    parser.add_argument("--context-model", default=CONTEXT_MODEL)
    parser.add_argument("--combined-model", default=COMBINED_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    args = parser.parse_args()
    result = analyze(
        pairs_path=args.pairs,
        predictions_path=args.predictions,
        out_dir=args.out_dir,
        sequence_model=args.sequence_model,
        context_model=args.context_model,
        combined_model=args.combined_model,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
