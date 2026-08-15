#!/usr/bin/env python3
"""Plot the frozen complexity definition and performance-independent examples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/private/tmp/pangenomefm-mpl"))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = {"low": "#2F6FBB", "medium": "#C9821A", "high": "#B94A62"}
COMPONENT_LABELS = {
    "complexity_component__log1p_edges_per_kb": "edge density",
    "complexity_component__branching_fraction": "branching",
    "complexity_component__log1p_max_degree": "maximum degree",
    "complexity_component__log1p_cycle_rank_per_kb": "cycle rank",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT
        / "results/complexity/graph_window_complexity_v1/complexity_features.tsv",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=ROOT
        / "results/complexity/graph_window_complexity_v1/complexity_thresholds.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "paper/figures/next_stage",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.features, sep="\t")
    core = data.loc[data["context"].eq("strict")].copy()
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    representatives = []
    for category in ("low", "medium", "high"):
        group = core.loc[core["locus_complexity_category"].eq(category)].copy()
        median = group["locus_complexity_score"].median()
        representative = group.loc[
            (group["locus_complexity_score"] - median).abs().idxmin()
        ].copy()
        representative["selection_rule"] = (
            "strict/core window closest to within-category median score; model performance unused"
        )
        representatives.append(representative)
    reps = pd.DataFrame(representatives)

    source_out = args.features.parent / "representative_complexity_windows.tsv"
    reps.to_csv(source_out, sep="\t", index=False)

    fig, (ax_score, ax_components) = plt.subplots(
        1, 2, figsize=(12.6, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    fig.subplots_adjust(wspace=0.32, top=0.82, bottom=0.20, left=0.07, right=0.98)
    fig.suptitle(
        "Graph-window complexity v1 is frozen without model performance",
        fontsize=16,
        weight="bold",
        x=0.07,
        ha="left",
    )

    ordered = core.sort_values("locus_complexity_score").reset_index(drop=True)
    for category in ("low", "medium", "high"):
        subset = ordered.loc[ordered["locus_complexity_category"].eq(category)]
        ax_score.scatter(
            subset.index,
            subset["locus_complexity_score"],
            s=18,
            color=COLORS[category],
            alpha=0.8,
            label=f"{category} (n={len(subset)})",
        )
    low_threshold = thresholds["thresholds"]["low_to_medium"]
    high_threshold = thresholds["thresholds"]["medium_to_high"]
    for threshold in (low_threshold, high_threshold):
        ax_score.axhline(threshold, color="#52606D", lw=1.0, ls="--")
    ax_score.set_title("A  Core-window score and frozen tertiles", loc="left", weight="bold")
    ax_score.set_xlabel("Strict/core windows sorted by score")
    ax_score.set_ylabel("Robust composite complexity score")
    ax_score.grid(axis="y", alpha=0.2)
    ax_score.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")

    component_columns = list(COMPONENT_LABELS)
    matrix = reps[component_columns].to_numpy(dtype=float)
    image = ax_components.imshow(matrix, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax_components.set_title(
        "B  Performance-independent representative windows", loc="left", weight="bold"
    )
    ax_components.set_xticks(range(len(component_columns)))
    ax_components.set_xticklabels(
        [COMPONENT_LABELS[column] for column in component_columns],
        rotation=32,
        ha="right",
        fontsize=8,
    )
    ax_components.set_yticks(range(3))
    ax_components.set_yticklabels(
        [
            f"{row.locus_complexity_category}\n{row.chromosome}:{row.start / 1e6:.2f}–{row.end / 1e6:.2f} Mb"
            for row in reps.itertuples(index=False)
        ],
        fontsize=8.5,
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax_components.text(
                j,
                i,
                f"{matrix[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(matrix[i, j]) > 1.3 else "#17212B",
            )
    colorbar = fig.colorbar(image, ax=ax_components, fraction=0.045, pad=0.03)
    colorbar.set_label("Robust normalized component", fontsize=8)
    fig.text(
        0.07,
        0.035,
        "Score = equal-weight mean of robust-normalized edge density, branching fraction, maximum degree, and cycle-rank density.\nThe strict/core locus label is joined to expanded contexts; no prediction column is read. SR-based alternate fraction is reported but has zero fit-population variance.",
        fontsize=8.5,
        color="#607080",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / "complexity_framework_draft"
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.{{svg,pdf,png}}")
    print(f"wrote {source_out}")


if __name__ == "__main__":
    main()
