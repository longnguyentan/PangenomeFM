#!/usr/bin/env python3
"""Generate paper-review figures from verified machine-readable outputs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pangenomefm-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


COLORS = {
    "full": "#2D6A9F",
    "graph": "#4A9D78",
    "coordinate": "#C47A3A",
    "nogate": "#8B6BB1",
    "sequence": "#D99A2B",
    "serialized": "#8B6BB1",
    "structural": "#6B7280",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_task_definitions(out_dir: Path, source_dir: Path) -> None:
    pd.DataFrame(
        [
            {
                "panel": "unmasked",
                "scientific_label": "target-exposed structural graph",
                "query_edge_visible_to_encoder": True,
                "primary_evidence": False,
            },
            {
                "panel": "core",
                "scientific_label": "core-node-induced context",
                "query_edge_visible_to_encoder": False,
                "primary_evidence": True,
            },
            {
                "panel": "expanded",
                "scientific_label": "endpoint-expanded-induced context",
                "query_edge_visible_to_encoder": False,
                "primary_evidence": True,
            },
        ]
    ).to_csv(source_dir / "figure1_task_definitions.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.15))
    titles = [
        "A  Unmasked structural graph\n(target visible; invalid headline)",
        "B  Core-node induced context\n(query edge masked)",
        "C  Endpoint-expanded context\n(query edge masked)",
    ]
    descriptions = [
        "Positive target edge remains in\nmessage-passing graph",
        "Visible nodes overlap target\ncoordinate window",
        "Add every endpoint touching a core\nnode; induce all retained-node edges",
    ]
    for axis, title, description in zip(axes, titles, descriptions):
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlim(-0.5, 4.5)
        axis.set_ylim(-1.1, 2.2)
        axis.axis("off")
        axis.text(2.0, -0.88, description, ha="center", va="top", fontsize=8)

    positions = np.asarray([[0, 0], [1, 0.65], [2, 0], [3, 0.75], [4, 0]])
    core = {1, 2, 3}
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)]
    target = (2, 3)
    for panel, axis in enumerate(axes):
        visible = set(range(5)) if panel in {0, 2} else core
        for left, right in edges:
            if left not in visible or right not in visible:
                continue
            is_target = (left, right) == target
            if is_target and panel > 0:
                axis.plot(
                    positions[[left, right], 0],
                    positions[[left, right], 1],
                    color="#C83E4D",
                    lw=2.3,
                    ls="--",
                    alpha=0.75,
                )
                axis.text(2.35, 0.45, "masked query", color="#C83E4D", fontsize=7)
            else:
                axis.plot(
                    positions[[left, right], 0],
                    positions[[left, right], 1],
                    color="#425466" if not is_target else "#C83E4D",
                    lw=2.0 if is_target else 1.35,
                    zorder=1,
                )
        for index, (x, y) in enumerate(positions):
            if index not in visible:
                continue
            color = "#E8F0F7" if index in core else "#DCEFE4"
            axis.scatter(x, y, s=360, color=color, edgecolor="#31465A", zorder=3)
            axis.text(x, y, f"v{index}", ha="center", va="center", fontsize=8, zorder=4)
        axis.add_patch(
            FancyBboxPatch(
                (0.65, -0.35),
                2.7,
                1.45,
                boxstyle="round,pad=0.08",
                edgecolor="#6A8CAF",
                facecolor="none",
                linestyle=":",
                linewidth=1.2,
            )
        )
        axis.text(0.76, 1.18, "target interval", fontsize=7, color="#557A9E")
    fig.suptitle(
        "Graph-theoretic task definitions: query edges must be removed before encoding",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    _save(fig, out_dir, "figure1_task_definitions")


def figure_link_results(review_root: Path, out_dir: Path, source_dir: Path) -> None:
    original_metrics = pd.read_csv(
        review_root / "masked_edge_statistics" / "overall_metrics.csv"
    )
    multiseed_root = review_root / "matched_training_v2" / "multiseed_fixed_split"
    metrics = pd.read_csv(multiseed_root / "aggregate_metrics.csv")
    chromosome = pd.read_csv(
        multiseed_root / "aggregate_chromosome_metrics.csv"
    )
    external = pd.read_csv(review_root / "external_hprc_to_hgsvc_statistics" / "overall_metrics.csv")
    context = pd.read_csv(
        review_root / "context" / "hprc_matched_nonoverlap_v2" / "context_summary_by_regime.csv"
    )
    coverage = pd.read_csv(
        review_root / "context" / "hprc_matched_nonoverlap_v2" / "window_coverage.csv"
    )
    metrics.to_csv(source_dir / "figure2_multiseed_metrics.csv", index=False)
    chromosome.to_csv(source_dir / "figure2_chromosome_metrics.csv", index=False)
    pd.concat(
        [
            original_metrics.assign(source="existing_hprc_checkpoint"),
            external.assign(source="frozen_hprc_to_hgsvc3"),
        ],
        ignore_index=True,
    ).to_csv(source_dir / "figure2_transfer_metrics.csv", index=False)
    context.to_csv(source_dir / "figure2_context_exposure.csv", index=False)
    coverage.to_csv(source_dir / "figure2_window_coverage.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.8))
    axis = axes[0, 0]
    order = ["full", "graph", "coordinate", "nogate"]
    widths = 0.18
    for closure_index, closure in enumerate(["strict", "1hop"]):
        subset = metrics[metrics["closure"] == closure].copy()
        subset.index = subset["model"].str.replace(f"_{closure}", "", regex=False)
        values = [subset.loc[name, "auroc_mean"] for name in order]
        errors = [subset.loc[name, "auroc_sd"] for name in order]
        x = np.arange(len(order)) + (closure_index - 0.5) * widths
        axis.bar(
            x,
            values,
            widths,
            yerr=errors,
            capsize=2,
            color=[COLORS[name] for name in order],
            alpha=1.0 if closure == "1hop" else 0.62,
            label="Core induced" if closure == "strict" else "Endpoint expanded",
        )
    axis.set_xticks(np.arange(len(order)), ["Full", "Graph", "Coordinate", "No gate"])
    axis.set_ylim(0.5, 0.98)
    axis.set_ylabel("AUROC")
    axis.set_title(
        "A  Locus-matched, non-overlap HPRC ablations",
        loc="left",
        fontweight="bold",
    )
    axis.legend(frameon=False, fontsize=8)
    axis.text(
        0.02,
        0.02,
        "three seeds; error bars = training-seed SD\n"
        "fixed candidate split; 30 core / 37 expanded slices",
        transform=axis.transAxes,
        fontsize=7,
    )

    axis = axes[0, 1]
    full = original_metrics[original_metrics["model"].str.startswith("full_")].copy()
    full["cohort"] = "HPRC held-out chromosomes"
    external = external.copy()
    external["cohort"] = "Frozen HPRC → HGSVC3"
    combined = pd.concat([full, external], ignore_index=True)
    combined["regime"] = combined["closure"].map(
        {"strict": "Core induced", "1hop": "Endpoint expanded"}
    )
    for index, cohort_name in enumerate(combined["cohort"].unique()):
        subset = combined[combined["cohort"] == cohort_name].set_index("regime")
        x = np.arange(2) + (index - 0.5) * 0.28
        vals = [subset.loc[name, "auroc"] for name in ["Core induced", "Endpoint expanded"]]
        low = [subset.loc[name, "auroc"] - subset.loc[name, "auroc_ci95_low"] for name in ["Core induced", "Endpoint expanded"]]
        high = [subset.loc[name, "auroc_ci95_high"] - subset.loc[name, "auroc"] for name in ["Core induced", "Endpoint expanded"]]
        axis.bar(x, vals, 0.28, yerr=[low, high], capsize=2, label=cohort_name)
    axis.axhline(0.5, color="#777777", ls=":", lw=1)
    axis.set_xticks(np.arange(2), ["Core induced", "Endpoint\nexpanded"])
    axis.set_ylim(0.5, 1.0)
    axis.set_ylabel("AUROC")
    axis.set_title(
        "B  Existing-checkpoint frozen transfer",
        loc="left",
        fontweight="bold",
    )
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    full_chr = chromosome[chromosome["model"].str.startswith("full_")].copy()
    chromosome_order = ["chr1", "chr8", "chr19", "chrY"]
    for closure, marker in [("strict", "o"), ("1hop", "s")]:
        subset = full_chr[full_chr["closure"] == closure].set_index("chromosome")
        axis.errorbar(
            chromosome_order,
            [subset.loc[name, "auroc_mean"] for name in chromosome_order],
            yerr=[subset.loc[name, "auroc_sd"] for name in chromosome_order],
            marker=marker,
            capsize=2,
            label="Core induced" if closure == "strict" else "Endpoint expanded",
        )
    axis.set_ylim(0.74, 0.98)
    axis.set_ylabel("AUROC")
    axis.set_title(
        "C  Three-seed held-out chromosome variation",
        loc="left",
        fontweight="bold",
    )
    axis.legend(frameon=False, fontsize=8, loc="lower right")

    axis = axes[1, 1]
    context = context.set_index("closure_legacy_label")
    x = np.arange(2)
    nodes = [context.loc[name, "mean_visible_nodes"] for name in ["strict", "1hop"]]
    edges = [context.loc[name, "mean_visible_edges"] for name in ["strict", "1hop"]]
    axis.bar(x - 0.17, nodes, 0.34, label="Visible nodes", color="#6A8CAF")
    axis.bar(x + 0.17, edges, 0.34, label="Visible edge rows", color="#74A57F")
    axis.set_xticks(x, ["Core induced", "Endpoint\nexpanded"])
    axis.set_ylabel("Mean count per slice")
    axis.set_title("D  Context is not exposure matched", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.02, 0.69))
    unique_ratio = coverage.groupby("closure")[["nominal_bp", "unique_bp"]].sum()
    axis.text(
        0.02,
        0.96,
        "240 exact interval pairs; zero within-regime overlap\n"
        f"Unique/nominal bp: core {unique_ratio.loc['strict','unique_bp']/unique_ratio.loc['strict','nominal_bp']:.1%}, "
        f"expanded {unique_ratio.loc['1hop','unique_bp']/unique_ratio.loc['1hop','nominal_bp']:.1%}",
        transform=axis.transAxes,
        va="top",
        fontsize=7.5,
    )
    fig.suptitle(
        "Masked graph-link prediction: matched loci support topology, not a broad foundation claim",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, out_dir, "figure2_masked_link_prediction")


def figure_ccre(
    review_root: Path, repository_root: Path, out_dir: Path, source_dir: Path
) -> None:
    summary = json.loads(
        (repository_root / "results/analysis/ccre_matched_window_gat_block_stats/summary.json").read_text()
    )
    model_metrics = pd.DataFrame(summary["metrics"])
    span_summary = json.loads((review_root / "ccre_graph_span" / "summary.json").read_text())
    strata = pd.read_csv(review_root / "ccre_matched_strata" / "stratified_metrics.csv")
    model_metrics.to_csv(source_dir / "figure3_model_metrics.csv", index=False)
    strata.to_csv(source_dir / "figure3_stratified_metrics.csv", index=False)
    (source_dir / "figure3_ccre_span_summary.json").write_text(
        json.dumps(span_summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.55))
    axis = axes[0]
    labels = ["frozen_gat", "sequence", "serialized", "structural"]
    subset = model_metrics.set_index("model").loc[labels]
    x = np.arange(len(labels))
    axis.bar(x - 0.17, subset["auroc"], 0.34, label="AUROC", color="#2D6A9F")
    axis.bar(x + 0.17, subset["auprc"], 0.34, label="AUPRC", color="#74A57F")
    axis.set_xticks(x, ["Frozen\nGraphFM", "3-mer\nsequence", "Serialized\ngraph", "Structural"])
    axis.set_ylim(0.35, 0.75)
    axis.set_title("A  Exact matched universe (n=1,322)", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    multi = span_summary["fraction_spanning_multiple_nodes"]
    axis.bar([0, 1], [1 - multi, multi], color=["#BFD7EA", "#2D6A9F"])
    axis.set_xticks([0, 1], ["One graph\nnode", "Multiple graph\nnodes"])
    axis.set_ylabel("Fraction of ENCODE cCREs")
    axis.set_title("B  Reference-path graph span", loc="left", fontweight="bold")
    axis.text(1, multi + 0.03, f"{multi:.2%}\n(31,966)", ha="center", fontsize=8)

    axis = axes[2]
    length = strata[(strata["stratum"] == "node_length") & (strata["model"] == "frozen_gat")]
    order = ["<=10", "11-50", "51-200", "201-1000", ">1000"]
    length = length.set_index("stratum_value").loc[order]
    axis.plot(order, length["auroc"], marker="o", color="#2D6A9F")
    for x_value, (_, row) in enumerate(length.iterrows()):
        axis.text(x_value, row["auroc"] + 0.018, f"n={int(row['n'])}", ha="center", fontsize=7)
    axis.axhline(0.5, color="#777777", ls=":", lw=1)
    axis.set_ylim(0.43, 0.87)
    axis.set_xlabel("Graph-node length (bp)")
    axis.set_ylabel("Frozen GraphFM AUROC")
    axis.set_title("C  Performance depends on node length", loc="left", fontweight="bold")
    fig.suptitle("cCRE evidence is preliminary and restricted to five genomic blocks", fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "figure3_ccre_matched_analysis")


def figure_haplotype(repository_root: Path, out_dir: Path, source_dir: Path) -> None:
    base = repository_root / "results/hprc/haplotype_methylation_chr8_cohort_v2"
    hierarchy = json.loads(
        (base / "hierarchical_donor_histgb_sequence_antisymmetric_v1/summary.json").read_text()
    )
    macro = pd.DataFrame(hierarchy["macro_metrics"]).set_index("model")
    marginal = pd.read_csv(
        base / "nonlinear_incremental_residual_v1/paired_graph_annotation_vs_annotation.csv"
    )
    association = json.loads((base / "adjusted_biological_audit_v1/summary.json").read_text())
    per_donor = pd.DataFrame(hierarchy["per_group_metrics"])
    macro.reset_index().to_csv(source_dir / "figure4_macro_metrics.csv", index=False)
    marginal.to_csv(source_dir / "figure4_marginal_graph_effects.csv", index=False)
    per_donor.to_csv(source_dir / "figure4_per_donor_metrics.csv", index=False)
    (source_dir / "figure4_adjusted_association.json").write_text(
        json.dumps(association, indent=2), encoding="utf-8"
    )

    models = [
        "whole_window_antisymmetric_histgb",
        "hierarchical_annotation_only_shrunk_residual",
        "hierarchical_graph_context_shrunk_residual",
    ]
    labels = ["Whole-window\nsequence", "Sequence +\nannotation", "Sequence + graph\n+ annotation"]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.5))
    axis = axes[0, 0]
    axis.bar(labels, macro.loc[models, "spearman"], color=["#D99A2B", "#6B7280", "#2D6A9F"])
    axis.set_ylim(0.30, 0.33)
    axis.set_ylabel("Donor-macro Spearman")
    axis.set_title("A  Donor-held-out methylation delta", loc="left", fontweight="bold")

    axis = axes[0, 1]
    axis.bar(labels, macro.loc[models, "mae"], color=["#D99A2B", "#6B7280", "#2D6A9F"])
    axis.set_ylim(0.0458, 0.0470)
    axis.set_ylabel("MAE (lower is better)")
    axis.set_title("B  Graph context does not beat annotation", loc="left", fontweight="bold")

    axis = axes[1, 0]
    metric_order = ["spearman", "mae", "r2", "direction_accuracy"]
    marginal = marginal.set_index("metric").loc[metric_order]
    scale = pd.Series({"spearman": 1, "mae": 100, "r2": 1, "direction_accuracy": 1})
    point = marginal["difference"] * scale
    low = marginal["ci95_lower"] * scale
    high = marginal["ci95_upper"] * scale
    x = np.arange(len(metric_order))
    axis.errorbar(x, point, yerr=[point - low, high - point], fmt="o", capsize=3, color="#2D6A9F")
    axis.axhline(0, color="#777777", ls=":", lw=1)
    axis.set_xticks(x, ["Spearman", "MAE ×100", "R²", "Direction"])
    axis.set_ylabel("Graph marginal improvement")
    axis.set_title("C  Increment beyond annotations is tiny", loc="left", fontweight="bold")

    axis = axes[1, 1]
    selected = per_donor[per_donor["model"].isin(models)].copy()
    for model, label, color in zip(models, labels, ["#D99A2B", "#6B7280", "#2D6A9F"]):
        group = selected[selected["model"] == model].sort_values("heldout_group")
        axis.plot(group["heldout_group"], group["spearman"], marker="o", ms=3, label=label.replace("\n", " "), color=color)
    axis.tick_params(axis="x", rotation=55)
    axis.set_ylabel("Spearman")
    axis.set_title("D  All ten donors are positive", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=7)
    coefficient = association["standardized_graph_divergence_coefficient"]
    fig.suptitle(
        "Haplotype methylation is a chr8 pilot; graph-divergence association is reproducible but not predictive gain\n"
        f"Adjusted divergence coefficient {coefficient['donor_macro_mean']:.4f} "
        f"(95% CI {coefficient['ci95_lower']:.4f}–{coefficient['ci95_upper']:.4f}; 10/10 donors positive)",
        fontweight="bold",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, out_dir, "figure4_haplotype_methylation_pilot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, default=Path("results/review_20260804"))
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures/review_20260804"))
    args = parser.parse_args()
    args.repository_root = args.repository_root.resolve()
    args.review_root = args.review_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.review_root / "figure_source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    _style()
    figure_task_definitions(args.out_dir, source_dir)
    figure_link_results(args.review_root, args.out_dir, source_dir)
    figure_ccre(args.review_root, args.repository_root, args.out_dir, source_dir)
    figure_haplotype(args.repository_root, args.out_dir, source_dir)
    print(f"wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
