#!/usr/bin/env python3
"""Build new main and supplementary manuscript figures from audited tables."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pangenomefm-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


STRICT = "#28679D"
ONE_HOP = "#E96B4A"
GRAY = "#7A8288"
LIGHT = "#E5E9EC"
TEAL = "#168F82"
PURPLE = "#7454B8"
BLACK = "#1D2428"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "legend.fontsize": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        -0.08,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
    )
    ax.set_title(title, loc="left", pad=13)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for suffix, kwargs in [
        ("pdf", {}),
        ("png", {"dpi": 360}),
    ]:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths[suffix] = str(path.resolve())
    plt.close(fig)
    return paths


def context_legend(ax: plt.Axes, *, location: str = "best") -> None:
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=STRICT, lw=0, label="Strict context"),
            Line2D([0], [0], marker="D", color=ONE_HOP, lw=0, label="One-hop context"),
        ],
        loc=location,
        frameon=False,
        ncol=2,
        handletextpad=0.5,
        columnspacing=1.3,
    )


def dot_interval(
    ax: plt.Axes,
    *,
    mean: float,
    low: float,
    high: float,
    y: float,
    color: str,
    marker: str,
    size: float = 6.5,
) -> None:
    ax.errorbar(
        mean,
        y,
        xerr=[[mean - low], [high - mean]],
        fmt=marker,
        ms=size,
        color=color,
        ecolor=color,
        elinewidth=1.25,
        capsize=2.5,
        zorder=4,
    )


def source_schematic(ax: plt.Axes, *, task: str) -> None:
    panel(
        ax,
        "a",
        "Frozen sources and held-out prediction",
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    if task == "cCRE":
        ax.plot([0.5, 4.0], [4.1, 4.1], color=GRAY, lw=2)
        ax.add_patch(FancyBboxPatch((1.55, 3.72), 1.15, 0.76, boxstyle="round,pad=.06", fc="#EAF1F8", ec=STRICT, lw=1.4))
        ax.text(2.12, 4.10, "candidate\nregulatory segment", ha="center", va="center", fontsize=8.5, color=STRICT)
        task_text = "cCRE class\nprobability"
    else:
        nodes = [(0.6, 4.1), (1.55, 4.1), (2.5, 4.1), (3.45, 4.1)]
        for left, right in zip(nodes[:-1], nodes[1:]):
            ax.plot([left[0], right[0]], [left[1], right[1]], color=TEAL, lw=1.6)
        ax.plot([1.55, 2.05, 2.95, 3.45], [4.1, 3.25, 3.25, 4.1], color=ONE_HOP, lw=1.6)
        for x, y in nodes + [(2.05, 3.25), (2.95, 3.25)]:
            ax.add_patch(Circle((x, y), 0.12, fc="white", ec=TEAL, lw=1.4))
        ax.text(2.48, 2.83, "alternate traversal / breakpoint context", ha="center", fontsize=8.3, color=ONE_HOP)
        task_text = "breakpoint-pair\nprobability"

    boxes = [
        (0.55, 1.15, "C", "genomic\ncoordinates", GRAY),
        (3.10, 1.15, "S", "frozen sequence\nrepresentation", STRICT),
        (5.65, 1.15, "T", "frozen graph\nrepresentation", TEAL),
    ]
    for x, y, symbol, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 1.75, 1.12, boxstyle="round,pad=.08", fc="white", ec=color, lw=1.6))
        ax.text(x + 0.35, y + 0.56, symbol, color=color, fontsize=15, fontweight="bold", ha="center", va="center")
        ax.text(x + 1.05, y + 0.56, label, color=BLACK, fontsize=8.3, ha="center", va="center")
    for x in [2.35, 4.90, 8.00]:
        ax.add_patch(FancyArrowPatch((x, 1.72), (x + 0.56, 1.72), arrowstyle="-|>", mutation_scale=12, lw=1.1, color=GRAY))
    ax.add_patch(FancyBboxPatch((8.55, 1.13), 1.30, 1.16, boxstyle="round,pad=.08", fc="#FAF5F3", ec=ONE_HOP, lw=1.5))
    ax.text(9.20, 1.72, task_text, ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text(5.05, 0.47, "Only the downstream logistic classifier is fitted; both pretrained encoders remain frozen.", ha="center", color=GRAY, fontsize=8.4)


def absolute_panel(ax: plt.Axes, absolute: pd.DataFrame, task: str, letter: str) -> None:
    title = "Candidate regulatory-region classification" if task == "cCRE" else "Structural-variant breakpoint classification"
    panel(ax, letter, title)
    subset = absolute.loc[(absolute["task"] == task) & (absolute["metric"] == "auprc")]
    suffix = "_pair" if task == "SV" else ""
    base_name = f"coordinate_plus_frozen_sequence_fm{suffix}"
    full_name = f"coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm{suffix}"
    base = subset.loc[subset["feature_set"].eq(base_name)].iloc[0]
    strict = subset.loc[(subset["feature_set"].eq(full_name)) & subset["closure"].eq("strict")].iloc[0]
    onehop = subset.loc[(subset["feature_set"].eq(full_name)) & subset["closure"].eq("1hop")].iloc[0]
    rows = [
        ("Coordinates + sequence", base, GRAY, "o"),
        ("+ graph representation\n(strict context)", strict, STRICT, "o"),
        ("+ graph representation\n(one-hop context)", onehop, ONE_HOP, "D"),
    ]
    y = np.arange(3)[::-1]
    lower = min(float(row[1]["ci95_low"]) for row in rows)
    upper = max(float(row[1]["ci95_high"]) for row in rows)
    pad = max(0.006, (upper - lower) * 0.35)
    for yi, (_, row, color, marker) in zip(y, rows):
        dot_interval(
            ax,
            mean=float(row["mean"]),
            low=float(row["ci95_low"]),
            high=float(row["ci95_high"]),
            y=float(yi),
            color=color,
            marker=marker,
            size=7,
        )
        ax.text(float(row["mean"]) + pad * 0.12, yi + 0.17, f'{float(row["mean"]):.4f}', color=color, fontsize=8.6)
    ax.set_yticks(y, [row[0] for row in rows])
    ax.set_xlim(lower - pad, upper + pad)
    ax.grid(axis="x", color=LIGHT, lw=0.8)
    ax.set_xlabel("Mean chromosome-held-out AUPRC")
    ax.text(0, -0.25, "95% hierarchical bootstrap intervals; five chromosome folds × three seeds", transform=ax.transAxes, fontsize=7.8, color=GRAY)


CONTRAST_LABELS = {
    "frozen_sequence_fm_vs_kmer_composition": "Sequence representation\nover k-mer composition",
    "topology_given_coordinate_and_frozen_sequence_fm": "Graph representation beyond\ncoordinates + sequence",
    "frozen_sequence_fm_given_coordinate_and_topology": "Sequence representation beyond\ncoordinates + graph",
}


def contribution_panel(ax: plt.Axes, contributions: pd.DataFrame, task: str, letter: str) -> None:
    title = "Independent information-source contributions for cCRE" if task == "cCRE" else "Independent information-source contributions for structural variants"
    panel(ax, letter, title)
    subset = contributions.loc[(contributions["task"] == task) & contributions["contrast"].isin(CONTRAST_LABELS)].copy()
    order = list(CONTRAST_LABELS)
    ys = np.arange(len(order))[::-1]
    for y, contrast in zip(ys, order):
        rows = subset.loc[subset["contrast"].eq(contrast)]
        for closure, color, marker, offset in [
            ("strict", STRICT, "o", 0.12),
            ("1hop", ONE_HOP, "D", -0.12),
        ]:
            row = rows.loc[rows["closure"].eq(closure)].iloc[0]
            dot_interval(
                ax,
                mean=float(row["mean_gain"]),
                low=float(row["ci95_low"]),
                high=float(row["ci95_high"]),
                y=float(y + offset),
                color=color,
                marker=marker,
            )
            if contrast == "topology_given_coordinate_and_frozen_sequence_fm":
                ax.text(
                    float(row["mean_gain"]) + 0.007,
                    y + offset,
                    f'{float(row["mean_gain"]):+.4f}',
                    va="center",
                    color=color,
                    fontsize=8.0,
                )
    ax.axvline(0, color=GRAY, lw=1, ls="--")
    ax.set_yticks(ys, [CONTRAST_LABELS[key] for key in order])
    ax.grid(axis="x", color=LIGHT, lw=0.8)
    ax.set_xlabel("Paired ΔAUPRC (positive favors added information)")
    context_legend(ax, location="lower right")
    ax.text(0, -0.25, "95% hierarchical bootstrap intervals; comparisons use identical examples", transform=ax.transAxes, fontsize=7.8, color=GRAY)


def ccre_strata_panel(ax_left: plt.Axes, ax_right: plt.Axes, gains: pd.DataFrame | None) -> None:
    panel(ax_left, "d", "Where does the cCRE graph contribution occur?")
    if gains is None or gains.empty:
        for ax in (ax_left, ax_right):
            ax.axis("off")
        ax_left.text(0.02, 0.55, "Stratified results are generated after the server-side prediction audit.", transform=ax_left.transAxes, color=GRAY)
        return
    settings = [
        (ax_left, "ENCODE_cCRE_subtype_vs_background", ["PLS", "pELS", "dELS", "CTCF-only"], "ENCODE class vs background"),
        (ax_right, "native_graph_complexity_tertile", ["low", "medium", "high"], "Native-graph complexity tertile"),
    ]
    for ax, kind, order, subtitle in settings:
        subset = gains.loc[gains["stratification"].eq(kind)]
        ys = np.arange(len(order))[::-1]
        for y, stratum in zip(ys, order):
            rows = subset.loc[subset["stratum"].eq(stratum)]
            for closure, color, marker, offset in [
                ("strict", STRICT, "o", 0.12),
                ("1hop", ONE_HOP, "D", -0.12),
            ]:
                row = rows.loc[rows["closure"].eq(closure)]
                if row.empty:
                    continue
                row = row.iloc[0]
                dot_interval(
                    ax,
                    mean=float(row["mean_gain"]),
                    low=float(row["ci95_low"]),
                    high=float(row["ci95_high"]),
                    y=float(y + offset),
                    color=color,
                    marker=marker,
                    size=5.8,
                )
        ax.axvline(0, color=GRAY, lw=1, ls="--")
        ax.set_yticks(ys, order)
        ax.set_title(subtitle, fontsize=10.5, pad=8)
        ax.grid(axis="x", color=LIGHT, lw=0.8)
        ax.set_xlabel("Graph contribution, paired ΔAUPRC")


def build_ccre(absolute: pd.DataFrame, contributions: pd.DataFrame, gains: pd.DataFrame | None, out: Path) -> dict[str, str]:
    fig = plt.figure(figsize=(13.4, 9.3))
    grid = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.48)
    source_schematic(fig.add_subplot(grid[0, 0]), task="cCRE")
    absolute_panel(fig.add_subplot(grid[0, 1]), absolute, "cCRE", "b")
    contribution_panel(fig.add_subplot(grid[1, 0]), contributions, "cCRE", "c")
    subgrid = grid[1, 1].subgridspec(1, 2, wspace=0.58)
    ccre_strata_panel(fig.add_subplot(subgrid[0, 0]), fig.add_subplot(subgrid[0, 1]), gains)
    fig.suptitle("Graph-native representations complement sequence for regulatory prediction", fontsize=16, fontweight="bold", y=0.995)
    return save_figure(fig, out, "figure3_ccre")


def sv_strata_panel(ax: plt.Axes, strata: pd.DataFrame, *, stratum: str, letter: str, title: str) -> None:
    panel(ax, letter, title)
    subset = strata.loc[strata["stratum"].eq(stratum)].copy()
    if stratum == "allele_frequency_bin":
        replacements = {
            "af[0.001,0.01)": "0.001–0.01",
            "af[0.01,0.05)": "0.01–0.05",
            "af[0.05,0.5)": "0.05–0.5",
            "af[0.5,1)": "0.5–1.0",
        }
        order = list(replacements)
    else:
        replacements = {
            "bp[50,100)": "50–100 bp",
            "bp[100,500)": "100–500 bp",
            "bp[500,1000)": "500 bp–1 kb",
            "bp[1000,10000)": "1–10 kb",
            "bp[10000,100000)": "10–100 kb",
            "bp[100000,1e+06)": "100 kb–1 Mb",
        }
        order = list(replacements)
    ys = np.arange(len(order))[::-1]
    for y, value in zip(ys, order):
        rows = subset.loc[subset["stratum_value"].eq(value)]
        for closure, color, marker, offset in [
            ("strict", STRICT, "o", 0.12),
            ("1hop", ONE_HOP, "D", -0.12),
        ]:
            row = rows.loc[rows["closure"].eq(closure)]
            if row.empty:
                continue
            row = row.iloc[0]
            underpowered = "underpowered" in str(row.get("interpretation_status", ""))
            plot_color = "#AEB4B8" if underpowered else color
            dot_interval(
                ax,
                mean=float(row["mean_gain"]),
                low=float(row["ci95_low"]),
                high=float(row["ci95_high"]),
                y=float(y + offset),
                color=plot_color,
                marker=marker,
                size=5.6,
            )
            if underpowered and closure == "strict":
                ax.text(float(row["mean_gain"]), y + 0.42, f'underpowered (mean n≈{float(row["mean_examples_per_run"]):.0f}/run)', ha="center", color=GRAY, fontsize=7.6)
    ax.axvline(0, color=GRAY, lw=1, ls="--")
    ax.set_yticks(ys, [replacements[value] for value in order])
    ax.grid(axis="x", color=LIGHT, lw=0.8)
    ax.set_xlabel("Graph contribution, paired ΔAUPRC")


def build_sv(absolute: pd.DataFrame, contributions: pd.DataFrame, strata: pd.DataFrame, out: Path) -> dict[str, str]:
    fig = plt.figure(figsize=(13.4, 11.2))
    grid = fig.add_gridspec(3, 2, hspace=0.68, wspace=0.50)
    source_schematic(fig.add_subplot(grid[0, 0]), task="SV")
    absolute_panel(fig.add_subplot(grid[0, 1]), absolute, "SV", "b")
    contribution_panel(fig.add_subplot(grid[1, :]), contributions, "SV", "c")
    sv_strata_panel(fig.add_subplot(grid[2, 0]), strata, stratum="allele_frequency_bin", letter="d", title="Graph contribution across allele-frequency groups")
    sv_strata_panel(fig.add_subplot(grid[2, 1]), strata, stratum="length_bin", letter="e", title="Graph contribution across variant-length groups")
    fig.suptitle("Graph organization contributes strongly to structural-variant breakpoint prediction", fontsize=16, fontweight="bold", y=0.995)
    return save_figure(fig, out, "figure4_sv")


def chance_for(prevalence: pd.DataFrame | None, scope: str, name: str, closure: str) -> float:
    if prevalence is None or prevalence.empty:
        return 0.5
    row = prevalence.loc[
        prevalence["analysis_scope"].eq(scope)
        & prevalence["analysis_name"].eq(name)
        & prevalence["closure"].eq(closure)
    ]
    return float(row["chance_auprc"].iloc[0]) if not row.empty else 0.5


def build_generalization(reconstruction: pd.DataFrame, transfer: pd.DataFrame, prevalence: pd.DataFrame | None, out: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4), gridspec_kw={"wspace": 0.48})
    ax = axes[0]
    panel(ax, "a", "Within-resource chromosome-held-out reconstruction")
    requests = [
        ("HPRC R2", "hprc_r2", "hprc_r2"),
        ("HGSVC3", "hgsvc3", "hgsvc3"),
        ("Integrated → HPRC R2", "combined_hprc_r2_hgsvc3", "hprc_r2"),
        ("Integrated → HGSVC3", "combined_hprc_r2_hgsvc3", "hgsvc3"),
        ("Integrated graph", "official_hgsvc3_hprc1_integrated", None),
    ]
    ys = np.arange(len(requests))[::-1]
    for y, (_, regime, dataset) in zip(ys, requests):
        query = reconstruction.loc[reconstruction["regime"].eq(regime)]
        if dataset is not None:
            query = query.loc[query["dataset"].eq(dataset)]
        for closure, color, marker, offset in [
            ("strict", STRICT, "o", 0.12),
            ("1hop", ONE_HOP, "D", -0.12),
        ]:
            row = query.loc[query["closure"].eq(closure)].iloc[0]
            dot_interval(
                ax,
                mean=float(row["auprc_mean"]),
                low=float(row["auprc_chromosome_bootstrap_ci_lower"]),
                high=float(row["auprc_chromosome_bootstrap_ci_upper"]),
                y=float(y + offset),
                color=color,
                marker=marker,
            )
    chance = chance_for(prevalence, "chromosome_held_out_reconstruction", "hprc_r2", "strict")
    ax.axvline(chance, color=GRAY, lw=1.1, ls="--")
    ax.text(0.055, 0.03, f"chance AUPRC ≈ {chance:.2f}", transform=ax.transAxes, color=GRAY, fontsize=8.2, rotation=90, va="bottom")
    ax.set_yticks(ys, [item[0] for item in requests])
    ax.set_xlim(0.48, 1.01)
    ax.grid(axis="x", color=LIGHT, lw=0.8)
    ax.set_xlabel("Mean chromosome-held-out AUPRC")
    context_legend(ax, location="upper left")
    ax.text(0, -0.24, "Whiskers: 95% chromosome-block bootstrap intervals", transform=ax.transAxes, color=GRAY, fontsize=7.8)

    ax = axes[1]
    panel(ax, "b", "Frozen cross-resource and cross-release transfer")
    pairs = [
        ("HPRC R2 → HGSVC3", "hprc_r2_to_hgsvc3"),
        ("HGSVC3 → HPRC R2", "hgsvc3_to_hprc_r2"),
        ("Integrated → HPRC R2", "hgsvc3_hprc1_combined_to_hprc_r2"),
        ("HPRC R1.1 → R2", "hprc_r1_1_to_hprc_r2"),
        ("HPRC R2 → R1.1", "hprc_r2_to_hprc_r1_1"),
    ]
    ys = np.arange(len(pairs))[::-1]
    for y, (_, key) in zip(ys, pairs):
        query = transfer.loc[transfer["transfer_pair"].eq(key)]
        for closure, color, marker, offset in [
            ("strict", STRICT, "o", 0.12),
            ("1hop", ONE_HOP, "D", -0.12),
        ]:
            row = query.loc[query["closure"].eq(closure)].iloc[0]
            mean = float(row["auprc_mean"])
            sd = float(row["auprc_sd_across_seeds"])
            dot_interval(ax, mean=mean, low=mean - sd, high=mean + sd, y=float(y + offset), color=color, marker=marker)
            ax.text(mean + 0.012, y + offset, f"{mean:.3f}", va="center", color=color, fontsize=8.0)
    chance = chance_for(prevalence, "cross_resource_or_release_transfer", "hprc_r2_to_hgsvc3", "strict")
    ax.axvline(chance, color=GRAY, lw=1.1, ls="--")
    ax.text(0.055, 0.03, f"chance AUPRC = {chance:.2f}", transform=ax.transAxes, color=GRAY, fontsize=8.2, rotation=90, va="bottom")
    ax.set_yticks(ys, [item[0] for item in pairs])
    ax.set_xlim(0.48, 1.01)
    ax.grid(axis="x", color=LIGHT, lw=0.8)
    ax.set_xlabel("Mean chromosome-held-out AUPRC")
    ax.text(0, -0.24, "Whiskers: ±1 SD across three seeds (not confidence intervals)", transform=ax.transAxes, color=GRAY, fontsize=7.8)
    fig.suptitle("Graph-native pretraining generalizes across chromosomes and graph resources", fontsize=16, fontweight="bold", y=1.02)
    return save_figure(fig, out, "figure5_generalization_transfer")


def capacity_points(old_runs: pd.DataFrame, principal_metrics: pd.DataFrame | None) -> pd.DataFrame:
    selected = old_runs.loc[(old_runs["metric_scope"] == "split") & (old_runs["split"] == "heldout_chr_test")].copy()
    mapping = {
        "hprc_r2_capacity_tiny_h24_l1": (24, 1, "24 / 1"),
        "hprc_r2_capacity_medium_h96_l4": (96, 4, "96 / 4"),
        "hprc_r2_capacity_large_h192_l6": (192, 6, "192 / 6"),
    }
    rows = []
    for regime, (hidden, layers, label) in mapping.items():
        values = selected.loc[selected["regime"].eq(regime), "auprc"].dropna().to_numpy(float)
        rows.append({"hidden": hidden, "layers": layers, "label": label, "mean": values.mean(), "sd": values.std(ddof=1), "principal": False})
    if principal_metrics is not None and not principal_metrics.empty:
        values = principal_metrics.loc[
            principal_metrics["regime"].eq("hprc_r2_capacity_principal_h48_l2")
            & principal_metrics["metric_scope"].eq("split")
            & principal_metrics["split"].eq("heldout_chr_test"),
            "auprc",
        ].dropna().to_numpy(float)
        if len(values):
            rows.append({"hidden": 48, "layers": 2, "label": "48 / 2\n(principal)", "mean": values.mean(), "sd": values.std(ddof=1), "principal": True})
    return pd.DataFrame(rows).sort_values("hidden")


def build_controls(source_dir: Path, reconstruction: pd.DataFrame, principal_metrics: pd.DataFrame | None, out: Path) -> dict[str, str]:
    baselines = pd.read_csv(source_dir / "figure4_baselines.csv")
    ablation = pd.read_csv(source_dir / "figure4_ablation_runs.csv")
    complexity = pd.read_csv(source_dir / "figure4_complexity.csv")
    capacity = pd.read_csv(source_dir / "figure4_capacity_runs.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.2), gridspec_kw={"hspace": 0.60, "wspace": 0.48})

    ax = axes[0, 0]
    panel(ax, "a", "Hidden-connection reconstruction versus simple baselines")
    display = {
        "PangenomeFM": "PangenomeFM",
        "coordinate_sgd": "Coordinates",
        "sequence_composition_sgd": "Sequence composition",
        "topology_degree_sum": "Degree sum",
        "topology_preferential_attachment": "Preferential attachment",
    }
    ys = np.arange(len(display))[::-1]
    for y, (baseline, label) in zip(ys, display.items()):
        for closure, color, marker, offset in [("strict", STRICT, "o", .12), ("1hop", ONE_HOP, "D", -.12)]:
            if baseline == "PangenomeFM":
                row = reconstruction.loc[(reconstruction["regime"] == "hprc_r2") & (reconstruction["dataset"] == "hprc_r2") & (reconstruction["closure"] == closure)].iloc[0]
                mean = float(row["auprc_mean"])
                low = float(row["auprc_chromosome_bootstrap_ci_lower"])
                high = float(row["auprc_chromosome_bootstrap_ci_upper"])
            else:
                row = baselines.loc[(baselines["baseline"] == baseline) & (baselines["closure"] == closure) & (baselines["metric"] == "auprc")].iloc[0]
                mean = float(row["mean"])
                low = float(row["ci95_low"])
                high = float(row["ci95_high"])
            dot_interval(ax, mean=mean, low=low, high=high, y=float(y + offset), color=color, marker=marker)
    ax.set_yticks(ys, list(display.values()))
    ax.set_xlim(0.45, 1.01)
    ax.grid(axis="x", color=LIGHT)
    ax.set_xlabel("Mean held-out-chromosome AUPRC")
    context_legend(ax, location="lower right")

    ax = axes[0, 1]
    panel(ax, "b", "Context effect across native-region complexity")
    comp = complexity.loc[(complexity["baseline"] == "topology_preferential_attachment") & (complexity["metric"] == "auprc_advantage")]
    order = ["low", "medium", "high"]
    x = np.arange(3)
    for closure, color, marker in [("strict", STRICT, "o"), ("1hop", ONE_HOP, "D")]:
        rows = comp.loc[comp["context"].eq(closure)].set_index("locus_complexity_category").loc[order]
        means = rows["chromosome_block_mean"].to_numpy(float)
        lows = rows["ci95_low"].to_numpy(float)
        highs = rows["ci95_high"].to_numpy(float)
        ax.errorbar(x, means, yerr=[means - lows, highs - means], color=color, marker=marker, lw=1.5, capsize=3, label="Strict context" if closure == "strict" else "One-hop context")
    ax.axhline(0, color=GRAY, ls="--", lw=1)
    ax.set_xticks(x, [name.title() for name in order])
    ax.set_ylabel("ΔAUPRC vs preferential attachment")
    ax.set_xlabel("Native-region complexity")
    ax.grid(axis="y", color=LIGHT)
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 0]
    panel(ax, "c", "Model-component controls")
    selected = ablation.loc[(ablation["metric_scope"] == "split") & (ablation["split"] == "heldout_chr_test")]
    display_regimes = {
        "hprc_r2_graph_only": "Graph-only",
        "hprc_r2_coordinate_only": "Coordinate-only",
        "hprc_r2_no_adaptive_window": "No adaptive window",
        "hprc_r2_no_fusion_gate": "No fusion gate",
        "hprc_r2_no_positional_encoding": "No positional encoding",
    }
    ys = np.arange(len(display_regimes))[::-1]
    for y, (regime, label) in zip(ys, display_regimes.items()):
        for closure, color, marker, offset in [("strict", STRICT, "o", .12), ("1hop", ONE_HOP, "D", -.12)]:
            values = selected.loc[(selected["regime"] == regime) & (selected["closure"] == closure), "auprc"].to_numpy(float)
            ax.errorbar(values.mean(), y + offset, xerr=values.std(ddof=1), fmt=marker, color=color, capsize=2.5)
    ax.set_yticks(ys, list(display_regimes.values()))
    ax.set_xlim(0.72, 1.005)
    ax.grid(axis="x", color=LIGHT)
    ax.set_xlabel("Mean held-out-chromosome AUPRC (±1 SD across runs)")
    context_legend(ax, location="lower right")

    ax = axes[1, 1]
    panel(ax, "d", "Capacity sweep under strict context")
    points = capacity_points(capacity, principal_metrics)
    positions = np.arange(len(points))
    for i, row in points.reset_index(drop=True).iterrows():
        color = TEAL if bool(row["principal"]) else PURPLE
        ax.errorbar(i, row["mean"], yerr=row["sd"], fmt="o", color=color, capsize=3, ms=7)
    ax.plot(positions, points["mean"], color=PURPLE, lw=1.2, alpha=.75)
    ax.set_xticks(positions, points["label"])
    ax.set_ylim(0.89, 1.005)
    ax.set_ylabel("AUPRC")
    ax.set_xlabel("Hidden units / graph layers")
    ax.grid(axis="y", color=LIGHT)
    fig.suptitle("Controls distinguish reusable neighborhood information from reconstruction shortcuts", fontsize=16, fontweight="bold", y=0.995)
    return save_figure(fig, out, "supplementary_figure1_controls_capacity")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--ccre-strata", type=Path)
    parser.add_argument("--prevalence-summary", type=Path)
    parser.add_argument("--principal-capacity-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-ccre-strata", action="store_true")
    args = parser.parse_args()

    absolute = pd.read_csv(args.source_dir / "figure5_absolute.csv")
    contributions = pd.read_csv(args.source_dir / "figure5_contributions.csv")
    sv_strata = pd.read_csv(args.source_dir / "figure5_sv_strata.csv")
    reconstruction = pd.read_csv(args.source_dir / "figure3_reconstruction.csv")
    transfer = pd.read_csv(args.source_dir / "figure3_transfer.csv")
    if args.require_ccre_strata and (args.ccre_strata is None or not args.ccre_strata.is_file()):
        parser.error("--require-ccre-strata requires an existing --ccre-strata table")
    ccre_gains = pd.read_csv(args.ccre_strata) if args.ccre_strata and args.ccre_strata.is_file() else None
    prevalence = pd.read_csv(args.prevalence_summary) if args.prevalence_summary and args.prevalence_summary.is_file() else None
    principal = pd.read_csv(args.principal_capacity_metrics) if args.principal_capacity_metrics and args.principal_capacity_metrics.is_file() else None

    outputs = {
        "figure3_ccre": build_ccre(absolute, contributions, ccre_gains, args.output_dir),
        "figure4_sv": build_sv(absolute, contributions, sv_strata, args.output_dir),
        "figure5_generalization_transfer": build_generalization(reconstruction, transfer, prevalence, args.output_dir),
        "supplementary_figure1_controls_capacity": build_controls(args.source_dir, reconstruction, principal, args.output_dir),
    }
    audit = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_dir": str(args.source_dir.resolve()),
        "ccre_strata": str(args.ccre_strata.resolve()) if args.ccre_strata else None,
        "prevalence_summary": str(args.prevalence_summary.resolve()) if args.prevalence_summary else None,
        "principal_capacity_metrics": str(args.principal_capacity_metrics.resolve()) if args.principal_capacity_metrics else None,
        "context_terms": {"strict": "Strict context", "1hop": "One-hop context"},
        "outputs": outputs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
