#!/usr/bin/env python3
"""Generate proposed PNG figures for the PSB GraphGenome-FM revision.

This script intentionally writes to paper/figures/proposed/ so the current
submission figure set is not overwritten. All result panels are generated from
completed local artifacts in results/.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "proposed"
MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/private/tmp/graphgenomefm-mpl"))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd


COL = {
    "ink": "#17212b",
    "muted": "#64748b",
    "grid": "#d8dee8",
    "blue": "#2f6fbb",
    "teal": "#16817a",
    "green": "#4c956c",
    "gold": "#c9821a",
    "rose": "#b94a62",
    "purple": "#7057a3",
    "orange": "#c75d2c",
    "light_blue": "#eff6ff",
    "light_green": "#ecfdf3",
    "light_gold": "#fff7e6",
    "light_rose": "#fff1f2",
    "light_purple": "#f4f0ff",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.edgecolor": "#111827",
        "axes.linewidth": 0.9,
        "savefig.facecolor": "white",
    }
)


def save_png(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {path.relative_to(ROOT)}")
    plt.close(fig)


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def status_frame() -> pd.DataFrame:
    path = ROOT / "results" / "submission_gap_status.csv"
    return pd.read_csv(path)


def master_frame() -> pd.DataFrame:
    path = ROOT / "results" / "master_experiment_status.csv"
    return pd.read_csv(path)


def status_row(experiment: str) -> pd.Series:
    df = status_frame()
    rows = df.loc[df["experiment"].eq(experiment)]
    if rows.empty:
        raise KeyError(experiment)
    return rows.iloc[0]


def secondary_metric(text: str, key: str) -> float:
    match = re.search(rf"{re.escape(key)}=([0-9.]+)", str(text))
    if not match:
        return float("nan")
    return float(match.group(1))


def draw_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    *,
    fc: str,
    ec: str,
    title_color: str | None = None,
    title_size: float = 10.0,
    body_size: float = 8.4,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.025,
        y + h - 0.045,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=title_color or ec,
    )
    ax.text(
        x + 0.025,
        y + h - 0.095,
        body,
        ha="left",
        va="top",
        fontsize=body_size,
        color=COL["ink"],
        linespacing=1.22,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#475569") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def mini_graph(ax: plt.Axes, x: float, y: float, s: float = 1.0) -> None:
    nodes = {
        "a": (x, y),
        "b": (x + 0.08 * s, y),
        "c": (x + 0.16 * s, y),
        "d": (x + 0.24 * s, y),
        "e": (x + 0.09 * s, y + 0.07 * s),
        "f": (x + 0.18 * s, y - 0.07 * s),
    }
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("b", "e"), ("e", "d"), ("b", "f"), ("f", "d")]
    for u, v in edges:
        ax.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]], color="#8da2bd", lw=1.4)
    for i, (nx, ny) in enumerate(nodes.values()):
        color = COL["blue"] if i < 4 else COL["gold"]
        ax.add_patch(Circle((nx, ny), 0.012 * s, facecolor=color, edgecolor="white", lw=0.8, zorder=3))


def copy_user_overview_pngs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_dir_value = os.environ.get("GRAPHGENOMEFM_USER_FIGURE_DIR")
    if not source_dir_value:
        print(
            "GRAPHGENOMEFM_USER_FIGURE_DIR is unset; "
            "skipping optional user-supplied overview PNGs"
        )
        return
    source_dir = Path(source_dir_value).expanduser()
    copies = [
        (
            source_dir / "ChatGPT Image Jun 8, 2026, 08_09_43 PM.png",
            "proposed_fig1_model_architecture_user_improved.png",
        ),
        (
            source_dir / "ChatGPT Image Jun 8, 2026, 08_12_36 PM.png",
            "proposed_fig2_ccre_pipeline_user_improved.png",
        ),
    ]
    for src, dst_name in copies:
        if src.exists():
            dst = OUT / dst_name
            shutil.copyfile(src, dst)
            print(f"copied {dst.relative_to(ROOT)}")
        else:
            print(f"missing source image: {src}")


def fig2_ccre_pipeline_corrected() -> None:
    fig, ax = plt.subplots(figsize=(14.2, 7.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.96, "cCRE evaluation pipeline and reporting rule", fontsize=18, fontweight="bold", color=COL["ink"])
    ax.text(
        0.02,
        0.91,
        "Inputs, baselines, leakage controls, evaluation universes, and outputs are reported explicitly.",
        fontsize=10,
        color=COL["muted"],
    )

    draw_card(
        ax,
        0.03,
        0.58,
        0.19,
        0.26,
        "1. Inputs",
        "ENCODE cCRE BED intervals\nmapped to HPRC GRCh38-path\nnodes by interval overlap.\n\nBackground = no mapped cCRE.",
        fc=COL["light_blue"],
        ec=COL["blue"],
    )
    draw_card(
        ax,
        0.27,
        0.58,
        0.19,
        0.26,
        "2. Labels",
        "Binary cCRE vs background.\n\nReduced grouped tasks:\n3-class, 4-class, 5-class.\n\nCategory-specific binaries.",
        fc=COL["light_gold"],
        ec=COL["gold"],
    )
    draw_card(
        ax,
        0.51,
        0.58,
        0.19,
        0.26,
        "3. Models",
        "Leakage-safe logistic/MLP/RF\nfeature baselines.\n\nSerialized graph Transformer\nbaseline (DeepGene-style).\n\nGAT scratch, frozen, fine-tuned.",
        fc=COL["light_green"],
        ec=COL["green"],
    )
    draw_card(
        ax,
        0.75,
        0.58,
        0.20,
        0.26,
        "4. Outputs",
        "AUROC/AUPRC for binary tasks.\n\nMacro-F1 for grouped and\nnine-class tasks.\n\nPer-category AUPRC and AUROC.",
        fc=COL["light_rose"],
        ec=COL["rose"],
    )
    arrow(ax, (0.22, 0.71), (0.27, 0.71), COL["muted"])
    arrow(ax, (0.46, 0.71), (0.51, 0.71), COL["muted"])
    arrow(ax, (0.70, 0.71), (0.75, 0.71), COL["muted"])

    draw_card(
        ax,
        0.06,
        0.19,
        0.25,
        0.23,
        "Evaluation universe A",
        "All-node held-out test:\n31,560 labeled GRCh38 nodes.\n\nUsed by all-node feature and\nserialized baselines.",
        fc="#ffffff",
        ec=COL["purple"],
        title_color=COL["purple"],
    )
    draw_card(
        ax,
        0.37,
        0.19,
        0.25,
        0.23,
        "Evaluation universe B",
        "Window-restricted test:\n1,330 unique labeled nodes\ncovered by held-out graph windows.\n\nUsed by graph-window methods.",
        fc="#ffffff",
        ec=COL["purple"],
        title_color=COL["purple"],
    )
    draw_card(
        ax,
        0.68,
        0.19,
        0.25,
        0.23,
        "Leakage control",
        "Safe cCRE reruns exclude\nSR / is_grch38 reference-status\nfeatures.\n\nReport universes separately.",
        fc="#ffffff",
        ec=COL["purple"],
        title_color=COL["purple"],
    )
    save_png(fig, "proposed_fig2_ccre_pipeline_corrected")


def fig3_hprc_ablation_polished() -> None:
    raw = pd.read_csv(ROOT / "results/analysis/maskedtrain_full_ablation_summary.csv")
    raw = raw.loc[raw["split"].eq("heldout_chr")].copy()
    model_names = {
        "coordinate_only": "Coordinate only",
        "graph_only": "Graph only",
        "no_gate": "Dual no gate",
        "full": "Full gated",
    }
    data = raw.assign(
        model=raw["model"].map(model_names),
        closure=raw["closure"].replace({"1hop": "1-hop"}),
        auroc=raw["mean_test_auc"],
    )[["model", "closure", "auroc"]]
    order = ["Coordinate only", "Graph only", "Dual no gate", "Full gated"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    colors = {"strict": COL["blue"], "1-hop": COL["gold"]}
    width = 0.35
    for i, closure in enumerate(["strict", "1-hop"]):
        vals = [
            data.loc[
                data["model"].eq(model_name) & data["closure"].eq(closure),
                "auroc",
            ].iloc[0]
            for model_name in order
        ]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, color=colors[closure], edgecolor="#111827", lw=0.5, label=closure)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.006, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, color="#475569", lw=1.0, ls=":", label="chance")
    ax.set_xticks(x, order, rotation=15, ha="right")
    ax.set_ylim(0.48, 1.03)
    ax.set_ylabel("Held-out AUROC")
    ax.set_title("Leakage-audited ablation: topology drives the 1-hop gain")
    ax.grid(axis="y", color=COL["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.text(3.42, 0.507, "chance", ha="right", va="bottom", fontsize=8.5, color=COL["muted"])
    handles = [
        Patch(facecolor=COL["blue"], edgecolor="#111827", label="strict"),
        Patch(facecolor=COL["gold"], edgecolor="#111827", label="1-hop"),
    ]
    ax.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.54))
    save_png(fig, "proposed_fig3_hprc_ablation_polished")


def fig4_hgsvc_transfer_dotplot() -> None:
    rows = []
    for closure, rel in [
        ("strict", "results/hgsvc3/external_eval_maskedtrain_full_strict/run_001/per_target_metrics.csv"),
        ("1-hop", "results/hgsvc3/external_eval_maskedtrain_full_1hop/run_001/per_target_metrics.csv"),
    ]:
        df = pd.read_csv(ROOT / rel)
        for _, r in df.iterrows():
            target = str(r["target_sn"]).split("|")[-1]
            rows.append((target, closure, float(r["mean_auroc"]), float(r["mean_auprc"]), float(r["mean_brier"])))
    data = pd.DataFrame(rows, columns=["target", "closure", "auroc", "auprc", "brier"])
    order = ["chr19", "chr21", "chr22", "chrY"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9.3, 5.1))
    specs = [
        ("Strict AUROC", "strict", "auroc", COL["blue"]),
        ("Strict AUPRC", "strict", "auprc", "#6b8fc8"),
        ("1-hop AUROC", "1-hop", "auroc", COL["gold"]),
        ("1-hop AUPRC", "1-hop", "auprc", "#d8a33a"),
    ]
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(specs))
    for offset, (label, closure, metric, color) in zip(offsets, specs):
        vals = [
            data.loc[
                data["target"].eq(target) & data["closure"].eq(closure),
                metric,
            ].iloc[0]
            for target in order
        ]
        bars = ax.bar(x + offset, vals, width, color=color, edgecolor="#111827", lw=0.5, label=label)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.010, f"{val:.3f}", ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(x, order)
    ax.set_ylim(0.48, 0.96)
    ax.axhline(0.5, color=COL["muted"], lw=0.9, ls=":")
    ax.set_ylabel("Mean score")
    ax.set_title("Masked external transfer: frozen HPRC checkpoint on HGSVC3")
    ax.grid(axis="y", color=COL["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), columnspacing=1.4)
    fig.subplots_adjust(bottom=0.22)
    save_png(fig, "proposed_fig4_hgsvc_transfer_dotplot")


def fig5_ccre_fair_comparison() -> None:
    items_all = [
        ("Coord.\nlogit", "ccre_safe_allnode_coordinate_binary_logistic", COL["blue"]),
        ("Topology", "ccre_safe_graph_allnode_binary", "#64748b"),
        ("Sequence", "ccre_safe_sequence_allnode_binary", COL["orange"]),
        ("Struct.\nlogit", "ccre_safe_allnode_structural_binary_logistic", COL["green"]),
        ("Linearized", "ccre_safe_allnode_linearized_graph_binary_logistic", COL["gold"]),
        ("Serialized", "ccre_safe_serialized_allnode_binary", COL["purple"]),
    ]
    items_win = [
        ("Coord.\nlogit", "ccre_safe_window_coordinate_binary_logistic", COL["blue"]),
        ("Topology", "ccre_safe_graph_window_binary", "#64748b"),
        ("Sequence", "ccre_safe_sequence_window_binary", COL["orange"]),
        ("Struct.\nlogit", "ccre_safe_window_structural_binary_logistic", COL["green"]),
        ("Linearized", "ccre_safe_window_linearized_graph_binary_logistic", COL["gold"]),
        ("Serialized", "ccre_safe_serialized_window_binary", COL["purple"]),
        ("GAT\nscratch", "ccre_safe_gat_scratch_binary", "#58996f"),
        ("GAT\nfrozen", "ccre_safe_gat_frozen_binary", COL["teal"]),
        ("GAT\nfine-tuned", "ccre_safe_gat_finetune_binary", COL["rose"]),
    ]
    grouped = [
        ("3-class", "ccre_safe_gat_scratch_group3", "ccre_safe_gat_frozen_group3", "ccre_safe_gat_finetune_group3"),
        ("4-class", "ccre_safe_gat_scratch_group4", "ccre_safe_gat_frozen_group4", "ccre_safe_gat_finetune_group4"),
        ("5-class", "ccre_safe_gat_scratch_group5", "ccre_safe_gat_frozen_group5", "ccre_safe_gat_finetune_group5"),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 5.2),
        gridspec_kw={"width_ratios": [1.4, 2.0, 1.3]},
    )

    for ax, items, title, universe in [
        (axes[0], items_all, "A. All-node binary cCRE", "31,560 held-out nodes"),
        (axes[1], items_win, "B. Window binary cCRE", "1,330 held-out nodes"),
    ]:
        vals = [float(status_row(exp)["primary_value"]) for _, exp, _ in items]
        labels = [label for label, _, _ in items]
        colors = [color for _, _, color in items]
        bars = ax.bar(np.arange(len(vals)), vals, color=colors, edgecolor="#111827", lw=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.009, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(len(vals)), labels, rotation=0, ha="center")
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylim(0.45, 0.90)
        ax.set_ylabel("AUROC")
        ax.set_title(title)
        ax.text(0.02, 0.94, universe, transform=ax.transAxes, ha="left", va="top", fontsize=9, color=COL["muted"])
        ax.grid(axis="y", color=COL["grid"], lw=0.6)
        ax.set_axisbelow(True)

    ax = axes[2]
    x = np.arange(len(grouped))
    width = 0.24
    model_specs = [("scratch", 1, "#58996f"), ("frozen", 2, COL["teal"]), ("fine-tuned", 3, COL["rose"])]
    for j, (name, idx, color) in enumerate(model_specs):
        vals = [float(status_row(row[idx])["primary_value"]) for row in grouped]
        bars = ax.bar(x + (j - 1) * width, vals, width, color=color, edgecolor="#111827", lw=0.5, label=name)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.009, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [row[0] for row in grouped])
    ax.set_ylim(0.0, 0.50)
    ax.set_ylabel("Macro-F1")
    ax.set_title("C. Window grouped cCRE")
    ax.text(0.02, 0.94, "Pretraining helps modestly", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=COL["muted"])
    ax.grid(axis="y", color=COL["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper right")

    fig.suptitle("Leakage-safe cCRE comparison: report all-node and window universes separately", fontsize=15, fontweight="bold", y=1.03)
    save_png(fig, "proposed_fig5_ccre_fair_comparison")


def fig6_ccre_category_lift() -> None:
    runs = [
        ("dELS", "ccre_safe_allnode_category_dels_logistic"),
        ("Enhancer-like", "ccre_safe_allnode_category_enhancer_like_logistic"),
        ("pELS", "ccre_safe_allnode_category_pels_logistic"),
        ("TF/CTCF", "ccre_safe_allnode_category_tf_ctcf_associated_logistic"),
        ("Open chromatin", "ccre_safe_allnode_category_open_chromatin_logistic"),
        ("Promoter-like", "ccre_safe_allnode_category_promoter_like_logistic"),
        ("PLS", "ccre_safe_allnode_category_pls_logistic"),
    ]
    rows = []
    for label, exp in runs:
        r = status_row(exp)
        summary = read_json(ROOT / str(r["run"]) / "summary.json")
        metrics = summary["test_metrics"]
        rows.append(
            {
                "label": label,
                "auroc": float(r["primary_value"]),
                "auprc": secondary_metric(r["secondary_metrics"], "AUPRC"),
                "macro_f1": secondary_metric(r["secondary_metrics"], "macro_F1"),
                "positive_fraction": float(metrics["positive_fraction"]),
                "n_test": int(summary["n_test"]),
            }
        )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    sizes = 170 + 720 * (df["auroc"] - df["auroc"].min()) / (df["auroc"].max() - df["auroc"].min())
    scatter = ax.scatter(
        df["positive_fraction"],
        df["auprc"],
        s=sizes,
        c=df["auroc"],
        cmap="viridis",
        edgecolor="#111827",
        linewidth=0.7,
        alpha=0.92,
    )
    max_x = max(df["positive_fraction"].max(), df["auprc"].max())
    ax.plot([0, max_x], [0, max_x], color="#94a3b8", ls=":", lw=1.2, label="random AUPRC = prevalence")
    offsets = {
        "dELS": (12, -18),
        "Enhancer-like": (-78, 8),
        "pELS": (12, 14),
        "Open chromatin": (12, -10),
        "TF/CTCF": (12, 8),
        "Promoter-like": (12, -14),
        "PLS": (12, 6),
    }
    for _, r in df.iterrows():
        ax.annotate(
            r["label"],
            (r["positive_fraction"], r["auprc"]),
            xytext=offsets.get(r["label"], (10, 8)),
            textcoords="offset points",
            fontsize=8.6,
            color=COL["ink"],
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75),
        )
    ax.set_xlim(0, 0.49)
    ax.set_ylim(0, 0.92)
    ax.set_xlabel("Positive fraction in all-node test set")
    ax.set_ylabel("AUPRC")
    ax.set_title("Category-specific cCRE prediction: AUPRC relative to class rarity", fontsize=13)
    ax.grid(color=COL["grid"], lw=0.6)
    ax.set_axisbelow(True)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("AUROC")
    ax.legend(frameon=False, loc="lower right")
    save_png(fig, "proposed_fig6_ccre_category_lift")


def fig7_hgsvc_imputation_polished() -> None:
    candidates = pd.read_csv(ROOT / "results/hgsvc3/graph_imputation_1hop/run_003/candidate_edges_unique_by_pair.csv")
    summary = read_json(ROOT / "results/hgsvc3/graph_imputation_1hop/run_003/summary.json")
    candidates["target"] = candidates["target_sn"].astype(str).str.split("|").str[-1]
    thresholds = [0.5, 0.8, 0.9]
    target_order = ["chr19", "chr21", "chr22", "chrY"]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    ax.hist(candidates["max_p_edge"], bins=np.linspace(0, 1, 51), color=COL["teal"], edgecolor="white")
    ax.set_yscale("log")
    for thr, color in [(0.5, COL["gold"]), (0.8, COL["rose"]), (0.9, COL["purple"])]:
        ax.axvline(thr, color=color, ls="--", lw=1.8)
        ax.text(thr + 0.01, ax.get_ylim()[1] / 2.2, f">= {thr}", color=color, fontsize=9, rotation=90, va="center")
    ax.set_xlabel("Max predicted edge probability per unique candidate")
    ax.set_ylabel("Unique candidate pairs, log scale")
    ax.set_title("A. HGSVC candidate missing-edge scores")
    ax.grid(axis="y", color=COL["grid"], lw=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    x = np.arange(len(target_order))
    width = 0.24
    colors = [COL["blue"], COL["gold"], COL["rose"]]
    for j, (thr, color) in enumerate(zip(thresholds, colors)):
        vals = []
        for target in target_order:
            sub = candidates.loc[candidates["target"].eq(target)]
            vals.append(int((sub["max_p_edge"] >= thr).sum()))
        bars = ax.bar(x + (j - 1) * width, vals, width, color=color, edgecolor="#111827", lw=0.5, label=f">= {thr}")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 4, str(val), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, target_order)
    ax.set_ylabel("Unique candidate pairs")
    ax.set_title("B. High-confidence candidates by target")
    ax.grid(axis="y", color=COL["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    cal = summary["calibration_metrics_on_edge_pred"]
    ax.text(
        0.02,
        0.86,
        f"Calibration on edge-prediction set\nAUROC={cal['auroc']:.4f}, AUPRC={cal['auprc']:.4f}\nCandidate scoring only",
        transform=ax.transAxes,
        fontsize=9,
        color=COL["ink"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cbd5e1", alpha=0.95),
        va="top",
    )
    fig.suptitle("HGSVC graph-imputation output is a prioritized candidate list, not validated recovery", fontsize=14, fontweight="bold", y=1.03)
    save_png(fig, "proposed_fig7_hgsvc_imputation_polished")


def fig8_ccre_coverage_audit() -> None:
    coverage = pd.read_csv(ROOT / "results/hprc/ccre_window_coverage/coverage_by_split_closure.csv")
    all_n = int(status_row("ccre_safe_allnode_coordinate_binary_logistic")["n_test"])
    rows = [
        ("All-node test", all_n, 100.0, COL["blue"]),
        (
            "Strict windows",
            int(coverage.query("split == 'test' and closure == 'strict'")["unique_labeled_nodes"].iloc[0]),
            100 * int(coverage.query("split == 'test' and closure == 'strict'")["unique_labeled_nodes"].iloc[0]) / all_n,
            COL["green"],
        ),
        (
            "1-hop windows",
            int(coverage.query("split == 'test' and closure == '1hop'")["unique_labeled_nodes"].iloc[0]),
            100 * int(coverage.query("split == 'test' and closure == '1hop'")["unique_labeled_nodes"].iloc[0]) / all_n,
            COL["gold"],
        ),
        (
            "Strict or 1-hop",
            int(coverage.query("split == 'test' and closure == 'strict_or_1hop'")["unique_labeled_nodes"].iloc[0]),
            100 * int(coverage.query("split == 'test' and closure == 'strict_or_1hop'")["unique_labeled_nodes"].iloc[0]) / all_n,
            COL["rose"],
        ),
    ]
    labels, counts, percents, colors = zip(*rows)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bars = ax.barh(np.arange(len(rows)), percents, color=colors, edgecolor="#111827", lw=0.5)
    ax.set_yticks(np.arange(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Percent of all-node held-out cCRE test set")
    ax.set_title("cCRE evaluation-universe audit")
    for bar, count, pct in zip(bars, counts, percents):
        label = f"{count:,} nodes ({pct:.2f}%)"
        x = min(bar.get_width() + 1.5, 86)
        ax.text(x, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9, color=COL["ink"])
    ax.text(
        0.98,
        0.12,
        "This is why all-node baselines and\nwindow GAT results are separate panels.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=COL["muted"],
    )
    ax.grid(axis="x", color=COL["grid"], lw=0.6)
    ax.set_axisbelow(True)
    save_png(fig, "proposed_fig8_ccre_coverage_audit")


def fig9_negative_sampling_schematic() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.93, "Negative-sampling controls for link prediction", fontsize=17, fontweight="bold", color=COL["ink"])
    ax.text(0.03, 0.885, "The robustness story is about removing coordinate and degree shortcuts from sampled non-edges.", fontsize=10, color=COL["muted"])

    rows = [
        ("Random negatives", "Easy audit only", "Sample any non-edge.\nOften separable by coordinate distance.", COL["light_blue"], COL["blue"], "arbitrary"),
        ("Distance-matched", "Main paper hard negative", "For each positive edge with distance d,\nsample non-edges with similar d.", COL["light_gold"], COL["gold"], "same d"),
        ("Degree-matched", "Additional robustness", "Match distance and endpoint degree\nwhere possible.", COL["light_green"], COL["green"], "same d + degree"),
    ]
    ys = [0.64, 0.39, 0.14]
    for (title, tag, body, fc, ec, lab), y in zip(rows, ys):
        draw_card(ax, 0.05, y, 0.27, 0.18, title, f"{tag}\n{body}", fc=fc, ec=ec, title_size=9.6, body_size=7.9)
        ax.plot([0.42, 0.60], [y + 0.09, y + 0.09], color=COL["ink"], lw=2.0)
        ax.add_patch(Circle((0.42, y + 0.09), 0.015, facecolor=COL["blue"], edgecolor="white"))
        ax.add_patch(Circle((0.60, y + 0.09), 0.015, facecolor=COL["blue"], edgecolor="white"))
        ax.text(0.51, y + 0.125, "positive edge", ha="center", fontsize=8, color=COL["ink"])
        x1, x2, yy = 0.73, 0.93, y + 0.085
        ax.plot([x1, x2], [yy, yy], color=ec, lw=1.8, ls="--")
        ax.add_patch(Circle((x1, yy), 0.012, facecolor="white", edgecolor=ec, lw=1.2))
        ax.add_patch(Circle((x2, yy), 0.012, facecolor="white", edgecolor=ec, lw=1.2))
        ax.text((x1 + x2) / 2, yy + 0.035, lab, ha="center", fontsize=8, color=COL["muted"])
        arrow(ax, (0.32, y + 0.08), (0.40, y + 0.08), ec)
        arrow(ax, (0.62, y + 0.08), (0.70, y + 0.08), ec)
    save_png(fig, "proposed_fig9_negative_sampling_schematic")


def fig10_reliability_curves() -> None:
    bins = pd.read_csv(ROOT / "results/reliability_curves/reliability_bins.csv")
    summary = read_json(ROOT / "results/reliability_curves/summary.json")
    brier = {row["label"]: float(row["brier"]) for row in summary if "label" in row}
    labels = [
        ("hprc_seed7_strict", "HPRC strict", COL["blue"], "o"),
        ("hprc_seed7_1hop", "HPRC 1-hop", COL["gold"], "s"),
        ("hgsvc_strict", "HGSVC strict", COL["teal"], "^"),
        ("hgsvc_1hop", "HGSVC 1-hop", COL["rose"], "D"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], color="#94a3b8", ls=":", lw=1.4, label="perfect calibration")
    for key, label, color, marker in labels:
        sub = bins.loc[bins["label"].eq(key) & (bins["n"] > 0)].sort_values("bin")
        ax.plot(
            sub["mean_predicted"],
            sub["observed_fraction"],
            marker=marker,
            color=color,
            lw=1.8,
            ms=5,
            label=f"{label} (Brier {brier.get(key, float('nan')):.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed edge fraction")
    ax.set_title("Reliability curves for edge probabilities")
    ax.grid(color=COL["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    save_png(fig, "proposed_fig10_reliability_curves")


def fig11_robustness_summary() -> None:
    master = master_frame()
    status = status_frame()

    def master_value(experiment: str, closure: str) -> float:
        rows = master.loc[master["experiment"].eq(experiment) & master["closure"].eq(closure)]
        return float(rows.iloc[0]["primary_value"])

    def status_value(requirement: str, closure: str, metric: str = "heldout_AUROC") -> tuple[float, float | None]:
        rows = status.loc[
            status["requirement"].eq(requirement)
            & status["eval_universe"].eq(closure)
            & status["primary_metric"].eq(metric)
        ]
        if rows.empty:
            rows = status.loc[status["requirement"].eq(requirement) & status["eval_universe"].eq(closure)]
        row = rows.iloc[0]
        sd = None
        if "sd=" in str(row["secondary_metrics"]):
            sd = secondary_metric(row["secondary_metrics"], "sd")
        return float(row["primary_value"]), sd

    rows = []
    for label, source in [
        ("Distance-matched", "master_distance"),
        ("Degree + distance", "degree"),
        ("Seed mean, n=3", "seed"),
        ("Non-overlap pilot", "nonoverlap"),
    ]:
        for closure in ["strict", "1hop"]:
            if source == "master_distance":
                value = master_value("HPRC shared pretraining with distance-matched negatives", closure)
                sd = None
            elif source == "degree":
                value, sd = status_value("hard-negative robustness", closure)
            elif source == "seed":
                value, sd = status_value("seed variance", closure, "heldout_AUROC_mean")
            else:
                subset = status.loc[
                    status["experiment"].eq("Non-overlapping hard-negative HPRC pretraining")
                    & status["eval_universe"].eq(closure)
                ]
                value = float(subset.iloc[0]["primary_value"])
                sd = None
            rows.append((label, "1-hop" if closure == "1hop" else "strict", value, sd))
    df = pd.DataFrame(rows, columns=["condition", "closure", "auroc", "sd"])
    order = ["Distance-matched", "Degree + distance", "Seed mean, n=3", "Non-overlap pilot"]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.3, 4.9))
    for closure, color, marker, offset in [("strict", COL["blue"], "o", -0.09), ("1-hop", COL["gold"], "s", 0.09)]:
        vals = [
            df.loc[
                df["condition"].eq(condition) & df["closure"].eq(closure),
                "auroc",
            ].iloc[0]
            for condition in order
        ]
        sds = [
            df.loc[
                df["condition"].eq(condition) & df["closure"].eq(closure),
                "sd",
            ].iloc[0]
            for condition in order
        ]
        xerr = [0 if pd.isna(sd) else sd for sd in sds]
        ax.errorbar(vals, y + offset, xerr=xerr, fmt=marker, color=color, ms=7, lw=1.6, capsize=3, label=closure)
        for val, yy in zip(vals, y + offset):
            ax.text(val + 0.003, yy, f"{val:.3f}", va="center", fontsize=8, color=COL["ink"])
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlim(0.76, 1.012)
    ax.set_xlabel("Held-out AUROC")
    ax.set_title("Hard-negative and reproducibility checks")
    ax.grid(axis="x", color=COL["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    save_png(fig, "proposed_fig11_robustness_summary")


def fig12_graph_native_vs_serialized() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.94, "What comparison does the cCRE section need?", fontsize=17, fontweight="bold", color=COL["ink"])
    ax.text(
        0.03,
        0.89,
        "Prof's expected comparison is linear reference vs serialized graph vs graph-native message passing.",
        fontsize=10,
        color=COL["muted"],
    )

    columns = [
        ("Linear reference", "One coordinate-ordered path\nFeature baselines use node and\nlocal coordinate summaries.", COL["light_blue"], COL["blue"]),
        ("Serialized graph", "Flatten a local graph neighborhood\ninto a token sequence.\nDeepGene-style baseline;\nno edge message passing in\nour lightweight run.", COL["light_gold"], COL["gold"]),
        ("Graph-native", "Use observed pangenome edges for\nGAT message passing.\nGraphGenome-FM frozen/fine-tuned\nembeddings feed the classifier.", COL["light_green"], COL["green"]),
    ]
    xs = [0.05, 0.37, 0.69]
    for x, (title, body, fc, ec) in zip(xs, columns):
        draw_card(ax, x, 0.50, 0.26, 0.28, title, body, fc=fc, ec=ec, body_size=7.6)
    # Draw method-specific mini visuals.
    ax.plot([0.09, 0.27], [0.43, 0.43], color=COL["blue"], lw=2.0)
    for nx in np.linspace(0.09, 0.27, 5):
        ax.add_patch(Circle((nx, 0.43), 0.012, facecolor=COL["blue"], edgecolor="white", lw=0.8))
    ax.text(0.18, 0.37, "coordinate path", ha="center", fontsize=9, color=COL["muted"])

    mini_graph(ax, 0.405, 0.43, 0.8)
    ax.text(0.50, 0.37, "graph flattened to tokens", ha="center", fontsize=9, color=COL["muted"])
    for i, nx in enumerate(np.linspace(0.405, 0.595, 6)):
        ax.add_patch(FancyBboxPatch((nx, 0.30), 0.022, 0.035, boxstyle="round,pad=0.002", facecolor="#f8fafc", edgecolor=COL["gold"], lw=1.0))
        if i < 5:
            arrow(ax, (nx + 0.026, 0.318), (nx + 0.038, 0.318), COL["gold"])

    mini_graph(ax, 0.725, 0.43, 0.8)
    ax.text(0.82, 0.37, "edges used during encoding", ha="center", fontsize=9, color=COL["muted"])
    ax.add_patch(FancyBboxPatch((0.755, 0.28), 0.13, 0.055, boxstyle="round,pad=0.01", facecolor="white", edgecolor=COL["green"], lw=1.1))
    ax.text(0.82, 0.307, "GAT messages", ha="center", va="center", fontsize=9, color=COL["green"], fontweight="bold")

    draw_card(
        ax,
        0.08,
        0.055,
        0.84,
        0.16,
        "Paper framing",
        "The completed serialized baseline is a fair lightweight DeepGene-style comparison,\n"
        "not a full DeepGene reproduction. Say that directly in the manuscript.",
        fc="#ffffff",
        ec=COL["purple"],
        title_color=COL["purple"],
        title_size=10,
        body_size=8.0,
    )
    save_png(fig, "proposed_fig12_graph_native_vs_serialized_comparison")


def main() -> None:
    copy_user_overview_pngs()
    fig2_ccre_pipeline_corrected()
    fig3_hprc_ablation_polished()
    fig4_hgsvc_transfer_dotplot()
    fig5_ccre_fair_comparison()
    fig6_ccre_category_lift()
    fig7_hgsvc_imputation_polished()
    fig8_ccre_coverage_audit()
    fig9_negative_sampling_schematic()
    fig10_reliability_curves()
    fig11_robustness_summary()
    fig12_graph_native_vs_serialized()


if __name__ == "__main__":
    main()
