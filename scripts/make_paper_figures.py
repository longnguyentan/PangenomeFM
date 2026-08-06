#!/usr/bin/env python3
"""Generate manuscript-ready draft figures for GraphGenome-FM.

The figures are intentionally reproducible from local result artifacts. Two
schematics are drawn directly with matplotlib because Prof specifically asked
for visual explanations of the model and the cCRE classifier pipeline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/private/tmp/graphgenomefm-mpl"))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath


OUT = ROOT / "paper" / "figures"

COLORS = {
    "ink": "#1f2933",
    "muted": "#6b7280",
    "light": "#eef2f7",
    "line": "#c6ced8",
    "teal": "#16817a",
    "blue": "#2f6fbb",
    "gold": "#c9821a",
    "rose": "#b94a62",
    "green": "#4c956c",
    "purple": "#7057a3",
    "orange": "#c75d2c",
}


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        path = OUT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path.relative_to(ROOT)}")
    plt.close(fig)


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    label: str,
    *,
    fc: str = "white",
    ec: str = COLORS["line"],
    lw: float = 1.5,
    fontsize: int = 10,
    weight: str = "normal",
    radius: float = 0.06,
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        weight=weight,
        linespacing=1.15,
    )
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["muted"],
    lw: float = 1.6,
    mutation_scale: float = 14,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def _draw_small_graph(ax: plt.Axes, origin: tuple[float, float], scale: float = 1.0) -> None:
    ox, oy = origin
    nodes = {
        "a": (ox + 0.00 * scale, oy + 0.25 * scale),
        "b": (ox + 0.18 * scale, oy + 0.25 * scale),
        "c": (ox + 0.36 * scale, oy + 0.25 * scale),
        "d": (ox + 0.54 * scale, oy + 0.25 * scale),
        "u": (ox + 0.28 * scale, oy + 0.43 * scale),
        "v": (ox + 0.45 * scale, oy + 0.08 * scale),
    }
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("b", "u"), ("u", "d"), ("b", "v"), ("v", "d")]
    for u, v in edges:
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        ax.plot([x1, x2], [y1, y2], color=COLORS["line"], lw=2.0, zorder=1)
    for name, (x, y) in nodes.items():
        color = COLORS["blue"] if name in {"a", "b", "c", "d"} else COLORS["gold"]
        ax.add_patch(Circle((x, y), 0.035 * scale, facecolor=color, edgecolor="white", lw=1.2, zorder=3))


def fig_model_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.94, "A", fontsize=18, weight="bold", color=COLORS["ink"])
    ax.text(0.09, 0.94, "GraphGenome-FM graph-native pretraining", fontsize=16, weight="bold", color=COLORS["ink"])

    _box(ax, (0.04, 0.68), (0.21, 0.18), "Pangenome graph slice", fc="#f8fafc", ec=COLORS["blue"], weight="bold")
    _draw_small_graph(ax, (0.083, 0.675), scale=0.22)
    ax.text(0.145, 0.688, "nodes = DNA segments\nedges = adjacencies", fontsize=8, color=COLORS["muted"], ha="center", va="top")

    _box(ax, (0.04, 0.38), (0.21, 0.18), "Node and edge features\nlength, offset, strand,\ndegree, reference status", fc="#ffffff", ec=COLORS["line"])
    _box(ax, (0.04, 0.12), (0.21, 0.16), "Candidate edge pair\n(u, v)", fc="#ffffff", ec=COLORS["line"], weight="bold")

    _box(ax, (0.34, 0.66), (0.22, 0.18), "Coordinate stream\nself-attention over\nordered genomic offsets", fc="#eff6ff", ec=COLORS["blue"], weight="bold")
    _box(ax, (0.34, 0.39), (0.22, 0.18), "Graph-topology stream\nGAT message passing over\npangenome links", fc="#ecfdf3", ec=COLORS["green"], weight="bold")
    _box(ax, (0.34, 0.14), (0.22, 0.15), "Multi-scale position\nand orientation encoding", fc="#fff7e6", ec=COLORS["gold"])

    _box(ax, (0.65, 0.53), (0.20, 0.18), "Fusion gate\nlearns how much to use\ncoordinate vs graph signal", fc="#f4f0ff", ec=COLORS["purple"], weight="bold")
    _box(ax, (0.65, 0.25), (0.20, 0.18), "Node embeddings\nh(u), h(v)", fc="#ffffff", ec=COLORS["line"], weight="bold")

    _box(ax, (0.90, 0.55), (0.08, 0.15), "Edge\nscore", fc="#fff1f2", ec=COLORS["rose"], weight="bold")
    _box(ax, (0.90, 0.25), (0.08, 0.15), "Reusable\nembeddings", fc="#eef2ff", ec=COLORS["purple"], weight="bold", fontsize=9)

    _arrow(ax, (0.25, 0.77), (0.34, 0.75), color=COLORS["blue"])
    _arrow(ax, (0.25, 0.47), (0.34, 0.48), color=COLORS["green"])
    _arrow(ax, (0.25, 0.20), (0.34, 0.21), color=COLORS["gold"])
    _arrow(ax, (0.56, 0.75), (0.65, 0.63), color=COLORS["blue"])
    _arrow(ax, (0.56, 0.48), (0.65, 0.61), color=COLORS["green"])
    _arrow(ax, (0.75, 0.53), (0.75, 0.43), color=COLORS["purple"])
    _arrow(ax, (0.85, 0.34), (0.90, 0.33), color=COLORS["purple"])
    _arrow(ax, (0.85, 0.61), (0.90, 0.62), color=COLORS["rose"])

    ax.text(0.90, 0.73, "Self-supervised target:\nobserved edge vs sampled non-edge", fontsize=9, color=COLORS["muted"], ha="center")
    ax.text(0.90, 0.17, "Downstream use:\ncCRE labels, imputation,\nlatent-space analysis", fontsize=9, color=COLORS["muted"], ha="center")

    _save(fig, "fig1_model_architecture")


def fig_ccre_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.92, "B", fontsize=18, weight="bold", color=COLORS["ink"])
    ax.text(0.09, 0.92, "cCRE classifier built on graph-pretrained representations", fontsize=16, weight="bold", color=COLORS["ink"])

    _box(ax, (0.04, 0.62), (0.19, 0.19), "ENCODE cCRE\nintervals on GRCh38\nBED labels", fc="#fff7e6", ec=COLORS["gold"], weight="bold")
    _box(ax, (0.04, 0.25), (0.19, 0.19), "HPRC graph nodes\nwith GRCh38 path\ncoordinates", fc="#eff6ff", ec=COLORS["blue"], weight="bold")
    _box(ax, (0.30, 0.43), (0.19, 0.20), "Map intervals to\npangenome nodes\nby overlap", fc="#f8fafc", ec=COLORS["line"], weight="bold")
    _box(ax, (0.56, 0.62), (0.18, 0.19), "Frozen or fine-tuned\nGraphGenome-FM\nencoder", fc="#ecfdf3", ec=COLORS["green"], weight="bold")
    _box(ax, (0.56, 0.25), (0.18, 0.19), "Baseline feature sets\ncoordinate, structural,\nlinearized graph", fc="#ffffff", ec=COLORS["line"])
    _box(ax, (0.80, 0.53), (0.16, 0.19), "Classifier head\nbinary or grouped\ncCRE prediction", fc="#fff1f2", ec=COLORS["rose"], weight="bold")
    _box(ax, (0.80, 0.19), (0.16, 0.17), "Outputs\nP(cCRE), group label,\nper-category scores", fc="#f4f0ff", ec=COLORS["purple"], weight="bold", fontsize=9)

    _arrow(ax, (0.23, 0.71), (0.30, 0.56), color=COLORS["gold"])
    _arrow(ax, (0.23, 0.34), (0.30, 0.48), color=COLORS["blue"])
    _arrow(ax, (0.49, 0.55), (0.56, 0.71), color=COLORS["green"])
    _arrow(ax, (0.49, 0.51), (0.56, 0.35), color=COLORS["muted"])
    _arrow(ax, (0.74, 0.71), (0.80, 0.64), color=COLORS["green"])
    _arrow(ax, (0.74, 0.35), (0.80, 0.58), color=COLORS["muted"])
    _arrow(ax, (0.88, 0.53), (0.88, 0.36), color=COLORS["rose"])

    ax.text(0.06, 0.115, "Current main task:\nbinary cCRE vs background", fontsize=9, color=COLORS["ink"], weight="bold", ha="left")
    ax.text(0.35, 0.115, "Reduced tasks:\n3-class, 4-class, category-specific binaries", fontsize=9, color=COLORS["ink"], weight="bold", ha="left")
    ax.text(0.74, 0.075, "Key caveat: all-node baselines and window-restricted GAT results\nmust be reported as separate evaluation universes.", fontsize=8.5, color=COLORS["muted"], ha="center")

    _save(fig, "fig2_ccre_classifier_pipeline")


def _read_link_ablation() -> pd.DataFrame:
    paths = {
        "Full gated": ROOT / "results/hprc/pretrain_paper_hardneg/run_001/gat_results__shared_dual_mscale3_orient_adpwk32a4_focal2.0_dedge0.1_heldout_chr1_chr8_chr19_chrY_val_chr16_ep100_pat20.csv",
        "Coordinate only": ROOT / "results/hprc/ablation_coordinate_only/run_002/gat_results__shared_dual_coordinate_mscale3_orient_adpwk32a4_focal2.0_dedge0.1_heldout_chr1_chr8_chr19_chrY_val_chr16_ep100_pat20.csv",
        "Graph only": ROOT / "results/hprc/ablation_graph_only/run_001/gat_results__shared_dual_graph_mscale3_orient_adpwk32a4_focal2.0_dedge0.1_heldout_chr1_chr8_chr19_chrY_val_chr16_ep100_pat20.csv",
        "Dual no gate": ROOT / "results/hprc/ablation_dual_no_gate/run_001/gat_results__shared_dual_nogate_mscale3_orient_adpwk32a4_focal2.0_dedge0.1_heldout_chr1_chr8_chr19_chrY_val_chr16_ep100_pat20.csv",
    }
    rows = []
    for model, path in paths.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        held = df[df["split"].astype(str).str.lower().str.contains("held")]
        for closure in ("strict", "1hop"):
            s = held.loc[held["closure"].eq(closure), "test_auc"]
            rows.append({"model": model, "closure": "1-hop" if closure == "1hop" else "strict", "auroc": float(s.mean())})
    return pd.DataFrame(rows)


def fig_hprc_ablation() -> None:
    df = _read_link_ablation()
    order = ["Coordinate only", "Graph only", "Dual no gate", "Full gated"]
    closures = ["strict", "1-hop"]
    colors = {"strict": COLORS["blue"], "1-hop": COLORS["gold"]}
    x = np.arange(len(order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for i, closure in enumerate(closures):
        vals = [df[(df.model == m) & (df.closure == closure)]["auroc"].iloc[0] for m in order]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=closure, color=colors[closure], edgecolor="#222", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.006, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, order, rotation=15, ha="right")
    ax.set_ylim(0.50, 1.03)
    ax.set_ylabel("Held-out AUROC")
    ax.set_title("Ablation: graph topology carries the link-prediction signal")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    _save(fig, "fig3_hprc_ablation")


def fig_hgsvc_transfer() -> None:
    strict_path = ROOT / "results/hgsvc3/external_eval_hardneg_strict/run_004/per_target_metrics.csv"
    onehop_path = ROOT / "results/hgsvc3/external_eval_hardneg_1hop/run_003/per_target_metrics.csv"
    rows = []
    for closure, path in [("strict", strict_path), ("1-hop", onehop_path)]:
        if not path.exists():
            continue
        d = pd.read_csv(path)
        d["chrom"] = d["target_sn"].astype(str).str.split("|").str[-1]
        d["closure"] = closure
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    chroms = ["chr19", "chr21", "chr22", "chrY"]
    x = np.arange(len(chroms))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for i, closure in enumerate(["strict", "1-hop"]):
        vals = [float(df[(df.chrom == c) & (df.closure == closure)]["mean_auroc"].iloc[0]) for c in chroms]
        ax.bar(x + (i - 0.5) * width, vals, width, label=closure, color=COLORS["blue"] if closure == "strict" else COLORS["gold"], edgecolor="#222", linewidth=0.5)
    ax.set_xticks(x, chroms)
    ax.set_ylim(0.96, 1.002)
    ax.set_ylabel("Mean AUROC")
    ax.set_title("External transfer: frozen HPRC checkpoint on HGSVC")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    _save(fig, "fig4_hgsvc_external_transfer")


def _summary(path: str) -> dict:
    base = ROOT / path
    summaries = sorted(base.glob("run_*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No summary.json under {base}")
    return json.loads(summaries[-1].read_text())


def fig_ccre_binary_and_grouped() -> None:
    binary = [
        ("All-node\ncoord. logit", "results/hprc/ccre_safe_allnode_coordinate_binary_logistic", "auroc", COLORS["blue"]),
        ("All-node\nserialized", "results/hprc/ccre_safe_serialized_allnode_binary", "auroc", COLORS["purple"]),
        ("Window\ncoord. logit", "results/hprc/ccre_safe_window_coordinate_binary_logistic", "auroc", COLORS["gold"]),
        ("Window\nserialized", "results/hprc/ccre_safe_serialized_window_binary", "auroc", COLORS["purple"]),
        ("Window\nGAT scratch", "results/hprc/ccre_safe_gat_scratch_binary", "auroc", COLORS["green"]),
        ("Window\nGAT frozen", "results/hprc/ccre_safe_gat_frozen_binary", "auroc", COLORS["teal"]),
        ("Window\nGAT fine-tuned", "results/hprc/ccre_safe_gat_finetune_binary", "auroc", COLORS["rose"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    ax = axes[0]
    labels = [b[0] for b in binary]
    vals = [_summary(b[1])["test_metrics"][b[2]] for b in binary]
    colors = [b[3] for b in binary]
    bars = ax.bar(labels, vals, color=colors, edgecolor="#222", linewidth=0.5)
    ax.set_ylim(0.45, 0.90)
    ax.set_ylabel("AUROC")
    ax.set_title("Binary cCRE: leakage-safe comparison")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.axvline(1.5, color="#999999", lw=0.8, ls=":")
    ax.text(0.75, 0.47, "all-node", fontsize=8, color=COLORS["muted"], ha="center")
    ax.text(4.0, 0.47, "window", fontsize=8, color=COLORS["muted"], ha="center")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.3f}", ha="center", fontsize=8)
    ax.tick_params(axis="x", labelsize=8, rotation=18)

    ax = axes[1]
    tasks = ["group3", "group4", "group5"]
    labels = ["3-class", "4-class", "5-class"]
    model_specs = [
        ("scratch", "results/hprc/ccre_safe_gat_scratch_{task}", COLORS["green"]),
        ("frozen", "results/hprc/ccre_safe_gat_frozen_{task}", COLORS["teal"]),
        ("fine-tuned", "results/hprc/ccre_safe_gat_finetune_{task}", COLORS["rose"]),
    ]
    x = np.arange(len(tasks))
    width = 0.25
    for i, (model_label, path_tpl, color) in enumerate(model_specs):
        vals = [_summary(path_tpl.format(task=t))["test_metrics"]["macro_f1"] for t in tasks]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=model_label, color=color, edgecolor="#222", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.010, f"{val:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 0.50)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Window grouped cCRE: pretraining helps modestly")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=3, loc="upper right", fontsize=8)

    _save(fig, "fig5_ccre_binary_and_grouped")


def fig_ccre_category_performance() -> None:
    cats = [
        ("dELS", "dels"),
        ("enhancer-like", "enhancer_like"),
        ("pELS", "pels"),
        ("TF/CTCF", "tf_ctcf_associated"),
        ("open chromatin", "open_chromatin"),
        ("promoter-like", "promoter_like"),
        ("PLS", "pls"),
    ]
    rows = []
    for label, key in cats:
        s = _summary(f"results/hprc/ccre_safe_allnode_category_{key}_logistic")
        tm = s["test_metrics"]
        rows.append({"label": label, "auroc": tm["auroc"], "auprc": tm["auprc"], "positive_fraction": tm["positive_fraction"], "macro_f1": tm["macro_f1"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    y = np.arange(len(df))
    ax.barh(y - 0.18, df["auroc"], 0.34, label="AUROC", color=COLORS["blue"], edgecolor="#222", linewidth=0.4)
    ax.barh(y + 0.18, df["auprc"], 0.34, label="AUPRC", color=COLORS["rose"], edgecolor="#222", linewidth=0.4)
    ax.set_yticks(y, df["label"])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.95)
    ax.set_xlabel("Score")
    ax.set_title("Category-specific cCRE prediction")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    for i, frac in enumerate(df["positive_fraction"]):
        ax.text(0.02, i + 0.42, f"positive fraction={frac:.3f}", fontsize=8, color=COLORS["muted"])
    _save(fig, "fig6_ccre_category_performance")


def fig_hgsvc_imputation() -> None:
    runs = sorted((ROOT / "results/hgsvc3/graph_imputation_1hop").glob("run_*"))
    run_dir = runs[-1] if runs else None
    summary_path = run_dir / "summary.json" if run_dir else ROOT / "missing"
    candidates_path = run_dir / "candidate_edges_unique_by_pair.csv" if run_dir else ROOT / "missing"
    if not summary_path.exists() or not candidates_path.exists():
        print("skip fig7_hgsvc_imputation_candidates: imputation artifacts missing")
        return
    summary = json.loads(summary_path.read_text())
    df = pd.read_csv(candidates_path)
    counts = (
        df.assign(chrom=df["target_sn"].astype(str).str.split("|").str[-1])
        .groupby("chrom")
        .agg(
            unique_pairs=("max_p_edge", "size"),
            ge_05=("max_p_edge", lambda s: int((s >= 0.5).sum())),
            ge_08=("max_p_edge", lambda s: int((s >= 0.8).sum())),
            ge_09=("max_p_edge", lambda s: int((s >= 0.9).sum())),
        )
        .reindex(["chr19", "chr21", "chr22", "chrY"])
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    ax.hist(df["max_p_edge"], bins=bins, color=COLORS["teal"], edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color=COLORS["gold"], lw=1.5, ls="--", label="0.5")
    ax.axvline(0.8, color=COLORS["rose"], lw=1.5, ls="--", label="0.8")
    ax.set_yscale("log")
    ax.set_xlabel("Max predicted edge probability per unique candidate")
    ax.set_ylabel("Unique candidate pairs, log scale")
    ax.set_title("HGSVC candidate missing-edge scores")
    ax.legend(frameon=False)

    ax = axes[1]
    x = np.arange(len(counts))
    width = 0.26
    for i, (col, label, color) in enumerate([("ge_05", ">=0.5", COLORS["blue"]), ("ge_08", ">=0.8", COLORS["gold"]), ("ge_09", ">=0.9", COLORS["rose"])]):
        ax.bar(x + (i - 1) * width, counts[col], width, label=label, color=color, edgecolor="#222", linewidth=0.4)
    ax.set_xticks(x, counts["chrom"])
    ax.set_ylabel("Unique candidate pairs")
    ax.set_title("High-confidence candidates by HGSVC target")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=3, loc="upper right")
    metrics = summary["calibration_metrics_on_edge_pred"]
    ax.text(
        0.02,
        0.08,
        f"Calibration AUROC={metrics['auroc']:.4f}\nAUPRC={metrics['auprc']:.4f}",
        transform=ax.transAxes,
        va="bottom",
        fontsize=9,
        color=COLORS["ink"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
    )

    _save(fig, "fig7_hgsvc_imputation_candidates")


def main() -> None:
    fig_model_architecture()
    fig_ccre_pipeline()
    fig_hprc_ablation()
    fig_hgsvc_transfer()
    fig_ccre_binary_and_grouped()
    fig_ccre_category_performance()
    fig_hgsvc_imputation()


if __name__ == "__main__":
    main()
