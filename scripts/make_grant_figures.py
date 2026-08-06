#!/usr/bin/env python3
"""Generate compact grant figures for the Africa 6K GraphGenomeFM proposal."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "grant" / "figures"
MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/private/tmp/graphgenomefm-mpl"))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


COL = {
    "ink": "#17212b",
    "muted": "#64748b",
    "line": "#cbd5e1",
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
    "light_gray": "#f8fafc",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "savefig.facecolor": "white",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    *,
    fc: str = "white",
    ec: str = COL["line"],
    title_color: str | None = None,
    title_size: float = 9.5,
    body_size: float = 8.0,
    lw: float = 1.2,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.018,
        y + h - 0.032,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=title_color or ec,
    )
    if body:
        ax.text(
            x + 0.018,
            y + h - 0.078,
            body,
            ha="left",
            va="top",
            fontsize=body_size,
            color=COL["ink"],
            linespacing=1.18,
        )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = COL["muted"]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def mini_graph(ax: plt.Axes, x: float, y: float, s: float = 1.0) -> None:
    nodes = {
        "a": (x, y),
        "b": (x + 0.055 * s, y),
        "c": (x + 0.110 * s, y),
        "d": (x + 0.165 * s, y),
        "e": (x + 0.065 * s, y + 0.055 * s),
        "f": (x + 0.125 * s, y - 0.050 * s),
    }
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("b", "e"), ("e", "d"), ("b", "f"), ("f", "d")]
    for u, v in edges:
        ax.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]], color="#8da2bd", lw=1.3, zorder=1)
    for i, (nx, ny) in enumerate(nodes.values()):
        color = COL["blue"] if i < 4 else COL["gold"]
        ax.add_patch(Circle((nx, ny), 0.011 * s, facecolor=color, edgecolor="white", lw=0.8, zorder=3))


def token_stack(ax: plt.Axes, x: float, y: float, color: str, n: int = 5) -> None:
    for i in range(n):
        ax.add_patch(Rectangle((x, y + i * 0.017), 0.010, 0.014, facecolor=color, edgecolor="white", lw=0.25))


def fig_graphgenomefm_gijoe() -> None:
    fig, ax = plt.subplots(figsize=(13.8, 6.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.965, "Aim 1a. Graph-native and genotype foundation models for Africa 6K resources", fontsize=15, fontweight="bold", color=COL["ink"])

    # Inputs.
    box(
        ax,
        0.025,
        0.67,
        0.19,
        0.21,
        "Reference pangenomes",
        "HPRC and HGSVC graph\nassemblies in rGFA/GFA\nplus phased haplotypes",
        fc=COL["light_blue"],
        ec=COL["blue"],
    )
    mini_graph(ax, 0.065, 0.699, 0.55)

    box(
        ax,
        0.025,
        0.38,
        0.19,
        0.20,
        "Africa 6K WGS",
        "High-coverage TOPMed\nAfrican genomes with\npopulation metadata",
        fc=COL["light_gold"],
        ec=COL["gold"],
    )
    box(
        ax,
        0.025,
        0.12,
        0.19,
        0.17,
        "TOPMed/1000GP panel",
        "Dense phased genotypes\nand dosage/posterior\nprobabilities",
        fc=COL["light_gray"],
        ec=COL["muted"],
    )

    # GraphGenomeFM architecture.
    box(
        ax,
        0.275,
        0.69,
        0.18,
        0.16,
        "Coordinate stream",
        "Multi-scale RoPE\nself-attention over\nordered offsets",
        fc=COL["light_blue"],
        ec=COL["blue"],
    )
    ax.plot([0.305, 0.425], [0.725, 0.725], color=COL["blue"], lw=1.0)
    for p in [0.31, 0.35, 0.39, 0.42]:
        ax.add_patch(Circle((p, 0.725), 0.006, color=COL["blue"]))
    for i in range(3):
        ax.plot([0.31, 0.35 + 0.035 * i, 0.42], [0.733, 0.775 + 0.015 * i, 0.733], color=COL["blue"], lw=0.7, ls="--")

    box(
        ax,
        0.275,
        0.47,
        0.18,
        0.16,
        "Graph stream",
        "GAT message passing\non observed pangenome\nadjacencies",
        fc=COL["light_green"],
        ec=COL["green"],
    )
    mini_graph(ax, 0.305, 0.512, 1.1)

    box(
        ax,
        0.275,
        0.25,
        0.18,
        0.15,
        "Position/orientation",
        "Distance buckets,\nstrand, local branching,\nand reference status",
        fc=COL["light_gold"],
        ec=COL["gold"],
    )

    box(
        ax,
        0.505,
        0.48,
        0.20,
        0.22,
        "Fusion gate",
        "Learns per-node weights\nfor coordinate and graph\nsignals, then produces\nembeddings h(v)",
        fc=COL["light_purple"],
        ec=COL["purple"],
    )
    token_stack(ax, 0.535, 0.545, "#8bb8e8")
    token_stack(ax, 0.595, 0.545, "#8fd19e")
    token_stack(ax, 0.655, 0.545, "#9f7aea")
    box(
        ax,
        0.755,
        0.59,
        0.205,
        0.17,
        "Graph edge imputation",
        "Score missing/uncertain\ncandidate adjacencies;\ncalibrate and prioritize",
        fc=COL["light_rose"],
        ec=COL["rose"],
    )
    box(
        ax,
        0.755,
        0.40,
        0.205,
        0.16,
        "Graph-aware genomes",
        "Individual graph paths,\nedge confidence scores,\nand reusable embeddings",
        fc=COL["light_purple"],
        ec=COL["purple"],
    )

    # GI-Joe pathway.
    box(
        ax,
        0.275,
        0.055,
        0.22,
        0.14,
        "GI-Joe long-context imputation",
        "Long-context Transformer\nfor phased variants without\nhard segmentation",
        fc=COL["light_gray"],
        ec=COL["muted"],
        title_color=COL["ink"],
    )
    box(
        ax,
        0.545,
        0.055,
        0.20,
        0.14,
        "Full-spectrum genotypes",
        "SNVs, indels, SVs,\ndosages, posterior\nuncertainty",
        fc=COL["light_green"],
        ec=COL["green"],
    )
    box(
        ax,
        0.790,
        0.055,
        0.17,
        0.14,
        "Reference panel",
        "Africa 6K phased\nhaplotypes for downstream\nQTL and risk analyses",
        fc=COL["light_blue"],
        ec=COL["blue"],
    )

    # Output ribbon.
    ax.text(0.790, 0.895, "Community deliverables", fontsize=10.5, fontweight="bold", color=COL["ink"])
    for i, (label, color) in enumerate(
        [
            ("GFA/GBZ graphs", COL["blue"]),
            ("VCF/BCF variants", COL["green"]),
            ("NPZ/HDF5 embeddings", COL["purple"]),
            ("cCRE/QTL maps", COL["rose"]),
        ]
    ):
        yy = 0.880 - i * 0.024
        ax.add_patch(Circle((0.800, yy), 0.007, facecolor=color, edgecolor="white", lw=0.5))
        ax.text(0.815, yy, label, va="center", fontsize=8.5, color=COL["ink"])

    arrow(ax, (0.215, 0.78), (0.275, 0.77), COL["blue"])
    arrow(ax, (0.215, 0.47), (0.275, 0.54), COL["gold"])
    arrow(ax, (0.455, 0.77), (0.505, 0.62), COL["blue"])
    arrow(ax, (0.455, 0.55), (0.505, 0.58), COL["green"])
    arrow(ax, (0.455, 0.33), (0.505, 0.52), COL["gold"])
    arrow(ax, (0.705, 0.60), (0.755, 0.70), COL["rose"])
    arrow(ax, (0.858, 0.62), (0.858, 0.56), COL["rose"])
    arrow(ax, (0.215, 0.205), (0.275, 0.105), COL["muted"])
    arrow(ax, (0.495, 0.108), (0.545, 0.108), COL["muted"])
    arrow(ax, (0.745, 0.108), (0.790, 0.108), COL["blue"])
    arrow(ax, (0.858, 0.40), (0.858, 0.175), COL["purple"])

    save(fig, "graphgenomefm_gijoe_africa6k")


def fig_multimodal_resource() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.955, "Aim 1b/2. Map functional genomics to graph coordinates and release AI-ready resources", fontsize=13.5, fontweight="bold", color=COL["ink"])

    inputs = [
        ("RNA-seq / GTEx", "FASTQ/BAM/CRAM\nexpression + ASE", COL["blue"], COL["light_blue"]),
        ("ENCODE / EN-TEx", "cCREs, ATAC, ChIP\nTF/histone marks", COL["green"], COL["light_green"]),
        ("4DN Hi-C", "Hi-C contacts\nloops and TADs", COL["gold"], COL["light_gold"]),
        ("Africa 6K traits", "phenotypes\nancestry metadata", COL["rose"], COL["light_rose"]),
    ]
    for i, (title, body, ec, fc) in enumerate(inputs):
        y = 0.70 - i * 0.17
        box(ax, 0.03, y, 0.20, 0.135, title, body, fc=fc, ec=ec, body_size=7.8)
        arrow(ax, (0.22, y + 0.065), (0.30, 0.50), ec)

    box(
        ax,
        0.30,
        0.64,
        0.20,
        0.17,
        "Graph-coordinate mapping",
        "Project functional intervals,\nreads, contacts, and variants\nto GRCh38/T2T and graph paths",
        fc=COL["light_gray"],
        ec=COL["muted"],
        title_color=COL["ink"],
    )
    box(
        ax,
        0.30,
        0.40,
        0.20,
        0.17,
        "Multimodal graph",
        "Variant, segment,\nregulatory and gene nodes;\nspatial/statistical edges",
        fc=COL["light_purple"],
        ec=COL["purple"],
    )
    box(
        ax,
        0.30,
        0.16,
        0.20,
        0.15,
        "Normalization and QC",
        "Ancestry-aware harmonization,\nbatch correction,\nmissing-modality handling",
        fc=COL["light_blue"],
        ec=COL["blue"],
    )

    arrow(ax, (0.50, 0.725), (0.58, 0.62), COL["muted"])
    arrow(ax, (0.50, 0.485), (0.58, 0.51), COL["purple"])
    arrow(ax, (0.50, 0.235), (0.58, 0.39), COL["blue"])

    box(
        ax,
        0.58,
        0.52,
        0.18,
        0.19,
        "Pretrained models",
        "GraphGenomeFM encoder,\nmultimodal contrastive\nlearning, task adapters",
        fc=COL["light_green"],
        ec=COL["green"],
    )
    box(
        ax,
        0.58,
        0.27,
        0.18,
        0.17,
        "Evaluation tasks",
        "cCRE prediction,\ngene expression/splicing,\n3D contacts, QTL recovery",
        fc=COL["light_gold"],
        ec=COL["gold"],
    )

    arrow(ax, (0.76, 0.61), (0.82, 0.69), COL["green"])
    arrow(ax, (0.76, 0.35), (0.82, 0.51), COL["gold"])

    box(
        ax,
        0.82,
        0.60,
        0.16,
        0.15,
        "Standard formats",
        "GFA/GBZ, VCF/BCF,\nBED/bigWig, .hic/.cool,\nAnnData/HDF5/Zarr",
        fc=COL["light_gray"],
        ec=COL["muted"],
        title_color=COL["ink"],
    )
    box(
        ax,
        0.82,
        0.39,
        0.16,
        0.15,
        "AI-ready outputs",
        "NPZ/HDF5 embeddings,\nPyTorch checkpoints,\nmodel/data cards, APIs",
        fc=COL["light_purple"],
        ec=COL["purple"],
    )
    box(
        ax,
        0.82,
        0.18,
        0.16,
        0.15,
        "Access model",
        "Public aggregates;\ncontrolled data;\nprivacy adapters",
        fc=COL["light_rose"],
        ec=COL["rose"],
    )

    ax.text(
        0.03,
        0.045,
        "The resource links pangenome-aware variants, regulatory elements, expression, and 3D genome contacts into machine-readable representations for reusable AI analysis.",
        fontsize=8.7,
        color=COL["muted"],
    )

    save(fig, "multimodal_graph_ai_resource")


def main() -> None:
    fig_graphgenomefm_gijoe()
    fig_multimodal_resource()


if __name__ == "__main__":
    main()
