#!/usr/bin/env python3
"""Generate the next-stage conceptual Figure 1 and method Figure 2."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures/next_stage"
MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/private/tmp/pangenomefm-mpl"))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402


COLORS = {
    "ink": "#17212B",
    "muted": "#607080",
    "line": "#C8D2DC",
    "blue": "#2F6FBB",
    "blue_light": "#EAF3FD",
    "teal": "#16817A",
    "teal_light": "#E8F7F4",
    "gold": "#C9821A",
    "gold_light": "#FFF5E3",
    "purple": "#7057A3",
    "purple_light": "#F2EEFA",
    "rose": "#B94A62",
    "rose_light": "#FCEEF1",
    "green": "#4C956C",
    "green_light": "#EBF7F0",
    "gray_light": "#F5F7F9",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": COLORS["ink"],
        "axes.edgecolor": COLORS["ink"],
        "savefig.facecolor": "white",
    }
)


def save_all(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(
            OUT / f"{stem}.{suffix}",
            dpi=320,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}/{stem}.{{svg,pdf,png}}")


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    *,
    face: str = "white",
    edge: str = COLORS["line"],
    title_color: str | None = None,
    fontsize: float = 9.2,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.35,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    if body:
        ax.text(
            x + w / 2,
            y + h * 0.64,
            title,
            ha="center",
            va="center",
            fontsize=fontsize,
            weight="bold",
            color=title_color or edge,
        )
        ax.text(
            x + w / 2,
            y + h * 0.32,
            body,
            ha="center",
            va="center",
            fontsize=fontsize - 1.0,
            color=COLORS["ink"],
            linespacing=1.2,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=fontsize,
            weight="bold",
            color=title_color or COLORS["ink"],
            linespacing=1.18,
        )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = COLORS["muted"],
    *,
    dashed: bool = False,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            linestyle="--" if dashed else "-",
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def mini_graph(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    masked: bool = False,
    show_paths: bool = True,
) -> None:
    nodes = {
        "a": (x, y + height / 2),
        "b": (x + width * 0.25, y + height / 2),
        "c": (x + width * 0.55, y + height / 2),
        "d": (x + width, y + height / 2),
        "u": (x + width * 0.48, y + height),
        "v": (x + width * 0.48, y),
    }
    reference = [("a", "b"), ("b", "c"), ("c", "d")]
    alternate = [("b", "u"), ("u", "d"), ("b", "v"), ("v", "d")]
    for left, right in reference:
        if masked and (left, right) == ("b", "c"):
            continue
        ax.plot(
            [nodes[left][0], nodes[right][0]],
            [nodes[left][1], nodes[right][1]],
            color=COLORS["blue"],
            lw=2.4,
            zorder=1,
        )
    if show_paths:
        for left, right in alternate:
            ax.plot(
                [nodes[left][0], nodes[right][0]],
                [nodes[left][1], nodes[right][1]],
                color=COLORS["gold"],
                lw=1.8,
                zorder=1,
            )
    for name, (nx, ny) in nodes.items():
        if not show_paths and name in {"u", "v"}:
            continue
        color = COLORS["blue"] if name in {"a", "b", "c", "d"} else COLORS["gold"]
        ax.add_patch(
            Circle((nx, ny), width * 0.035, facecolor=color, edgecolor="white", lw=1, zorder=3)
        )
    if masked:
        midx = (nodes["b"][0] + nodes["c"][0]) / 2
        midy = nodes["b"][1]
        ax.plot(
            [midx - width * 0.025, midx + width * 0.025],
            [midy - height * 0.11, midy + height * 0.11],
            color=COLORS["rose"],
            lw=2.2,
        )
        ax.plot(
            [midx - width * 0.025, midx + width * 0.025],
            [midy + height * 0.11, midy - height * 0.11],
            color=COLORS["rose"],
            lw=2.2,
        )


def figure1() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 6.6))
    fig.subplots_adjust(wspace=0.12, top=0.84, bottom=0.08, left=0.03, right=0.98)
    fig.suptitle(
        "Three genomic modeling paradigms",
        fontsize=20,
        weight="bold",
        x=0.04,
        ha="left",
        y=0.96,
    )
    fig.text(
        0.04,
        0.905,
        "PangenomeFM learns a reusable representation from population-graph topology rather than treating sequence as its model tokens.",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    panels = [
        (
            "A",
            "Linear genomic modeling",
            COLORS["blue"],
            COLORS["blue_light"],
            [
                ("Reference / haplotype", "A C G T sequence"),
                ("Sequence representation", "tokens or k-mers"),
                ("Sequence model", "genomic foundation model"),
                ("Embedding / prediction", "sequence-native output"),
            ],
        ),
        (
            "B",
            "Task-specific graph modeling",
            COLORS["gold"],
            COLORS["gold_light"],
            [
                ("Pangenome graph", "nodes, links, paths"),
                ("Graph features / GNN", "task-designed representation"),
                ("Supervised training", "one biological endpoint"),
                ("Task prediction", "task-specific output"),
            ],
        ),
        (
            "C",
            "PangenomeFM",
            COLORS["teal"],
            COLORS["teal_light"],
            [
                ("Population pangenome", "oriented topology + coordinates"),
                ("Masked pretraining", "hide and reconstruct relations"),
                ("Topology representation", "pretrained node embeddings"),
                ("Frozen transfer / probes", "graphs, cCRE, SV"),
            ],
        ),
    ]
    for ax, (letter, title, edge, face, steps) in zip(axes, panels):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.02, 0.97, letter, fontsize=16, weight="bold", color=edge, va="top")
        ax.text(0.10, 0.97, title, fontsize=13.5, weight="bold", va="top")
        y_positions = [0.73, 0.52, 0.31, 0.10]
        for index, ((step_title, body), y) in enumerate(zip(steps, y_positions)):
            box(
                ax,
                0.12,
                y,
                0.76,
                0.14,
                step_title,
                body,
                face=face if index in {0, 2} else "white",
                edge=edge if index in {0, 2} else COLORS["line"],
                title_color=edge,
            )
            if index < 3:
                arrow(ax, (0.50, y), (0.50, y_positions[index + 1] + 0.14), edge)
        if letter == "A":
            ax.text(0.5, 0.885, "A C G T  ·  A C G T", ha="center", color=edge, weight="bold")
        else:
            mini_graph(ax, 0.36, 0.865, 0.28, 0.075, show_paths=True)
        if letter == "C":
            ax.add_patch(
                FancyBboxPatch(
                    (0.05, 0.055),
                    0.90,
                    0.87,
                    boxstyle="round,pad=0.01,rounding_size=0.03",
                    fill=False,
                    edgecolor=edge,
                    linewidth=2.0,
                )
            )
    save_all(fig, "figure1_three_paradigms")


def figure2() -> None:
    fig, ax = plt.subplots(figsize=(15.2, 8.3))
    fig.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.04)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.97, "PangenomeFM pretraining and representation reuse", fontsize=20, weight="bold", va="top")
    ax.text(
        0.02,
        0.925,
        "The queried relation is removed from both directed and reverse-complement message-passing traversals before scoring.",
        fontsize=10.5,
        color=COLORS["muted"],
        va="top",
    )

    # Panel A
    ax.text(0.02, 0.84, "A", fontsize=15, weight="bold", color=COLORS["blue"])
    ax.text(0.055, 0.84, "Population graph input", fontsize=12.5, weight="bold")
    box(ax, 0.02, 0.56, 0.20, 0.23, "", face=COLORS["blue_light"], edge=COLORS["blue"])
    ax.text(0.12, 0.735, "Oriented segments", ha="center", weight="bold", color=COLORS["blue"], fontsize=9.2)
    mini_graph(ax, 0.055, 0.635, 0.13, 0.075, show_paths=True)
    ax.text(0.12, 0.59, "reference + alternate paths\ngenomic coordinates", ha="center", va="center", fontsize=8.1, linespacing=1.15)
    ax.text(0.045, 0.575, "+", color=COLORS["blue"], weight="bold")
    ax.text(0.185, 0.575, "−", color=COLORS["gold"], weight="bold")

    # Panel B
    ax.text(0.26, 0.84, "B", fontsize=15, weight="bold", color=COLORS["rose"])
    ax.text(0.295, 0.84, "Query-edge masking", fontsize=12.5, weight="bold")
    box(ax, 0.26, 0.56, 0.20, 0.23, "", face=COLORS["rose_light"], edge=COLORS["rose"])
    ax.text(0.36, 0.735, "Hidden target", ha="center", weight="bold", color=COLORS["rose"], fontsize=9.2)
    mini_graph(ax, 0.295, 0.635, 0.13, 0.075, masked=True, show_paths=True)
    ax.text(0.36, 0.59, "edge absent during message passing", ha="center", va="center", fontsize=8.0)
    ax.text(0.36, 0.57, "score later", ha="center", fontsize=8, color=COLORS["rose"], weight="bold")

    # Panel C
    ax.text(0.50, 0.84, "C", fontsize=15, weight="bold", color=COLORS["purple"])
    ax.text(0.535, 0.84, "Two-stream encoder", fontsize=12.5, weight="bold")
    box(ax, 0.50, 0.675, 0.20, 0.105, "Coordinate / context stream", "ordered offsets + position", face=COLORS["blue_light"], edge=COLORS["blue"], fontsize=8.6)
    box(ax, 0.50, 0.535, 0.20, 0.105, "Graph-attention stream", "message passing on links", face=COLORS["green_light"], edge=COLORS["green"], fontsize=8.6)
    box(ax, 0.735, 0.605, 0.095, 0.105, "Fusion gate", "node\nembedding", face=COLORS["purple_light"], edge=COLORS["purple"], fontsize=8.5)
    arrow(ax, (0.70, 0.725), (0.735, 0.675), COLORS["blue"])
    arrow(ax, (0.70, 0.585), (0.735, 0.64), COLORS["green"])

    # Panel D
    ax.text(0.86, 0.84, "D", fontsize=15, weight="bold", color=COLORS["gold"])
    ax.text(0.895, 0.84, "Reconstruct", fontsize=12.5, weight="bold")
    box(ax, 0.855, 0.58, 0.125, 0.18, "Endpoint pair", "h(u), h(v)\n→ edge score", face=COLORS["gold_light"], edge=COLORS["gold"], fontsize=8.8)
    arrow(ax, (0.83, 0.655), (0.855, 0.655), COLORS["purple"])
    arrow(ax, (0.22, 0.675), (0.26, 0.675), COLORS["muted"])
    arrow(ax, (0.46, 0.675), (0.50, 0.675), COLORS["muted"])

    # Separation between pretraining and reuse.
    ax.plot([0.03, 0.97], [0.46, 0.46], color=COLORS["line"], lw=1.2)
    ax.text(0.03, 0.475, "SELF-SUPERVISED PRETRAINING", fontsize=8.5, color=COLORS["muted"], weight="bold")
    ax.text(0.03, 0.425, "E", fontsize=15, weight="bold", color=COLORS["teal"])
    ax.text(0.068, 0.425, "Freeze once, reuse across analyses", fontsize=13, weight="bold")

    box(ax, 0.05, 0.19, 0.18, 0.15, "Pretrained encoder", "fixed topology representation", face=COLORS["teal_light"], edge=COLORS["teal"])
    arrow(ax, (0.23, 0.265), (0.31, 0.265), COLORS["teal"])
    box(ax, 0.31, 0.19, 0.16, 0.15, "Embeddings", "node / region features", face="white", edge=COLORS["purple"])
    destinations = [
        (0.55, 0.31, "Cross-graph transfer", COLORS["blue"], COLORS["blue_light"]),
        (0.76, 0.31, "cCRE probe", COLORS["gold"], COLORS["gold_light"]),
        (0.55, 0.10, "SV probe", COLORS["rose"], COLORS["rose_light"]),
        (0.76, 0.10, "Future biological probes", COLORS["green"], COLORS["green_light"]),
    ]
    for x, y, label, edge, face in destinations:
        box(ax, x, y, 0.18, 0.105, label, face=face, edge=edge, fontsize=8.7)
        arrow(ax, (0.47, 0.265), (x, y + 0.052), COLORS["purple"])
    ax.text(
        0.05,
        0.08,
        "No nucleotide A/C/G/T strings are used as encoder tokens in the current model.",
        fontsize=9,
        color=COLORS["muted"],
    )
    save_all(fig, "figure2_pretraining_and_reuse")


def main() -> None:
    figure1()
    figure2()


if __name__ == "__main__":
    main()
