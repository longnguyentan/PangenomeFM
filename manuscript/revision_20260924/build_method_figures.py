#!/usr/bin/env python3
"""Rebuild the manuscript's method-first diagrams as editable vector artwork.

Only schematic information is drawn; no result tables or biological test outputs
are read. Figure 2 describes the original topology-pretrained v1 encoder. Its
historical checkpoints masked exact directed query rows; reciprocal-relation
masking was repaired later. Degree and component inputs are calculated before
query-edge masking, and graph messages follow incoming edges (without reverse-
message augmentation).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent / "figures"
INK = "#263543"
MUTED = "#637381"
BORDER = "#CDD5DC"
BLUE = "#315F9B"
TEAL = "#137C78"
PURPLE = "#6B4AA0"
ORANGE = "#B16A23"
RED = "#B74D53"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "text.color": INK,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "pangenomefm-method-first-v1",
        "savefig.facecolor": "white",
    }
)


def canvas(height):
    fig = plt.figure(figsize=(6.7, height), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def txt(ax, x, y, s, *, size=8.5, weight="normal", color=INK, ha="center", **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                   ha=ha, va="center", linespacing=1.3, **kw)


def box(ax, x, y, w, h, label=None, *, color=BORDER, fill="white", size=8.5,
        weight="normal", lw=0.8, rounding=0.008):
    patch = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={rounding}",
                          linewidth=lw, edgecolor=color, facecolor=fill)
    ax.add_patch(patch)
    if label is not None:
        txt(ax, x + w / 2, y + h / 2, label, size=size, weight=weight)
    return patch


def arrow(ax, p, q, *, color=MUTED, lw=1.0, style="-|>", mutation=8, **kw):
    patch = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=mutation,
                            linewidth=lw, color=color, shrinkA=0, shrinkB=0, **kw)
    ax.add_patch(patch)
    return patch


def line(ax, points, *, color=MUTED, lw=0.9, **kw):
    ax.plot(*zip(*points), color=color, linewidth=lw, **kw)


def panel(ax, x, y, w, h, letter, title):
    box(ax, x, y, w, h, color=BORDER, rounding=0.009)
    txt(ax, x + 0.015, y + h - 0.028, letter, size=10.5, weight="bold", ha="left")
    txt(ax, x + 0.044, y + h - 0.028, title, size=9.2, weight="bold", ha="left")


def sequence(ax, cx, cy, width=0.105, *, color=BLUE, labels=True):
    letters = "ACGT"
    for i, letter in enumerate(letters):
        x = cx - width / 2 + i * width / 4
        box(ax, x, cy - 0.017, width / 4 - 0.002, 0.034,
            letter if labels else None, color=color, fill="#F3F6FA", size=7,
            rounding=0.003, lw=0.65)


def graph(ax, cx, cy, width=0.12, height=0.067, *, color=TEAL):
    pts = [(cx - width / 2, cy), (cx - width / 6, cy + height / 2),
           (cx - width / 6, cy - height / 2), (cx + width / 6, cy),
           (cx + width / 2, cy)]
    for i, j in ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4)):
        line(ax, [pts[i], pts[j]], color=color)
    for x, y in pts:
        ax.add_patch(Circle((x, y), 0.008, facecolor="white", edgecolor=color, lw=0.9))


def embedding(ax, cx, cy, *, color=PURPLE, width=0.12, height=0.032):
    shades = ["#E4DAF2", "#F6F2FA", "#BFAAD9", "#D6C8E6", "#F6F2FA", "#C8B6DF"]
    for i in range(6):
        ax.add_patch(Rectangle((cx - width / 2 + i * width / 6, cy - height / 2),
                               width / 6, height, facecolor=shades[i],
                               edgecolor=color, linewidth=0.55))


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        metadata = {"Creator": "PangenomeFM manuscript vector-figure builder"}
        if suffix == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        else:
            metadata.update(Date=None)
        target = OUT / f"{stem}.{suffix}"
        fig.savefig(target, metadata=metadata)
        if suffix == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
    plt.close(fig)


def paradigms():
    fig, ax = canvas(4.75)
    txt(ax, 0.025, 0.964, "Four routes to learned genomic representations", size=12,
        weight="bold", ha="left")
    cols = [0.35, 0.535, 0.715, 0.902]
    for x, label in zip(cols, ["Source", "Model input", "Learning", "Representation"]):
        txt(ax, x, 0.895, label, size=8.5, weight="bold", color=MUTED)
    rows = [
        ("a", "Linear sequence\nmodels", "DNABERT · NT", "sequence", "Tokens",
         "Sequence\nself-supervision", "Sequence\nembedding", BLUE),
        ("b", "Graph-derived\nsequence models", "DeepGene", "graph", "Linearized\nsequence",
         "Sequence\nlearning", "Sequence\nembedding", BLUE),
        ("c", "Task-specific\ngraph models", "PangenomeX", "population", "CNV\nnetwork",
         "Task\nsupervision", "Graph\nembedding", ORANGE),
        ("d", "Topology-native\npretraining", "PangenomeFM", "graph", "Oriented\nnative graph",
         "Masked\nadjacency", "Graph\nembedding", TEAL),
    ]
    for k, (letter, title, example, source, model_input, learning, output, color) in enumerate(rows):
        y = 0.69 - k * 0.202
        box(ax, 0.02, y, 0.96, 0.181, color=BORDER, fill="#FCFDFE")
        txt(ax, 0.039, y + 0.148, letter, weight="bold", size=10.5, ha="left")
        txt(ax, 0.077, y + 0.116, title, size=9, weight="bold", ha="left")
        txt(ax, 0.077, y + 0.040, example, size=7.8, color=color, ha="left")
        glyph_y = y + 0.113
        if source == "sequence":
            sequence(ax, cols[0], glyph_y, width=0.10)
            source_label = "Sequence"
        else:
            graph(ax, cols[0], glyph_y, width=0.10, height=0.055, color=color)
            source_label = "Population\nrelationships" if source == "population" else "Pangenome graph"
        txt(ax, cols[0], y + 0.043, source_label, size=7.7)
        box(ax, cols[1] - 0.071, y + 0.065, 0.142, 0.092,
            model_input, size=8.5, color=color)
        box(ax, cols[2] - 0.074, y + 0.065, 0.148, 0.092,
            learning, size=8.2, color=color)
        embedding(ax, cols[3], glyph_y, width=0.111)
        txt(ax, cols[3], y + 0.042, output, size=7.8)
        arrow(ax, (0.410, glyph_y), (0.456, glyph_y))
        arrow(ax, (0.613, glyph_y), (0.634, glyph_y))
        arrow(ax, (0.796, glyph_y), (0.836, glyph_y))
    save(fig, "figure1_paradigms_methodfirst")


def method(stem="figure2_method_methodfirst"):
    fig, ax = canvas(6.8)
    txt(ax, 0.02, 0.971, "PangenomeFM: topology-pretrained representation", size=12,
        weight="bold", ha="left")

    # a. The minibatch candidate set defines which positive relations are hidden.
    panel(ax, 0.02, 0.703, 0.96, 0.239, "a", "Oriented graph, query masking and shared inputs")
    for x in (0.306, 0.584):
        line(ax, [(x, 0.718), (x, 0.885)], color=BORDER, lw=0.65)
    txt(ax, 0.164, 0.874, "Oriented handles", weight="bold", size=8.2, color=TEAL)
    pts = [(0.068, 0.804), (0.142, 0.841), (0.142, 0.768),
           (0.218, 0.804), (0.276, 0.804)]
    for i, j in ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4)):
        arrow(ax, (pts[i][0] + 0.014, pts[i][1]),
              (pts[j][0] - 0.016, pts[j][1]), color=TEAL, mutation=6, lw=0.8)
    for (x, y), label in zip(pts, ["+", "+", "−", "+", "+"]):
        box(ax, x - 0.017, y - 0.013, 0.034, 0.026, label, color=TEAL,
            fill="#EFF8F6", size=7.5, rounding=0.004)
    txt(ax, 0.163, 0.731, "Segments + genomic positions", size=7.5, color=MUTED)

    txt(ax, 0.443, 0.874, "Candidate minibatch", weight="bold", size=8.2)
    txt(ax, 0.443, 0.853, "≤512 ordered pairs", size=7.6, color=MUTED)
    txt(ax, 0.342, 0.816, r"$u^+$", size=9)
    txt(ax, 0.543, 0.816, r"$v^+$", size=9)
    arrow(ax, (0.365, 0.816), (0.52, 0.816), color=RED, lw=1.1, ls=(0, (4, 3)))
    txt(ax, 0.442, 0.816, "×", color=RED, weight="bold", size=12,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.2))
    txt(ax, 0.443, 0.781, "Exact directed query rows", size=7.3, color=RED)
    txt(ax, 0.443, 0.759, "Remove before message passing", size=7.1, color=RED)

    txt(ax, 0.78, 0.874, "Seven shared structural inputs", weight="bold", size=8.2)
    txt(ax, 0.781, 0.841, "Offset · length · source rank", size=8)
    txt(ax, 0.781, 0.813, "Reference · degree · orientation", size=8)
    txt(ax, 0.781, 0.785, "Component ID", size=8)
    txt(ax, 0.781, 0.759, "Degree / component: unmasked graph", size=7.15, color=RED)
    box(ax, 0.647, 0.714, 0.273, 0.032, "Shared projection: 7 → 48D",
        color=PURPLE, fill="#F6F2FA", size=8)

    # b. Both streams consume the same evolving hidden state at every layer.
    panel(ax, 0.02, 0.335, 0.96, 0.346, "b", "Coupled encoder")
    txt(ax, 0.96, 0.653, "2 layers · 4 heads · 48D", size=8.5, weight="bold",
        color=PURPLE, ha="right")
    box(ax, 0.21, 0.556, 0.252, 0.068, "Coordinate attention", color=ORANGE,
        fill="#FCF6EE", size=8.5, weight="bold")
    txt(ax, 0.336, 0.573, "Genomic position + orientation", size=7.0, color=ORANGE)
    # Lift main label to leave room for its keyword sublabel.
    ax.texts[-2].set_y(0.603)
    box(ax, 0.21, 0.419, 0.252, 0.067, "Incoming-edge GAT", color=TEAL,
        fill="#EFF8F6", size=8.5, weight="bold")
    txt(ax, 0.336, 0.435, "Masked message graph", size=7.2, color=TEAL)
    ax.texts[-2].set_y(0.465)
    box(ax, 0.055, 0.494, 0.082, 0.052, r"$h^{(\ell)}$", color=PURPLE,
        fill="#F6F2FA", size=11)
    line(ax, [(0.137, 0.52), (0.167, 0.52)], color=PURPLE)
    line(ax, [(0.167, 0.453), (0.167, 0.59)], color=PURPLE)
    arrow(ax, (0.167, 0.59), (0.21, 0.59), color=PURPLE)
    arrow(ax, (0.167, 0.453), (0.21, 0.453), color=PURPLE)
    txt(ax, 0.102, 0.454, "Shared\nhidden state", size=7.5, color=PURPLE)
    txt(ax, 0.044, 0.389, r"$h^{(0)}$: node projection", size=7.2, color=MUTED, ha="left")

    box(ax, 0.543, 0.482, 0.128, 0.077, "Elementwise\ngate", color=PURPLE,
        fill="#F6F2FA", size=7.8, weight="bold")
    arrow(ax, (0.462, 0.59), (0.566, 0.56), color=ORANGE)
    arrow(ax, (0.462, 0.453), (0.566, 0.481), color=TEAL)
    txt(ax, 0.607, 0.429, r"$g=\sigma(W[h_c\Vert h_g]+b)$", size=8)
    txt(ax, 0.607, 0.398, r"$z=g\odot h_c+(1-g)\odot h_g$", size=8)

    box(ax, 0.736, 0.482, 0.159, 0.077, "Residual\n+ LayerNorm", color=PURPLE,
        fill="#F6F2FA", size=8.3, weight="bold")
    arrow(ax, (0.671, 0.52), (0.736, 0.52), color=PURPLE)
    line(ax, [(0.096, 0.546), (0.096, 0.635), (0.815, 0.635), (0.815, 0.578)],
         color=PURPLE, lw=0.85)
    arrow(ax, (0.815, 0.578), (0.815, 0.559), color=PURPLE)
    txt(ax, 0.607, 0.622, r"Residual $h^{(\ell)}$", size=7.5, color=PURPLE)
    arrow(ax, (0.895, 0.52), (0.946, 0.52), color=PURPLE)
    txt(ax, 0.932, 0.481, r"$h^{(\ell+1)}$", size=9, color=PURPLE)
    txt(ax, 0.816, 0.429, r"$h^{(\ell+1)}=\mathrm{LN}(z+h^{(\ell)})$", size=7.8)
    txt(ax, 0.50, 0.357, "Repeat the coupled layer twice → segment vectors", size=8,
        color=MUTED)

    # c. The decoder is ordered: swapping the pair changes its MLP input.
    panel(ax, 0.02, 0.025, 0.447, 0.284, "c", "Ordered edge reconstruction")
    embedding(ax, 0.132, 0.244, width=0.113, height=0.020)
    embedding(ax, 0.352, 0.244, width=0.113, height=0.020)
    txt(ax, 0.132, 0.267, r"$h_u$", size=9, color=PURPLE)
    txt(ax, 0.352, 0.267, r"$h_v$", size=9, color=PURPLE)
    line(ax, [(0.132, 0.234), (0.132, 0.217), (0.353, 0.217), (0.353, 0.234)],
         color=PURPLE)
    arrow(ax, (0.243, 0.217), (0.243, 0.202), color=PURPLE)
    box(ax, 0.112, 0.159, 0.263, 0.042, r"Ordered $[h_u\Vert h_v]$ → MLP → $\sigma$",
        color=PURPLE, fill="#F6F2FA", size=8.0)
    arrow(ax, (0.243, 0.159), (0.243, 0.137), color=PURPLE)
    txt(ax, 0.243, 0.122, r"$p(u\to v)$", size=10, color=PURPLE)
    txt(ax, 0.243, 0.090, "Focal binary cross-entropy", size=8.3, weight="bold")
    txt(ax, 0.243, 0.060, "+ adjacency    − distance-matched non-edge", size=7.3)
    txt(ax, 0.243, 0.037, "Self-supervision from graph relations", size=7.2, color=MUTED)

    # d. Sequence embeddings are a separate frozen downstream branch.
    panel(ax, 0.489, 0.025, 0.491, 0.284, "d", "Frozen downstream reuse")
    for x, label, color, fill in (
        (0.513, "T\nGraph encoder", TEAL, "#EFF8F6"),
        (0.672, "S\nNT encoder", BLUE, "#F1F5FB"),
        (0.831, "C\nGenomic inputs", ORANGE, "#FCF6EE"),
    ):
        box(ax, x, 0.203, 0.125, 0.057, label, color=color, fill=fill, size=7.5)
        if label[0] in ("T", "S"):
            txt(ax, x + 0.0625, 0.188, "frozen", size=7.4, color=color)
        line(ax, [(x + 0.0625, 0.178), (x + 0.0625, 0.157)], color=color)
    line(ax, [(0.5755, 0.157), (0.8935, 0.157)], color=MUTED)
    arrow(ax, (0.7345, 0.157), (0.7345, 0.137), color=MUTED)
    box(ax, 0.614, 0.093, 0.241, 0.044, "Linear probe", color=PURPLE,
        fill="#F6F2FA", size=9, weight="bold")
    txt(ax, 0.7345, 0.071, "Fixed feature combinations", size=7.3, color=MUTED)
    txt(ax, 0.7345, 0.043, "cCRE classification · insertion / deletion", size=7.6)
    save(fig, stem)


if __name__ == "__main__":
    paradigms()
    method()
