"""
src/analysis/latent.py

Latent space visualization of pangenome node embeddings.

What we produce:
  1. UMAP scatter: nodes colored by degree (hub vs linear)
  2. UMAP scatter: nodes colored by is_grch38 (reference vs alt)
  3. UMAP scatter: nodes colored by clustering coefficient
  4. UMAP scatter: nodes colored by connected component ID
  5. UMAP scatter: nodes colored by attention radius (branching proximity)
  6. Degree distribution plot (log-log for scale-free check)
  7. Clustering coefficient distribution
  8. Summary stats panel
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# UMAP embedding of node features
# ---------------------------------------------------------------------------


def compute_umap_embedding(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute 2D UMAP embedding of node feature matrix X (N, D).
    Returns (N, 2) array.

    Requires: pip install umap-learn
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn not installed. Run: pip install umap-learn")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric=metric,
    )
    return reducer.fit_transform(X)


def compute_pca_fallback(X: np.ndarray) -> np.ndarray:
    """
    PCA fallback if UMAP not available. Returns (N, 2).
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------


def _categorical_colors(values: np.ndarray, cmap_name: str = "tab20") -> np.ndarray:
    """Map integer category IDs to RGBA colors."""
    uniq = np.unique(values)
    cmap = cm.get_cmap(cmap_name, max(len(uniq), 1))
    id_map = {v: i for i, v in enumerate(uniq)}
    colors = np.array([cmap(id_map[v]) for v in values])
    return colors


def _continuous_colors(values: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """Map continuous values to RGBA colors."""
    vmin, vmax = values.min(), values.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    return cmap(norm(values))


# ---------------------------------------------------------------------------
# Individual plot helpers
# ---------------------------------------------------------------------------


def _scatter(
    ax: "plt.Axes",
    emb: np.ndarray,
    colors: np.ndarray,
    title: str,
    alpha: float = 0.5,
    s: float = 6.0,
):
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=s, alpha=alpha, linewidths=0)
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("UMAP-1", fontsize=7)
    ax.set_ylabel("UMAP-2", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)


def _add_colorbar(fig, ax, values: np.ndarray, cmap_name: str, label: str):
    norm = Normalize(vmin=values.min(), vmax=values.max())
    sm = cm.ScalarMappable(cmap=cm.get_cmap(cmap_name), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)


# ---------------------------------------------------------------------------
# Main visualization functions
# ---------------------------------------------------------------------------


def plot_latent_space(
    emb: np.ndarray,
    degrees: np.ndarray,
    is_grch38: np.ndarray,
    clustering_coeffs: np.ndarray,
    component_ids: np.ndarray,
    attention_radii: np.ndarray,
    hub_mask: np.ndarray,  # bool array, True = hub node
    out_path: Optional[str | Path] = None,
) -> "Optional[plt.Figure]":
    """
    6-panel UMAP scatter plot.

    Panels:
      1. Colored by log(degree) — hub nodes highlighted
      2. Colored by is_grch38 — reference vs alt haplotypes
      3. Colored by clustering coefficient — triangle density
      4. Colored by connected component — community structure
      5. Colored by attention radius — branching proximity
      6. Hub nodes only (large markers on top of full scatter)

    Args:
        emb:               (N, 2) UMAP embedding
        degrees:           (N,) degree per node
        is_grch38:         (N,) binary float
        clustering_coeffs: (N,) float clustering coefficient
        component_ids:     (N,) int connected component ID
        attention_radii:   (N,) int attention radius from branching nodes
        hub_mask:          (N,) bool, True for hub/branching nodes
        out_path:          If given, save figure here
    """
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib not installed.")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Pangenome Graph — Latent Space Analysis", fontsize=11, y=0.98)

    log_deg = np.log1p(degrees.astype(float))

    # Panel 1: degree
    c1 = _continuous_colors(log_deg, "plasma")
    _scatter(axes[0, 0], emb, c1, "Node Degree (log scale)", alpha=0.45)
    _add_colorbar(fig, axes[0, 0], log_deg, "plasma", "log(degree+1)")

    # Panel 2: is_grch38
    c2 = np.array(
        [
            [0.12, 0.47, 0.71, 0.7] if v > 0.5 else [0.84, 0.15, 0.16, 0.7]
            for v in is_grch38
        ]
    )
    _scatter(axes[0, 1], emb, c2, "Reference (GRCh38) vs Alt Haplotypes", alpha=0.5)
    # Manual legend
    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=[0.12, 0.47, 0.71, 1],
            markersize=6,
            label="GRCh38",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=[0.84, 0.15, 0.16, 1],
            markersize=6,
            label="Alt haplotype",
        ),
    ]
    axes[0, 1].legend(handles=legend_elems, fontsize=6, loc="upper right")

    # Panel 3: clustering coefficient
    c3 = _continuous_colors(clustering_coeffs, "coolwarm")
    _scatter(axes[0, 2], emb, c3, "Local Clustering Coefficient", alpha=0.5)
    _add_colorbar(fig, axes[0, 2], clustering_coeffs, "coolwarm", "Clustering coeff")

    # Panel 4: connected component
    # Cap at top 20 components, rest grey
    comp_display = component_ids.copy().astype(float)
    top_comps = np.unique(component_ids)[:20]
    comp_display[~np.isin(component_ids, top_comps)] = -1
    c4 = _categorical_colors(comp_display.astype(int), "tab20")
    _scatter(axes[1, 0], emb, c4, "Connected Component (top 20)", alpha=0.5)

    # Panel 5: attention radius
    c5 = _continuous_colors(attention_radii.astype(float), "RdYlGn")
    _scatter(
        axes[1, 1], emb, c5, "Attention Radius (0=branching, high=linear)", alpha=0.5
    )
    _add_colorbar(
        fig, axes[1, 1], attention_radii.astype(float), "RdYlGn", "Attention radius"
    )

    # Panel 6: hub nodes overlaid
    _scatter(
        axes[1, 2],
        emb,
        np.full((len(emb), 4), [0.8, 0.8, 0.8, 0.3]),
        "Hubs / Branching Nodes",
        alpha=0.3,
        s=4,
    )
    if hub_mask.any():
        axes[1, 2].scatter(
            emb[hub_mask, 0],
            emb[hub_mask, 1],
            c="red",
            s=20,
            alpha=0.8,
            linewidths=0,
            zorder=5,
            label=f"Hubs (n={hub_mask.sum()})",
        )
        axes[1, 2].legend(fontsize=6)

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[latent_viz] Saved: {out_path}")

    return fig


def plot_degree_distribution(
    degrees: np.ndarray,
    out_path: Optional[str | Path] = None,
    title: str = "Degree Distribution",
) -> "Optional[plt.Figure]":
    """
    Log-log degree distribution plot.
    A straight line in log-log space suggests scale-free (power-law) behavior.
    Prof: "When I look at it, it should be scale-free, you know, like a hairball.
           And it's a network, right? So you show that it's a small-network."
    """
    if not MPL_AVAILABLE:
        return None

    deg_vals, counts = np.unique(degrees, return_counts=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title, fontsize=10)

    # Linear scale
    axes[0].bar(deg_vals, counts, color="steelblue", alpha=0.7)
    axes[0].set_xlabel("Degree", fontsize=8)
    axes[0].set_ylabel("Count", fontsize=8)
    axes[0].set_title("Linear scale", fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Log-log scale
    mask = (deg_vals > 0) & (counts > 0)
    axes[1].loglog(
        deg_vals[mask],
        counts[mask],
        "o",
        markersize=4,
        color="steelblue",
        alpha=0.7,
        label="Empirical",
    )

    # Fit power law if enough points
    if mask.sum() >= 5:
        log_d = np.log(deg_vals[mask].astype(float))
        log_c = np.log(counts[mask].astype(float))
        coeffs = np.polyfit(log_d, log_c, 1)
        gamma = -coeffs[0]
        d_range = np.logspace(
            np.log10(deg_vals[mask].min()), np.log10(deg_vals[mask].max()), 50
        )
        c_fit = np.exp(coeffs[1]) * d_range ** coeffs[0]
        axes[1].loglog(
            d_range,
            c_fit,
            "--",
            color="red",
            linewidth=1.5,
            label=f"Power-law fit γ={gamma:.2f}",
        )
        scale_free_note = (
            "Scale-free ✓" if 2.0 < gamma < 3.0 else f"γ={gamma:.2f} (check manually)"
        )
        axes[1].text(
            0.05,
            0.05,
            scale_free_note,
            transform=axes[1].transAxes,
            fontsize=7,
            color="darkgreen" if 2.0 < gamma < 3.0 else "darkorange",
        )

    axes[1].set_xlabel("Degree (log)", fontsize=8)
    axes[1].set_ylabel("Count (log)", fontsize=8)
    axes[1].set_title("Log-log scale (power-law check)", fontsize=8)
    axes[1].legend(fontsize=7)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[latent_viz] Saved: {out_path}")
    return fig


def plot_network_summary(
    stats_dict: Dict,
    out_path: Optional[str | Path] = None,
) -> "Optional[plt.Figure]":
    """
    Text + bar summary panel of complex network stats.
    Useful as a supplementary figure or sanity check panel.
    """
    if not MPL_AVAILABLE:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Pangenome Graph — Network Statistics", fontsize=10)

    # Left: key scalar stats as text
    ax = axes[0]
    ax.axis("off")
    lines = [
        ("Nodes", f"{stats_dict.get('n_nodes', '?'):,}"),
        ("Edges", f"{stats_dict.get('n_edges', '?'):,}"),
        ("Density", f"{stats_dict.get('density', 0):.4f}"),
        ("Mean degree", f"{stats_dict.get('mean_degree', 0):.2f}"),
        ("Max degree", f"{stats_dict.get('max_degree', 0):,}"),
        ("Branching frac", f"{stats_dict.get('branching_frac', 0):.3f}"),
        ("# components", f"{stats_dict.get('n_components', '?'):,}"),
        ("LCC frac", f"{stats_dict.get('largest_component_frac', 0):.4f}"),
        ("Mean clustering", f"{stats_dict.get('mean_clustering_coeff', 0):.4f}"),
        ("Transitivity", f"{stats_dict.get('transitivity', 0):.4f}"),
        ("Degree exponent γ", f"{stats_dict.get('degree_exponent', 'N/A')}"),
        ("Scale-free?", "Yes ✓" if stats_dict.get("is_scale_free_proxy") else "No"),
        ("Avg path length", f"{stats_dict.get('sampled_avg_shortest_path', 'N/A')}"),
    ]
    y = 0.97
    for label, val in lines:
        ax.text(
            0.05,
            y,
            f"{label}:",
            fontsize=8,
            ha="left",
            va="top",
            transform=ax.transAxes,
            color="gray",
        )
        ax.text(
            0.55,
            y,
            val,
            fontsize=8,
            ha="left",
            va="top",
            transform=ax.transAxes,
            color="black",
            fontweight="bold",
        )
        y -= 0.072
    ax.set_title("Summary Statistics", fontsize=8)

    # Middle: hub degrees
    ax = axes[1]
    hub_oids = stats_dict.get("hub_oids", [])
    hub_degs = stats_dict.get("hub_degrees", [])
    if hub_oids and hub_degs:
        labels = [f"oid={h}" for h in hub_oids[:10]]
        bars = ax.barh(labels, hub_degs[:10], color="tomato", alpha=0.8)
        ax.set_xlabel("Degree", fontsize=8)
        ax.set_title("Top Hub Nodes (Branching Points)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No hub data", ha="center", va="center", fontsize=8)

    # Right: component size distribution
    ax = axes[2]
    comp_sizes = stats_dict.get("component_sizes_top5", [])
    if comp_sizes:
        x_pos = np.arange(len(comp_sizes))
        ax.bar(x_pos, comp_sizes, color="steelblue", alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"C{i}" for i in x_pos], fontsize=7)
        ax.set_ylabel("Size (# nodes)", fontsize=8)
        ax.set_title("Top Component Sizes", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    else:
        ax.axis("off")

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[latent_viz] Saved: {out_path}")
    return fig


def plot_attention_radius_distribution(
    oid_to_radius: Dict[int, int],
    out_path: Optional[str | Path] = None,
) -> "Optional[plt.Figure]":
    """
    Distribution of attention radii across all nodes.
    Shows how many nodes are near branching points vs deep in linear chains.
    """
    if not MPL_AVAILABLE:
        return None

    radii = np.array(list(oid_to_radius.values()), dtype=int)
    vals, counts = np.unique(radii, return_counts=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(vals, counts, color="mediumseagreen", alpha=0.8)
    ax.axvline(
        x=0, color="red", linestyle="--", linewidth=1.5, label="Branching nodes (r=0)"
    )
    ax.set_xlabel("Attention Radius (hops to nearest branching node)", fontsize=9)
    ax.set_ylabel("Number of Nodes", fontsize=9)
    ax.set_title(
        "Distribution of Attention Radii\n"
        "(r=0 = branching node; high r = deep in linear chain)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate percentages
    total = len(radii)
    for v, c in zip(vals, counts):
        ax.text(
            v,
            c + total * 0.005,
            f"{100*c/total:.0f}%",
            ha="center",
            fontsize=7,
            color="black",
        )

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[latent_viz] Saved: {out_path}")
    return fig


# ---------------------------------------------------------------------------
# High-level: run all visualizations from model output
# ---------------------------------------------------------------------------


def visualize_all(
    node_features: np.ndarray,  # (N, D) — raw or GAT-embedded
    nodes: np.ndarray,  # (N,) oid array
    degrees: np.ndarray,  # (N,)
    is_grch38: np.ndarray,  # (N,)
    clustering_coeffs: np.ndarray,  # (N,)
    component_ids: np.ndarray,  # (N,) int
    oid_to_radius: Dict[int, int],
    hub_oids: List[int],
    out_dir: str | Path,
    label: str = "slice",
    use_umap: bool = True,
) -> Dict[str, str]:
    """
    Run all visualizations and save to out_dir.
    Returns dict of {plot_name: file_path}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    # 1. Compute embedding
    print("[latent_viz] Computing 2D embedding...")
    if use_umap and UMAP_AVAILABLE:
        emb = compute_umap_embedding(node_features)
        method = "umap"
    else:
        if use_umap:
            print("[latent_viz] UMAP not available, falling back to PCA.")
        emb = compute_pca_fallback(node_features)
        method = "pca"

    # Build aligned arrays
    oid_list = nodes.astype(int).tolist()
    attention_radii = np.array([oid_to_radius.get(o, 8) for o in oid_list], dtype=int)
    hub_set = set(hub_oids)
    hub_mask = np.array([o in hub_set for o in oid_list], dtype=bool)

    # 2. Latent space 6-panel
    p = out_dir / f"{label}_{method}_latent_space.png"
    plot_latent_space(
        emb,
        degrees,
        is_grch38,
        clustering_coeffs,
        component_ids,
        attention_radii,
        hub_mask,
        out_path=p,
    )
    paths["latent_space"] = str(p)

    # 3. Degree distribution
    p = out_dir / f"{label}_degree_distribution.png"
    plot_degree_distribution(
        degrees, out_path=p, title=f"Degree Distribution ({label})"
    )
    paths["degree_dist"] = str(p)

    # 4. Attention radius distribution
    p = out_dir / f"{label}_attention_radii.png"
    plot_attention_radius_distribution(oid_to_radius, out_path=p)
    paths["attention_radii"] = str(p)

    print(f"[latent_viz] All plots saved to {out_dir}")
    return paths
