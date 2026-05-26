"""
src/models/attention_window.py

Branching-point aware attention window computation.

Prof's advice (from meeting):
  "You want to pay attention from the last common node to the next common node."
  "Everything in between is the same, so you don't need to pay much attention."
  "You should pay attention to the heterogeneity of the data."
  "If the neighborhood is linear/uniform, maybe you don't have to pay too much attention.
   But if it's more complex, attention should help."

Key insight:
  In a pangenome graph, most of the graph is linear (simple chains).
  The structurally INTERESTING places are:
    (a) Branching nodes: degree > 2 → where haplotypes diverge or converge
    (b) Junctions: where multiple SN paths meet

  Prof's prescription for attention window:
    For each node i, its "attention radius" = distance to the nearest branching node.
    - If i is itself branching → radius = 0 (attend only immediate neighbors)
    - If i is in a long linear chain → radius grows until the next branch
    - This means: at branches, attend locally (fine-grained); in linear regions, attend broader

  This is implemented as:
    1. BFS from all branching nodes simultaneously (multi-source BFS)
    2. Each non-branching node's distance = hops to nearest branching node
    3. Attention radius = min(distance, max_radius)

  In the GAT, this radius is used to:
    (a) Expand the neighbor set (k-hop neighborhood) per node
    (b) OR used as a per-node attention temperature (scale attention logits)

Implementation A (neighbor expansion) → used in GAT sparse attention
Implementation B (temperature scaling) → simpler, can run on 1-hop GAT
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Multi-source BFS to compute distance to nearest branching node
# ---------------------------------------------------------------------------


def compute_branching_distances(
    adj: Dict[int, List[int]],
    branching_nodes: List[int],
    max_radius: int = 8,
) -> Dict[int, int]:
    """
    Multi-source BFS from all branching nodes simultaneously.
    Returns oid -> distance_to_nearest_branching_node.

    - Branching nodes themselves get distance 0.
    - Nodes not reachable within max_radius get distance max_radius.
    - Isolated nodes (degree 0) get distance max_radius.

    Args:
        adj:              Undirected adjacency list in oid space
        branching_nodes:  Nodes with degree > 2
        max_radius:       Cap on distance (also caps attention radius)
    """
    dist: Dict[int, int] = {}
    queue: List[int] = []

    # Initialize: all branching nodes start at distance 0
    for bn in branching_nodes:
        dist[bn] = 0
        queue.append(bn)

    # BFS
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        d = dist[node]
        if d >= max_radius:
            continue
        for nb in adj.get(node, []):
            if nb not in dist:
                dist[nb] = d + 1
                queue.append(nb)

    # Nodes not reached → max_radius
    for node in adj:
        if node not in dist:
            dist[node] = max_radius

    return dist


# ---------------------------------------------------------------------------
# K-hop neighborhood expansion (for sparse attention)
# ---------------------------------------------------------------------------


def compute_khop_neighborhoods(
    adj: Dict[int, List[int]],
    oid_to_radius: Dict[int, int],
    max_radius: int = 8,
) -> Dict[int, Set[int]]:
    """
    For each node, find all neighbors within its attention radius.
    The radius per node = oid_to_radius[node] (capped at max_radius).

    Returns oid -> set of oids within k hops (not including self).

    This defines the attention mask: for node i, only attend to nodes
    in khop_neighborhoods[i] during GAT message passing.

    Note: this can be expensive for large graphs. The function
    short-circuits at radius 1 for branching nodes (radius=0 maps to
    immediate neighbors only).
    """
    neighborhoods: Dict[int, Set[int]] = {}

    for node in adj:
        radius = min(oid_to_radius.get(node, max_radius), max_radius)
        # BFS up to radius hops
        visited: Set[int] = set()
        frontier = {node}
        for hop in range(radius + 1):
            next_frontier: Set[int] = set()
            for n in frontier:
                for nb in adj.get(n, []):
                    if nb != node and nb not in visited:
                        visited.add(nb)
                        next_frontier.add(nb)
            frontier = next_frontier
        neighborhoods[node] = visited

    return neighborhoods


# ---------------------------------------------------------------------------
# Attention temperature (simpler alternative to full k-hop expansion)
# ---------------------------------------------------------------------------


def compute_attention_temperature(
    oid_to_radius: Dict[int, int],
    min_temp: float = 0.3,
    max_temp: float = 2.0,
) -> Dict[int, float]:
    """
    Maps attention radius to a per-node temperature for softmax in GAT.

    Convention:
        - High radius (far from branching) → higher temperature → softer attention
          (spread attention broadly; the region is more homogeneous)
        - Low radius (near or at branching) → lower temperature → sharper attention
          (discriminate more finely; the region is structurally complex)

    Temperature is linearly scaled from min_temp (radius=0) to max_temp (radius=max_radius).
    """
    if not oid_to_radius:
        return {}

    max_r = max(oid_to_radius.values()) or 1
    temps: Dict[int, float] = {}
    for oid, r in oid_to_radius.items():
        frac = r / max_r  # 0 at branching nodes, 1 at far nodes
        temps[oid] = min_temp + frac * (max_temp - min_temp)
    return temps


def compute_attention_temperature_v2(
    oid_to_radius: Dict[int, int],
    min_temp: float = 0.3,
    max_temp: float = 2.0,
) -> Dict[int, float]:
    """
    INVERTED temperature: nodes AT branching get HIGH temp (soft/wide attention),
    nodes FAR from branching get LOW temp (sharp/focused attention).

    Motivation from experimental results (ep150 ablation):
      --no_topo_temp (flat=1.0) outperformed v1 temperature by +0.010 overall,
      and by up to +0.074 on high-branching chromosomes (chr18, chr11, chr8).

      Root cause: v1 assigns SHARP attention (low temp) to branching nodes.
      In high-branching graphs (branching_frac > 45%), most nodes are near
      branching points, so most nodes get very sharp attention → each node
      only aggregates from 1-2 neighbors → GAT loses multi-hop context.

    v2 fix: nodes AT branching need WIDE attention to aggregate from ALL
    incoming paths (understanding topology requires seeing all options).
    Nodes in LINEAR chains only have 1-2 neighbors, so sharp vs soft
    barely matters — but soft is still fine since all neighbors are similar.

    Temperature formula (inverted):
        radius=0 (branching node) → max_temp (soft/wide)
        radius=max (linear chain)  → min_temp (sharp/focused)
    """
    if not oid_to_radius:
        return {}

    max_r = max(oid_to_radius.values()) or 1
    temps: Dict[int, float] = {}
    for oid, r in oid_to_radius.items():
        frac = r / max_r  # 0 at branching, 1 at far nodes
        # Inverted: branching → max_temp, far → min_temp
        temps[oid] = max_temp - frac * (max_temp - min_temp)
    return temps


def compute_attention_temperature_adaptive(
    oid_to_radius: Dict[int, int],
    oid_to_degree: Dict[int, int],
    min_temp: float = 0.3,
    max_temp: float = 2.0,
) -> Dict[int, float]:
    """
    Adaptive temperature that combines branching distance AND local degree.

    Nodes with both high degree AND near branching get the highest temperature
    (widest attention — these are the complex junction nodes that need to
    aggregate from many neighbors simultaneously).

    Nodes with low degree AND far from branching get the lowest temperature
    (sharpest attention — they have few neighbors and no structural complexity).

    Formula:
        branch_score = 1 - (radius / max_radius)   # 1 at branching, 0 far
        degree_score = log1p(degree) / log1p(max_degree)  # normalized degree
        combined     = 0.6 * branch_score + 0.4 * degree_score
        temp         = min_temp + combined * (max_temp - min_temp)
    """
    if not oid_to_radius:
        return {}

    import math

    max_r = max(oid_to_radius.values()) or 1
    max_deg = max(oid_to_degree.values()) if oid_to_degree else 1
    log_max_deg = math.log1p(max_deg)

    temps: Dict[int, float] = {}
    for oid, r in oid_to_radius.items():
        branch_score = 1.0 - (r / max_r)
        deg = oid_to_degree.get(oid, 1)
        degree_score = math.log1p(deg) / log_max_deg if log_max_deg > 0 else 0.0
        combined = 0.6 * branch_score + 0.4 * degree_score
        temps[oid] = min_temp + combined * (max_temp - min_temp)
    return temps


# ---------------------------------------------------------------------------
# Compact arrays for use in GAT (oid → dense index)
# ---------------------------------------------------------------------------


def build_attention_arrays(
    nodes: np.ndarray,
    oid_to_radius: Dict[int, int],
    oid_to_temp: Dict[int, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert dicts to dense arrays aligned to a sorted node list.

    Returns:
        radius_arr:  shape (N,) int32  — attention radius per node
        temp_arr:    shape (N,) float32 — attention temperature per node
    """
    radius_arr = np.array([oid_to_radius.get(int(n), 8) for n in nodes], dtype=np.int32)
    temp_arr = np.array([oid_to_temp.get(int(n), 1.0) for n in nodes], dtype=np.float32)
    return radius_arr, temp_arr


# ---------------------------------------------------------------------------
# Build sparse edge index for k-hop attention (COO format)
# ---------------------------------------------------------------------------


def build_khop_edge_index(
    u: np.ndarray,
    v: np.ndarray,
    adj: Dict[int, List[int]],
    oid_to_radius: Dict[int, int],
    nodes: np.ndarray,
    max_radius: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an expanded edge index where each node also attends to
    nodes within its attention radius (beyond just direct neighbors).

    Returns:
        src, dst arrays in oid space (can be re-indexed to dense after)

    NOTE: This can significantly expand the edge count. Use with care
    for large slices. For slices with > 10k nodes, prefer temperature
    scaling over full k-hop expansion.
    """
    src_list: List[int] = list(u.astype(int))
    dst_list: List[int] = list(v.astype(int))
    existing = set(zip(src_list, dst_list))

    node_set = set(nodes.astype(int).tolist())

    for node in nodes.astype(int).tolist():
        radius = min(oid_to_radius.get(node, 1), max_radius)
        if radius <= 1:
            continue  # already covered by direct edges
        # BFS to collect k-hop neighbors
        visited: Set[int] = {node}
        frontier = {node}
        for hop in range(radius):
            next_f: Set[int] = set()
            for n in frontier:
                for nb in adj.get(n, []):
                    if nb not in visited and nb in node_set:
                        visited.add(nb)
                        next_f.add(nb)
            frontier = next_f
        for nb in visited:
            if nb == node:
                continue
            if (node, nb) not in existing:
                src_list.append(node)
                dst_list.append(nb)
                existing.add((node, nb))

    return np.array(src_list, dtype=np.int64), np.array(dst_list, dtype=np.int64)


# ---------------------------------------------------------------------------
# Summary statistics on attention windows
# ---------------------------------------------------------------------------


def summarize_attention_windows(
    oid_to_radius: Dict[int, int],
    branching_nodes: List[int],
) -> Dict:
    radii = list(oid_to_radius.values())
    branch_set = set(branching_nodes)
    return {
        "mean_attention_radius": float(np.mean(radii)) if radii else 0.0,
        "median_attention_radius": float(np.median(radii)) if radii else 0.0,
        "max_attention_radius": int(np.max(radii)) if radii else 0,
        "frac_radius_0": sum(1 for r in radii if r == 0) / len(radii) if radii else 0.0,
        "frac_radius_gt_4": sum(1 for r in radii if r > 4) / len(radii)
        if radii
        else 0.0,
        "n_branching_nodes": len(branching_nodes),
        "n_linear_nodes": len(radii) - len(branching_nodes),
    }
