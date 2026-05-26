"""
src/analysis/network.py

Complex network analysis of the oriented pangenome graph.

  - Check if the graph is scale-free (power-law degree distribution)
  - Compute clustering coefficient → small-world property
  - Identify hubs (high-degree nodes = branching nodes → targets for attention)
  - Identify isolated nodes (disconnected from main component)
  - Check connectivity and component sizes

  - If scale-free: hubs dominate the graph → graph attention should focus on hub neighborhoods
  - If small-world: most paths are short → aggressive multi-hop attention is affordable
  - Isolated nodes: low utility for training, can be flagged/filtered
  - Branching fraction already computed in qc.py but here we do a full network science treatment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NetworkStats:
    """Results of complex network analysis on a pangenome slice."""

    n_nodes: int
    n_edges: int
    density: float

    # Degree
    mean_degree: float
    max_degree: int
    degree_distribution: Dict[int, int]  # degree -> count
    branching_nodes: List[int]  # oids with degree > 2
    branching_frac: float
    hub_oids: List[int]  # top-k by degree
    hub_degrees: List[int]
    isolated_oids: List[int]  # degree == 0 (unreachable)

    # Connectivity
    n_components: int
    largest_component_frac: float  # fraction of nodes in LCC
    component_sizes: List[int]  # sorted desc

    # Small-world / clustering
    mean_clustering_coeff: float
    transitivity: float  # global clustering (triads)

    # Scale-free proxy
    degree_exponent: Optional[float]  # fitted power-law exponent
    is_scale_free_proxy: bool  # exponent in (2, 3)

    # Path-length proxy (sampled, expensive on large graphs)
    sampled_avg_shortest_path: Optional[float]

    # Per-node outputs (for downstream use)
    oid_to_degree: Dict[int, int]
    oid_to_component_id: Dict[int, int]
    oid_to_clustering: Dict[int, float]


# ---------------------------------------------------------------------------
# Graph construction from CSVs
# ---------------------------------------------------------------------------


def build_adjacency(
    u: np.ndarray,
    v: np.ndarray,
    nodes: np.ndarray,
) -> Dict[int, List[int]]:
    """Build undirected adjacency list on oriented node IDs."""
    adj: Dict[int, List[int]] = {int(n): [] for n in nodes}
    for a, b in zip(u.tolist(), v.tolist()):
        a, b = int(a), int(b)
        adj[a].append(b)
        adj[b].append(a)
    return adj


# ---------------------------------------------------------------------------
# Component analysis (BFS)
# ---------------------------------------------------------------------------


def find_connected_components(
    adj: Dict[int, List[int]],
) -> Tuple[Dict[int, int], List[List[int]]]:
    """
    BFS to find connected components.
    Returns:
        oid_to_comp: mapping from oid to component index
        components:  list of lists (each is a component's oid list), sorted desc by size
    """
    visited: Dict[int, int] = {}
    comp_id = 0
    components: List[List[int]] = []

    for start in adj:
        if start in visited:
            continue
        queue = [start]
        comp: List[int] = []
        visited[start] = comp_id
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            comp.append(node)
            for nb in adj[node]:
                if nb not in visited:
                    visited[nb] = comp_id
                    queue.append(nb)
        components.append(comp)
        comp_id += 1

    components.sort(key=len, reverse=True)
    # remap ids so 0 = largest component
    old_to_new: Dict[int, int] = {}
    for new_id, comp in enumerate(components):
        for node in comp:
            old_to_new[node] = new_id

    return old_to_new, components


# ---------------------------------------------------------------------------
# Clustering coefficient (local)
# ---------------------------------------------------------------------------


def local_clustering(adj: Dict[int, List[int]], node: int) -> float:
    """
    Fraction of pairs among node's neighbors that are themselves connected.
    Returns 0.0 if degree < 2.
    """
    neighbors = adj[node]
    k = len(neighbors)
    if k < 2:
        return 0.0
    nb_set = set(neighbors)
    triangles = sum(
        1
        for i, nb in enumerate(neighbors)
        for other in neighbors[i + 1 :]
        if other in adj.get(nb, [])
    )
    return 2.0 * triangles / (k * (k - 1))


def mean_clustering(adj: Dict[int, List[int]]) -> Tuple[float, Dict[int, float]]:
    oid_to_cc: Dict[int, float] = {n: local_clustering(adj, n) for n in adj}
    mean_cc = float(np.mean(list(oid_to_cc.values()))) if oid_to_cc else 0.0
    return mean_cc, oid_to_cc


def transitivity(adj: Dict[int, List[int]]) -> float:
    """
    Global clustering coefficient = 3 * triangles / triads.
    Ranges from 0 (no clustering) to 1 (complete graph).
    """
    triangles = 0
    triads = 0
    for node in adj:
        neighbors = adj[node]
        k = len(neighbors)
        if k < 2:
            continue
        triads += k * (k - 1) // 2
        nb_set = set(neighbors)
        for i, nb in enumerate(neighbors):
            for other in neighbors[i + 1 :]:
                if other in adj.get(nb, []):
                    triangles += 1
    return 3.0 * triangles / triads if triads > 0 else 0.0


# ---------------------------------------------------------------------------
# Scale-free proxy: power-law degree exponent (MLE)
# ---------------------------------------------------------------------------


def fit_powerlaw_exponent(degrees: List[int], d_min: int = 2) -> Optional[float]:
    """
    MLE for power-law exponent on degrees >= d_min.
    Formula: gamma = 1 + n * (sum log(d/d_min - 0.5))^-1
    Returns None if too few samples.
    """
    d = np.array([x for x in degrees if x >= d_min], dtype=float)
    if len(d) < 5:
        return None
    gamma = 1.0 + len(d) * np.sum(np.log(d / (d_min - 0.5))) ** -1
    return float(gamma)


# ---------------------------------------------------------------------------
# Shortest path sampling (BFS from random sources)
# ---------------------------------------------------------------------------


def sample_avg_shortest_path(
    adj: Dict[int, List[int]],
    components: List[List[int]],
    n_samples: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> Optional[float]:
    """
    Sample average shortest path within the largest connected component.
    BFS from n_samples random sources; average over all reached pairs.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    lcc = components[0] if components else []
    if len(lcc) < 2:
        return None

    sources = rng.choice(lcc, size=min(n_samples, len(lcc)), replace=False).tolist()
    total_dist = 0
    total_pairs = 0

    for src in sources:
        # BFS
        dist: Dict[int, int] = {src: 0}
        queue = [src]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for nb in adj[node]:
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        for d in dist.values():
            if d > 0:
                total_dist += d
                total_pairs += 1

    return float(total_dist / total_pairs) if total_pairs > 0 else None


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------


def analyze(
    u: np.ndarray,
    v: np.ndarray,
    nodes: np.ndarray,
    top_k_hubs: int = 10,
    sample_paths: bool = True,
    path_samples: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> NetworkStats:
    """
    Run full network analysis on a set of oriented edges (u, v) over nodes.

    Args:
        u, v:        oriented node ID arrays (oid space, same as neg_sampling)
        nodes:       all unique oids present
        top_k_hubs:  how many top-degree nodes to return as hubs
        sample_paths: whether to sample shortest paths (expensive for large graphs)
        path_samples: number of BFS sources for path sampling
        rng:         numpy Generator for reproducibility
    """
    if rng is None:
        rng = np.random.default_rng(42)

    adj = build_adjacency(u, v, nodes)
    n_nodes = len(nodes)
    n_edges = len(
        u
    )  # directed edges; undirected edge count = n_edges if no duplication

    # --- Degree ---
    degree_map: Dict[int, int] = {n: len(adj[n]) for n in adj}
    degrees = list(degree_map.values())
    mean_deg = float(np.mean(degrees)) if degrees else 0.0
    max_deg = int(np.max(degrees)) if degrees else 0

    deg_dist: Dict[int, int] = {}
    for d in degrees:
        deg_dist[d] = deg_dist.get(d, 0) + 1

    branching = [n for n, d in degree_map.items() if d > 2]
    isolated = [n for n, d in degree_map.items() if d == 0]
    branch_frac = len(branching) / n_nodes if n_nodes > 0 else 0.0

    sorted_by_deg = sorted(degree_map.items(), key=lambda x: -x[1])
    hub_oids = [n for n, _ in sorted_by_deg[:top_k_hubs]]
    hub_degs = [d for _, d in sorted_by_deg[:top_k_hubs]]

    # --- Connectivity ---
    oid_to_comp, components = find_connected_components(adj)
    n_comp = len(components)
    lcc_frac = len(components[0]) / n_nodes if components and n_nodes > 0 else 0.0
    comp_sizes = [len(c) for c in components]

    # --- Clustering ---
    mean_cc, oid_to_cc = mean_clustering(adj)
    trans = transitivity(adj)

    # --- Scale-free proxy ---
    gamma = fit_powerlaw_exponent(degrees, d_min=2)
    is_sf = (gamma is not None) and (2.0 < gamma < 3.0)

    # --- Shortest path sampling ---
    avg_path = None
    if sample_paths and n_nodes <= 50_000:
        avg_path = sample_avg_shortest_path(adj, components, path_samples, rng)

    density = (2 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

    return NetworkStats(
        n_nodes=n_nodes,
        n_edges=n_edges,
        density=density,
        mean_degree=mean_deg,
        max_degree=max_deg,
        degree_distribution=deg_dist,
        branching_nodes=branching,
        branching_frac=branch_frac,
        hub_oids=hub_oids,
        hub_degrees=hub_degs,
        isolated_oids=isolated,
        n_components=n_comp,
        largest_component_frac=lcc_frac,
        component_sizes=comp_sizes,
        mean_clustering_coeff=mean_cc,
        transitivity=trans,
        degree_exponent=gamma,
        is_scale_free_proxy=is_sf,
        sampled_avg_shortest_path=avg_path,
        oid_to_degree=degree_map,
        oid_to_component_id=oid_to_comp,
        oid_to_clustering=oid_to_cc,
    )


def stats_to_dict(s: NetworkStats) -> Dict:
    """Serialize NetworkStats for JSON output."""
    return {
        "n_nodes": s.n_nodes,
        "n_edges": s.n_edges,
        "density": s.density,
        "mean_degree": s.mean_degree,
        "max_degree": s.max_degree,
        "branching_frac": s.branching_frac,
        "n_branching_nodes": len(s.branching_nodes),
        "n_isolated": len(s.isolated_oids),
        "n_components": s.n_components,
        "largest_component_frac": s.largest_component_frac,
        "component_sizes_top5": s.component_sizes[:5],
        "mean_clustering_coeff": s.mean_clustering_coeff,
        "transitivity": s.transitivity,
        "degree_exponent": s.degree_exponent,
        "is_scale_free_proxy": s.is_scale_free_proxy,
        "sampled_avg_shortest_path": s.sampled_avg_shortest_path,
        "hub_oids": s.hub_oids,
        "hub_degrees": s.hub_degrees,
    }
