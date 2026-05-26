from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def oriented_ids_from_links(
    links_sub: pd.DataFrame, seg_index: pd.Index
) -> Tuple[np.ndarray, np.ndarray]:
    from_id = seg_index.get_indexer(links_sub["from_seg"].astype("string")).astype(
        np.int64
    )
    to_id = seg_index.get_indexer(links_sub["to_seg"].astype("string")).astype(np.int64)
    from_bit = (links_sub["from_orient"].astype(str) == "-").astype(np.int8).to_numpy()
    to_bit = (links_sub["to_orient"].astype(str) == "-").astype(np.int8).to_numpy()
    u = from_id * 2 + from_bit
    v = to_id * 2 + to_bit
    return u, v


def build_pos_set(u: np.ndarray, v: np.ndarray) -> Set[Tuple[int, int]]:
    return set(map(tuple, np.stack([u, v], axis=1).tolist()))


def slice_oriented_node_set(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.unique(np.concatenate([u, v]))


def compute_oriented_degrees(
    u: np.ndarray, v: np.ndarray, nodes: np.ndarray
) -> Dict[int, int]:
    """
    Degree within slice at oriented-level: outdeg + indeg.
    Return dict oid->deg.
    """
    # compact mapping
    idx = {int(o): i for i, o in enumerate(nodes.tolist())}
    outdeg = np.zeros(len(nodes), dtype=np.int32)
    indeg = np.zeros(len(nodes), dtype=np.int32)
    for a in u.tolist():
        outdeg[idx[int(a)]] += 1
    for b in v.tolist():
        indeg[idx[int(b)]] += 1
    deg = outdeg + indeg
    return {int(nodes[i]): int(deg[i]) for i in range(len(nodes))}


def neg_random(
    nodes: np.ndarray,
    pos_set: Set[Tuple[int, int]],
    n_neg: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    neg: List[Tuple[int, int]] = []
    tries = 0
    max_tries = max(10_000, n_neg * 80)
    n_nodes = len(nodes)

    while len(neg) < n_neg and tries < max_tries:
        uu = int(nodes[rng.integers(0, n_nodes)])
        vv = int(nodes[rng.integers(0, n_nodes)])
        if uu == vv:
            tries += 1
            continue
        if (uu, vv) in pos_set:
            tries += 1
            continue
        neg.append((uu, vv))
        tries += 1

    return neg


def neg_hard_coord_degree(
    nodes: np.ndarray,
    pos_pairs: np.ndarray,
    pos_set: Set[Tuple[int, int]],
    oid_to_sn: Dict[int, str],
    oid_to_so: Dict[int, int],
    oid_to_deg: Dict[int, int],
    n_neg: int,
    rng: np.random.Generator,
    same_sn: bool,
    coord_band: int,
    degree_matched: bool,
) -> List[Tuple[int, int]]:
    """
    Hard negatives:
      - sample around positives
      - optionally enforce same SN
      - enforce |delta_SO_neg - delta_SO_pos| <= coord_band (loose band)
      - optionally degree-matched to endpoint degrees (within slice)

    This is your v2 "hardneg".
    """
    neg: List[Tuple[int, int]] = []
    tries = 0
    max_tries = max(20_000, n_neg * 200)

    # precompute nodes by SN for faster sameSN sampling
    nodes_by_sn: Dict[str, np.ndarray] = {}
    if same_sn:
        for o in nodes.tolist():
            sn = oid_to_sn[int(o)]
            nodes_by_sn.setdefault(sn, []).append(int(o))
        nodes_by_sn = {k: np.array(v, dtype=np.int64) for k, v in nodes_by_sn.items()}

    # sample by anchoring to random positive edge
    n_pos = len(pos_pairs)
    while len(neg) < n_neg and tries < max_tries:
        i = int(rng.integers(0, n_pos))
        u_pos, v_pos = int(pos_pairs[i, 0]), int(pos_pairs[i, 1])

        sn_u = oid_to_sn[u_pos]
        sn_v = oid_to_sn[v_pos]
        so_u = oid_to_so[u_pos]
        so_v = oid_to_so[v_pos]
        delta_pos = abs(int(so_u) - int(so_v))

        if same_sn:
            # choose a single SN to sample within; prefer sn_u (often equals sn_v)
            sn = sn_u
            cand_nodes = nodes_by_sn.get(sn, nodes)
        else:
            cand_nodes = nodes

        uu = int(cand_nodes[rng.integers(0, len(cand_nodes))])
        vv = int(cand_nodes[rng.integers(0, len(cand_nodes))])

        if uu == vv:
            tries += 1
            continue
        if (uu, vv) in pos_set:
            tries += 1
            continue

        # coord band constraint
        delta_neg = abs(int(oid_to_so[uu]) - int(oid_to_so[vv]))
        if abs(delta_neg - delta_pos) > coord_band:
            tries += 1
            continue

        if same_sn:
            if oid_to_sn[uu] != oid_to_sn[vv]:
                tries += 1
                continue

        if degree_matched:
            # match degrees of endpoints to positive endpoints (unordered)
            deg_u = oid_to_deg[uu]
            deg_v = oid_to_deg[vv]
            deg_u_pos = oid_to_deg[u_pos]
            deg_v_pos = oid_to_deg[v_pos]
            ok = (deg_u == deg_u_pos and deg_v == deg_v_pos) or (
                deg_u == deg_v_pos and deg_v == deg_u_pos
            )
            if not ok:
                tries += 1
                continue

        neg.append((uu, vv))
        tries += 1

    return neg


def neg_distance_matched(
    nodes: np.ndarray,
    pos_pairs: np.ndarray,
    pos_set: Set[Tuple[int, int]],
    oid_to_sn: Dict[int, str],
    oid_to_so: Dict[int, int],
    oid_to_deg: Dict[int, int],
    n_neg: int,
    rng: np.random.Generator,
    same_sn: bool,
    tol_bp: int,
    tol_frac: float,
    degree_matched: bool,
) -> List[Tuple[int, int]]:
    """
    Distance-matched negatives (your v3):
      For each sampled positive (u_pos,v_pos) with d_pos = |SO(u)-SO(v)|,
      sample (u,v) such that:
        |d_neg - d_pos| <= max(tol_bp, tol_frac * d_pos)
      and optionally same SN and degree-matched.

    This is what made AUC drop in strict slices in your results.
    """
    neg: List[Tuple[int, int]] = []
    tries = 0
    max_tries = max(50_000, n_neg * 400)

    nodes_by_sn: Dict[str, np.ndarray] = {}
    if same_sn:
        for o in nodes.tolist():
            sn = oid_to_sn[int(o)]
            nodes_by_sn.setdefault(sn, []).append(int(o))
        nodes_by_sn = {k: np.array(v, dtype=np.int64) for k, v in nodes_by_sn.items()}

    n_pos = len(pos_pairs)
    while len(neg) < n_neg and tries < max_tries:
        i = int(rng.integers(0, n_pos))
        u_pos, v_pos = int(pos_pairs[i, 0]), int(pos_pairs[i, 1])

        sn_u = oid_to_sn[u_pos]
        so_u = oid_to_so[u_pos]
        so_v = oid_to_so[v_pos]
        d_pos = abs(int(so_u) - int(so_v))
        tol = max(int(tol_bp), int(tol_frac * d_pos))

        if same_sn:
            cand_nodes = nodes_by_sn.get(sn_u, nodes)
        else:
            cand_nodes = nodes

        uu = int(cand_nodes[rng.integers(0, len(cand_nodes))])
        vv = int(cand_nodes[rng.integers(0, len(cand_nodes))])

        if uu == vv:
            tries += 1
            continue
        if (uu, vv) in pos_set:
            tries += 1
            continue
        if same_sn and (oid_to_sn[uu] != oid_to_sn[vv]):
            tries += 1
            continue

        d_neg = abs(int(oid_to_so[uu]) - int(oid_to_so[vv]))
        if abs(d_neg - d_pos) > tol:
            tries += 1
            continue

        if degree_matched:
            deg_u = oid_to_deg[uu]
            deg_v = oid_to_deg[vv]
            deg_u_pos = oid_to_deg[u_pos]
            deg_v_pos = oid_to_deg[v_pos]
            ok = (deg_u == deg_u_pos and deg_v == deg_v_pos) or (
                deg_u == deg_v_pos and deg_v == deg_u_pos
            )
            if not ok:
                tries += 1
                continue

        neg.append((uu, vv))
        tries += 1

    return neg
