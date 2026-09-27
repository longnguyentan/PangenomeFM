"""Shortcut-free masked-junction pretraining candidates.

Background
----------
The v1 pretraining benchmark pairs each observed connection with a negative
matched on genomic separation and on endpoint degree measured in the
*unmasked* slice.  Only the positive query connection is then hidden before
message passing.  Every positive endpoint therefore loses exactly one visible
connection while its matched negative loses none, so "visible degree" (or the
difference between the unmasked-degree node feature and the number of visible
messages) identifies positives almost perfectly.  ``scripts/audit_masking_
degree_shortcut.py`` quantifies this on real slices.

Junction re-pairing
-------------------
Here every candidate endpoint is a dangling end created by the same masking
operation.  Observed connections (junctions) are grouped into short
genomic spans.  All junctions in a span are hidden together, and negatives are
cross-pairings ``(u_i, v_j)`` of an out-end from one hidden junction with an
in-end from another hidden junction in the same span, chosen to match the
genomic separation of ``(u_i, v_i)`` as closely as possible.  Positive and
negative endpoints thus carry an identical one-connection deficit, and the
model has to decide *which* dangling ends were joined, which is the
pangenome analogue of resolving breakpoint junctions at a bubble.

Everything in this module is NumPy-only so that the candidate construction can
be audited without PyTorch and reused by heuristic baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from graph.neg_sampling import canonical_oriented_pair

JUNCTION_SCOPES = ("all", "branching")
SPLIT_NAMES = ("train", "val", "test")


def bidirected_degrees(
    u: np.ndarray, v: np.ndarray
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Out/in degree of oriented handles in a bidirected graph.

    A stored link ``u -> v`` is the same traversal as ``(v^1) -> (u^1)``, so it
    contributes an out-edge to ``u`` and ``v^1`` and an in-edge to ``v`` and
    ``u^1``.  Reverse-equivalent duplicate rows are counted once.
    """

    out_deg: Dict[int, int] = {}
    in_deg: Dict[int, int] = {}
    seen: Set[Tuple[int, int]] = set()
    for a, b in zip(np.asarray(u).tolist(), np.asarray(v).tolist()):
        key = canonical_oriented_pair(int(a), int(b))
        if key in seen:
            continue
        seen.add(key)
        a, b = int(a), int(b)
        for src, dst in ((a, b), (b ^ 1, a ^ 1)):
            out_deg[src] = out_deg.get(src, 0) + 1
            in_deg[dst] = in_deg.get(dst, 0) + 1
    return out_deg, in_deg


def junction_mask(
    u: np.ndarray,
    v: np.ndarray,
    out_deg: Mapping[int, int],
    in_deg: Mapping[int, int],
    scope: str,
) -> np.ndarray:
    """Select which observed connections are pretraining targets.

    ``all`` keeps every connection.  ``branching`` keeps connections that leave
    a handle with more than one successor or enter a handle with more than one
    predecessor, i.e. bubble entry/exit junctions.  This removes the
    non-branching chop links that dominate minigraph-cactus edge sets and
    carry no information about alternative haplotype structure.
    """

    if scope not in JUNCTION_SCOPES:
        raise ValueError(f"junction scope must be one of {JUNCTION_SCOPES}, got {scope!r}")
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    if scope == "all":
        return np.ones(len(u), dtype=bool)
    return np.asarray(
        [out_deg.get(int(a), 0) > 1 or in_deg.get(int(b), 0) > 1 for a, b in zip(u, v)],
        dtype=bool,
    )


def split_positive_indices(n_positive: int, split_seed: int) -> np.ndarray:
    """Deterministic 70/10/20 split labels for positive junctions."""

    rng = np.random.default_rng(split_seed)
    order = rng.permutation(n_positive)
    n_test = int(n_positive * 0.2)
    n_val = int(n_positive * 0.1)
    labels = np.empty(n_positive, dtype=object)
    labels[order[:n_test]] = "test"
    labels[order[n_test : n_test + n_val]] = "val"
    labels[order[n_test + n_val :]] = "train"
    return labels


def _spans(order: np.ndarray, span_size: int) -> List[np.ndarray]:
    groups = [order[i : i + span_size] for i in range(0, len(order), span_size)]
    if len(groups) > 1 and len(groups[-1]) < 2:
        groups[-2] = np.concatenate([groups[-2], groups[-1]])
        groups.pop()
    return [g for g in groups if len(g) >= 2]


@dataclass(frozen=True)
class JunctionRepairAudit:
    n_observed_connections: int
    n_scope_connections: int
    n_positive_candidates: int
    n_negative_candidates: int
    n_groups: int
    median_abs_distance_mismatch_bp: float
    fraction_distance_within_tolerance: float

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


def build_junction_repair_candidates(
    u: np.ndarray,
    v: np.ndarray,
    oid_to_so: Mapping[int, int],
    *,
    scope: str = "branching",
    span_size: int = 16,
    split_seed: int = 20260806,
    rng_seed: int = 0,
    tol_bp: int = 1_000,
    tol_frac: float = 0.10,
) -> Tuple[pd.DataFrame, JunctionRepairAudit]:
    """Return junction re-pairing candidates for one slice.

    Parameters
    ----------
    u, v:
        Stored oriented link endpoints of the slice (``u -> v``).
    oid_to_so:
        Oriented handle -> genomic offset.

    Returns
    -------
    candidates:
        Columns ``u_oid, v_oid, label, split, group_id, source_i, source_j,
        distance_mismatch_bp``.  ``group_id`` identifies the span whose
        junctions must be hidden together whenever any of its candidates is
        scored.  ``source_i``/``source_j`` index the positive junctions that
        contributed the out-end and in-end.
    audit:
        Counts and distance-matching quality.
    """

    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    if len(u) != len(v):
        raise ValueError("u and v must have equal length")
    if span_size < 2:
        raise ValueError("span_size must be at least 2")

    # One representative stored direction per biological junction.
    seen: Dict[Tuple[int, int], int] = {}
    keep_rows: List[int] = []
    for row, (a, b) in enumerate(zip(u.tolist(), v.tolist())):
        if a // 2 == b // 2:
            continue  # self-loops/inversions of one segment are not re-paired
        key = canonical_oriented_pair(a, b)
        if key in seen:
            continue
        seen[key] = row
        keep_rows.append(row)
    pos_set: Set[Tuple[int, int]] = set(seen)
    pu, pv = u[keep_rows], v[keep_rows]

    out_deg, in_deg = bidirected_degrees(u, v)
    in_scope = junction_mask(pu, pv, out_deg, in_deg, scope)
    pu, pv = pu[in_scope], pv[in_scope]
    n_pos = len(pu)

    columns = [
        "u_oid", "v_oid", "label", "split", "group_id",
        "source_i", "source_j", "distance_mismatch_bp",
    ]
    if n_pos < 2:
        empty = pd.DataFrame(columns=columns)
        return empty, JunctionRepairAudit(len(pos_set), n_pos, 0, 0, 0, float("nan"), float("nan"))

    so_u = np.asarray([int(oid_to_so.get(int(x), 0)) for x in pu], dtype=np.int64)
    so_v = np.asarray([int(oid_to_so.get(int(x), 0)) for x in pv], dtype=np.int64)
    distance = np.abs(so_u - so_v)
    anchor = np.minimum(so_u, so_v)
    split = split_positive_indices(n_pos, split_seed)
    rng = np.random.default_rng(rng_seed)

    rows: List[Tuple] = []
    used_negatives: Set[Tuple[int, int]] = set()
    group_id = 0
    mismatches: List[float] = []
    within_tol: List[bool] = []
    for split_name in SPLIT_NAMES:
        members = np.flatnonzero(split == split_name)
        if len(members) < 2:
            continue
        # Stable genomic order; random jitter breaks exact ties reproducibly.
        order = members[np.lexsort((rng.random(len(members)), anchor[members]))]
        for group in _spans(order, span_size):
            used_in_end: Dict[int, int] = {}
            group_rows: List[Tuple] = []
            for i in group.tolist():
                best = None
                for j in group.tolist():
                    if j == i:
                        continue
                    a, b = int(pu[i]), int(pv[j])
                    key_ab = canonical_oriented_pair(a, b)
                    if a // 2 == b // 2 or key_ab in pos_set or key_ab in used_negatives:
                        continue
                    mismatch = abs(abs(int(so_u[i]) - int(so_v[j])) - int(distance[i]))
                    key = (mismatch, used_in_end.get(j, 0), float(rng.random()))
                    if best is None or key < best[0]:
                        best = (key, j)
                # Every junction of the span is emitted as a positive row so it
                # is hidden whenever the span is scored, including junctions
                # whose own out-end found no admissible partner.
                group_rows.append((int(pu[i]), int(pv[i]), 1, split_name, group_id, i, i, 0.0))
                if best is None:
                    continue
                (mismatch, _, _), j = best
                used_in_end[j] = used_in_end.get(j, 0) + 1
                used_negatives.add(canonical_oriented_pair(int(pu[i]), int(pv[j])))
                tolerance = max(int(tol_bp), int(tol_frac * int(distance[i])))
                mismatches.append(float(mismatch))
                within_tol.append(mismatch <= tolerance)
                group_rows.append((int(pu[i]), int(pv[j]), 0, split_name, group_id, i, j, float(mismatch)))
            if group_rows:
                rows.extend(group_rows)
                group_id += 1

    frame = pd.DataFrame(rows, columns=columns)
    audit = JunctionRepairAudit(
        n_observed_connections=len(pos_set),
        n_scope_connections=n_pos,
        n_positive_candidates=int((frame["label"] == 1).sum()) if len(frame) else 0,
        n_negative_candidates=int((frame["label"] == 0).sum()) if len(frame) else 0,
        n_groups=group_id,
        median_abs_distance_mismatch_bp=float(np.median(mismatches)) if mismatches else float("nan"),
        fraction_distance_within_tolerance=float(np.mean(within_tol)) if within_tol else float("nan"),
    )
    return frame, audit


def group_batches(
    groups: Sequence[np.ndarray], batch_size: int, rng: np.random.Generator
) -> List[np.ndarray]:
    """Pack whole span groups into batches of roughly ``batch_size`` candidates.

    Groups are never split: a negative is only shortcut-free when both source
    junctions are hidden in the same forward pass.
    """

    batches: List[np.ndarray] = []
    current: List[np.ndarray] = []
    size = 0
    for index in rng.permutation(len(groups)).tolist():
        group = np.asarray(groups[index], dtype=np.int64)
        current.append(group)
        size += len(group)
        if size >= batch_size:
            batches.append(np.concatenate(current))
            current, size = [], 0
    if current:
        batches.append(np.concatenate(current))
    return batches


def visible_degree_after_masking(
    u: np.ndarray, v: np.ndarray, hidden: Iterable[Tuple[int, int]]
) -> Dict[int, int]:
    """Stored-direction out+in degree after hiding canonical junctions.

    Matches ``graph.neg_sampling.compute_oriented_degrees`` on the visible graph.
    """

    hidden_set = {canonical_oriented_pair(int(a), int(b)) for a, b in hidden}
    deg: Dict[int, int] = {}
    for a, b in zip(np.asarray(u).tolist(), np.asarray(v).tolist()):
        if canonical_oriented_pair(int(a), int(b)) in hidden_set:
            continue
        deg[int(a)] = deg.get(int(a), 0) + 1
        deg[int(b)] = deg.get(int(b), 0) + 1
    return deg
