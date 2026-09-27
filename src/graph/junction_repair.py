"""Endpoint-balanced masked-junction pretraining candidates.

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
genomic separation of ``(u_i, v_i)`` as closely as possible.  Positive and negative candidates have identical endpoint marginals within
each retained group. This removes marginal endpoint reuse as a label cue;
shared endpoints and pairwise structure still require empirical shortcut audits.
Only complete cycles of an admissible cross-pairing are retained; unmatched
junctions are excluded and counted. No unpaired positive is emitted.

Candidate construction uses NumPy, pandas and SciPy assignment so that the candidate construction can
be audited without PyTorch and reused by heuristic baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

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
    n_missing_coordinate_connections: int
    n_rejected_groups: int
    n_cross_coordinate_groups: int
    median_abs_distance_mismatch_bp: float
    fraction_distance_within_tolerance: float

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


def build_junction_repair_candidates(
    u: np.ndarray,
    v: np.ndarray,
    oid_to_so: Mapping[int, int],
    *,
    oid_to_sn: Mapping[int, str],
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

    # Canonical directions and order are invariant to stored reverse rows.
    pos_set = {
        canonical_oriented_pair(int(a), int(b))
        for a, b in zip(u, v) if a // 2 != b // 2
    }
    pairs = np.asarray(sorted(pos_set), dtype=np.int64).reshape(-1, 2)
    pu, pv = pairs[:, 0], pairs[:, 1]

    out_deg, in_deg = bidirected_degrees(u, v)
    in_scope = junction_mask(pu, pv, out_deg, in_deg, scope)
    pu, pv = pu[in_scope], pv[in_scope]
    n_pos = len(pu)

    columns = [
        "u_oid", "v_oid", "label", "split", "group_id",
        "source_i", "source_j", "distance_mismatch_bp",
    ]
    # Never interpret offsets from distinct SN coordinate systems as a distance.
    # Cross-system links remain eligible, but are grouped by the ordered SN pair
    # and matched on offsets within each corresponding system separately.
    known = np.array([
        a in oid_to_so and b in oid_to_so
        and isinstance(oid_to_sn.get(int(a)), str)
        and isinstance(oid_to_sn.get(int(b)), str)
        and oid_to_sn[int(a)] not in ("", "*", "nan")
        and oid_to_sn[int(b)] not in ("", "*", "nan")
        for a, b in zip(pu, pv)
    ], dtype=bool)
    missing = int((~known).sum())
    pu, pv = pu[known], pv[known]
    so_u = np.asarray([int(oid_to_so[int(x)]) for x in pu], dtype=np.int64)
    so_v = np.asarray([int(oid_to_so[int(x)]) for x in pv], dtype=np.int64)
    signatures = [(oid_to_sn[int(a)], oid_to_sn[int(b)], int(a % 2), int(b % 2))
                  for a, b in zip(pu, pv)]
    split = split_positive_indices(len(pu), split_seed)
    rng = np.random.default_rng(rng_seed)
    rows: List[Tuple] = []
    used_negatives: Set[Tuple[int, int]] = set()
    group_id = 0
    rejected = 0
    cross_coordinate_groups = 0
    mismatches: List[float] = []
    within_tol: List[bool] = []
    for split_name in SPLIT_NAMES:
        by_signature: Dict[Tuple, List[int]] = {}
        for i in np.flatnonzero(split == split_name):
            by_signature.setdefault(signatures[i], []).append(int(i))
        for signature, indices in sorted(by_signature.items()):
            members = np.asarray(indices, dtype=np.int64)
            if len(members) < 2:
                rejected += 1
                continue
            order = members[np.lexsort((rng.random(len(members)), so_u[members]))]
            same_system = signature[0] == signature[1]
            for group in _spans(order, span_size):
                cost = np.full((len(group), len(group)), np.inf)
                mismatch_matrix = np.full_like(cost, np.nan)
                for ii, i in enumerate(group):
                    for jj, j in enumerate(group):
                        a, b = int(pu[i]), int(pv[j])
                        key = canonical_oriented_pair(a, b)
                        if i == j or a // 2 == b // 2 or key in pos_set or key in used_negatives:
                            continue
                        if same_system:
                            distance = abs(int(so_u[i]) - int(so_v[i]))
                            mismatch = abs(abs(int(so_u[i]) - int(so_v[j])) - distance)
                            if mismatch > max(tol_bp, tol_frac * distance):
                                continue
                            mismatch_matrix[ii, jj] = mismatch
                            cost[ii, jj] = mismatch
                        else:
                            # Both differences below stay inside a single SN.
                            cost[ii, jj] = abs(int(so_u[i]) - int(so_u[j])) + abs(int(so_v[i]) - int(so_v[j]))
                # Allow a junction to remain unused through a diagonal dummy.
                # A large cost first maximizes the number of re-paired ends;
                # distance mismatch breaks ties. Nontrivial permutation cycles
                # preserve endpoint counts even if the full span is infeasible.
                finite = cost[np.isfinite(cost)]
                penalty = (float(finite.max()) + 1.0) * (len(group) + 1) if len(finite) else 1.0
                np.fill_diagonal(cost, penalty)
                try:
                    ri, ci = linear_sum_assignment(cost)
                    selected = ri != ci
                    ri, ci = ri[selected], ci[selected]
                    if len(ri) < 2:
                        rejected += 1
                        continue
                except ValueError:  # no complete admissible permutation
                    rejected += 1
                    continue
                keys = [canonical_oriented_pair(int(pu[group[ii]]), int(pv[group[jj]]))
                        for ii, jj in zip(ri, ci)]
                if len(set(keys)) != len(keys):
                    rejected += 1
                    continue
                # Emit complete cycles, preserving each retained endpoint count.
                for ii, jj in zip(ri, ci):
                    i, j = int(group[ii]), int(group[jj])
                    mismatch = float(mismatch_matrix[ii, jj])
                    rows.append((int(pu[i]), int(pv[i]), 1, split_name, group_id, i, i, 0.0))
                    rows.append((int(pu[i]), int(pv[j]), 0, split_name, group_id, i, j, mismatch))
                    if same_system:
                        mismatches.append(mismatch)
                        within_tol.append(True)
                used_negatives.update(keys)
                cross_coordinate_groups += int(not same_system)
                group_id += 1

    frame = pd.DataFrame(rows, columns=columns)
    audit = JunctionRepairAudit(
        n_observed_connections=len(pos_set),
        n_scope_connections=n_pos,
        n_positive_candidates=int((frame["label"] == 1).sum()) if len(frame) else 0,
        n_negative_candidates=int((frame["label"] == 0).sum()) if len(frame) else 0,
        n_groups=group_id,
        n_missing_coordinate_connections=missing,
        n_rejected_groups=rejected,
        n_cross_coordinate_groups=cross_coordinate_groups,
        median_abs_distance_mismatch_bp=float(np.median(mismatches)) if mismatches else float("nan"),
        fraction_distance_within_tolerance=float(np.mean(within_tol)) if within_tol else float("nan"),
    )
    return frame, audit


def group_batches(
    groups: Sequence[np.ndarray], batch_size: int, rng: np.random.Generator
) -> List[np.ndarray]:
    """Pack whole span groups into batches of roughly ``batch_size`` candidates.

    Groups are never split: balanced masking requires that both source
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
