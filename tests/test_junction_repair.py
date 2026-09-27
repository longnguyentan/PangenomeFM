"""Tests for the shortcut-free junction re-pairing objective (v2)."""

from __future__ import annotations

import numpy as np
import pytest

from graph.junction_repair import (
    bidirected_degrees,
    build_junction_repair_candidates,
    group_batches,
    junction_mask,
    visible_degree_after_masking,
)
from graph.neg_sampling import (
    build_pos_set,
    canonical_oriented_pair,
    canonicalize_oriented_pairs,
    compute_oriented_degrees,
    neg_distance_matched_paired,
    slice_oriented_node_set,
)


def _bubble_chain(n_bubbles: int = 30, seg_len: int = 100):
    """Reference chain r0 -> r1 -> ... with a deletion bypass every 3 segments.

    Oriented ids are 2*segid (forward).  Returns u, v, oid_to_so.
    """
    u, v, so = [], [], {}
    n_ref = 3 * n_bubbles + 2
    for i in range(n_ref):
        so[2 * i] = i * seg_len
        so[2 * i + 1] = i * seg_len
    for i in range(n_ref - 1):
        u.append(2 * i)
        v.append(2 * (i + 1))
    for b in range(n_bubbles):
        start = 3 * b
        u.append(2 * start)
        v.append(2 * (start + 2))  # deletion edge skipping one segment
    return np.asarray(u), np.asarray(v), so


def test_branching_scope_keeps_only_bubble_junctions():
    u, v, _ = _bubble_chain(5)
    out_deg, in_deg = bidirected_degrees(u, v)
    keep = junction_mask(u, v, out_deg, in_deg, "branching")
    assert keep.any() and not keep.all()
    for a, b, k in zip(u, v, keep):
        assert k == (out_deg[a] > 1 or in_deg[b] > 1)


def test_candidates_are_balanced_unique_and_cross_paired():
    u, v, so = _bubble_chain(40)
    cand, audit = build_junction_repair_candidates(u, v, so, oid_to_sn={x: "chrT" for x in so}, scope="branching", span_size=8)
    assert audit.n_positive_candidates == audit.n_negative_candidates > 0
    observed = {canonical_oriented_pair(a, b) for a, b in zip(u, v)}
    keys = [canonical_oriented_pair(a, b) for a, b in cand[["u_oid", "v_oid"]].itertuples(index=False)]
    assert len(keys) == len(set(keys)), "candidates must be unique"
    for (group, split), sub in cand.groupby(["group_id", "split"]):
        pos = sub[sub.label == 1]
        neg = sub[sub.label == 0]
        out_ends = set(pos.u_oid)
        in_ends = set(pos.v_oid)
        assert set(neg.u_oid) <= out_ends and set(neg.v_oid) <= in_ends
        for a, b in neg[["u_oid", "v_oid"]].itertuples(index=False):
            assert canonical_oriented_pair(a, b) not in observed
    assert cand.groupby("group_id")["split"].nunique().max() == 1


def test_every_candidate_endpoint_is_masked_and_marginals_are_balanced():
    u, v, so = _bubble_chain(40)
    cand, _ = build_junction_repair_candidates(u, v, so, oid_to_sn={x: "chrT" for x in so}, scope="branching", span_size=8)
    full = compute_oriented_degrees(u, v, slice_oriented_node_set(u, v))
    for _, sub in cand.groupby("group_id"):
        from collections import Counter

        for column in ("u_oid", "v_oid"):
            assert Counter(sub.loc[sub.label == 1, column]) == Counter(sub.loc[sub.label == 0, column])
        hidden = sub.loc[sub.label == 1, ["u_oid", "v_oid"]].itertuples(index=False)
        visible = visible_degree_after_masking(u, v, list(hidden))
        for a, b in sub[["u_oid", "v_oid"]].itertuples(index=False):
            assert full[a] - visible.get(a, 0) >= 1
            assert full[b] - visible.get(b, 0) >= 1


def test_v1_degree_matched_sampler_has_deterministic_deficit_shortcut():
    """Documents the v1 artifact: single-query masking + unmasked degree matching."""
    u, v, so = _bubble_chain(40)
    nodes = slice_oriented_node_set(u, v)
    pos = canonicalize_oriented_pairs(np.stack([u, v], axis=1))
    deg = compute_oriented_degrees(u, v, nodes)
    sn = {int(n): "chrT" for n in nodes}
    neg, pidx = neg_distance_matched_paired(
        nodes, pos, build_pos_set(u, v), sn, so, deg, np.random.default_rng(0),
        same_sn=True, tol_bp=1000, tol_frac=0.1, degree_matched=True,
    )
    assert len(neg) > 0
    pos_deficit = [
        (deg[a] - visible_degree_after_masking(u, v, [(a, b)]).get(a, 0)) for a, b in pos[pidx]
    ]
    neg_deficit = [
        (deg[a] - visible_degree_after_masking(u, v, []).get(a, 0)) for a, b in neg
    ]
    assert min(pos_deficit) >= 1 and max(neg_deficit) == 0


def test_group_batches_never_split_groups():
    groups = [np.arange(i * 5, i * 5 + 5) for i in range(20)]
    batches = group_batches(groups, batch_size=12, rng=np.random.default_rng(1))
    seen = np.concatenate(batches)
    assert sorted(seen.tolist()) == list(range(100))
    members = {tuple(g.tolist()) for g in groups}
    for batch in batches:
        chunks = [tuple(batch[i : i + 5].tolist()) for i in range(0, len(batch), 5)]
        assert all(chunk in members for chunk in chunks)


def test_visible_structure_features_matches_unmasked_definition():
    torch = pytest.importorskip("torch")
    from training.node_inputs import visible_structure_features

    src = torch.tensor([0, 1, 2, 0])
    dst = torch.tensor([1, 2, 3, 2])
    deg = torch.bincount(src, minlength=4) + torch.bincount(dst, minlength=4)
    norm = float(torch.log1p(deg.double()).max())
    X = torch.zeros(4, 7)
    X[:, 4] = torch.log1p(deg.float()) / norm
    same = visible_structure_features(X, src, dst, norm)
    assert torch.allclose(same, X)
    masked = visible_structure_features(X, src[:3], dst[:3], norm)  # hide 0->2
    assert masked[0, 4] < X[0, 4] and masked[2, 4] < X[2, 4]
    assert torch.isclose(masked[1, 4], X[1, 4])


def test_coordinate_systems_orientation_and_distance_tolerance_are_preserved():
    u, v, so = _bubble_chain(80)
    sn = {x: ("chr1" if x // 2 < 120 else "chr2") for x in so}
    cand, audit = build_junction_repair_candidates(u, v, so, oid_to_sn=sn, span_size=8)
    assert len(cand) and audit.fraction_distance_within_tolerance == 1
    for _, group in cand.groupby("group_id"):
        signatures = {(sn[a], sn[b], a % 2, b % 2)
                      for a, b in group[["u_oid", "v_oid"]].itertuples(index=False)}
        assert len(signatures) == 1
    missing, audit = build_junction_repair_candidates(u, v, so, oid_to_sn={})
    assert missing.empty and audit.n_missing_coordinate_connections > 0


def test_candidate_construction_is_invariant_to_reverse_equivalent_storage():
    u, v, so = _bubble_chain(80)
    kw = dict(oid_to_sn={x: "chrT" for x in so}, span_size=8)
    forward, _ = build_junction_repair_candidates(u, v, so, **kw)
    reverse, _ = build_junction_repair_candidates(v[::-1] ^ 1, u[::-1] ^ 1, so, **kw)
    import pandas as pd
    pd.testing.assert_frame_equal(forward, reverse)


def test_feasible_cycle_survives_an_unmatchable_span_member():
    u, v = np.array([0, 4, 8]), np.array([2, 6, 10])
    so = {0: 0, 2: 100, 4: 300, 6: 400, 8: 100000, 10: 100100}
    c, audit = build_junction_repair_candidates(
        u, v, so, oid_to_sn={x: "chrT" for x in so}, scope="all", span_size=16)
    assert audit.n_positive_candidates == audit.n_negative_candidates == 2
    assert set(c.u_oid) == {0, 4} and set(c.v_oid) == {2, 6}
