from __future__ import annotations

import numpy as np

from graph.neg_sampling import (
    build_pos_set,
    canonical_oriented_pair,
    canonicalize_oriented_pairs,
    neg_distance_matched,
    neg_distance_matched_paired,
    neg_hard_coord_degree,
    neg_random,
    oriented_reverse_complement,
)


def assert_canonical_negative_integrity(
    negatives: list[tuple[int, int]],
    positives: set[tuple[int, int]],
) -> None:
    canonical = [canonical_oriented_pair(*pair) for pair in negatives]
    assert len(canonical) == len(set(canonical))
    assert not (set(canonical) & positives)


def test_reverse_complement_and_canonical_pair_are_one_identity() -> None:
    assert oriented_reverse_complement(10, 20) == (21, 11)
    assert canonical_oriented_pair(10, 20) == canonical_oriented_pair(21, 11)
    pairs = np.asarray([[10, 20], [21, 11], [30, 32]], dtype=np.int64)
    assert canonicalize_oriented_pairs(pairs).tolist() == [[10, 20], [30, 32]]


def test_positive_set_includes_reverse_equivalent_traversal() -> None:
    positives = build_pos_set(np.asarray([10]), np.asarray([20]))
    assert positives == {canonical_oriented_pair(10, 20)}
    assert canonical_oriented_pair(21, 11) in positives


def test_random_negatives_are_unique_under_reverse_equivalence() -> None:
    nodes = np.arange(12, dtype=np.int64)
    positives = build_pos_set(np.asarray([0, 4]), np.asarray([2, 6]))
    negatives = neg_random(
        nodes, positives, n_neg=20, rng=np.random.default_rng(20260815)
    )
    assert len(negatives) == 20
    assert_canonical_negative_integrity(negatives, positives)


def test_constrained_negative_samplers_enforce_canonical_identity() -> None:
    nodes = np.arange(20, dtype=np.int64)
    positive_pairs = np.asarray([[0, 2], [4, 6], [8, 10]], dtype=np.int64)
    positives = build_pos_set(positive_pairs[:, 0], positive_pairs[:, 1])
    oid_to_sn = {int(node): "chr1" for node in nodes}
    oid_to_so = {int(node): int(node // 2) * 100 for node in nodes}
    oid_to_deg = {int(node): 1 for node in nodes}

    hard = neg_hard_coord_degree(
        nodes=nodes,
        pos_pairs=positive_pairs,
        pos_set=positives,
        oid_to_sn=oid_to_sn,
        oid_to_so=oid_to_so,
        oid_to_deg=oid_to_deg,
        n_neg=8,
        rng=np.random.default_rng(1),
        same_sn=True,
        coord_band=10_000,
        degree_matched=True,
    )
    distance = neg_distance_matched(
        nodes=nodes,
        pos_pairs=positive_pairs,
        pos_set=positives,
        oid_to_sn=oid_to_sn,
        oid_to_so=oid_to_so,
        oid_to_deg=oid_to_deg,
        n_neg=8,
        rng=np.random.default_rng(2),
        same_sn=True,
        tol_bp=10_000,
        tol_frac=1.0,
        degree_matched=True,
    )
    paired, retained = neg_distance_matched_paired(
        nodes=nodes,
        pos_pairs=positive_pairs,
        pos_set=positives,
        oid_to_sn=oid_to_sn,
        oid_to_so=oid_to_so,
        oid_to_deg=oid_to_deg,
        rng=np.random.default_rng(3),
        same_sn=True,
        tol_bp=10_000,
        tol_frac=1.0,
        degree_matched=True,
    )

    assert len(hard) == 8
    assert len(distance) == 8
    assert len(paired) == len(retained) == len(positive_pairs)
    assert_canonical_negative_integrity(hard, positives)
    assert_canonical_negative_integrity(distance, positives)
    assert_canonical_negative_integrity(paired, positives)
