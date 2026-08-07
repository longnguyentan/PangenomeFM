from __future__ import annotations

import numpy as np

from graph.neg_sampling import neg_distance_matched_paired


def test_paired_distance_sampler_returns_unique_constraint_matched_pairs():
    nodes = np.arange(6, dtype=np.int64)
    positives = np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int64)
    positive_set = {tuple(pair) for pair in positives.tolist()}
    coordinates = {0: 0, 1: 10, 2: 20, 3: 30, 4: 40, 5: 50}
    sequence_names = {node: "chr1" for node in nodes.tolist()}
    degrees = {node: 1 for node in nodes.tolist()}

    negatives, positive_indices = neg_distance_matched_paired(
        nodes=nodes,
        pos_pairs=positives,
        pos_set=positive_set,
        oid_to_sn=sequence_names,
        oid_to_so=coordinates,
        oid_to_deg=degrees,
        rng=np.random.default_rng(7),
        same_sn=True,
        tol_bp=0,
        tol_frac=0.0,
        degree_matched=True,
        max_tries_per_positive=2_000,
    )

    assert len(negatives) == len(positive_indices) > 0
    assert len(negatives) == len(set(negatives))
    for negative, positive_index in zip(negatives, positive_indices, strict=True):
        positive = positives[int(positive_index)]
        assert negative not in positive_set
        assert negative[0] != negative[1]
        assert abs(coordinates[negative[0]] - coordinates[negative[1]]) == abs(
            coordinates[int(positive[0])] - coordinates[int(positive[1])]
        )
