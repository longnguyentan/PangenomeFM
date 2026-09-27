"""Tests for handcrafted topology controls, pair geometry and factorial sets."""

from __future__ import annotations

import numpy as np

from evaluation.modality_factorial import build_modality_factorial, factorial_feature_access
from tasks.transfer.topology_controls import FEATURE_NAMES, compute_topology_features
from training.node_inputs import PAIR_GEOMETRY_DIM, pair_geometry


def _deletion_bubble():
    # 0 -> 1 -> 2 -> 3 with bypass 0 -> 2 (deletion of segment 1)
    from_id = np.array([0, 1, 2, 0])
    to_id = np.array([1, 2, 3, 2])
    rev = np.zeros(4, dtype=bool)
    return from_id, rev, to_id, rev


def test_topology_features_shape_and_bubble_flags():
    f, fr, t, tr = _deletion_bubble()
    segids, feats = compute_topology_features(4, np.array([10, 20, 30, 40.0]), f, fr, t, tr)
    assert feats.shape == (4, len(FEATURE_NAMES)) and np.isfinite(feats).all()
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}
    assert feats[0, col["is_branching"]] == 1.0  # two successors
    assert feats[2, col["is_branching"]] == 1.0  # two predecessors
    assert feats[3, col["is_branching"]] == 0.0
    assert feats[1, col["deletion_bypass"]] == 1.0  # predecessor 0 links to successor 2
    assert feats[3, col["hops_to_branching_capped8"]] == 1.0


def test_topology_controls_restrict_to_targets():
    f, fr, t, tr = _deletion_bubble()
    segids, feats = compute_topology_features(4, np.ones(4), f, fr, t, tr, target_segids=[3, 1])
    assert segids.tolist() == [1, 3] and feats.shape[0] == 2


def test_branch_distance_includes_four_to_seven_hops_and_disconnected_nodes():
    # Branch at 0: a chain to 10 and a second outgoing edge to 11; 12 isolated.
    f = np.array(list(range(10)) + [0])
    t = np.array(list(range(1, 11)) + [11])
    rev = np.zeros(len(f), dtype=bool)
    ids, features = compute_topology_features(13, np.ones(13), f, rev, t, rev)
    distance = features[:, FEATURE_NAMES.index("hops_to_branching_capped8")]
    assert distance.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 1, 8]


def test_factorial_adds_topology_control_sets_with_access_text():
    n = 5
    comps = {name: np.random.default_rng(0).normal(size=(n, 3)) for name in
             ("coordinate", "sequence_kmer", "frozen_pangenomefm", "frozen_sequence_fm", "topology_control")}
    feats = build_modality_factorial(comps, include_external_sequence=True, include_topology_control=True)
    key = "coordinate_plus_frozen_sequence_fm_plus_topology_control_plus_frozen_pangenomefm"
    assert key in feats and feats[key].shape == (n, 12)
    assert key in factorial_feature_access()
    plain = build_modality_factorial(comps, include_external_sequence=True)
    assert "topology_control" not in plain


def test_pair_geometry_contiguity_gap_is_zero_for_adjacent_segments():
    md = {"oid_to_so": {0: 100, 2: 150, 4: 400}, "oid_to_ln": {0: 50, 2: 250, 4: 10},
          "oid_to_sn": {0: "chr", 2: "chr", 4: "alt"}}
    g = pair_geometry(np.array([0, 0]), np.array([2, 4]), md)
    assert g.shape == (2, PAIR_GEOMETRY_DIM)
    assert g[0, 1] == 0.0 and g[0, 3] == 1.0
    assert g[1, 3] == 0.0 and g[1, 0] == 0.0  # different coordinate systems are gated


def test_pair_geometry_does_not_treat_missing_coordinates_as_shared():
    md = {"oid_to_so": {0: 100, 2: 150}, "oid_to_ln": {0: 50}}
    g = pair_geometry(np.array([0]), np.array([2]), md)
    assert g[0, 3] == 0 and g[0, 0] == 0 and g[0, 1] == 0
