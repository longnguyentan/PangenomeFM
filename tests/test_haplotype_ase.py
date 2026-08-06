from __future__ import annotations

import numpy as np

from tasks.haplotype.ase import paired_delta_features


def test_paired_delta_features_are_antisymmetric():
    ids = np.array(["a", "b", "c"])
    embeddings = np.array([[1.0, 2.0], [4.0, 1.0], [2.0, 5.0]])
    forward = paired_delta_features(["a", "b"], ["b", "c"], ids, embeddings)
    reverse = paired_delta_features(["b", "c"], ["a", "b"], ids, embeddings)
    np.testing.assert_allclose(forward, -reverse)
