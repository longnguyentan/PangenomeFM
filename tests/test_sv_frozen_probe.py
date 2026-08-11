from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.server.run_sv_frozen_probe_fold import (
    complete_feature_mask,
    pair_features,
    stratified_metrics,
)


def test_pair_features() -> None:
    left = np.array([[1.0, 2.0]], dtype=np.float32)
    right = np.array([[3.0, 1.0]], dtype=np.float32)
    result = pair_features(left, right)
    assert result.shape == (1, 8)
    np.testing.assert_allclose(result[0], [1, 2, 3, 1, 2, 1, 3, 2])


def test_complete_feature_mask_is_writable_and_requires_both_endpoints() -> None:
    examples = pd.DataFrame(
        {
            "start_segid": [1, 1, 4, 5],
            "end_segid": [2, 3, 2, 6],
        }
    )
    result = complete_feature_mask(
        examples,
        embedded_segids={1, 2, 4, 5, 6},
        cached_segids={1, 2, 3, 5, 6},
    )

    np.testing.assert_array_equal(result, [True, False, False, True])
    assert result.flags.writeable

    # Regression check for the server failure: callers can safely refine the
    # returned mask with in-place boolean operations.
    result &= np.array([True, True, True, False])
    np.testing.assert_array_equal(result, [True, False, False, False])


def test_sv_stratified_metrics() -> None:
    examples = pd.DataFrame(
        {
            "example_id": [1, 2, 3, 4],
            "svtype": ["DEL", "INS", "DEL", "INS"],
            "svlen": [60, 70, 600, 700],
            "length_bin": ["short", "short", "long", "long"],
            "allele_frequency": [0.1, 0.1, 0.01, 0.01],
            "af_bin": ["common", "common", "rare", "rare"],
            "start_mapping_rule": ["overlap"] * 4,
            "end_mapping_rule": ["overlap"] * 4,
            "start_mapping_distance": [0] * 4,
            "end_mapping_distance": [0] * 4,
        }
    )
    predictions = pd.DataFrame(
        {
            "example_id": [1, 2, 3, 4],
            "chromosome": ["chr1", "chr1", "chr2", "chr2"],
            "y_true": [0, 1, 0, 1],
            "feature_set": ["toy"] * 4,
            "p_calibrated": [0.1, 0.9, 0.2, 0.8],
            "threshold": [0.5] * 4,
        }
    )
    result = stratified_metrics(predictions, examples)
    assert set(result["stratum"]) == {
        "chromosome", "sv_class", "length_bin", "allele_frequency_bin", "breakpoint_mapping_distance"
    }
    assert result.loc[result["stratum"].eq("length_bin"), "accuracy"].eq(1.0).all()
