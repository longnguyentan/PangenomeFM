import numpy as np
import pandas as pd
import pytest

from scripts.server.matched_locus_enrichment import (
    annotate_overlaps,
    build_matched_sets,
    matched_inference,
    validate_intervals,
    REQUIRED_REGION_COLUMNS,
    REQUIRED_SIGNAL_COLUMNS,
)


def synthetic_regions() -> pd.DataFrame:
    rows = []
    for index in range(18):
        rows.append(
            {
                "region_id": f"r{index}",
                "chromosome": "8",
                "start": index * 100,
                "end": index * 100 + 50,
                "is_prioritized": index < 3,
                "region_length": 50,
                "gc_content": 0.40 + index * 0.001,
            }
        )
    return pd.DataFrame(rows)


def test_overlap_matching_and_inference_are_reproducible() -> None:
    regions = validate_intervals(synthetic_regions(), REQUIRED_REGION_COLUMNS, "region")
    signals = validate_intervals(
        pd.DataFrame(
            {
                "signal_id": ["s0", "s1", "s2"],
                "chromosome": ["chr8", "chr8", "chr8"],
                "start": [1, 101, 201],
                "end": [2, 102, 202],
            }
        ),
        REQUIRED_SIGNAL_COLUMNS,
        "signal",
    )
    annotated, overlaps = annotate_overlaps(regions, signals)
    assignments, exclusions = build_matched_sets(
        annotated,
        priority_column="is_prioritized",
        match_columns=["region_length", "gc_content"],
        controls_per_case=2,
        relative_caliper=1.0,
    )
    summary_a, draws_a = matched_inference(
        assignments, n_permutations=199, n_bootstrap=100, seed=42
    )
    summary_b, draws_b = matched_inference(
        assignments, n_permutations=199, n_bootstrap=100, seed=42
    )

    assert len(overlaps) == 3
    assert exclusions.empty
    assert assignments["matched_set_id"].nunique() == 3
    assert np.isclose(summary_a.loc[0, "case_hit_fraction"], 1.0)
    assert np.isclose(summary_a.loc[0, "control_hit_fraction"], 0.0)
    pd.testing.assert_frame_equal(summary_a, summary_b)
    pd.testing.assert_frame_equal(draws_a, draws_b)


def test_missing_requested_covariate_is_a_hard_error() -> None:
    regions = synthetic_regions().assign(signal_hit=0, signal_count=0)
    with pytest.raises(ValueError, match="refusing an unmatched enrichment"):
        build_matched_sets(
            regions,
            priority_column="is_prioritized",
            match_columns=["gc_content", "mappability"],
            controls_per_case=2,
            relative_caliper=0.25,
        )


def test_overlap_search_handles_variable_length_signals() -> None:
    regions = validate_intervals(
        pd.DataFrame(
            {
                "region_id": ["left", "hit", "right"],
                "chromosome": ["chr1"] * 3,
                "start": [0, 100, 300],
                "end": [50, 110, 350],
            }
        ),
        REQUIRED_REGION_COLUMNS,
        "region",
    )
    signals = validate_intervals(
        pd.DataFrame(
            {
                "signal_id": ["long", "point"],
                "chromosome": ["chr1", "chr1"],
                "start": [75, 340],
                "end": [105, 341],
            }
        ),
        REQUIRED_SIGNAL_COLUMNS,
        "signal",
    )
    annotated, overlaps = annotate_overlaps(regions, signals)
    assert overlaps[["region_id", "signal_id"]].values.tolist() == [
        ["hit", "long"],
        ["right", "point"],
    ]
    assert annotated["signal_hit"].tolist() == [0, 1, 1]
