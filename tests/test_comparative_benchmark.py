from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.comparative_benchmark import (
    canonical_oriented_pair,
    candidate_partitions,
    method_fairness_row,
    oriented_reverse_complement,
    stable_example_id,
    validate_benchmark_manifest,
)


def test_candidate_partitions_match_expected_sizes_and_seed() -> None:
    first = candidate_partitions(101, 20260806)
    second = candidate_partitions(101, 20260806)
    other = candidate_partitions(101, 42)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert dict(zip(*np.unique(first, return_counts=True))) == {
        "test": 20,
        "train": 71,
        "validation": 10,
    }


def test_stable_ids_and_reverse_complement() -> None:
    assert stable_example_id(["chr22", 10, 20]) == stable_example_id(
        ["chr22", 10, 20]
    )
    assert stable_example_id(["chr22", 10, 20]) != stable_example_id(
        ["chr22", 10, 21]
    )
    assert oriented_reverse_complement(10, 20) == (21, 11)
    assert canonical_oriented_pair(10, 20) == canonical_oriented_pair(21, 11)


def tiny_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["a", "b"],
            "dataset": ["hprc", "hprc"],
            "graph_release": ["r2", "r2"],
            "coordinate_system": ["GRCh38", "GRCh38"],
            "chromosome": ["chr22", "chr22"],
            "start": [10, 10],
            "end": [20, 20],
            "fold": ["fold_b", "fold_b"],
            "chromosome_split": ["test", "test"],
            "candidate_partition": ["train", "test"],
            "context": ["strict", "strict"],
            "source_oid": [0, 2],
            "destination_oid": [2, 4],
            "source_orientation": ["+", "+"],
            "destination_orientation": ["+", "+"],
            "source_segment_name": ["s0", "s1"],
            "destination_segment_name": ["s1", "s2"],
            "source_coordinate": [10, 11],
            "destination_coordinate": [11, 12],
            "label": [1, 0],
        }
    )


def test_manifest_and_fairness_audit_surface_conversion_loss() -> None:
    manifest = tiny_manifest()
    assert validate_benchmark_manifest(manifest) == []
    row = method_fairness_row(
        method="DeepGene",
        compatibility="approximate",
        requested_examples=2,
        exact_example_ids=[],
        approximate_example_ids=["a"],
        benchmark=manifest,
        status="BLOCKED_EXACT",
        blocker="task mismatch",
    )
    assert row["exact_conversion_success_pct"] == 0.0
    assert row["approximate_conversion_success_pct"] == 50.0
    assert row["positive_count"] == 1
    assert row["negative_count"] == 1
