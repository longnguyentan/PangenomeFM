import numpy as np
import pandas as pd
import pytest

from graph.slicing import choose_window_gap_aware
from data.make_benchmark import plan_tiled_ranges


def test_distributed_sampling_fills_nonoverlapping_windows() -> None:
    intervals = pd.DataFrame(
        {
            "SO": np.arange(0, 10_000_000, 10_000, dtype=np.int64),
            "END": np.arange(0, 10_000_000, 10_000, dtype=np.int64) + 500,
        }
    )
    rng = np.random.default_rng(20260804)
    selected: list[tuple[int, int]] = []
    attempts = 0
    while len(selected) < 10 and attempts < 150:
        attempts += 1
        start, end, _ = choose_window_gap_aware(
            intervals, 50_000, rng, prefer_dense=False
        )
        if any(max(start, old_start) < min(end, old_end) for old_start, old_end in selected):
            continue
        selected.append((start, end))

    assert len(selected) == 10


def test_deterministic_tiling_covers_reference_extent() -> None:
    intervals = pd.DataFrame(
        {"SO": [10, 4_999_999, 10_000_001], "END": [100, 5_000_010, 10_000_100]}
    )
    ranges = plan_tiled_ranges(intervals, window_bp=5_000_000, stride_bp=5_000_000)
    assert ranges == [
        (0, 5_000_000),
        (5_000_000, 10_000_000),
        (10_000_000, 15_000_000),
    ]


def test_deterministic_tiling_rejects_coordinate_gaps() -> None:
    intervals = pd.DataFrame({"SO": [0], "END": [20_000_000]})
    with pytest.raises(ValueError, match="cannot exceed"):
        plan_tiled_ranges(
            intervals, window_bp=5_000_000, stride_bp=10_000_000
        )
