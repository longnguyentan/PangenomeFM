import numpy as np
import pandas as pd
import pytest

import data.make_benchmark as benchmark_module
from graph.slicing import choose_window_gap_aware
from data.make_benchmark import (
    build_manifest,
    plan_tiled_ranges,
    retain_locus_matched_closures,
)


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


def test_locus_matching_removes_closure_specific_successes() -> None:
    manifest = pd.DataFrame(
        [
            {
                "slice_id": "strict_shared",
                "target_sn": "GRCh38#0#chr22",
                "closure": "strict",
                "start": 10_000_000,
                "end": 15_000_000,
            },
            {
                "slice_id": "onehop_unmatched",
                "target_sn": "GRCh38#0#chr22",
                "closure": "1hop",
                "start": 5_000_000,
                "end": 10_000_000,
            },
            {
                "slice_id": "onehop_shared",
                "target_sn": "GRCh38#0#chr22",
                "closure": "1hop",
                "start": 10_000_000,
                "end": 15_000_000,
            },
        ]
    )

    retained, excluded = retain_locus_matched_closures(
        manifest, ["strict", "1hop"]
    )

    assert retained["slice_id"].tolist() == ["strict_shared", "onehop_shared"]
    assert excluded["slice_id"].tolist() == ["onehop_unmatched"]
    assert excluded.loc[0, "missing_closures"] == "strict"
    assert (
        excluded.loc[0, "exclusion_reason"]
        == "successful_slice_missing_required_closure"
    )


def test_build_manifest_reports_ineligible_coordinate_tiles(
    tmp_path, monkeypatch
) -> None:
    target = "GRCh38#0#chr22"
    segments = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "name": ["a", "b", "c", "d"],
            "seq": ["AAAAA", "CCCCC", "GGGGG", "TTTTT"],
            "LN": [5, 5, 5, 5],
            "SN": [target, "sample#1#chr22", target, target],
            "SO": [0, 0, 10, 15],
            "SR": [0, 1, 0, 0],
        }
    )
    links = pd.DataFrame(
        {
            "from_seg": ["a", "c"],
            "from_orient": ["+", "+"],
            "to_seg": ["b", "d"],
            "to_orient": ["+", "+"],
            "overlap": ["0M", "0M"],
        }
    )

    def paired_stub(*, pos_pairs, **_kwargs):
        return [(int(pos_pairs[0, 1]), int(pos_pairs[0, 0]))], np.asarray([0])

    monkeypatch.setattr(
        benchmark_module, "neg_distance_matched_paired", paired_stub
    )

    manifest = build_manifest(
        segs=segments,
        links=links,
        targets=[target],
        out_dir=tmp_path,
        seed=7,
        window_bp=10,
        n_windows=1,
        closures=["strict", "1hop"],
        negative_sampler="distance_matched",
        negative_degree_matched=True,
        matched_closure_windows=True,
        tile_stride_bp=10,
        negative_shortfall_policy="paired_subsample",
    )

    assert len(manifest) == 2
    assert set(manifest["closure"]) == {"strict", "1hop"}
    assert set(zip(manifest["start"], manifest["end"])) == {(10, 20)}

    exclusions = pd.read_csv(tmp_path / "exclusions.csv")
    assert set(exclusions["exclusion_reason"]) == {
        "no_induced_subgraph_links",
        "successful_slice_missing_required_closure",
    }

    coverage = pd.read_json(tmp_path / "coverage_report.json", typ="series")
    assert coverage["planned_tile_count"] == 2
    assert coverage["retained_locus_matched_tile_count"] == 1
    assert coverage["paired_task_eligible_tile_fraction"] == 0.5
