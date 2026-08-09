from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.server.prepare_sv_breakpoint_examples import map_coordinate, prepare


def test_coordinate_mapping_rules() -> None:
    import numpy as np

    starts = np.array([0, 100, 200])
    ends = np.array([50, 150, 250])
    segids = np.array([1, 2, 3])
    assert map_coordinate(120, starts, ends, segids, maximum_nearest_distance=20) == (2, 0, "overlap")
    assert map_coordinate(175, starts, ends, segids, maximum_nearest_distance=30) == (3, 25, "nearest")
    assert map_coordinate(500, starts, ends, segids, maximum_nearest_distance=10)[0] is None


def test_prepare_sv_examples(tmp_path: Path) -> None:
    segments = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "SN": ["GRCh38#0#chr1"] * 3,
            "SO": [0, 100, 200],
            "LN": [100, 100, 100],
        }
    )
    segment_path = tmp_path / "segments.csv.gz"
    segments.to_csv(segment_path, index=False, compression="gzip")
    variants = pd.DataFrame(
        {
            "chrom": ["1", "chr1", "chr1"],
            "pos": [11, 121, 250],
            "end": [20, 180, 260],
            "variant_id": ["ins", "del", "small"],
            "svtype": ["INS", "DEL", "INS"],
            "svlen": [100, 60, 20],
            "length_bin": ["bp[100,500)", "bp[50,100)", "bp[0,50)"],
            "allele_frequency": [0.1, 0.2, 0.3],
            "af_bin": ["af[0.05,0.5)"] * 3,
            "filter": ["PASS"] * 3,
        }
    )
    variant_path = tmp_path / "variants.csv.gz"
    variants.to_csv(variant_path, index=False, compression="gzip")
    audit = prepare(
        normalized_variants=variant_path,
        full_segments=segment_path,
        out_dir=tmp_path / "out",
        reference_prefix="GRCh38#0",
        svtypes={"INS", "DEL"},
        minimum_length=50,
        maximum_nearest_distance=10,
    )
    assert audit["mapped_example_count"] == 2
    examples = pd.read_csv(tmp_path / "out/sv_breakpoint_examples.csv.gz")
    assert set(examples["binary_svtype_label"]) == {0, 1}
    assert set(examples["start_mapping_rule"]) == {"overlap"}
    nodes = pd.read_csv(tmp_path / "out/feature_nodes.csv.gz")
    assert set(nodes["ccre_label"]) == {0}
