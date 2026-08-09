from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "server"
    / "prepare_ccre_feature_cache.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_ccre_cache", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_feature_cache_is_aligned_and_machine_readable(tmp_path: Path) -> None:
    segments = tmp_path / "segments.csv"
    links = tmp_path / "links.csv"
    labels = tmp_path / "labels.csv"
    output = tmp_path / "features.npz"
    pd.DataFrame(
        {
            "id": ["s1", "s2", "s3"],
            "name": ["s1", "s2", "s3"],
            "seq": ["ACGT", "AAAA", "CGCG"],
            "LN": [4, 4, 4],
            "SN": ["GRCh38#0#chr1"] * 3,
            "SO": [0, 4, 8],
            "SR": [0, 0, 0],
        }
    ).to_csv(segments, index=False)
    pd.DataFrame(
        {
            "from_seg": ["s1", "s2"],
            "from_orient": ["+", "+"],
            "to_seg": ["s2", "s3"],
            "to_orient": ["+", "+"],
            "overlap": ["0M", "0M"],
        }
    ).to_csv(links, index=False)
    pd.DataFrame(
        {
            "segid": [0, 1, 2],
            "chrom": ["chr1"] * 3,
            "SO": [0, 4, 8],
            "LN": [4, 4, 4],
            "ccre_label": [0, 1, 0],
        }
    ).to_csv(labels, index=False)
    audit = MODULE.prepare_cache(
        full_segments=segments,
        full_links=links,
        node_labels=labels,
        output=output,
        force=False,
    )
    assert audit["status"] == "complete"
    with np.load(output, allow_pickle=False) as cache:
        assert cache["segid"].tolist() == [0, 1, 2]
        assert cache["sequence_kmer"].shape == (3, 86)
        assert cache["structural"].shape[0] == 3
