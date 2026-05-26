from __future__ import annotations

import pandas as pd

from graph.features import build_oid_metadata_from_segments
from graph.slicing import build_global_index


def test_reference_feature_includes_chm13_sr0():
    segments = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "name": ["ref", "alt", "grch"],
            "seq": ["*", "*", "*"],
            "LN": [10, 10, 10],
            "SN": ["id=CHM13|chr22", "HG001#1#chr22", "GRCh38#0#chr22"],
            "SO": [0, 10, 20],
            "SR": [0, 1, 0],
        }
    )
    seg_index, _ = build_global_index(segments)
    md = build_oid_metadata_from_segments(segments, seg_index)

    assert md["oid_to_is_grch38"][0] == 1
    assert md["oid_to_is_grch38"][2] == 0
    assert md["oid_to_is_grch38"][4] == 1
