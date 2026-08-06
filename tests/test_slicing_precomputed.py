from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graph.slicing import (
    build_global_index,
    build_incident_edge_index,
    induced_subgraph,
    map_links_to_segids,
)


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    segments = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "name": ["a", "b", "c"],
            "seq": ["A", "C", "G"],
            "LN": [1, 1, 1],
            "SN": ["ref", "ref", "alt"],
            "SO": [0, 1, 1],
            "SR": [0, 0, 1],
        }
    )
    links = pd.DataFrame(
        {
            "from_seg": ["a", "a"],
            "from_orient": ["+", "+"],
            "to_seg": ["b", "c"],
            "to_orient": ["+", "+"],
            "overlap": ["0M", "0M"],
        }
    )
    return segments, links


def test_precomputed_endpoints_match_legacy_slicing() -> None:
    segments, links = _tables()
    index, aligned = build_global_index(segments)
    source, target = map_links_to_segids(links, index)
    indptr, edge_ids = build_incident_edge_index(source, target, len(index))
    legacy_segments, legacy_links, legacy_ids = induced_subgraph(
        segments, links, index, np.array([0]), add_one_hop=True
    )
    fast_segments, fast_links, fast_ids = induced_subgraph(
        aligned,
        links,
        index,
        np.array([0]),
        add_one_hop=True,
        from_id=source,
        to_id=target,
        incident_indptr=indptr,
        incident_edge_ids=edge_ids,
        segments_aligned_to_index=True,
    )
    assert fast_segments["name"].tolist() == legacy_segments["name"].tolist()
    assert fast_links.equals(legacy_links)
    assert fast_ids.tolist() == legacy_ids.tolist()


def test_precomputed_endpoints_require_link_alignment() -> None:
    segments, links = _tables()
    index, aligned = build_global_index(segments)
    with pytest.raises(ValueError, match="must align"):
        induced_subgraph(
            aligned,
            links,
            index,
            np.array([0]),
            add_one_hop=False,
            from_id=np.array([0]),
            to_id=np.array([1]),
            segments_aligned_to_index=True,
        )


def test_incident_index_rejects_mismatched_endpoints() -> None:
    with pytest.raises(ValueError, match="equal length"):
        build_incident_edge_index(np.array([0, 1]), np.array([1]), 2)
