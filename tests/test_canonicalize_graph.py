import csv
import gzip
from pathlib import Path

import pandas as pd

from data.canonicalize_graph import canonicalize_graph_tables


def _write(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_canonicalize_preserves_oriented_endpoints(tmp_path: Path) -> None:
    segments = tmp_path / "segments.csv"
    links = tmp_path / "links.csv"
    _write(
        segments,
        [
            {"id": 0, "name": "a", "seq": "AACCGG", "LN": 6, "SN": "chr1", "SO": 10, "SR": 0},
            {"id": 1, "name": "b", "seq": "TTAA", "LN": 4, "SN": "chr1", "SO": 20, "SR": 0},
        ],
        ["id", "name", "seq", "LN", "SN", "SO", "SR"],
    )
    _write(
        links,
        [
            {
                "from_seg": "a",
                "from_orient": "-",
                "to_seg": "b",
                "to_orient": "+",
                "overlap": "0M",
                "SR": 0,
                "L1": 6,
                "L2": 4,
            }
        ],
        ["from_seg", "from_orient", "to_seg", "to_orient", "overlap", "SR", "L1", "L2"],
    )

    out = tmp_path / "out"
    stats = canonicalize_graph_tables(
        segments_path=segments,
        links_path=links,
        out_dir=out,
        max_node_length=3,
    )
    seg = pd.read_csv(out / "full_segments.csv.gz")
    edge = pd.read_csv(out / "full_links.csv.gz")

    assert stats["input_segments"] == 2
    assert stats["output_segments"] == 4
    assert seg["LN"].max() <= 3
    assert seg.loc[seg["name"] == "a~chop000001", "SO"].item() == 13
    external = edge[edge["from_seg"].eq("a~chop000000") & edge["to_seg"].eq("b~chop000000")]
    assert len(external) == 1
    assert external.iloc[0]["from_orient"] == "-"
    assert external.iloc[0]["to_orient"] == "+"
    with gzip.open(out / "segment_chop_map.csv.gz", "rt", encoding="utf-8") as handle:
        assert sum(1 for _ in handle) == 5
