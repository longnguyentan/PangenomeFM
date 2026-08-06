from pathlib import Path

import pandas as pd

from scripts.server.compute_chromosome_graph_statistics import statistics


def test_chromosome_statistics_counts_within_and_cross_edges(tmp_path: Path):
    segments = tmp_path / "segments.csv"
    links = tmp_path / "links.csv"
    pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "name": ["a", "b", "c"],
            "SN": ["GRCh38#0#chr1", "GRCh38#0#chr1", "GRCh38#0#chr2"],
            "SO": [0, 10, 0],
            "LN": [10, 5, 8],
        }
    ).to_csv(segments, index=False)
    pd.DataFrame(
        {
            "from_seg": ["a", "b", "b"],
            "to_seg": ["b", "a", "c"],
        }
    ).to_csv(links, index=False)

    frame, summary = statistics("tiny", segments, links, chunksize=2)
    by_chromosome = frame.set_index("chromosome")
    assert by_chromosome.loc["chr1", "nodes"] == 2
    assert by_chromosome.loc["chr1", "within_chromosome_edges"] == 2
    assert by_chromosome.loc["chr2", "within_chromosome_edges"] == 0
    assert summary["links_total"] == 3
    assert summary["cross_chromosome_edges"] == 1
    assert summary["link_rows_with_missing_or_noncanonical_endpoint"] == 0
