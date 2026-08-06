from __future__ import annotations

import csv
from pathlib import Path

from graph.validation import validate_graph_tables, validate_path_metadata


def _write(path: Path, rows: list[dict[str, object]], fields: list[str], delimiter: str = ",") -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def test_graph_validation_detects_endpoint_orientation_and_length_problems(tmp_path: Path) -> None:
    segments = tmp_path / "segments.csv"
    links = tmp_path / "links.csv"
    _write(
        segments,
        [
            {"id": 0, "name": "a", "seq": "AACCGG", "LN": 6, "SN": "HG001.1#1#chr1", "SO": 0, "SR": 1},
            {"id": 1, "name": "b", "seq": "TT", "LN": 3, "SN": "HG001.2#2#chr1", "SO": 6, "SR": 2},
        ],
        ["id", "name", "seq", "LN", "SN", "SO", "SR"],
    )
    _write(
        links,
        [
            {
                "from_seg": "a",
                "from_orient": "+",
                "to_seg": "missing",
                "to_orient": "?",
                "overlap": "0M",
                "SR": 1,
                "L1": 6,
                "L2": 3,
            }
        ],
        ["from_seg", "from_orient", "to_seg", "to_orient", "overlap", "SR", "L1", "L2"],
    )

    summary, issues = validate_graph_tables(
        segments_path=segments,
        links_path=links,
        label="toy",
        max_node_length=4,
    )

    issue_counts = {row["issue_type"]: row["count"] for row in issues}
    assert summary["validation_status"] == "fail"
    assert summary["donors"] == 1
    assert summary["phased_haplotypes"] == 2
    assert issue_counts["missing_link_endpoints"] == 1
    assert issue_counts["invalid_orientation_fields"] == 1
    assert issue_counts["sequence_length_mismatch"] == 1
    assert issue_counts["nodes_over_configured_length"] == 1


def test_path_metadata_duplicate_detection(tmp_path: Path) -> None:
    metadata = tmp_path / "paths.tsv"
    _write(
        metadata,
        [
            {"path_name": "p1", "haplotype": "1", "step_count": 2},
            {"path_name": "p1", "haplotype": "x", "step_count": -1},
        ],
        ["path_name", "haplotype", "step_count"],
        delimiter="\t",
    )
    summary, issues = validate_path_metadata(metadata)
    counts = {row["issue_type"]: row["count"] for row in issues}
    assert summary["validation_status"] == "fail"
    assert counts["duplicate_path_identifiers"] == 1
    assert counts["invalid_path_step_counts"] == 1
    assert counts["noncanonical_haplotype_labels"] == 1
