from __future__ import annotations

import gzip
import json

import pandas as pd

from scripts.server.index_gfa_segments import build_index, segment_length


def test_segment_length_prefers_ln_and_audits_disagreement() -> None:
    assert segment_length("ACGT", []) == (4, False)
    assert segment_length("*", ["LN:i:17"]) == (17, False)
    assert segment_length("ACGT", ["LN:i:5"]) == (5, True)


def test_build_minimal_gfa_segment_index(tmp_path) -> None:
    gfa = tmp_path / "graph.gfa.gz"
    with gzip.open(gfa, "wt") as handle:
        handle.write("H\tVN:Z:1.1\n")
        handle.write("S\t10\tACGT\tLN:i:4\n")
        handle.write("S\t11\t*\tLN:i:7\n")
        handle.write("L\t10\t+\t11\t+\t0M\n")
        handle.write("P\tref#0#chr1\t10+,11+\t*\n")
        handle.write("W\tHG001\t1\tchr1\t0\t11\t>10>11\n")
    output = tmp_path / "segments.csv.gz"
    audit = tmp_path / "segments.audit.json"

    result = build_index(gfa=gfa, output=output, audit=audit)

    assert result["status"] == "complete"
    assert result["counters"]["segments"] == 2
    assert result["counters"]["paths"] == 1
    assert result["counters"]["walks"] == 1
    frame = pd.read_csv(output)
    assert frame.astype({"name": str}).to_dict("records") == [
        {"id": 0, "name": "10", "LN": 4},
        {"id": 1, "name": "11", "LN": 7},
    ]
    assert json.loads(audit.read_text())["success_criteria"]["unique_segment_names"]
