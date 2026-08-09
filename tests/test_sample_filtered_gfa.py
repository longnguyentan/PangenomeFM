from __future__ import annotations

import gzip
from pathlib import Path

from scripts.server.materialize_sample_filtered_gfa import (
    canonical_edge,
    materialize,
)


def test_bidirected_edge_canonicalization() -> None:
    assert canonical_edge("1", "+", "2", "-") == canonical_edge("2", "+", "1", "-")


def test_materialize_excludes_unique_sample_topology(tmp_path: Path) -> None:
    source = tmp_path / "input.gfa.gz"
    with gzip.open(source, "wt") as handle:
        handle.write("H\tVN:Z:1.1\n")
        for node in ("1", "2", "3", "4"):
            handle.write(f"S\t{node}\tA\n")
        handle.write("L\t1\t+\t2\t+\t0M\n")
        handle.write("L\t2\t+\t3\t+\t0M\n")
        handle.write("L\t2\t+\t4\t+\t0M\n")
        handle.write("W\tKEEP\t1\tchr1\t0\t3\t>1>2>3\n")
        handle.write("W\tDROP\t1\tchr1\t0\t3\t>1>2>4\n")
    output = tmp_path / "filtered.gfa.gz"
    audit_path = tmp_path / "audit.json"
    audit = materialize(
        source_gfa=source,
        output_gfa=output,
        audit_path=audit_path,
        exclude_samples={"DROP"},
        include_samples=None,
        include_p_paths=True,
    )
    assert audit["status"] == "complete"
    assert audit["path_scan"]["selected_samples"] == ["KEEP"]
    with gzip.open(output, "rt") as handle:
        text = handle.read()
    assert "S\t3\tA" in text
    assert "S\t4\tA" not in text
    assert "L\t2\t+\t3\t+" in text
    assert "L\t2\t+\t4\t+" not in text
    assert "W\t" not in text
