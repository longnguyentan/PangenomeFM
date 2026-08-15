from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from scripts.server import build_path_branch_choice_examples as branch


def write_gzip(path: Path, text: str) -> None:
    with gzip.open(path, "wt") as handle:
        handle.write(text)


def test_collect_adjacency_includes_reverse_complement(tmp_path: Path) -> None:
    gfa = tmp_path / "tiny.gfa.gz"
    write_gzip(gfa, "L\t1\t+\t2\t-\t0M\nL\t1\t+\t3\t+\t0M\n")
    targets = {("1", "+"): {("2", "-")}, ("2", "+"): {("1", "-")}}
    adjacency, counters = branch.collect_adjacency(
        gfa=gfa,
        positive_targets=targets,
        alternatives_per_positive=2,
    )
    assert ("2", "-") in adjacency[("1", "+")]
    assert ("3", "+") in adjacency[("1", "+")]
    assert ("1", "-") in adjacency[("2", "+")]
    assert counters["gfa_link_rows"] == 2


def test_branch_choice_cli_writes_one_positive_per_group(
    tmp_path: Path, monkeypatch
) -> None:
    gfa = tmp_path / "tiny.gfa.gz"
    write_gzip(gfa, "L\t1\t+\t2\t+\t0M\nL\t1\t+\t3\t+\t0M\n")
    graph_audit = tmp_path / "graph_audit.json"
    graph_audit.write_text(
        json.dumps(
            {
                "status": "complete",
                "gfa_sha256": branch.sha256_file(gfa),
            }
        )
    )
    positives = tmp_path / "positive.csv.gz"
    pd.DataFrame(
        {
            "dataset": ["toy"],
            "donor_split": ["test"],
            "sample": ["sample1"],
            "haplotype": [1],
            "locus": ["chr1"],
            "chromosome": ["chr1"],
            "path_name": ["sample1#1#chr1"],
            "window_start": [0],
            "window_end": [100],
            "path_edge_index": [4],
            "path_position": [20],
            "u_segment": [1],
            "u_orientation": ["+"],
            "v_segment": [2],
            "v_orientation": ["+"],
            "deterministic_rank": [7],
        }
    ).to_csv(positives, index=False)
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_path_branch_choice_examples.py",
            "--positive-edges",
            str(positives),
            "--gfa",
            str(gfa),
            "--graph-audit",
            str(graph_audit),
            "--out-dir",
            str(out),
            "--alternatives-per-positive",
            "2",
        ],
    )
    assert branch.main() == 0
    candidates = pd.read_csv(out / "branch_choice_candidates.csv.gz")
    assert len(candidates) == 2
    assert candidates.groupby("group_id")["label"].sum().eq(1).all()
    audit = json.loads((out / "audit.json").read_text())
    assert audit["candidate_groups"] == 1
    assert audit["observed_edges_absent_from_gfa"] == 0
