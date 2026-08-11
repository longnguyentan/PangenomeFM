from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from scripts.server.materialize_path_resolved_examples import (
    canonical_chromosome,
    iter_path_handles,
    load_path_records,
    materialize,
)


def test_path_handle_parser_and_chromosome_aliases() -> None:
    assert list(iter_path_handles(">s1>s2<s3")) == [
        ("s1", "+"),
        ("s2", "+"),
        ("s3", "-"),
    ]
    assert canonical_chromosome("GRCh38#0#chr22") == "chr22"
    assert canonical_chromosome("id=CHM13|chrY") == "chrY"


def test_path_selection_falls_back_to_chromosome_in_path_name(tmp_path: Path) -> None:
    records = pd.DataFrame(
        {
            "donor_split": ["train", "train"],
            "sample": ["HG001", "HG002"],
            "haplotype": ["1", "2"],
            "locus": ["assembly_contig_1", "chr21"],
            "path_name": ["HG001#1#chr22[0-100]", "HG002#2#chr21"],
        }
    )
    path = tmp_path / "paths.csv.gz"
    records.to_csv(path, index=False, compression="gzip")

    selected = load_path_records(
        path,
        chromosomes={"chr22"},
        splits={"train"},
        max_paths=100,
    )

    assert selected["path_name"].tolist() == ["HG001#1#chr22[0-100]"]
    assert selected["chromosome"].tolist() == ["chr22"]


def test_materialize_from_cached_gaf(tmp_path: Path) -> None:
    segments = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "name": ["s1", "s2", "s3", "s4"],
            "LN": [10, 10, 10, 10],
        }
    )
    segment_path = tmp_path / "segments.csv.gz"
    segments.to_csv(segment_path, index=False, compression="gzip")
    records = pd.DataFrame(
        {
            "dataset": ["toy"],
            "donor_split": ["test"],
            "sample": ["HG001"],
            "haplotype": ["1"],
            "locus": ["chr1"],
            "path_name": ["HG001#1#chr1#0"],
        }
    )
    records_path = tmp_path / "paths.csv.gz"
    records.to_csv(records_path, index=False, compression="gzip")
    gaf = tmp_path / "paths.gaf.gz"
    with gzip.open(gaf, "wt") as handle:
        handle.write("HG001#1#chr1#0\t40\t0\t40\t+\t>s1>s2<s3>s4\t40\t0\t40\t40\t40\t60\n")

    audit = materialize(
        gbz=None,
        gaf_input=gaf,
        path_records=records_path,
        full_segments=segment_path,
        out_dir=tmp_path / "out",
        chromosomes={"chr1"},
        splits={"test"},
        max_paths=None,
        window_bp=20,
        edges_per_window=1,
        vg_executable="vg",
    )
    assert audit["status"] == "complete"
    assert audit["counters"]["path_edges_seen"] == 3
    assert audit["counters"]["sampled_positive_edges"] == 2
    output = pd.read_csv(tmp_path / "out/positive_path_edges.csv.gz")
    assert len(output) == 2
    assert set(output["donor_split"]) == {"test"}
    assert set(output["chromosome"]) == {"chr1"}
    assert not output[["path_name", "window_start"]].duplicated().any()
