from pathlib import Path
import gzip

import pandas as pd
import pytest

from tasks.haplotype.annotations import (
    _load_labeled_bed,
    annotate_paired_windows,
    interval_overlap_fraction,
)


def test_interval_overlap_fraction_merges_overlaps() -> None:
    assert interval_overlap_fraction([(0, 30), (20, 50), (80, 120)], 0, 100) == 0.7


def test_hmmflagger_labels_are_retained(tmp_path: Path) -> None:
    path = tmp_path / "hmm.bed.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("D#1#chr8\t10\t20\tDup\n")
        handle.write("D#1#chr8\t30\t40\tCol\n")
    labels = _load_labeled_bed(path)
    assert labels["dup"]["D#1#chr8"] == [(10, 20)]
    assert labels["col"]["D#1#chr8"] == [(30, 40)]


def test_annotation_proxies_are_explicit_when_external_files_missing(
    tmp_path: Path,
) -> None:
    pairs = tmp_path / "pairs.csv.gz"
    pd.DataFrame(
        [
            {
                "donor_id": "D",
                "h1_contig": "chr8",
                "h2_contig": "chr8",
                "h1_window_start": 0,
                "h2_window_start": 0,
                "h1_path_bp": 10_000,
                "h2_path_bp": 9_900,
                "h1_mean_path_node_multiplicity": 1.2,
                "h2_mean_path_node_multiplicity": 1.0,
                "h1_multicopy_fraction": 0.25,
                "h2_multicopy_fraction": 0.0,
                "node_jaccard": 0.8,
            }
        ]
    ).to_csv(pairs, index=False, compression="gzip")
    output = tmp_path / "annotated.csv.gz"
    summary = annotate_paired_windows(
        pairs_path=pairs,
        epigenome_root=tmp_path / "missing",
        out_path=output,
        strict=False,
    )
    row = pd.read_csv(output).iloc[0]
    assert row["h1_graph_mappability_proxy"] == pytest.approx(0.75)
    assert row["delta_copy_number_proxy"] == pytest.approx(0.2)
    assert row["sv_event_proxy"] == 1
    assert "mappability" in summary["proxies_not_ground_truth"]
