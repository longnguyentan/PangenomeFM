from pathlib import Path

import pandas as pd

from scripts.prepare_hprc_chr_paths import (
    annotate_reference_chromosome_blocks,
    prepare_paths,
)


def test_reference_blocks_follow_primary_reference_rows() -> None:
    metadata = pd.DataFrame(
        [
            {"SENSE": "REFERENCE", "LOCUS": "chr7"},
            {"SENSE": "HAPLOTYPE", "LOCUS": "CM000100.1"},
            {"SENSE": "REFERENCE", "LOCUS": "chr8"},
            {"SENSE": "HAPLOTYPE", "LOCUS": "JBI001.1"},
            {"SENSE": "REFERENCE", "LOCUS": "chr8_random"},
            {"SENSE": "HAPLOTYPE", "LOCUS": "JBI002.1"},
        ]
    )
    assert annotate_reference_chromosome_blocks(metadata).tolist() == [
        "chr7",
        "chr7",
        "chr8",
        "chr8",
        pd.NA,
        pd.NA,
    ]


def test_prepare_paths_writes_both_haplotypes(tmp_path: Path) -> None:
    metadata = tmp_path / "paths.tsv"
    rows = [
        {
            "#NAME": "GRCh38#0#chr7",
            "SENSE": "REFERENCE",
            "SAMPLE": "GRCh38",
            "HAPLOTYPE": "0",
            "LOCUS": "chr7",
        },
        {
            "#NAME": "D#1#CM000107.1#0",
            "SENSE": "HAPLOTYPE",
            "SAMPLE": "D",
            "HAPLOTYPE": "1",
            "LOCUS": "CM000107.1",
        },
        {
            "#NAME": "GRCh38#0#chr8",
            "SENSE": "REFERENCE",
            "SAMPLE": "GRCh38",
            "HAPLOTYPE": "0",
            "LOCUS": "chr8",
        },
        {
            "#NAME": "D#1#JBIA000001.1#0",
            "SENSE": "HAPLOTYPE",
            "SAMPLE": "D",
            "HAPLOTYPE": "1",
            "LOCUS": "JBIA000001.1",
        },
        {
            "#NAME": "D#1#JBIA000002.1#100",
            "SENSE": "HAPLOTYPE",
            "SAMPLE": "D",
            "HAPLOTYPE": "1",
            "LOCUS": "JBIA000002.1",
        },
        {
            "#NAME": "D#2#CM000207.1#0",
            "SENSE": "HAPLOTYPE",
            "SAMPLE": "D",
            "HAPLOTYPE": "2",
            "LOCUS": "CM000207.1",
        },
        {
            "#NAME": "GRCh38#0#chr9",
            "SENSE": "REFERENCE",
            "SAMPLE": "GRCh38",
            "HAPLOTYPE": "0",
            "LOCUS": "chr9",
        },
        {
            "#NAME": "D#2#CM000208.1#0",
            "SENSE": "HAPLOTYPE",
            "SAMPLE": "D",
            "HAPLOTYPE": "2",
            "LOCUS": "CM000208.1",
        },
    ]
    pd.DataFrame(rows).to_csv(metadata, sep="\t", index=False)
    cohort = tmp_path / "cohort.tsv"
    pd.DataFrame([{"donor_id": "D"}]).to_csv(cohort, sep="\t", index=False)
    summary = prepare_paths(
        metadata_path=metadata,
        cohort_path=cohort,
        out_dir=tmp_path / "out",
        chromosome="chr8",
    )
    selected = (tmp_path / "out" / "D.chr8.paths.txt").read_text().splitlines()
    assert summary["n_haplotypes"] == 2
    assert selected == [
        "D#1#JBIA000001.1#0",
        "D#1#JBIA000002.1#100",
        "D#2#CM000207.1#0",
    ]
    manifest = pd.read_csv(
        tmp_path / "out" / "chr8.path_manifest.tsv", sep="\t"
    )
    assert manifest["resolution_method"].eq("reference_chromosome_block").all()
    assert manifest["n_loci"].tolist() == [2, 1]
