from __future__ import annotations

import pandas as pd

from data.gfa_parser import parse_gfa_to_tables


def test_parse_gfa_to_project_tables(tmp_path):
    gfa = tmp_path / "tiny.gfa"
    gfa.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                "S\tseg1\tACGT\tLN:i:4\tSN:Z:GRCh38#0#chr1\tSO:i:10\tSR:i:0",
                "S\tseg2\tTT\tLN:i:2\tSN:Z:GRCh38#0#chr1\tSO:i:14\tSR:i:0",
                "L\tseg1\t+\tseg2\t+\t0M",
                "P\tGRCh38#0#chr1\tseg1+,seg2+\t0M,0M",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    segments_out = tmp_path / "segments.csv"
    links_out = tmp_path / "links.csv"
    summary_out = tmp_path / "summary.json"

    stats = parse_gfa_to_tables(
        gfa_path=gfa,
        segments_out=segments_out,
        links_out=links_out,
        summary_out=summary_out,
    )

    segments = pd.read_csv(segments_out)
    links = pd.read_csv(links_out)
    assert stats["segments"] == 2
    assert stats["links"] == 1
    assert stats["paths"] == 1
    assert list(segments.columns) == ["id", "name", "seq", "LN", "SN", "SO", "SR"]
    assert list(links.columns) == [
        "from_seg",
        "from_orient",
        "to_seg",
        "to_orient",
        "overlap",
        "SR",
        "L1",
        "L2",
    ]
    assert segments.loc[0, "SN"] == "GRCh38#0#chr1"
    assert links.loc[0, "from_seg"] == "seg1"


def test_parse_gfa_target_sns_with_neighbors(tmp_path):
    gfa = tmp_path / "tiny.gfa"
    gfa.write_text(
        "\n".join(
            [
                "S\tref1\tACGT\tLN:i:4\tSN:Z:id=CHM13|chr22\tSO:i:10\tSR:i:0",
                "S\talt1\tTT\tLN:i:2\tSN:Z:HG001#1#chr22\tSO:i:11\tSR:i:1",
                "S\tother\tGG\tLN:i:2\tSN:Z:id=CHM13|chr1\tSO:i:10\tSR:i:0",
                "L\tref1\t+\talt1\t+\t0M",
                "L\tother\t+\tref1\t+\t0M",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    segments_out = tmp_path / "segments.csv"
    links_out = tmp_path / "links.csv"

    stats = parse_gfa_to_tables(
        gfa_path=gfa,
        segments_out=segments_out,
        links_out=links_out,
        target_sns={"id=CHM13|chr22"},
        include_link_neighbors=True,
    )

    segments = pd.read_csv(segments_out)
    links = pd.read_csv(links_out)
    assert stats["target_segments"] == 1
    assert set(segments["name"]) == {"ref1", "alt1", "other"}
    assert len(links) == 2
