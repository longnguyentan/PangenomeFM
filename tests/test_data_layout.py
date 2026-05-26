from __future__ import annotations

import pytest

from data.layout import find_dataset


def test_find_dataset_accepts_cleaned_hprc_layout(tmp_path):
    data_dir = tmp_path / "hprc"
    data_dir.mkdir()
    (data_dir / "full_segments.csv").write_text("id,name,seq,LN,SN,SO,SR\n", encoding="utf-8")
    (data_dir / "full_links.csv").write_text(
        "from_seg,from_orient,to_seg,to_orient,overlap\n",
        encoding="utf-8",
    )

    dataset = find_dataset(data_dir)

    assert dataset.name == "hprc"
    assert dataset.segments.name == "full_segments.csv"
    assert dataset.links.name == "full_links.csv"
    assert dataset.benchmark_dir == data_dir / "benchmark"
    assert dataset.ccre_dir == data_dir / "ccre"


def test_find_dataset_requires_segments_and_links(tmp_path):
    data_dir = tmp_path / "hprc"
    data_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="full_segments"):
        find_dataset(data_dir)

