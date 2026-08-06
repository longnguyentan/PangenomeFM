import gzip
from pathlib import Path

from scripts.verify_hprc_chr_chunks import verify


def test_verify_accepts_overlapping_bounded_path_slices(tmp_path: Path) -> None:
    path_list = tmp_path / "D.chr8.paths.txt"
    path_list.write_text("D#1#CM1#100\n", encoding="utf-8")
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    for index, (left, right) in enumerate([(0, 105), (100, 200)]):
        name = f"chunk_{index}_D#1#CM1#100_{left}_{right}.gfa.gz"
        with gzip.open(chunks / name, "wt") as handle:
            handle.write(f"P\tD#1#CM1#100[{left}-{right}]\t1+\t*\n")
            handle.write("W\tD\t1\tCM1\t1000\t1100\t>2\n")
    result = verify(path_list, chunks, tmp_path / "coverage.json")
    assert result["failures"] == []
    assert result["paths"]["D#1#CM1#100"]["merged_relative_intervals"] == [
        (0, 200)
    ]


def test_verify_restores_phase_block_suffix_for_walk_records(
    tmp_path: Path,
) -> None:
    path_list = tmp_path / "D.chr8.paths.txt"
    path_list.write_text("D#1#CONTIG#0\n", encoding="utf-8")
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    chunk = chunks / "chunk_0_D#1#CONTIG#0_0_500.gfa.gz"
    with gzip.open(chunk, "wt") as handle:
        handle.write("W\tD\t1\tCONTIG\t0\t500\t>1\n")
    result = verify(path_list, chunks, tmp_path / "coverage.json")
    assert result["failures"] == []
    assert result["paths"]["D#1#CONTIG#0"]["merged_relative_intervals"] == [
        (0, 500)
    ]
