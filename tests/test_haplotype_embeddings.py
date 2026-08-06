from pathlib import Path

from tasks.haplotype.embeddings import (
    _load_chunk,
    _non_special_mask,
    _target_p_walk,
    _uncovered_length,
    _walk_base_matches_active_target,
)
import torch


def test_target_p_walk_uses_stable_track_and_path_offset() -> None:
    fields = [
        "P",
        "HG00438#1#CM089174.1#9507",
        "10+,11-,12+",
        "*",
    ]
    assert _target_p_walk(fields, "HG00438") == {
        "track": "HG00438#1#CM089174.1",
        "start": 9507,
        "steps": [("10", "+"), ("11", "-"), ("12", "+")],
    }


def test_target_p_walk_rejects_context_fragment() -> None:
    fields = [
        "P",
        "HG00438#1#CM089174.1#9507[100-200]",
        "10+,11+",
        "*",
    ]
    assert _target_p_walk(fields, "HG00438") is None


def test_target_p_walk_accepts_manifest_selected_chunk_slice() -> None:
    fields = [
        "P",
        "HG00438#1#CM089174.1#9507[2000000-4000000]",
        "10+,11+",
        "*",
    ]
    assert _target_p_walk(
        fields,
        "HG00438",
        target_path_names={"HG00438#1#CM089174.1#9507"},
    ) == {
        "track": "HG00438#1#CM089174.1",
        "start": 2_009_507,
        "steps": [("10", "+"), ("11", "+")],
    }


def test_load_chunk_reads_vg_chunk_p_path(tmp_path: Path) -> None:
    chunk_path = tmp_path / "chunk.gfa"
    chunk_path.write_text(
        "\n".join(
            [
                "S\t10\tAC",
                "S\t11\tGT",
                "S\t12\tAA",
                "L\t10\t+\t11\t-\t0M",
                "L\t11\t-\t12\t+\t0M",
                "P\tHG00438#1#CM089174.1#9507\t10+,11-,12+\t*",
                "P\tHG00438#2#CM089193.1#15424[0-2]\t10+\t*",
                "",
            ]
        ),
        encoding="utf-8",
    )
    chunk = _load_chunk(chunk_path, "HG00438")
    assert chunk["target_walks"] == [
        {
            "track": "HG00438#1#CM089174.1",
            "start": 9507,
            "steps": [("10", "+"), ("11", "-"), ("12", "+")],
        }
    ]
    assert chunk["lengths"] == {"10": 2, "11": 2, "12": 2}


def test_uncovered_length_deduplicates_chunk_boundary_bases() -> None:
    covered: dict[str, list[tuple[int, int]]] = {}
    assert _uncovered_length(covered, "path:0", 0, 100) == 100
    assert _uncovered_length(covered, "path:0", 90, 150) == 50
    assert _uncovered_length(covered, "path:0", 20, 30) == 0
    assert covered["path:0"] == [(0, 150)]


def test_load_chunk_uses_filename_target_not_same_donor_context(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunk_0_D#1#CM1#0_0_4.gfa"
    chunk_path.write_text(
        "\n".join(
            [
                "S\t1\tAA",
                "S\t2\tCC",
                "P\tD#1#CM1#0[0-2]\t1+\t*",
                "P\tD#2#CM2#0[0-2]\t2+\t*",
                "",
            ]
        ),
        encoding="utf-8",
    )
    chunk = _load_chunk(
        chunk_path,
        "D",
        target_path_names={"D#1#CM1#0", "D#2#CM2#0"},
    )
    assert [walk["track"] for walk in chunk["target_walks"]] == ["D#1#CM1"]
    assert chunk["lengths"] == {"1": 2}


def test_load_chunk_restores_walk_phase_block_from_filename(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunk_0_D#1#CONTIG#0_0_4.gfa"
    chunk_path.write_text(
        "\n".join(
            [
                "S\t1\tACGT",
                "S\t2\tTTTT",
                "W\tD\t1\tCONTIG\t0\t4\t>1",
                "W\tD\t2\tOTHER\t0\t4\t>2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    chunk = _load_chunk(
        chunk_path,
        "D",
        target_path_names={"D#1#CONTIG#0"},
    )
    assert chunk["target_walks"] == [
        {
            "track": "D#1#CONTIG",
            "start": 0,
            "steps": [("1", "+")],
        }
    ]
    assert chunk["lengths"] == {"1": 4}


def test_walk_target_restoration_rejects_nonzero_phase_context() -> None:
    assert _walk_base_matches_active_target(
        "D#1#CONTIG", {"D#1#CONTIG#0"}
    )
    assert not _walk_base_matches_active_target(
        "D#1#CONTIG", {"D#1#CONTIG#100"}
    )


def test_non_special_mask_uses_actual_input_shape() -> None:
    mask = _non_special_mask(
        torch.tensor([[3, 10, 11, 1]]),
        torch.tensor([[1, 1, 1, 0]]),
        [0, 1, 2, 3],
    )
    assert mask.tolist() == [[False, True, True, False]]
