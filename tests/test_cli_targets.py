from __future__ import annotations

from cli import _expand_chroms


def test_expand_chroms_supports_hash_prefix():
    assert _expand_chroms(["chr22"], "GRCh38#0") == ["GRCh38#0#chr22"]


def test_expand_chroms_supports_pipe_prefix_and_full_sn():
    assert _expand_chroms(["chr22"], "id=CHM13|") == ["id=CHM13|chr22"]
    assert _expand_chroms(["id=CHM13|chr22"], "GRCh38#0") == ["id=CHM13|chr22"]


def test_expand_chroms_supports_presets():
    all_targets = _expand_chroms(["all"], "GRCh38#0")
    assert all_targets[0] == "GRCh38#0#chr1"
    assert all_targets[-1] == "GRCh38#0#chrY"
    assert len(all_targets) == 24

    assert _expand_chroms(["ccre-paper"], "GRCh38#0") == [
        "GRCh38#0#chr16",
        "GRCh38#0#chr8",
        "GRCh38#0#chr19",
        "GRCh38#0#chr22",
    ]
