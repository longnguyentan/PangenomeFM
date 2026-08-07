from __future__ import annotations

import pytest

from data.make_benchmark import resolve_reference_target_aliases


def test_resolves_hash_targets_to_release_pipe_spelling():
    available = {"id=CHM13|chr1", "id=GRCh38|chr22"}
    assert resolve_reference_target_aliases(
        ["CHM13#0#chr1", "GRCh38#0#chr22"], available
    ) == ["id=CHM13|chr1", "id=GRCh38|chr22"]


def test_preserves_exact_and_unrecognized_targets():
    available = {"CHM13#0#chr1"}
    assert resolve_reference_target_aliases(
        ["CHM13#0#chr1", "sample#1#chr1"], available
    ) == ["CHM13#0#chr1", "sample#1#chr1"]


def test_rejects_ambiguous_reference_aliases():
    available = {"CHM13#0#chr1", "id=CHM13|chr1"}
    with pytest.raises(ValueError, match="Ambiguous reference target"):
        resolve_reference_target_aliases(["CHM13|chr1"], available)
