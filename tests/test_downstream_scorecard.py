import pytest
from scripts.server.build_downstream_scorecard import paired_row


def test_scorecard_refuses_unpaired_arithmetic_and_preserves_nulls():
    row = paired_row("task", "strict", .7, .69, -.01, -.04, .02, 15, "source")
    assert row["interpretation"] == "inconclusive" and row["delta_t"] == -.01
    with pytest.raises(ValueError, match="arithmetic"):
        paired_row("task", "strict", .7, .69, .01, -.04, .02, 15, "source")
