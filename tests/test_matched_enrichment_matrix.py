from pathlib import Path

import pytest

from scripts.server.run_matched_enrichment_matrix import (
    EnrichmentJob,
    build_command,
    parse_signal,
)


def test_signal_spec_and_command_are_explicit(tmp_path: Path) -> None:
    job = parse_signal(f"parkinson={tmp_path / 'signals.csv.gz'}")
    command = build_command(
        job,
        regions=tmp_path / "regions.csv",
        out_root=tmp_path / "out",
        controls_per_case=5,
        relative_caliper=0.25,
        n_permutations=10_000,
        n_bootstrap=2_000,
        seed=20260806,
    )
    assert job == EnrichmentJob("parkinson", tmp_path / "signals.csv.gz")
    assert command[command.index("--priority-column") + 1] == "is_prioritized"
    assert command[command.index("--controls-per-case") + 1] == "5"
    assert "distance_to_gene" in command


def test_signal_spec_rejects_unsafe_name() -> None:
    with pytest.raises(Exception, match="only letters"):
        parse_signal("../bad=/tmp/signals.csv.gz")

