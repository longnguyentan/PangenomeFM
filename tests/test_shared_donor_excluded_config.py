from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_full_multicohort_all_chromosomes import load_config


def test_shared_donor_excluded_config_is_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PANGENOMEFM_DATA_ROOT", "/tmp/pangenomefm-data")
    monkeypatch.setenv("PANGENOMEFM_RESULTS_ROOT", "/tmp/pangenomefm-results")
    path = Path("configs/server_shared_donor_excluded_transfer_20260809.json")
    config = load_config(path)
    dataset = config["datasets"]["hprc_r2_no_hgsvc_overlap"]
    assert dataset["graph_donors"] == 227
    assert dataset["graph_haplotypes"] == 454
    assert set(dataset["excluded_donors"]) == {
        "HG00733",
        "HG02818",
        "NA19036",
        "NA19240",
    }
    assert [row["name"] for row in config["experiment_matrix"]["final_regimes"]] == [
        "hprc_r2_no_hgsvc_overlap"
    ]
    assert [row["name"] for row in config["experiment_matrix"]["cohort_transfer_pairs"]] == [
        "hprc_r2_no_hgsvc_overlap_to_hgsvc3"
    ]
    assert config["experiment_matrix"]["release_transfer_pairs"] == []
