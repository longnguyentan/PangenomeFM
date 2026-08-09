import argparse
from pathlib import Path

from scripts.run_full_multicohort_all_chromosomes import load_config
from scripts.server.run_gpu_matrix import build_jobs


def test_capacity_scaling_matrix_has_45_strict_fold_jobs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PANGENOMEFM_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("PANGENOMEFM_RESULTS_ROOT", str(tmp_path / "results"))
    config = load_config(Path("configs/server_capacity_scaling_hprc_20260809.json"))
    args = argparse.Namespace(
        seeds=None,
        contexts=None,
        regimes=None,
        folds=None,
        pairs=None,
    )
    jobs = build_jobs(config, "folds", args)
    assert len(jobs) == 45
    contexts = {
        job.filters[job.filters.index("--contexts") + 1]
        for job in jobs
    }
    assert contexts == {"strict"}
    variants = config["experiment_matrix"]["fold_regimes"]
    assert [item["training_variant"]["hidden_dim"] for item in variants] == [24, 96, 192]
