from __future__ import annotations

from scripts.server.run_ccre_frozen_probe_matrix import build_jobs


def test_sv_matrix_uses_same_complete_rotating_design() -> None:
    config = {
        "rotating_chromosome_folds": [
            {"name": f"fold_{index}", "test": [f"chr{index}"], "validation": ["chr22"]}
            for index in range(1, 6)
        ],
        "training": {"seeds": [42, 314159, 20260806], "contexts": ["strict", "1hop"]},
    }
    jobs = build_jobs(config)
    assert len(jobs) == 30
    assert len({job.name for job in jobs}) == 30
