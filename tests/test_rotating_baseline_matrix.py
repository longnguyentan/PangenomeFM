from __future__ import annotations

from scripts.server.run_rotating_baseline_matrix import build_jobs


def test_baseline_matrix_matches_folds_seeds_and_contexts() -> None:
    config = {
        "rotating_chromosome_folds": [
            {"name": "a", "test": ["chr1"], "validation": ["chr2"]},
            {"name": "b", "test": ["chr2"], "validation": ["chr3"]},
        ],
        "training": {"seeds": [42, 7], "contexts": ["strict", "1hop"]},
    }
    jobs = build_jobs(config)
    assert len(jobs) == 8
    assert len({job.name for job in jobs}) == 8
    assert jobs[0].test == ("chr1",)
    assert jobs[-1].context == "1hop"
