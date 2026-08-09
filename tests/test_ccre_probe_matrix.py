from __future__ import annotations

from scripts.server.run_ccre_frozen_probe_matrix import ProbeJob, build_jobs


def test_ccre_probe_matrix_filters_seeds_and_contexts() -> None:
    config = {
        "rotating_chromosome_folds": [
            {"name": "fold_a", "test": ["chr1"], "validation": ["chr2"]},
            {"name": "fold_b", "test": ["chr2"], "validation": ["chr3"]},
        ],
        "training": {"seeds": [42, 7], "contexts": ["strict", "1hop"]},
    }
    jobs = build_jobs(config, seeds={42}, contexts={"1hop"})
    assert jobs == [
        ProbeJob("fold_a", ("chr1",), ("chr2",), 42, "1hop"),
        ProbeJob("fold_b", ("chr2",), ("chr3",), 42, "1hop"),
    ]
