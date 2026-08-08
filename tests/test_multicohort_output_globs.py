from __future__ import annotations

from scripts.run_full_multicohort_all_chromosomes import Runner


def test_output_glob_expands_intermediate_version_directory(tmp_path) -> None:
    checkpoint = tmp_path / "final_models" / "strict" / "run_001" / "ckpt_strict__model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert Runner._glob_exists(
        [str(tmp_path / "final_models" / "strict" / "run_*" / "ckpt_strict__*.pt")]
    )


def test_output_glob_requires_every_declared_pattern(tmp_path) -> None:
    checkpoint = tmp_path / "run_001" / "ckpt_1hop__model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert not Runner._glob_exists(
        [
            str(tmp_path / "run_*" / "ckpt_1hop__*.pt"),
            str(tmp_path / "run_*" / "predictions_*.csv.gz"),
        ]
    )
