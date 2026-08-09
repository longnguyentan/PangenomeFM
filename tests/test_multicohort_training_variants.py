from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_full_multicohort_all_chromosomes import Runner, load_config


def _runner(tmp_path: Path) -> Runner:
    config = {
        "analysis_id": "test",
        "datasets": {
            "hprc_r2": {
                "pretrain_benchmark": str(tmp_path / "benchmark"),
                "segments": str(tmp_path / "segments.csv.gz"),
            }
        },
        "training": {
            "device_default": "cpu",
            "hidden_dim": 48,
            "heads": 4,
            "layers": 2,
            "epochs": 100,
            "patience": 20,
            "split_seed": 7,
            "recovery_every_epochs": 1,
            "candidate_mask_batch_size": 512,
        },
        "outputs": {
            "root": str(tmp_path / "results"),
            "logs": str(tmp_path / "results" / "logs"),
            "state": str(tmp_path / "results" / "state.json"),
        },
    }
    return Runner(config, execute=False, phases={"folds"})


@pytest.mark.parametrize(
    ("variant", "required", "forbidden"),
    [
        ({"stream_mode": "coordinate"}, ["--stream_mode", "coordinate"], []),
        ({"stream_mode": "graph"}, ["--stream_mode", "graph"], []),
        ({"no_fusion_gate": True}, ["--no_fusion_gate"], []),
        (
            {
                "no_rope": True,
                "multiscale_rope": False,
                "orientation_rope": False,
            },
            ["--no_rope"],
            ["--multiscale_rope", "--orientation_rope"],
        ),
        ({"adaptive_window": False}, [], ["--adaptive_window"]),
    ],
)
def test_training_variant_flags(
    tmp_path: Path,
    variant: dict,
    required: list[str],
    forbidden: list[str],
) -> None:
    command = _runner(tmp_path).training_command(
        primary="hprc_r2",
        output_root=str(tmp_path / "out"),
        closure="strict",
        seed=42,
        training_variant=variant,
    )
    joined = " ".join(command)
    assert all(token in joined for token in required)
    assert all(token not in command for token in forbidden)


def test_training_variant_rejects_unknown_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown training_variant"):
        _runner(tmp_path).training_command(
            primary="hprc_r2",
            output_root=str(tmp_path / "out"),
            closure="strict",
            seed=42,
            training_variant={"typo": True},
        )


def test_training_variant_supports_capacity_scaling(tmp_path: Path) -> None:
    command = _runner(tmp_path).training_command(
        primary="hprc_r2",
        output_root=str(tmp_path / "out"),
        closure="strict",
        seed=42,
        training_variant={
            "hidden_dim": 96,
            "heads": 8,
            "layers": 4,
            "candidate_mask_batch_size": 128,
        },
    )
    values = dict(zip(command[:-1], command[1:]))
    assert values["--hidden_dim"] == "96"
    assert values["--n_heads"] == "8"
    assert values["--n_layers"] == "4"
    assert values["--batch_size"] == "128"


def test_training_variant_rejects_incompatible_head_width(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        _runner(tmp_path).training_command(
            primary="hprc_r2",
            output_root=str(tmp_path / "out"),
            closure="strict",
            seed=42,
            training_variant={"hidden_dim": 50, "heads": 8},
        )


def test_load_config_deep_merges_relative_parent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PANGENOMEFM_RESULTS_ROOT", str(tmp_path / "results"))
    parent = tmp_path / "base.json"
    child = tmp_path / "child.json"
    parent.write_text(
        json.dumps(
            {
                "training": {"seeds": [42], "contexts": ["strict", "1hop"]},
                "outputs": {"root": "${PANGENOMEFM_RESULTS_ROOT}/base"},
            }
        )
    )
    child.write_text(
        json.dumps(
            {
                "extends": "base.json",
                "training": {"seeds": [314159]},
                "outputs": {"root": "${PANGENOMEFM_RESULTS_ROOT}/child"},
            }
        )
    )

    config = load_config(child)
    assert config["training"] == {"seeds": [314159], "contexts": ["strict", "1hop"]}
    assert config["outputs"]["root"] == str(tmp_path / "results" / "child")
