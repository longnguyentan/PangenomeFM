from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "server"
    / "calibrate_rotating_fold_predictions.py"
)
SPEC = importlib.util.spec_from_file_location("calibrate_rotating", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_threshold_is_deterministic_and_validation_optimized() -> None:
    labels = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.6, 0.9])
    threshold = MODULE.choose_f1_threshold(labels, probability)
    assert np.isclose(threshold, 0.6)


def test_calibration_uses_validation_and_evaluates_heldout(tmp_path: Path) -> None:
    rotating = tmp_path / "rotating_folds"
    path = (
        rotating
        / "hprc_r2"
        / "fold_a"
        / "seed_42"
        / "strict"
        / "run_001"
        / "strict_pooled_predictions.csv.gz"
    )
    path.parent.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "dataset": ["hprc_r2"] * 8,
            "split": ["val_chr_test"] * 4 + ["heldout_chr_test"] * 4,
            "y_true": [0, 0, 1, 1, 0, 0, 1, 1],
            "p_edge": [0.2, 0.3, 0.7, 0.8, 0.1, 0.4, 0.6, 0.9],
        }
    )
    frame.to_csv(path, index=False, compression="gzip")

    rows = MODULE.calibrate_prediction_file(
        path,
        rotating_root=rotating,
        calibration_split="val_chr_test",
        evaluation_split="heldout_chr_test",
        n_bins=5,
    )

    assert len(rows) == 6  # all + dataset, each under three evaluation policies
    assert {row["evaluation"] for row in rows} == {
        "uncalibrated_threshold_0.5",
        "temperature_threshold_0.5",
        "temperature_validation_f1_threshold",
    }
    assert all(row["n_calibration"] == 4 for row in rows)
    assert all(row["n_targets"] == 4 for row in rows)


def test_hierarchical_summary_records_fold_resampling() -> None:
    rows = []
    for fold, value in (("fold_a", 0.7), ("fold_b", 0.9)):
        for seed in (1, 2):
            rows.append(
                {
                    "regime": "hprc_r2",
                    "closure": "strict",
                    "dataset": "all",
                    "evaluation": "temperature_validation_f1_threshold",
                    "fold": fold,
                    "seed": seed,
                    **{metric: value for metric in (
                        "auroc", "auprc", "accuracy", "precision", "recall",
                        "f1", "nll", "brier", "ece"
                    )},
                }
            )
    summary = MODULE.hierarchical_bootstrap_summary(
        pd.DataFrame(rows), n_bootstrap=50, seed=3
    )
    f1 = summary.loc[summary["metric"].eq("f1")].iloc[0]
    assert np.isclose(f1["mean"], 0.8)
    assert f1["n_runs"] == 4
    assert f1["n_folds"] == 2
    assert f1["n_seeds"] == 2
    assert f1["resampling_unit"] == "chromosome_fold_then_seed"
