from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.server.aggregate_ccre_frozen_probes import (
    PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST,
)
from scripts.server.audit_manuscript_decision_experiments import matrix_gate


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def build_complete_matrix(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "matrix"
    aggregate = root / "paper_source_data_v2"
    write_json(
        root / "matrix_summary.json",
        {"jobs_requested": 30, "jobs_recorded": 30, "failures": 0},
    )
    folds = [f"fold_{letter}" for letter in "abcde"]
    seeds = [42, 314159, 20260806]
    closures = ["strict", "1hop"]
    for fold in folds:
        for seed in seeds:
            for closure in closures:
                write_json(
                    root / fold / f"seed_{seed}" / closure / "audit.json",
                    {"status": "complete"},
                )
    write_json(
        aggregate / "audit.json",
        {
            "status": "complete",
            "files": 30,
            "n_bootstrap": 10_000,
            "exact_feature_universe": {"status": "pass"},
        },
    )
    contributions = []
    for closure in closures:
        contributions.append(
            {
                "closure": closure,
                "contrast": PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST,
                "metric": "auprc",
                "mean_gain": 0.01,
                "ci95_low": 0.005,
                "ci95_high": 0.015,
                "n_paired_runs": 15,
                "n_folds": 5,
                "n_seeds": 3,
            }
        )
    pd.DataFrame(contributions).to_csv(
        aggregate / "ccre_modality_contributions.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "contrast": (
                    "one_hop_minus_strict__"
                    + PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST
                ),
                "metric": "auprc",
                "mean_gain": 0.002,
                "ci95_low": -0.001,
                "ci95_high": 0.004,
                "n_paired_runs": 15,
                "n_folds": 5,
                "n_seeds": 3,
            }
        ]
    ).to_csv(aggregate / "ccre_context_contributions.csv", index=False)
    metric_rows = []
    for fold in folds:
        for seed in seeds:
            for closure in closures:
                for feature_set in ("smaller", "larger"):
                    metric_rows.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "closure": closure,
                            "feature_set": feature_set,
                            "n_train": 100,
                            "n_validation": 20,
                            "n_test": 30,
                        }
                    )
    pd.DataFrame(metric_rows).to_csv(
        aggregate / "ccre_fold_metrics.csv", index=False
    )
    return root, aggregate


def test_matrix_gate_requires_named_primary_contrast(tmp_path: Path) -> None:
    root, aggregate = build_complete_matrix(tmp_path)
    result = matrix_gate(
        "ccre_modality_factorial",
        root,
        aggregate,
        expected_bootstrap=10_000,
    )
    assert result["status"] == "pass"
    assert result["primary_auprc_rows"] == 2
    assert result["primary_context_interaction_rows"] == 1
    assert result["exact_feature_universe"] is True

    contributions = pd.read_csv(
        aggregate / "ccre_modality_contributions.csv"
    )
    contributions["contrast"] = "wrong_contrast"
    contributions.to_csv(
        aggregate / "ccre_modality_contributions.csv", index=False
    )
    failed = matrix_gate(
        "ccre_modality_factorial",
        root,
        aggregate,
        expected_bootstrap=10_000,
    )
    assert failed["status"] == "fail"
    assert failed["primary_auprc_rows"] == 0


def test_matrix_gate_rejects_mismatched_feature_counts(tmp_path: Path) -> None:
    root, aggregate = build_complete_matrix(tmp_path)
    metrics = pd.read_csv(aggregate / "ccre_fold_metrics.csv")
    metrics.loc[0, "n_test"] += 1
    metrics.to_csv(aggregate / "ccre_fold_metrics.csv", index=False)
    result = matrix_gate(
        "ccre_modality_factorial",
        root,
        aggregate,
        expected_bootstrap=10_000,
    )
    assert result["status"] == "fail"
    assert result["exact_feature_universe"] is False
