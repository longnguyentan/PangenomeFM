from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.server.prepare_dense_region_scores import (
    aggregate_region_scores,
    align_heldout_predictions,
    load_tile_inventory,
    score_one_run,
    validate_fold_map,
    main,
)


def synthetic_neural() -> pd.DataFrame:
    validation_y = [0, 0, 1, 1]
    heldout_y = [0, 1] * 6
    return pd.DataFrame(
        {
            "dataset": ["hprc_r2"] * 16,
            "slice": ["validation"] * 4 + ["tile_chr1_0"] * 12,
            "target_sn": ["GRCh38#0#chr2"] * 4 + ["GRCh38#0#chr1"] * 12,
            "split": ["val_chr_test"] * 4 + ["heldout_chr_test"] * 12,
            "u_local": [0, 2, 4, 6] + list(range(0, 24, 2)),
            "v_local": [1, 3, 5, 7] + list(range(1, 25, 2)),
            "y_true": validation_y + heldout_y,
            "p_edge": [0.2, 0.3, 0.7, 0.8] + [0.1, 0.9] * 6,
        }
    )


def synthetic_baseline() -> pd.DataFrame:
    labels = np.asarray([0, 1] * 6, dtype=np.int8)
    return pd.DataFrame(
        {
            "slice": ["tile_chr1_0"] * 12,
            "chromosome": ["chr1"] * 12,
            "fold_evaluation_split": ["heldout"] * 12,
            "candidate_row_index": np.arange(12),
            "u_oid": np.arange(0, 24, 2),
            "v_oid": np.arange(1, 25, 2),
            "y_true": labels,
            "baseline": ["sequence_composition_sgd"] * 12,
            "p_calibrated": np.where(labels == 1, 0.7, 0.3),
        }
    )


def test_fold_map_rejects_duplicate_holdout_assignment() -> None:
    with pytest.raises(ValueError, match="held out by both"):
        validate_fold_map(
            [
                {"name": "a", "test": ["chr1"], "validation": ["chr2"]},
                {"name": "b", "test": ["1"], "validation": ["chr3"]},
            ]
        )


def test_native_inventory_rejects_overlapping_tiles(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "name": ["a", "b"],
            "target_sn": ["GRCh38#0#chr1"] * 2,
            "closure": ["strict"] * 2,
            "start": [0, 9],
            "end": [10, 20],
            "n_segments": [4, 5],
            "n_links": [3, 4],
            "links_path": ["a.links.csv.gz", "b.links.csv.gz"],
        }
    ).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="overlap"):
        load_tile_inventory(
            manifest,
            closure="strict",
            chromosome_to_fold={"chr1": "fold_a"},
            chromosomes={"chr1"},
        )


def test_alignment_refuses_label_mismatch() -> None:
    baseline = synthetic_baseline()
    baseline.loc[0, "y_true"] = 1
    with pytest.raises(ValueError, match="labels differ"):
        align_heldout_predictions(
            synthetic_neural(),
            baseline,
            baseline_name="sequence_composition_sgd",
            chromosomes={"chr1"},
            slice_nodes={"tile_chr1_0": np.arange(24)},
        )


def test_alignment_uses_exact_edge_common_support_and_audits_exclusion() -> None:
    baseline = pd.concat(
        [
            synthetic_baseline(),
            pd.DataFrame(
                {
                    "slice": ["tile_chr1_missing"],
                    "chromosome": ["chr1"],
                    "fold_evaluation_split": ["heldout"],
                    "candidate_row_index": [12],
                    "u_oid": [100],
                    "v_oid": [101],
                    "y_true": [0],
                    "baseline": ["sequence_composition_sgd"],
                    "p_calibrated": [0.3],
                }
            ),
        ],
        ignore_index=True,
    )
    matched, coverage, exclusions = align_heldout_predictions(
        synthetic_neural(),
        baseline,
        baseline_name="sequence_composition_sgd",
        chromosomes={"chr1"},
        slice_nodes={"tile_chr1_0": np.arange(24)},
    )
    assert len(matched) == 12
    assert coverage["neural_slices"] == 1
    assert coverage["baseline_slices"] == 2
    assert coverage["common_slices"] == 1
    assert coverage["baseline_only_slices"] == 1
    assert coverage["baseline_only_candidates"] == 1
    assert coverage["common_fraction"] == pytest.approx(12 / 13)
    assert coverage["common_slice_exact_fraction"] == 1
    assert len(exclusions) == 1
    assert exclusions.iloc[0]["_merge"] == "right_only"

    baseline.loc[len(baseline)] = {
        "slice": "tile_chr1_0",
        "chromosome": "chr1",
        "fold_evaluation_split": "heldout",
        "candidate_row_index": 13,
        "u_oid": 100,
        "v_oid": 101,
        "y_true": 0,
        "baseline": "sequence_composition_sgd",
        "p_calibrated": 0.3,
    }
    with pytest.raises(ValueError, match="within common slices"):
        align_heldout_predictions(
            synthetic_neural(),
            baseline,
            baseline_name="sequence_composition_sgd",
            chromosomes={"chr1"},
            slice_nodes={"tile_chr1_0": np.arange(24)},
        )


def test_scoring_uses_validation_calibration_and_identical_candidates(
    tmp_path: Path,
) -> None:
    neural_path = tmp_path / "neural.csv.gz"
    baseline_path = tmp_path / "baseline.csv.gz"
    synthetic_neural().to_csv(neural_path, index=False, compression="gzip")
    synthetic_baseline().to_csv(baseline_path, index=False, compression="gzip")
    scores, temperature, coverage, exclusions = score_one_run(
        neural_path,
        baseline_path,
        baseline_name="sequence_composition_sgd",
        chromosomes={"chr1"},
        dataset="hprc_r2",
        slice_nodes={"tile_chr1_0": np.arange(24)},
    )
    assert len(scores) == 1
    assert scores.loc[0, "region_id"] == "tile_chr1_0"
    assert scores.loc[0, "n_test_candidates"] == 12
    assert scores.loc[0, "n_positive_test_candidates"] == 6
    assert scores.loc[0, "log_loss_advantage"] > 0
    assert np.isfinite(scores.loc[0, "auprc_advantage"])
    assert np.isfinite(scores.loc[0, "auroc_advantage"])
    assert np.isfinite(temperature) and temperature > 0
    assert coverage["common_fraction"] == 1
    assert exclusions.empty


def test_canonical_conflicts_fail_or_are_explicitly_excluded() -> None:
    neural = synthetic_neural()
    neural.loc[4, ["u_local", "v_local"]] = [10, 20]
    neural.loc[5, ["u_local", "v_local"]] = [21, 11]
    baseline = synthetic_baseline()
    baseline.loc[0, ["u_oid", "v_oid"]] = [10, 20]
    baseline.loc[1, ["u_oid", "v_oid"]] = [21, 11]
    nodes = np.arange(24)

    with pytest.raises(ValueError, match="orientation-equivalent"):
        align_heldout_predictions(
            neural,
            baseline,
            baseline_name="sequence_composition_sgd",
            chromosomes={"chr1"},
            slice_nodes={"tile_chr1_0": nodes},
        )

    matched, coverage, exclusions = align_heldout_predictions(
        neural,
        baseline,
        baseline_name="sequence_composition_sgd",
        chromosomes={"chr1"},
        slice_nodes={"tile_chr1_0": nodes},
        canonical_conflict_policy="exclude",
    )
    assert len(matched) == 10
    assert coverage["canonical_label_conflict_identities"] == 1
    assert coverage["canonical_label_conflict_rows"] == 2
    assert len(exclusions) == 2
    assert set(exclusions["_merge"]) == {"canonical_label_conflict"}


def test_aggregation_requires_seed_coverage_and_two_classes() -> None:
    inventory = pd.DataFrame(
        {
            "region_id": ["good", "missing"],
            "name": ["good", "missing"],
            "chromosome": ["chr1", "chr1"],
            "start": [0, 10],
            "end": [10, 20],
            "region_length": [10, 10],
            "fold": ["fold_a", "fold_a"],
            "target_sn": ["GRCh38#0#chr1"] * 2,
            "n_segments": [10, 10],
            "n_links": [9, 9],
            "graph_complexity": [1.8, 1.8],
        }
    )
    rows = []
    for seed in (1, 2, 3):
        rows.append(
            {
                "region_id": "good",
                "seed": seed,
                "baseline": "sequence_composition_sgd",
                "closure": "strict",
                "fold": "fold_a",
                "n_test_candidates": 12,
                "n_positive_test_candidates": 6,
                "model_nll": 0.2,
                "baseline_nll": 0.4,
                "log_loss_advantage": 0.2,
                "model_auprc": 0.8,
                "baseline_auprc": 0.7,
                "auprc_advantage": 0.1,
                "model_auroc": 0.8,
                "baseline_auroc": 0.7,
                "auroc_advantage": 0.1,
                "mean_model_probability": 0.5,
                "mean_baseline_probability": 0.5,
            }
        )
    result = aggregate_region_scores(
        inventory,
        pd.DataFrame(rows),
        expected_seeds=[1, 2, 3],
        minimum_test_candidates=10,
    ).set_index("region_id")
    assert bool(result.loc["good", "is_eligible"])
    assert result.loc["good", "model_score"] == pytest.approx(0.2)
    assert result.loc["missing", "eligibility_reason"] == "missing_heldout_score"


def test_command_line_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    segments = tmp_path / "full_segments.csv.gz"
    pd.DataFrame({"name": [f"s{index}" for index in range(12)]}).to_csv(
        segments, index=False, compression="gzip"
    )
    links = benchmark / "slice.links.csv.gz"
    pd.DataFrame(
        {
            "from_seg": [f"s{index}" for index in range(12)],
            "from_orient": ["+"] * 12,
            "to_seg": [f"s{index}" for index in range(12)],
            "to_orient": ["-"] * 12,
        }
    ).to_csv(links, index=False, compression="gzip")
    pd.DataFrame(
        {
            "name": ["tile_chr1_0", "validation"],
            "target_sn": ["GRCh38#0#chr1", "GRCh38#0#chr2"],
            "closure": ["strict", "strict"],
            "start": [0, 0],
            "end": [5_000_000, 5_000_000],
            "n_segments": [100, 100],
            "n_links": [120, 120],
            "links_path": [str(links), str(links)],
        }
    ).to_csv(benchmark / "manifest.csv", index=False)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "datasets": {
                    "hprc_r2": {
                        "pretrain_benchmark": str(benchmark),
                        "segments": str(segments),
                    }
                },
                "training": {"seeds": [42]},
                "rotating_chromosome_folds": [
                    {
                        "name": "fold_a",
                        "test": ["chr1"],
                        "validation": ["chr2"],
                    }
                ],
            }
        )
    )
    results = tmp_path / "main"
    neural_dir = (
        results / "rotating_folds/hprc_r2/fold_a/seed_42/strict/run_001"
    )
    neural_dir.mkdir(parents=True)
    synthetic_neural().to_csv(
        neural_dir / "strict_pooled_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    baselines = tmp_path / "baselines"
    baseline_dir = baselines / "hprc_r2/fold_a/seed_42/strict"
    baseline_dir.mkdir(parents=True)
    synthetic_baseline().to_csv(
        baseline_dir / "heldout_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    output = tmp_path / "scores"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_dense_region_scores.py",
            "--config", str(config),
            "--results-root", str(results),
            "--baseline-root", str(baselines),
            "--out-dir", str(output),
            "--chromosomes", "chr1",
            "--seeds", "42",
        ],
    )
    assert main() == 0
    region_scores = pd.read_csv(output / "region_scores.csv")
    assert region_scores["region_id"].tolist() == ["tile_chr1_0"]
    assert bool(region_scores.loc[0, "is_eligible"])
    audit = json.loads((output / "audit.json").read_text())
    assert audit["status"] == "complete"
    assert audit["downstream_signal_access"].startswith("none")
    assert audit["candidate_coverage_exclusions"] == 0
    coverage = pd.read_csv(output / "candidate_coverage_summary.csv")
    assert coverage.loc[0, "common_fraction"] == 1
    assert (output / "SHA256SUMS").is_file()
