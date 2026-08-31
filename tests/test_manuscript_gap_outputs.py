from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from scripts.server.build_manuscript_gap_figures import capacity_points
from scripts.server.summarize_link_prediction_prevalence import select_primary_rows, summarize


SOURCE_DIR = Path("data/manuscript_gap_sources_20260830")


def test_tracked_manuscript_source_bundle_is_complete() -> None:
    required = {
        "figure3_reconstruction.csv",
        "figure3_transfer.csv",
        "figure4_ablation_runs.csv",
        "figure4_baselines.csv",
        "figure4_capacity_runs.csv",
        "figure4_complexity.csv",
        "figure5_absolute.csv",
        "figure5_contributions.csv",
        "figure5_sv_strata.csv",
        "source_manifest.json",
        "SHA256SUMS",
    }
    assert SOURCE_DIR.is_dir()
    assert required <= {path.name for path in SOURCE_DIR.iterdir() if path.is_file()}

    transfer = pd.read_csv(SOURCE_DIR / "figure3_transfer.csv")
    assert not transfer.empty
    assert {"transfer_pair", "closure", "auprc_mean"} <= set(transfer.columns)

    for line in (SOURCE_DIR / "SHA256SUMS").read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        payload = (SOURCE_DIR / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected


def test_prevalence_summary_uses_exact_evaluation_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "experiment_type": "rotating_folds",
                "metric_scope": "split",
                "split": "heldout_chr_test",
                "regime": "hprc_r2",
                "fold": "fold_a",
                "transfer_pair": None,
                "closure": "strict",
                "seed": 42,
                "n_targets": 100,
                "positive_fraction": 0.48,
            },
            {
                "experiment_type": "cohort_transfer",
                "metric_scope": "split",
                "split": "all",
                "regime": "hprc_r2_to_hgsvc3",
                "fold": None,
                "transfer_pair": "hprc_r2_to_hgsvc3",
                "closure": "strict",
                "seed": 42,
                "n_targets": 200,
                "positive_fraction": 0.50,
            },
            {
                "experiment_type": "cohort_transfer",
                "metric_scope": "all_saved_splits",
                "split": "all_saved",
                "regime": "hprc_r2_to_hgsvc3",
                "fold": None,
                "transfer_pair": "hprc_r2_to_hgsvc3",
                "closure": "strict",
                "seed": 42,
                "n_targets": 200,
                "positive_fraction": 0.50,
            },
        ]
    )
    selected = select_primary_rows(frame)
    assert len(selected) == 2
    result = summarize(selected)
    transfer = result.loc[result["analysis_name"].eq("hprc_r2_to_hgsvc3")].iloc[0]
    assert transfer["chance_auprc"] == 0.5
    assert transfer["total_connected"] == 100


def test_capacity_points_inserts_principal_configuration() -> None:
    cells = [("fold_a", 42, 10), ("fold_b", 42, 12)]
    old = pd.DataFrame(
        [
            {
                "metric_scope": "split",
                "split": "heldout_chr_test",
                "closure": "strict",
                "fold": fold,
                "seed": seed,
                "n_targets": targets,
                "regime": regime,
                "auprc": value,
            }
            for fold, seed, targets in cells
            for regime, value in [
                ("hprc_r2_capacity_tiny_h24_l1", 0.92),
                ("hprc_r2_capacity_medium_h96_l4", 0.98),
                ("hprc_r2_capacity_large_h192_l6", 0.97),
            ]
        ]
    )
    principal = pd.DataFrame(
        [
            {
                "metric_scope": "split",
                "split": "heldout_chr_test",
                "closure": "strict",
                "fold": fold,
                "seed": seed,
                "n_targets": targets,
                "regime": "hprc_r2",
                "auprc": value,
            }
            for (fold, seed, targets), value in zip(cells, [0.95, 0.96])
        ]
    )
    result = capacity_points(old, principal)
    assert result["hidden"].tolist() == [24, 48, 96, 192]
    assert result.loc[result["principal"], "mean"].iloc[0] == 0.955
    assert result.loc[result["principal"], "runs"].iloc[0] == 2
