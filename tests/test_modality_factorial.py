from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.modality_factorial import (
    build_modality_factorial,
    load_frozen_node_embedding_cache,
    select_feature_sets,
)
from scripts.server.aggregate_ccre_frozen_probes import (
    PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST,
    modality_contrasts,
    paired_context_summary,
    paired_modality_contribution_summary,
    validate_exact_feature_universe,
)
from scripts.server.aggregate_sv_frozen_probes import summarize_strata


def test_builds_complete_c_s_t_factorial_on_identical_rows() -> None:
    components = {
        "coordinate": np.ones((4, 2), dtype=np.float32),
        "sequence_kmer": np.full((4, 3), 2, dtype=np.float32),
        "frozen_pangenomefm": np.full((4, 5), 3, dtype=np.float32),
    }
    result = build_modality_factorial(components)
    assert set(result) == {
        "coordinate",
        "sequence_kmer",
        "frozen_pangenomefm",
        "coordinate_plus_sequence_kmer",
        "coordinate_plus_frozen_pangenomefm",
        "sequence_plus_frozen_pangenomefm",
        "coordinate_plus_sequence_plus_frozen_pangenomefm",
    }
    assert result["coordinate_plus_sequence_plus_frozen_pangenomefm"].shape == (
        4,
        10,
    )
    np.testing.assert_array_equal(
        result["coordinate_plus_sequence_kmer"][:, :2],
        components["coordinate"],
    )


def test_feature_selection_is_ordered_and_strict() -> None:
    features = {
        "a": np.ones((2, 1), dtype=np.float32),
        "b": np.ones((2, 2), dtype=np.float32),
    }
    assert list(select_feature_sets(features, ["b", "a"])) == ["b", "a"]
    with pytest.raises(ValueError, match="Unknown feature sets"):
        select_feature_sets(features, ["missing"])


def test_external_cache_requires_label_blind_complete_audit(tmp_path: Path) -> None:
    cache = tmp_path / "sequence.npz"
    np.savez(
        cache,
        segid=np.array([2, 7]),
        embeddings=np.ones((2, 3), dtype=np.float32),
    )
    audit = {"status": "complete", "downstream_label_access": "none"}
    Path(f"{cache}.audit.json").write_text(json.dumps(audit))
    segids, embeddings, loaded_audit = load_frozen_node_embedding_cache(cache)
    np.testing.assert_array_equal(segids, [2, 7])
    assert embeddings.shape == (2, 3)
    assert loaded_audit == audit


def test_paired_modality_summary_uses_loss_improvement_direction() -> None:
    rows = []
    feature_values = {
        "coordinate": (0.60, 0.50),
        "sequence_kmer": (0.62, 0.48),
        "frozen_pangenomefm": (0.64, 0.46),
        "coordinate_plus_sequence_kmer": (0.66, 0.44),
        "coordinate_plus_frozen_pangenomefm": (0.68, 0.42),
        "sequence_plus_frozen_pangenomefm": (0.70, 0.40),
        "coordinate_plus_sequence_plus_frozen_pangenomefm": (0.75, 0.35),
        "frozen_sequence_fm": (0.71, 0.39),
        "coordinate_plus_frozen_sequence_fm": (0.73, 0.37),
        "frozen_sequence_fm_plus_frozen_pangenomefm": (0.77, 0.33),
        "coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm": (
            0.80,
            0.30,
        ),
    }
    for fold in ("fold_a", "fold_b"):
        for seed in (42, 314159):
            for feature, (auprc, nll) in feature_values.items():
                row = {
                    "fold": fold,
                    "seed": seed,
                    "closure": "strict",
                    "feature_set": feature,
                }
                for metric in (
                    "auroc",
                    "auprc",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                ):
                    row[metric] = auprc
                row.update({"nll": nll, "brier": nll, "ece": nll})
                rows.append(row)
    result = paired_modality_contribution_summary(
        pd.DataFrame(rows), n_bootstrap=50, seed=1
    )
    topology = result.loc[
        result["contrast"].eq("topology_given_coordinate_and_sequence")
    ]
    assert topology.loc[topology["metric"].eq("auprc"), "mean_gain"].item() > 0
    assert topology.loc[topology["metric"].eq("nll"), "mean_gain"].item() > 0
    sequence_controlled = result.loc[
        result["contrast"].eq(PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST)
    ]
    assert sequence_controlled.loc[
        sequence_controlled["metric"].eq("auprc"), "mean_gain"
    ].item() == pytest.approx(0.07)
    assert sequence_controlled.loc[
        sequence_controlled["metric"].eq("nll"), "mean_gain"
    ].item() == pytest.approx(0.07)


def test_external_sequence_contrasts_are_prespecified_for_pair_suffix() -> None:
    contrasts = modality_contrasts(suffix="_pair")
    assert contrasts[PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST] == (
        "coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm_pair",
        "coordinate_plus_frozen_sequence_fm_pair",
    )
    assert "topology_given_frozen_sequence_fm" in contrasts
    assert "frozen_sequence_fm_given_coordinate_and_topology" in contrasts


def test_context_summary_reports_feature_and_modality_interactions() -> None:
    rows = []
    values = {
        "coordinate_plus_frozen_sequence_fm": {"strict": 0.70, "1hop": 0.70},
        "coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm": {
            "strict": 0.74,
            "1hop": 0.77,
        },
    }
    for fold in ("fold_a", "fold_b"):
        for seed in (42, 314159):
            for closure in ("strict", "1hop"):
                for feature_set, context_values in values.items():
                    score = context_values[closure]
                    row = {
                        "fold": fold,
                        "seed": seed,
                        "closure": closure,
                        "feature_set": feature_set,
                    }
                    for metric in (
                        "auroc",
                        "auprc",
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                    ):
                        row[metric] = score
                    row.update(
                        {
                            "nll": 1.0 - score,
                            "brier": 1.0 - score,
                            "ece": 1.0 - score,
                        }
                    )
                    rows.append(row)
    result = paired_context_summary(pd.DataFrame(rows), n_bootstrap=50, seed=1)
    interaction = result.loc[
        result["contrast"].eq(
            "one_hop_minus_strict__"
            + PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST
        )
        & result["metric"].eq("auprc")
    ]
    assert interaction["mean_gain"].item() == pytest.approx(0.03)
    assert interaction["n_paired_runs"].item() == 4


def test_exact_feature_universe_rejects_mismatched_test_counts() -> None:
    frame = pd.DataFrame(
        [
            {
                "fold": "fold_a",
                "seed": 42,
                "closure": "strict",
                "feature_set": "a",
                "n_train": 10,
                "n_validation": 4,
                "n_test": 5,
            },
            {
                "fold": "fold_a",
                "seed": 42,
                "closure": "strict",
                "feature_set": "b",
                "n_train": 10,
                "n_validation": 4,
                "n_test": 6,
            },
        ]
    )
    with pytest.raises(ValueError, match="different evaluation counts"):
        validate_exact_feature_universe(frame)


def test_sv_stratum_summary_flags_single_class_and_single_level() -> None:
    rows = []
    for stratum, stratum_value, positive_fraction, auroc in (
        ("sv_class", "INS", 1.0, np.nan),
        ("breakpoint_mapping_distance", "overlap", 0.5, 0.7),
        ("length_bin", "bp[100000,1e+06)", 0.5, 0.7),
        ("length_bin", "bp[100,500)", 0.5, 0.8),
    ):
        for fold in ("fold_a", "fold_b"):
            rows.append(
                {
                    "fold": fold,
                    "seed": 42,
                    "closure": "strict",
                    "feature_set": "feature",
                    "stratum": stratum,
                    "stratum_value": stratum_value,
                    "n": 20 if "100000" in stratum_value else 200,
                    "positive_fraction": positive_fraction,
                    "auroc": auroc,
                    "auprc": auroc,
                    "accuracy": 0.7,
                    "f1": 0.7,
                }
            )
    summary = summarize_strata(pd.DataFrame(rows))
    statuses = dict(
        zip(
            summary["stratum_value"],
            summary["interpretation_status"],
            strict=True,
        )
    )
    assert statuses["INS"] == "non_informative_single_level"
    assert statuses["overlap"] == "non_informative_single_level"
    assert statuses["bp[100000,1e+06)"] == "underpowered_mean_n_lt_100"
    assert statuses["bp[100,500)"] == "descriptive_ranking_supported"
