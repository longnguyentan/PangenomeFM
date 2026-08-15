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
    paired_modality_contribution_summary,
)


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
