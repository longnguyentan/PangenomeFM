from __future__ import annotations

import pandas as pd
import pytest

from analysis.complexity import (
    apply_complexity_definition,
    compute_slice_complexity,
    fit_complexity_definition,
)
from scripts.extract_graph_complexity import build_integrity_audit


def test_slice_complexity_is_deterministic_and_includes_isolates() -> None:
    segments = pd.DataFrame(
        {
            "name": ["a", "b", "c", "d", "isolate"],
            "LN": [10, 20, 30, 40, 50],
            "SR": [0, 0, 1, 1, 1],
        }
    )
    links = pd.DataFrame(
        {
            "from_seg": ["a", "b", "b", "a", "c"],
            "from_orient": ["+"] * 5,
            "to_seg": ["b", "c", "d", "c", "d"],
            "to_orient": ["+"] * 5,
        }
    )
    candidates = pd.DataFrame(
        {"u_oid": [0, 2, 0], "v_oid": [2, 4, 2], "label": [1, 0, 1]}
    )

    first = compute_slice_complexity(segments, links, candidates, window_bp=1000)
    second = compute_slice_complexity(segments, links, candidates, window_bp=1000)

    assert first == second
    assert first["node_count"] == 5
    assert first["unique_edge_count"] == 5
    assert first["connected_components"] == 2
    assert first["cycle_rank"] == 2
    assert first["branching_node_count"] == 2
    assert first["alternate_node_fraction"] == pytest.approx(0.6)
    assert first["duplicate_candidate_pairs"] == 1
    assert first["reverse_equivalent_candidate_duplicates"] == 1
    assert first["orientation_equivalent_candidate_label_conflicts"] == 0
    assert first["node_length_iqr_over_median"] == pytest.approx(2 / 3)


def test_complexity_surfaces_reverse_equivalent_label_conflicts() -> None:
    segments = pd.DataFrame(
        {"name": ["a", "b"], "LN": [10, 20], "SR": [0, 1]}
    )
    links = pd.DataFrame(
        {
            "from_seg": ["a"],
            "from_orient": ["+"],
            "to_seg": ["b"],
            "to_orient": ["+"],
        }
    )
    candidates = pd.DataFrame(
        {"u_oid": [10, 21], "v_oid": [20, 11], "label": [1, 0]}
    )
    result = compute_slice_complexity(
        segments, links, candidates, window_bp=1000
    )
    assert result["reverse_equivalent_candidate_duplicates"] == 1
    assert result["orientation_equivalent_candidate_label_conflicts"] == 1


def test_legacy_candidate_policy_never_waives_graph_or_label_failures() -> None:
    features = pd.DataFrame(
        {
            "duplicate_segment_ids": [0, 0],
            "missing_link_endpoint_rows": [0, 0],
            "invalid_orientation_rows": [0, 0],
            "invalid_candidate_labels": [0, 0],
            "duplicate_candidate_pairs": [0, 0],
            "reverse_equivalent_candidate_duplicates": [5, 7],
            "orientation_equivalent_candidate_label_conflicts": [4, 5],
        }
    )

    strict = build_integrity_audit(
        features, canonical_conflict_policy="error"
    )
    assert strict["status"] == "FAIL"
    assert strict["integrity_failures"][
        "orientation_equivalent_candidate_label_conflicts"
    ] == 9

    legacy = build_integrity_audit(
        features, canonical_conflict_policy="exclude"
    )
    assert legacy["status"] == "PASS"
    assert legacy["legacy_candidate_affected_slices"] == 2
    assert legacy["legacy_candidate_findings"] == {
        "duplicate_candidate_pairs": 0,
        "reverse_equivalent_candidate_duplicates": 12,
        "orientation_equivalent_candidate_label_conflicts": 9,
    }
    assert not any(legacy["integrity_failures"].values())

    features.loc[0, "missing_link_endpoint_rows"] = 1
    assert build_integrity_audit(
        features, canonical_conflict_policy="exclude"
    )["status"] == "FAIL"

    features.loc[0, "missing_link_endpoint_rows"] = 0
    features.loc[0, "invalid_candidate_labels"] = 1
    assert build_integrity_audit(
        features, canonical_conflict_policy="exclude"
    )["status"] == "FAIL"


def test_frozen_complexity_ignores_model_performance_columns() -> None:
    features = pd.DataFrame(
        {
            "edges_per_kb": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            "branching_fraction": [0.0, 0.1, 0.2, 0.4, 0.6, 0.8],
            "max_degree": [2, 2, 3, 4, 6, 10],
            "cycle_rank_per_kb": [0, 0, 1, 2, 4, 8],
            "alternate_node_fraction": [0.0, 0.1, 0.2, 0.5, 0.7, 0.9],
            "PangenomeFM_AUPRC": [0.99, 0.1, 0.7, 0.3, 0.4, 0.2],
        }
    )
    definition = {
        "name": "test",
        "independence_rule": "no performance",
        "fit_population": {"reference_context": "strict"},
        "normalization": {"winsorize": [-5, 5]},
        "composite": {
            "features": [
                {"name": "edge", "source_feature": "edges_per_kb", "transform": "log1p", "weight": 1},
                {"name": "branch", "source_feature": "branching_fraction", "transform": "identity", "weight": 1},
                {"name": "degree", "source_feature": "max_degree", "transform": "log1p", "weight": 1},
                {"name": "cycle", "source_feature": "cycle_rank_per_kb", "transform": "log1p", "weight": 1},
                {"name": "alternate", "source_feature": "alternate_node_fraction", "transform": "identity", "weight": 1},
            ]
        },
        "categories": {"method": "tertiles", "labels": ["low", "medium", "high"]},
    }

    scored, frozen = fit_complexity_definition(features, definition)
    changed = features.copy()
    changed["PangenomeFM_AUPRC"] = list(reversed(changed["PangenomeFM_AUPRC"]))
    reapplied = apply_complexity_definition(changed, frozen)

    assert scored["complexity_score"].tolist() == pytest.approx(
        reapplied["complexity_score"].tolist()
    )
    assert set(scored["complexity_category"]) == {"low", "medium", "high"}
