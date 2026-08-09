from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "server"
    / "build_submission_readiness_tables.py"
)
SPEC = importlib.util.spec_from_file_location("readiness_tables", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_exclusion_table_records_exact_loader_reason(tmp_path: Path) -> None:
    path = tmp_path / "context_audit" / "hprc_r2" / "context_statistics.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "slice_id": "keep",
                "target_sn": "GRCh38#0#chr1",
                "chromosome": "chr1",
                "split": "test",
                "closure_legacy_label": "strict",
                "context_regime": "core-node-induced-subgraph",
                "start": 0,
                "end": 50_000,
                "visible_nodes": 10,
                "visible_link_rows": 10,
                "candidate_targets": 20,
                "default_training_candidates": 14,
                "default_validation_candidates": 2,
                "default_test_candidates": 4,
                "default_loader_eligible": True,
            },
            {
                "slice_id": "drop",
                "target_sn": "GRCh38#0#chr1",
                "chromosome": "chr1",
                "split": "test",
                "closure_legacy_label": "strict",
                "context_regime": "core-node-induced-subgraph",
                "start": 50_000,
                "end": 100_000,
                "visible_nodes": 4,
                "visible_link_rows": 3,
                "candidate_targets": 6,
                "default_training_candidates": 5,
                "default_validation_candidates": 0,
                "default_test_candidates": 1,
                "default_loader_eligible": False,
            },
        ]
    ).to_csv(path, index=False)

    exclusions = MODULE.build_exclusion_table(tmp_path)

    assert exclusions["slice_id"].tolist() == ["drop"]
    assert exclusions["dataset"].tolist() == ["hprc_r2"]
    assert exclusions["exclusion_reason"].tolist() == [
        "fewer_than_10_default_training_candidates"
    ]


def test_baseline_table_marks_path_ablation_not_applicable() -> None:
    table = pd.DataFrame(MODULE.BASELINES).set_index("baseline")
    path = table.loc["No path encoding"]
    assert not bool(path["implemented"])
    assert path["submission_use"] == "not an ablation of this model"
