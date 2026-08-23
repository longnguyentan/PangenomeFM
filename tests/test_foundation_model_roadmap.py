from __future__ import annotations

from pathlib import Path

from scripts.server.run_foundation_model_roadmap import ordered_tasks, task_status


def test_roadmap_dependency_order_includes_transitive_dependencies() -> None:
    config = {
        "tasks": [
            {"id": "a", "depends_on": []},
            {"id": "b", "depends_on": ["a"]},
            {"id": "c", "depends_on": ["b"]},
        ]
    }
    assert [task["id"] for task in ordered_tasks(config, {"c"})] == ["a", "b", "c"]


def test_task_preflight_reports_missing_env_and_resource_gates(tmp_path: Path) -> None:
    task = {
        "id": "gpu_task",
        "command": ["python", "work.py"],
        "requires_env": ["NEEDED"],
        "requires_paths": ["${NEEDED}/input.tsv"],
        "resource": "gpu",
        "cost": "expensive",
        "external": True,
    }
    status, reasons = task_status(
        task,
        variables={},
        result_root=tmp_path,
        allow_gpu=False,
        allow_expensive=False,
        allow_external=False,
    )
    assert status == "blocked"
    assert any("NEEDED" in reason for reason in reasons)
    assert any("--allow-gpu" in reason for reason in reasons)
    assert any("--allow-expensive" in reason for reason in reasons)
    assert any("--allow-external" in reason for reason in reasons)


def test_designed_only_task_preserves_scientific_blocker(tmp_path: Path) -> None:
    task = {
        "id": "external",
        "command": None,
        "blocker": "same-sample truth data unavailable",
    }
    status, reasons = task_status(
        task,
        variables={},
        result_root=tmp_path,
        allow_gpu=True,
        allow_expensive=True,
        allow_external=True,
    )
    assert status == "designed_data_gated"
    assert reasons == ["same-sample truth data unavailable"]

