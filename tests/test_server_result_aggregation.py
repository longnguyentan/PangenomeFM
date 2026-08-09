from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "server"
    / "aggregate_server_results.py"
)
SPEC = importlib.util.spec_from_file_location("aggregate_server_results", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_execution_steps_are_found_by_payload_not_basename(tmp_path: Path) -> None:
    state_dir = tmp_path / "gpu_queue_states" / "folds"
    state_dir.mkdir(parents=True)
    state_path = state_dir / "folds_hprc_r2_fold_a_seed42_strict.json"
    state_path.write_text(
        json.dumps(
            {
                "steps": {
                    "train_fold": {
                        "status": "completed",
                        "wall_seconds": 12.5,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )

    rows, failures = MODULE.collect_execution_steps(tmp_path)

    assert failures == []
    assert len(rows) == 1
    assert rows[0]["step"] == "train_fold"
    assert rows[0]["status"] == "completed"
    assert rows[0]["wall_seconds"] == 12.5
