#!/usr/bin/env python3
"""Plan and execute the staged PangenomeFM foundation-model roadmap.

The roadmap mixes runnable repository jobs with experiments that require new
cohorts, assays, graph builds, or official external-model installations.  This
runner never disguises those prerequisites: ``--plan`` records every task as
ready, completed, resource-locked, or blocked and ``--execute`` runs only
explicit argv commands whose requirements pass.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/foundation_model_v2_roadmap_20260823.json"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def expand(value: str, variables: dict[str, str]) -> str:
    output = value
    for key, replacement in variables.items():
        output = output.replace("${" + key + "}", replacement)
    return os.path.expandvars(output)


def task_status(
    task: dict[str, Any],
    *,
    variables: dict[str, str],
    result_root: Path,
    allow_gpu: bool,
    allow_expensive: bool,
    allow_external: bool,
    ignore_marker: bool = False,
) -> tuple[str, list[str]]:
    marker = result_root / "task_status" / f"{task['id']}.json"
    if marker.exists() and not ignore_marker:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if payload.get("status") == "complete":
            return "completed", []
    reasons: list[str] = []
    if task.get("command") is None:
        reasons.append(task.get("blocker", "no executable command is defined"))
        return "designed_data_gated", reasons
    for name in task.get("requires_env", []):
        if not variables.get(name):
            reasons.append(f"missing environment variable {name}")
    for raw_path in task.get("requires_paths", []):
        path_text = expand(raw_path, variables)
        if "${" in path_text:
            reasons.append(f"unresolved path {raw_path}")
        elif not Path(path_text).exists():
            reasons.append(f"missing path {path_text}")
    resource = task.get("resource", "cpu")
    if resource == "gpu" and not allow_gpu:
        reasons.append("GPU task requires --allow-gpu")
    if task.get("cost") == "expensive" and not allow_expensive:
        reasons.append("expensive task requires --allow-expensive")
    if task.get("external", False) and not allow_external:
        reasons.append("external-model/data task requires --allow-external")
    return ("ready" if not reasons else "blocked"), reasons


def closure(selected: set[str], tasks: dict[str, dict[str, Any]]) -> set[str]:
    expanded = set(selected)
    changed = True
    while changed:
        changed = False
        for task_id in list(expanded):
            for dependency in tasks[task_id].get("depends_on", []):
                if dependency not in expanded:
                    expanded.add(dependency)
                    changed = True
    return expanded


def ordered_tasks(config: dict[str, Any], selected: set[str]) -> list[dict[str, Any]]:
    tasks = {task["id"]: task for task in config["tasks"]}
    selected = closure(selected, tasks)
    result: list[dict[str, Any]] = []
    remaining = set(selected)
    while remaining:
        ready = sorted(
            task_id
            for task_id in remaining
            if set(tasks[task_id].get("depends_on", [])).isdisjoint(remaining)
        )
        if not ready:
            raise ValueError(f"Roadmap contains a dependency cycle: {sorted(remaining)}")
        for task_id in ready:
            result.append(tasks[task_id])
            remaining.remove(task_id)
    return result


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_command(
    task: dict[str, Any], variables: dict[str, str], result_root: Path
) -> int:
    command = [expand(str(value), variables) for value in task["command"]]
    if any("${" in value for value in command):
        raise ValueError(f"Task {task['id']} has unresolved command variables")
    log_path = result_root / "logs" / f"{task['id']}.log"
    marker = result_root / "task_status" / f"{task['id']}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    print(f"[{task['id']}] {' '.join(command)}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{started}] COMMAND {json.dumps(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    payload = {
        "task_id": task["id"],
        "stage": task["stage"],
        "status": "complete" if return_code == 0 else "failed",
        "started_at": started,
        "finished_at": utc_now(),
        "return_code": return_code,
        "command": command,
        "log": str(log_path.resolve()),
    }
    atomic_json(marker, payload)
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--allow-expensive", action="store_true")
    parser.add_argument("--allow-external", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args()
    if args.plan == args.execute:
        parser.error("choose exactly one of --plan or --execute")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    tasks_by_id = {task["id"]: task for task in config["tasks"]}
    if len(tasks_by_id) != len(config["tasks"]):
        raise ValueError("Roadmap task IDs must be unique")
    unknown = set(args.task) - set(tasks_by_id)
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")
    known_stages = {task["stage"] for task in config["tasks"]}
    unknown_stages = set(args.stage) - known_stages
    if unknown_stages:
        raise ValueError(f"Unknown stages: {sorted(unknown_stages)}")

    variables = dict(os.environ)
    variables["REPO_ROOT"] = str(ROOT)
    default_result = variables.get(
        "PANGENOMEFM_RESULTS_ROOT", str(ROOT / "server_workspace" / "results")
    )
    result_root = args.result_root or Path(default_result) / config["run_name"]
    variables["ROADMAP_RESULT_ROOT"] = str(result_root)
    if args.task or args.stage:
        selected = set(args.task)
        selected.update(
            task["id"] for task in config["tasks"] if task["stage"] in args.stage
        )
    else:
        selected = set(tasks_by_id)
    ordered = ordered_tasks(config, selected)

    plan_rows: list[dict[str, Any]] = []
    for task in ordered:
        current, reasons = task_status(
            task,
            variables=variables,
            result_root=result_root,
            allow_gpu=args.allow_gpu,
            allow_expensive=args.allow_expensive,
            allow_external=args.allow_external,
            ignore_marker=args.rerun,
        )
        marker = result_root / "task_status" / f"{task['id']}.json"
        if args.rerun and marker.exists():
            reasons.append("existing completion marker ignored by --rerun")
        plan_rows.append(
            {
                "id": task["id"],
                "stage": task["stage"],
                "status": current,
                "resource": task.get("resource", "cpu"),
                "cost": task.get("cost", "small"),
                "description": task["description"],
                "reasons": reasons,
            }
        )

    result_root.mkdir(parents=True, exist_ok=True)
    plan_payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "result_root": str(result_root.resolve()),
        "tasks": plan_rows,
    }
    atomic_json(result_root / "execution_plan.json", plan_payload)
    for row in plan_rows:
        reason = "; ".join(row["reasons"])
        print(f"{row['status']:24s} {row['id']:36s} {reason}")
    if args.plan:
        return int(args.fail_on_blocked and any(row["status"] not in {"ready", "completed"} for row in plan_rows))

    completed_now: set[str] = set()
    failed = False
    blocked = False
    rows_by_id = {row["id"]: row for row in plan_rows}
    for task in ordered:
        row = rows_by_id[task["id"]]
        if row["status"] == "completed" and not args.rerun:
            continue
        unmet = [
            dependency
            for dependency in task.get("depends_on", [])
            if rows_by_id[dependency]["status"] != "completed" and dependency not in completed_now
        ]
        if row["status"] != "ready" or unmet:
            blocked = True
            print(f"[{task['id']}] BLOCKED: {row['reasons'] + (['unmet dependencies: ' + ','.join(unmet)] if unmet else [])}")
            continue
        return_code = run_command(task, variables, result_root)
        if return_code:
            failed = True
            print(f"[{task['id']}] FAILED with exit code {return_code}")
        else:
            completed_now.add(task["id"])
    final = {
        **plan_payload,
        "finished_at": utc_now(),
        "status": "failed" if failed else ("partial_blocked" if blocked else "complete"),
        "completed_this_run": sorted(completed_now),
    }
    atomic_json(result_root / "final_status.json", final)
    if failed:
        return 1
    if blocked and args.fail_on_blocked:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
