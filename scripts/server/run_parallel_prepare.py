#!/usr/bin/env python3
"""Run independent dataset preparation stages with bounded CPU parallelism."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def expand(value):
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, str):
        item = os.path.expandvars(value)
        if "${" in item:
            raise ValueError(f"Unresolved environment variable: {value}")
        return item
    return value


def run_one(command: list[str], log_path: Path, execute: bool) -> dict[str, object]:
    started = time.monotonic()
    if not execute:
        return {"command": command, "returncode": None, "wall_seconds": 0.0}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "command": command,
        "returncode": process.returncode,
        "wall_seconds": time.monotonic() - started,
        "log": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/server_full_multicohort_20260806.json"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["parse", "paths", "validate", "benchmark", "pretrain_benchmark"],
        default=["parse", "paths", "validate", "benchmark", "pretrain_benchmark"],
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = expand(json.loads(config_path.read_text(encoding="utf-8")))
    datasets = args.datasets or list(config["datasets"])
    unknown = sorted(set(datasets) - set(config["datasets"]))
    if unknown:
        parser.error(f"unknown datasets: {unknown}")
    output_root = Path(config["outputs"]["root"])
    state_root = output_root / "prepare_states"
    log_root = output_root / "prepare_driver_logs"
    records = []
    failures = []

    for stage in args.stages:
        futures = {}
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for dataset in datasets:
                state = state_root / f"{stage}_{dataset}.json"
                command = [
                    sys.executable,
                    str(REPO_ROOT / "scripts/run_full_multicohort_all_chromosomes.py"),
                    "--config",
                    str(config_path),
                    "--phases",
                    stage,
                    "--datasets",
                    dataset,
                    "--state-file",
                    str(state),
                ]
                if args.execute:
                    command.append("--execute")
                log_path = log_root / f"{stage}_{dataset}.log"
                print(f"[{stage}:{dataset}] {' '.join(command)}", flush=True)
                future = pool.submit(run_one, command, log_path, args.execute)
                futures[future] = dataset
            for future in as_completed(futures):
                dataset = futures[future]
                record = {"stage": stage, "dataset": dataset, **future.result()}
                records.append(record)
                if record["returncode"] not in (None, 0):
                    failures.append(record)
        if failures:
            break

    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "execute": args.execute,
        "jobs": args.jobs,
        "stages": args.stages,
        "datasets": datasets,
        "records": records,
        "failures": failures,
        "status": "failed" if failures else ("completed" if args.execute else "planned"),
    }
    summary_path = output_root / "parallel_prepare_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    print(f"Preparation summary: {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
