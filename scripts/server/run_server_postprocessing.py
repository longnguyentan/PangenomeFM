#!/usr/bin/env python3
"""Generate validation, context, chromosome, and paper-source result tables."""

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


def invoke(name: str, command: list[str], log_dir: Path) -> dict[str, object]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment.setdefault("MPLCONFIGDIR", str(log_dir / "matplotlib_cache"))
    started = time.monotonic()
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
        "name": name,
        "command": command,
        "log": str(log_path),
        "returncode": process.returncode,
        "wall_seconds": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/server_full_multicohort_20260806.json")
    parser.add_argument("--context-jobs", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--skip-context-audit", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = expand(json.loads(config_path.read_text(encoding="utf-8")))
    root = Path(config["outputs"]["root"])
    logs = root / "postprocessing_logs"
    records = []

    commands = [
        (
            "resource_inventory",
            [
                sys.executable,
                str(REPO_ROOT / "scripts/summarize_full_multicohort_resources.py"),
                "--config", str(config_path),
                "--out-dir", str(root / "resource_inventory"),
            ],
        ),
        (
            "chromosome_graph_statistics",
            [
                sys.executable,
                str(REPO_ROOT / "scripts/server/compute_chromosome_graph_statistics.py"),
                "--config", str(config_path),
                "--out-dir", str(root / "chromosome_graph_statistics"),
            ],
        ),
    ]
    for name, command in commands:
        record = invoke(name, command, logs)
        records.append(record)
        if record["returncode"] != 0:
            break

    if not any(record["returncode"] != 0 for record in records) and not args.skip_context_audit:
        futures = {}
        with ThreadPoolExecutor(max_workers=args.context_jobs) as pool:
            for dataset, values in config["datasets"].items():
                command = [
                    sys.executable,
                    str(REPO_ROOT / "scripts/audit_benchmark_context.py"),
                    "--manifest", str(Path(values["benchmark"]) / "manifest.csv"),
                    "--out-dir", str(root / "context_audit" / dataset),
                    "--test-chromosomes", "chr1", "chr6", "chr11", "chr16", "chr21",
                    "--validation-chromosomes", "chr2", "chr7", "chr12", "chr17", "chr22",
                    "--seed", "20260806",
                ]
                future = pool.submit(invoke, f"context_audit_{dataset}", command, logs)
                futures[future] = dataset
            for future in as_completed(futures):
                records.append(future.result())

    if not any(record["returncode"] != 0 for record in records):
        aggregate = invoke(
            "aggregate_server_results",
            [
                sys.executable,
                str(REPO_ROOT / "scripts/server/aggregate_server_results.py"),
                "--results-root", str(root),
                "--out-dir", str(root / "paper_source_data"),
                "--n-boot", str(args.n_boot),
                "--seed", "20260806",
            ],
            logs,
        )
        records.append(aggregate)
    if not any(record["returncode"] != 0 for record in records):
        adjusted_context = invoke(
            "analyze_context_adjusted_performance",
            [
                sys.executable,
                str(REPO_ROOT / "scripts/server/analyze_context_adjusted_performance.py"),
                "--per-slice", str(root / "paper_source_data/per_slice_metrics.csv.gz"),
                "--context-root", str(root / "context_audit"),
                "--out-dir", str(root / "context_adjusted_performance"),
                "--relative-caliper", "0.10",
            ],
            logs,
        )
        records.append(adjusted_context)
    if not any(record["returncode"] != 0 for record in records):
        gap = invoke(
            "analyze_hgsvc_performance_gap",
            [
                sys.executable,
                str(REPO_ROOT / "scripts/server/analyze_hgsvc_performance_gap.py"),
                "--graph-statistics", str(root / "chromosome_graph_statistics/chromosome_graph_statistics.csv"),
                "--performance", str(root / "paper_source_data/chromosome_summary_across_seeds.csv"),
                "--out-dir", str(root / "hgsvc_performance_gap"),
            ],
            logs,
        )
        records.append(gap)

    failures = [record for record in records if record["returncode"] != 0]
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records": records,
        "failures": failures,
        "status": "failed" if failures else "completed",
    }
    path = root / "postprocessing_summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
