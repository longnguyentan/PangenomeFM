#!/usr/bin/env python3
"""Distribute resumable experiment-matrix shards across one or more GPUs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
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


@dataclass
class Job:
    name: str
    phase: str
    filters: list[str]


def build_jobs(config: dict, phase: str, args: argparse.Namespace) -> list[Job]:
    matrix = config["experiment_matrix"]
    seeds = args.seeds or config["training"]["seeds"]
    contexts = args.contexts or config["training"]["contexts"]
    jobs: list[Job] = []
    if phase in {"final", "folds"}:
        key = "final_regimes" if phase == "final" else "fold_regimes"
        regimes = [item["name"] for item in matrix[key]]
        if args.regimes:
            regimes = [item for item in regimes if item in set(args.regimes)]
        folds = [None]
        if phase == "folds":
            folds = [item["name"] for item in config["rotating_chromosome_folds"]]
            if args.folds:
                folds = [item for item in folds if item in set(args.folds)]
        for regime in regimes:
            for fold in folds:
                for seed in seeds:
                    for context in contexts:
                        bits = [phase, regime]
                        filters = ["--regimes", regime, "--seeds", str(seed), "--contexts", context]
                        if fold:
                            bits.append(fold)
                            filters.extend(["--folds", fold])
                        bits.extend([f"seed{seed}", context])
                        jobs.append(Job("_".join(bits), phase, filters))
    else:
        key = "cohort_transfer_pairs" if phase == "transfer" else "release_transfer_pairs"
        pairs = [item.get("name", f"{item['source']}_to_{item['target']}") for item in matrix[key]]
        if args.pairs:
            pairs = [item for item in pairs if item in set(args.pairs)]
        for pair in pairs:
            for seed in seeds:
                for context in contexts:
                    name = f"{phase}_{pair}_seed{seed}_{context}"
                    filters = ["--pairs", pair, "--seeds", str(seed), "--contexts", context]
                    jobs.append(Job(name, phase, filters))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["final", "folds", "transfer", "release"], required=True)
    parser.add_argument("--config", default="configs/server_full_multicohort_20260806.json")
    parser.add_argument("--gpus", default="0", help="Comma-separated physical GPU IDs.")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--contexts", nargs="+")
    parser.add_argument("--regimes", nargs="+")
    parser.add_argument("--folds", nargs="+")
    parser.add_argument("--pairs", nargs="+")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument(
        "--job-index",
        type=int,
        help="Run exactly one zero-based matrix job (for a SLURM array task).",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = expand(json.loads(config_path.read_text(encoding="utf-8")))
    output_root = Path(config["outputs"]["root"])
    jobs = build_jobs(config, args.phase, args)
    if args.job_index is not None:
        if args.job_index < 0 or args.job_index >= len(jobs):
            parser.error(f"--job-index must be in [0, {len(jobs) - 1}]")
        jobs = [jobs[args.job_index]]
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        parser.error("--gpus must include at least one GPU ID")

    work: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        work.put(job)
    records: list[dict[str, object]] = []
    records_lock = threading.Lock()
    logs = output_root / "gpu_queue_logs" / args.phase
    states = output_root / "gpu_queue_states" / args.phase

    def worker(gpu_id: str) -> None:
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            state = states / f"{job.name}.json"
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts/run_full_multicohort_all_chromosomes.py"),
                "--config",
                str(config_path),
                "--phases",
                job.phase,
                *job.filters,
                "--state-file",
                str(state),
            ]
            if args.execute:
                command.append("--execute")
            started = time.monotonic()
            print(f"[gpu {gpu_id}] {job.name}: {' '.join(command)}", flush=True)
            if args.execute:
                logs.mkdir(parents=True, exist_ok=True)
                state.parent.mkdir(parents=True, exist_ok=True)
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu_id
                environment["DEVICE"] = "cuda"
                environment["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
                log_path = logs / f"{job.name}.log"
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
                returncode = process.returncode
            else:
                log_path = logs / f"{job.name}.log"
                returncode = None
            with records_lock:
                records.append(
                    {
                        **asdict(job),
                        "gpu_id": gpu_id,
                        "returncode": returncode,
                        "wall_seconds": time.monotonic() - started,
                        "log": str(log_path),
                        "command": command,
                    }
                )
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu_id,)) for gpu_id in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failures = [record for record in records if record["returncode"] not in (None, 0)]
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": args.phase,
        "execute": args.execute,
        "gpu_ids": gpu_ids,
        "jobs_requested": len(jobs),
        "jobs_recorded": len(records),
        "failures": len(failures),
        "records": sorted(records, key=lambda row: str(row["name"])),
    }
    summary_path = output_root / f"gpu_queue_{args.phase}_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    print(f"GPU queue summary: {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
