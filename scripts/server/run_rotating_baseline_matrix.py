#!/usr/bin/env python3
"""Run the HPRC rotating-fold non-neural baseline matrix on multiple CPUs."""

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
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_full_multicohort_all_chromosomes import load_config


@dataclass(frozen=True)
class BaselineJob:
    fold: str
    test: tuple[str, ...]
    validation: tuple[str, ...]
    seed: int
    context: str

    @property
    def name(self) -> str:
        return f"hprc_r2_{self.fold}_seed{self.seed}_{self.context}"


def build_jobs(config: dict) -> list[BaselineJob]:
    return [
        BaselineJob(
            fold=fold["name"],
            test=tuple(fold["test"]),
            validation=tuple(fold["validation"]),
            seed=int(seed),
            context=str(context),
        )
        for fold in config["rotating_chromosome_folds"]
        for seed in config["training"]["seeds"]
        for context in config["training"]["contexts"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/server_full_multicohort_20260806.json"
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    config = load_config(args.config)
    dataset = config["datasets"]["hprc_r2"]
    matrix = build_jobs(config)
    if args.max_jobs is not None:
        matrix = matrix[: args.max_jobs]
    work: queue.Queue[BaselineJob] = queue.Queue()
    for job in matrix:
        work.put(job)
    records: list[dict[str, object]] = []
    lock = threading.Lock()
    logs = args.out_root / "logs"

    def worker(worker_id: int) -> None:
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            out_dir = args.out_root / "hprc_r2" / job.fold / f"seed_{job.seed}" / job.context
            audit = out_dir / "audit.json"
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts/server/evaluate_rotating_link_baselines.py"),
                "--manifest", f"{dataset['pretrain_benchmark']}/manifest.csv",
                "--full-segments", dataset["segments"],
                "--out-dir", str(out_dir),
                "--closure", job.context,
                "--test-chrs", *job.test,
                "--val-chrs", *job.validation,
                "--seed", str(job.seed),
                "--split-seed", str(config["training"]["split_seed"]),
            ]
            print(f"[cpu {worker_id}] {job.name}: {' '.join(command)}", flush=True)
            started = time.monotonic()
            status = "planned"
            returncode: int | None = None
            if args.execute and audit.is_file():
                try:
                    status = "verified_existing" if json.loads(audit.read_text()).get("status") == "complete" else "rerun"
                except json.JSONDecodeError:
                    status = "rerun"
            if args.execute and status != "verified_existing":
                logs.mkdir(parents=True, exist_ok=True)
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + str(REPO_ROOT / "src")
                environment["OMP_NUM_THREADS"] = "1"
                environment["MKL_NUM_THREADS"] = "1"
                with (logs / f"{job.name}.log").open("a") as handle:
                    process = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                returncode = process.returncode
                status = "complete" if returncode == 0 else "failed"
            with lock:
                records.append(
                    {
                        **asdict(job),
                        "name": job.name,
                        "worker_id": worker_id,
                        "command": command,
                        "status": status,
                        "returncode": returncode,
                        "wall_seconds": time.monotonic() - started,
                    }
                )
            work.task_done()

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(args.jobs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = [row for row in records if row["status"] == "failed"]
    args.out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "jobs_requested": len(matrix),
        "jobs_recorded": len(records),
        "failures": len(failures),
        "records": sorted(records, key=lambda row: str(row["name"])),
    }
    (args.out_root / "matrix_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ["jobs_requested", "jobs_recorded", "failures"]}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
