#!/usr/bin/env python3
"""Queue the chromosome-held-out frozen cCRE probe over GPUs."""

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
class ProbeJob:
    fold: str
    test: tuple[str, ...]
    validation: tuple[str, ...]
    seed: int
    closure: str

    @property
    def name(self) -> str:
        return f"ccre_{self.fold}_seed{self.seed}_{self.closure}"


def build_jobs(
    config: dict,
    *,
    seeds: set[int] | None = None,
    contexts: set[str] | None = None,
) -> list[ProbeJob]:
    return [
        ProbeJob(
            fold=fold["name"],
            test=tuple(fold["test"]),
            validation=tuple(fold["validation"]),
            seed=int(seed),
            closure=str(context),
        )
        for fold in config["rotating_chromosome_folds"]
        for seed in config["training"]["seeds"]
        if seeds is None or int(seed) in seeds
        for context in config["training"]["contexts"]
        if contexts is None or str(context) in contexts
    ]


def checkpoint_for(results_root: Path, job: ProbeJob) -> Path:
    directory = (
        results_root
        / "rotating_folds"
        / "hprc_r2"
        / job.fold
        / f"seed_{job.seed}"
        / job.closure
    )
    checkpoints = sorted(directory.glob(f"run_*/ckpt_{job.closure}__*.pt"))
    if len(checkpoints) != 1:
        raise FileNotFoundError(
            f"Expected exactly one checkpoint for {job.name}, found {len(checkpoints)} under {directory}"
        )
    return checkpoints[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--node-labels", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--contexts", nargs="+", choices=["strict", "1hop"])
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--max-slices", type=int)
    parser.add_argument("--external-sequence-cache", type=Path)
    parser.add_argument("--minimum-external-coverage", type=float, default=0.95)
    parser.add_argument("--feature-sets", nargs="+")
    parser.add_argument(
        "--canonical-conflict-policy",
        choices=["error", "exclude"],
        default="exclude",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = config["datasets"]["hprc_r2"]
    jobs = build_jobs(
        config,
        seeds=set(args.seeds) if args.seeds else None,
        contexts=set(args.contexts) if args.contexts else None,
    )
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        parser.error("--gpus must contain at least one GPU ID")
    if args.execute:
        required_paths = [args.results_root, args.node_labels, args.feature_cache]
        if args.external_sequence_cache is not None:
            required_paths.append(args.external_sequence_cache)
        for required in required_paths:
            if not required.exists():
                parser.error(f"required path does not exist: {required}")

    work: queue.Queue[ProbeJob] = queue.Queue()
    for job in jobs:
        work.put(job)
    records: list[dict[str, object]] = []
    lock = threading.Lock()
    logs = args.out_root / "logs"

    def worker(gpu: str) -> None:
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            checkpoint = checkpoint_for(args.results_root, job) if args.execute else Path(
                f"<checkpoint:{job.fold}:seed={job.seed}:{job.closure}>"
            )
            out_dir = args.out_root / job.fold / f"seed_{job.seed}" / job.closure
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts/server/run_ccre_frozen_probe_fold.py"),
                "--checkpoint", str(checkpoint),
                "--manifest", f"{dataset['pretrain_benchmark']}/manifest.csv",
                "--full-segments", dataset["segments"],
                "--node-labels", str(args.node_labels),
                "--feature-cache", str(args.feature_cache),
                "--out-dir", str(out_dir),
                "--fold", job.fold,
                "--test-chrs", *job.test,
                "--val-chrs", *job.validation,
                "--closure", job.closure,
                "--device", "cuda",
                "--seed", str(job.seed),
                "--canonical-conflict-policy", args.canonical_conflict_policy,
            ]
            if args.max_slices is not None:
                command.extend(["--max-slices", str(args.max_slices)])
            if args.external_sequence_cache is not None:
                command.extend(
                    [
                        "--external-sequence-cache",
                        str(args.external_sequence_cache),
                        "--minimum-external-coverage",
                        str(args.minimum_external_coverage),
                    ]
                )
            if args.feature_sets:
                command.extend(["--feature-sets", *args.feature_sets])
            print(f"[gpu {gpu}] {job.name}: {' '.join(command)}", flush=True)
            started = time.monotonic()
            status = "planned"
            returncode: int | None = None
            audit_path = out_dir / "audit.json"
            if args.execute and audit_path.exists():
                try:
                    status = (
                        "verified_existing"
                        if json.loads(audit_path.read_text()).get("status") == "complete"
                        else "rerun"
                    )
                except json.JSONDecodeError:
                    status = "rerun"
            if args.execute and status != "verified_existing":
                logs.mkdir(parents=True, exist_ok=True)
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + str(REPO_ROOT / "src")
                environment.setdefault("OMP_NUM_THREADS", "4")
                environment.setdefault("MKL_NUM_THREADS", "4")
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
                        "gpu": gpu,
                        "checkpoint": str(checkpoint),
                        "command": command,
                        "status": status,
                        "returncode": returncode,
                        "wall_seconds": time.monotonic() - started,
                    }
                )
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    args.out_root.mkdir(parents=True, exist_ok=True)
    failures = [record for record in records if record["status"] == "failed"]
    summary = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "jobs_requested": len(jobs),
        "jobs_recorded": len(records),
        "failures": len(failures),
        "records": sorted(records, key=lambda record: str(record["name"])),
    }
    (args.out_root / "matrix_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("jobs_requested", "jobs_recorded", "failures")}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
