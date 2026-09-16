#!/usr/bin/env python3
"""Dispatch existing EN-TEx fold probes using manuscript jobs and fixed GPU workers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from scripts.server.run_ccre_frozen_probe_matrix import build_jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/entex_v1.json"))
    for key in [
        "loci",
        "mapping-dir",
        "full-segments",
        "manifest",
        "feature-cache",
        "sequence-cache",
        "results-root",
        "out-root",
        "topology-cache-root",
    ]:
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument(
        "--sensitivity",
        choices=["primary", "exposure_matched", "h3k27ac", "ctcf"],
        default="primary",
    )
    ap.add_argument("--task", choices=["p0", "p1"], default="p0")
    ap.add_argument("--subtask")
    ap.add_argument("--cache-all-reference-targets", action="store_true")
    ap.add_argument("--gpus", default="0,1,2,3")
    args = ap.parse_args()
    if args.out_root.exists() and any(args.out_root.iterdir()):
        raise FileExistsError(f"Matrix output root must be new: {args.out_root}")
    config = json.loads(args.config.read_text())
    jobs = build_jobs(json.loads(Path(config["manuscript_config"]).read_text()))
    gpus = args.gpus.split(",")
    if len(gpus) != len(set(gpus)) or not all(x.isdigit() for x in gpus):
        raise ValueError("GPU IDs must be unique nonnegative integers")
    args.out_root.mkdir(parents=True, exist_ok=True)
    logs = args.out_root / "logs"
    logs.mkdir()
    tasks = queue.Queue()
    plans = []
    for job in jobs:
        command = [
            sys.executable,
            "-m",
            "tasks.entex.probe",
            "--config",
            str(args.config),
            "--sensitivity",
            args.sensitivity,
            "--device",
            "cuda",
            "--folds",
            job.fold,
            "--seeds",
            str(job.seed),
            "--contexts",
            job.closure,
        ]
        command.extend(["--task", args.task])
        if args.subtask:
            command.extend(["--subtask", args.subtask])
        if args.cache_all_reference_targets:
            command.append("--cache-all-reference-targets")
        for key in [
            "loci",
            "mapping-dir",
            "full-segments",
            "manifest",
            "feature-cache",
            "sequence-cache",
            "results-root",
            "out-root",
            "topology-cache-root",
        ]:
            command.extend(["--" + key, str(getattr(args, key.replace("-", "_")))])
        item = dict(job=f"{job.fold}_seed{job.seed}_{job.closure}", command=command)
        plans.append(item)
        tasks.put(item)
    (args.out_root / "matrix_plan.json").write_text(json.dumps(plans, indent=2) + "\n")
    records = []
    lock = threading.Lock()
    stop = threading.Event()

    def worker(gpu):
        while not stop.is_set():
            try:
                item = tasks.get_nowait()
            except queue.Empty:
                return
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=gpu,
                OMP_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                PYTHONUNBUFFERED="1",
            )
            started = time.monotonic()
            with (logs / (item["job"] + ".log")).open("w") as log:
                result = subprocess.run(
                    item["command"], env=env, stdout=log, stderr=subprocess.STDOUT
                )
            with lock:
                records.append(
                    dict(
                        job=item["job"],
                        gpu=gpu,
                        returncode=result.returncode,
                        wall_seconds=time.monotonic() - started,
                    )
                )
                (args.out_root / "matrix_results.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
            if result.returncode:
                stop.set()
                return

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, gpus))
    if stop.is_set() or len(records) != len(plans):
        raise RuntimeError(
            "Matrix incomplete: inspect logs and matrix_results.json; remaining jobs were not launched after failure"
        )


if __name__ == "__main__":
    main()
