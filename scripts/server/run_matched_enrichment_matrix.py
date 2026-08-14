#!/usr/bin/env python3
"""Run independent matched-enrichment signal sets with a resumable audit."""

from __future__ import annotations

import argparse
import hashlib
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


@dataclass(frozen=True)
class EnrichmentJob:
    name: str
    signals: Path


def sha256sum(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_signal(value: str) -> EnrichmentJob:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--signal must have NAME=PATH form")
    name, path_text = value.split("=", 1)
    if not name or not path_text:
        raise argparse.ArgumentTypeError("--signal must have nonempty NAME and PATH")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in name):
        raise argparse.ArgumentTypeError(
            "signal NAME may contain only letters, digits, underscore, and hyphen"
        )
    return EnrichmentJob(name=name, signals=Path(path_text))


def build_command(
    job: EnrichmentJob,
    *,
    regions: Path,
    out_root: Path,
    controls_per_case: int,
    relative_caliper: float,
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts/server/matched_locus_enrichment.py"),
        "--regions", str(regions),
        "--signals", str(job.signals),
        "--out-dir", str(out_root / job.name),
        "--priority-column", "is_prioritized",
        "--match-columns",
        "region_length", "gc_content", "mappability", "graph_complexity",
        "variant_density", "distance_to_gene",
        "--controls-per-case", str(controls_per_case),
        "--relative-caliper", str(relative_caliper),
        "--n-permutations", str(n_permutations),
        "--n-bootstrap", str(n_bootstrap),
        "--seed", str(seed),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--signal", action="append", type=parse_signal, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--controls-per-case", type=int, default=5)
    parser.add_argument("--relative-caliper", type=float, default=0.25)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not args.regions.is_file():
        parser.error(f"Missing regions table: {args.regions}")
    names = [job.name for job in args.signal]
    if len(names) != len(set(names)):
        parser.error("--signal names must be unique")
    for job in args.signal:
        if not job.signals.is_file():
            parser.error(f"Missing signal table for {job.name}: {job.signals}")

    work: queue.Queue[EnrichmentJob] = queue.Queue()
    for job in args.signal:
        work.put(job)
    records: list[dict[str, object]] = []
    lock = threading.Lock()
    logs = args.out_root / "logs"
    region_sha256 = sha256sum(args.regions)

    def worker(worker_id: int) -> None:
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            output = args.out_root / job.name
            audit_path = output / "audit.json"
            command = build_command(
                job,
                regions=args.regions,
                out_root=args.out_root,
                controls_per_case=args.controls_per_case,
                relative_caliper=args.relative_caliper,
                n_permutations=args.n_permutations,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            started = time.monotonic()
            status = "planned"
            returncode: int | None = None
            message: str | None = None
            print(f"[enrichment cpu {worker_id}] {job.name}: {' '.join(command)}", flush=True)
            if args.execute and audit_path.is_file():
                try:
                    audit = json.loads(audit_path.read_text())
                    if (
                        audit.get("region_source_sha256") == region_sha256
                        and audit.get("signal_source_sha256") == sha256sum(job.signals)
                        and audit.get("status", "complete") == "complete"
                    ):
                        status = "verified_existing"
                except (json.JSONDecodeError, OSError):
                    pass
            if args.execute and status != "verified_existing" and output.exists():
                status = "failed"
                returncode = 98
                message = (
                    f"Refusing to reuse incomplete output directory {output}; "
                    "archive it before resuming"
                )
            if args.execute and status == "planned":
                logs.mkdir(parents=True, exist_ok=True)
                environment = os.environ.copy()
                environment["PYTHONPATH"] = (
                    str(REPO_ROOT) + os.pathsep + str(REPO_ROOT / "src")
                )
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
                        "signals": str(job.signals),
                        "worker_id": worker_id,
                        "command": command,
                        "status": status,
                        "returncode": returncode,
                        "message": message,
                        "wall_seconds": time.monotonic() - started,
                    }
                )
            work.task_done()

    threads = [
        threading.Thread(target=worker, args=(index,))
        for index in range(min(args.jobs, len(args.signal)))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = [record for record in records if record["status"] == "failed"]
    args.out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "jobs_requested": len(args.signal),
        "jobs_recorded": len(records),
        "failures": len(failures),
        "regions": str(args.regions.resolve()),
        "controls_per_case": args.controls_per_case,
        "relative_caliper": args.relative_caliper,
        "n_permutations": args.n_permutations,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "records": sorted(records, key=lambda record: str(record["name"])),
    }
    (args.out_root / "matrix_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("jobs_requested", "jobs_recorded", "failures")
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
