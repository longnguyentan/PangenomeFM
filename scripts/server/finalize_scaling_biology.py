#!/usr/bin/env python3
"""Finish the already-running scaling workflow once all 360 audited probes exist.

This is a bounded dependency wait inside the current experiment, not a scheduler.
It never retries failed probes or writes a summary from an incomplete matrix.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--status-file", type=Path, required=True)
    ap.add_argument("--timeout-hours", type=float, default=24)
    args = ap.parse_args()
    cfg = json.loads(Path("configs/server_full_multicohort_20260806.json").read_text())
    expected = [
        args.probe_root
        / f"fraction_{fraction:g}"
        / task
        / fold["name"]
        / f"seed_{seed}"
        / context
        / "audit.json"
        for fraction, task, fold, seed, context in product(
            [0.125, 0.25, 0.5, 1],
            ["sv", "ccre", "ctcf"],
            cfg["rotating_chromosome_folds"],
            cfg["training"]["seeds"],
            cfg["training"]["contexts"],
        )
    ]
    started = time.monotonic()
    args.status_file.parent.mkdir(parents=True, exist_ok=True)

    def status(state, **extra):
        content = dict(
            status=state,
            expected_task_runs=len(expected),
            elapsed_seconds=time.monotonic() - started,
            **extra,
        )
        temporary = args.status_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(content, indent=2) + "\n")
        temporary.replace(args.status_file)

    try:
        while True:
            for path in args.probe_root.glob(
                "fraction_*/execution/fold_*/seed_*/*/*.json"
            ):
                try:
                    state = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue  # A writer can be between truncate and flush.
                if state.get("status") == "failed":
                    raise RuntimeError(f"Upstream probe failed: {path}")
            complete = 0
            for path in expected:
                if not path.exists():
                    continue
                try:
                    complete += json.loads(path.read_text()).get("status") == "complete"
                except json.JSONDecodeError:
                    continue
            status("waiting_for_probes", completed_task_runs=complete)
            if complete == len(expected):
                break
            if time.monotonic() - started > args.timeout_hours * 3600:
                raise TimeoutError(f"Only {complete}/{len(expected)} probes complete")
            time.sleep(60)
        status("auditing_and_summarizing", completed_task_runs=complete)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tasks.transfer.scaling_bio_summary",
                "--probe-root",
                str(args.probe_root),
                "--out-dir",
                str(args.out_dir),
            ],
            check=True,
        )
        status("complete", completed_task_runs=complete, summary=str(args.out_dir))
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
