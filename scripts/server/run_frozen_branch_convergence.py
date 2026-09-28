#!/usr/bin/env python3
"""Repeat all fixed-seed probes at the same prespecified solver budget."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tasks.entex.prepare import fingerprint

SEEDS = [314159, 20260806]
MODELS = ["v2", "v2_random", "nt", "nt_random", "T_Q", "T_Rq", "Rt_Q", "Rt_Rq"]


def convergence_command(source: list[str], out: Path, seed: int) -> list[str]:
    """Change only output location/interpreter and the uniform iteration ceiling."""
    if seed not in SEEDS or Path(source[1]).name != "run_v2_review_controls.py":
        raise ValueError("Unexpected source workflow or seed")
    for flag, value in [("--fold", "fold_a"), ("--seed", str(seed)),
                        ("--contexts", "1hop"), ("--extraction-candidate-policy", "manuscript")]:
        if source.count(flag) != 1 or source[source.index(flag) + 1] != value:
            raise ValueError("Source differs from the fixed development protocol")
    if any(flag not in source for flag in ["--validation-only", "--primary-features-only"]):
        raise ValueError("Require the complete validation-only source")
    if "--probe-max-iter" in source:
        raise ValueError("Source already overrides the historical optimizer budget")
    models = source[source.index("--models") + 1:]
    models = models[:next((i for i, x in enumerate(models) if x.startswith("--")), len(models))]
    if models != MODELS or source.count("--out-root") != 1:
        raise ValueError("All eight fixed model arms must be retained in order")
    result = source.copy()
    result[0] = sys.executable
    result[result.index("--out-root") + 1] = str(out)
    return result + ["--probe-max-iter", "4000"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replication-root", type=Path, required=True)
    ap.add_argument("--seed42-report", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--summary-script", type=Path, help="Pinned report source when running beside an active older checkout")
    ap.add_argument("--wait-hours", type=float, default=8)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if not 0 < args.wait_hours <= 24:
        raise ValueError("Require a bounded wait of at most 24 hours")
    out = args.out_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = dict(status="planned", seeds=[42, *SEEDS], max_iter=4000,
        scope="Uniform numerical sensitivity; no new pretraining or biological hyperparameter search",
        source_replication=str(args.replication_root.resolve()), seed42_report=str(args.seed42_report.resolve()),
        implementation=fingerprint(Path(__file__)), completed_seeds=[], commands={})
    status = out / "status.json"

    def save():
        temporary = status.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(status)

    save()
    if not args.execute:
        return
    env = dict(os.environ, PYTHONPATH="src:.", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    record.update(status="waiting_for_existing_runs")
    save()
    deadline = time.monotonic() + args.wait_hours * 3600
    try:
        source_path = args.replication_root / "status.json"
        while True:
            source = json.loads(source_path.read_text())
            if source["status"] == "failed":
                raise RuntimeError("Source replication failed; retain its diagnostic log")
            if source["status"] == "complete" and (args.seed42_report / "audit.json").is_file():
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Source runs did not complete within the bounded wait")
            time.sleep(30)
        record["source_receipt"] = fingerprint(source_path)
        record["native_code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        roots = [args.seed42_report]
        commands = {seed: convergence_command(source["probe_commands"][str(seed)], out / f"seed_{seed}", seed)
                    for seed in SEEDS}
        record.update(status="running", commands=commands)
        save()
        for seed, command in commands.items():
            with (out / f"seed_{seed}.log").open("w") as handle:
                subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
            root = out / f"seed_{seed}"
            report = out / f"seed_{seed}_analysis"
            report_command = [sys.executable, "-m", "tasks.transfer.frozen_branch_report",
                "--root", str(root), "--topology-root", str(root), "--sequence-root", str(root),
                "--out-dir", str(report)]
            with (out / f"seed_{seed}_report.log").open("w") as handle:
                subprocess.run(report_command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
            roots.append(report)
            record["completed_seeds"].append(seed)
            save()
        record["report_roots"] = list(map(str, roots))
        summary_command = [sys.executable]
        if args.summary_script:
            record["summary_implementation"] = fingerprint(args.summary_script)
            summary_command += [str(args.summary_script.resolve())]
        else:
            summary_command += ["-m", "tasks.transfer.frozen_branch_replication_report"]
        summary_command += ["--sources", *map(str, roots), "--out-dir", str(out / "analysis")]
        record["summary_command"] = summary_command
        save()
        with (out / "summary.log").open("w") as handle:
            subprocess.run(summary_command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        record.update(status="complete")
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
