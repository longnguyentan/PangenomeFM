#!/usr/bin/env python3
"""Replicate the fixed fold-A frozen-branch candidate at the two remaining seeds."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from tasks.entex.prepare import fingerprint


def pretraining_commands(out: Path, audit_root: Path, nt_cache: Path, seed: int, gpus: list[int]) -> dict:
    if seed not in {314159, 20260806} or len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("Two prespecified replication seeds and four distinct GPUs are required")
    result = {}
    for gpu, (name, sequence, random) in zip(gpus, [
        ("topology_trained", False, False), ("topology_random", False, True),
        ("sequence_trained", True, False), ("sequence_random", True, True)]):
        command = [sys.executable, "scripts/server/run_junction_geometry_pilot.py",
            "--incoming-audit", str(audit_root / "junction_geometry_matched"),
            "--bidirectional-audit", str(audit_root / "junction_geometry_bidirectional"),
            "--context", "1hop", "--arms", "bidirectional_linear" if sequence else "bidirectional_default",
            "--seed", str(seed), "--gpus", str(gpu), "--out-root", str(out / name), "--execute"]
        if sequence:
            command += ["--node-feature-cache", str(nt_cache)]
        if random:
            command.append("--random-encoder-control")
        result[name] = command
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main-checkout", type=Path, required=True)
    ap.add_argument("--audit-root", type=Path, required=True)
    ap.add_argument("--nt-cache", type=Path, required=True)
    ap.add_argument("--topology-control-cache", type=Path, required=True)
    ap.add_argument("--decision-gate", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 2, 3])
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    gate = json.loads(args.decision_gate.read_text())
    if gate.get("status") != "eligible_for_replication":
        raise ValueError("The recorded frozen-branch development gate did not pass")
    out = args.out_root.resolve()
    stages = {seed: pretraining_commands(out / f"seed_{seed}", args.audit_root.resolve(),
               args.nt_cache.resolve(), seed, args.gpus) for seed in [314159, 20260806]}
    out.mkdir(parents=True, exist_ok=False)
    record = dict(status="planned", scope="fold-A validation seed replication; not independent chromosome replication",
        seeds=list(stages), fold="fold_a", context="1hop", test_scoring=False,
        fixed_candidate="48D topology MLP-head pretraining + 48D NT-conditioned linear-head pretraining; frozen concatenation",
        checkpoint_selection="Each run's native reconstruction validation AUROC; no biological model selection",
        candidate_partition_seed=20260806, matching_audits_reused="Candidate generation uses split_seed, independent of model initialization seed",
        decision_gate=fingerprint(args.decision_gate), commands=stages,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        completed_stages=[])
    status = out / "status.json"

    def save():
        temporary = status.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(status)

    save()
    if not args.execute:
        return
    env = dict(os.environ, PYTHONPATH="src:.", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")

    def execute(command, log):
        with log.open("w") as handle:
            subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

    record["status"] = "running"
    save()
    try:
        for seed, commands in stages.items():
            directory = out / f"seed_{seed}"
            directory.mkdir()
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(execute, command, directory / (name + ".log")) for name, command in commands.items()]
                for future in futures:
                    future.result()
            record["completed_stages"].append(f"seed_{seed}/pretraining")
            save()
            checkpoints = {}
            for name, model in [("topology_trained", "v2"), ("topology_random", "v2_random"),
                                ("sequence_trained", "nt"), ("sequence_random", "nt_random")]:
                found = list((directory / name).glob("*/run_001/ckpt_*.pt"))
                if len(found) != 1:
                    raise ValueError("Expected one selected checkpoint per fixed branch")
                checkpoints[model] = found[0]
            for kind in ["topology", "sequence"]:
                report = [sys.executable, "-m", "tasks.transfer.junction_pilot_report", "--trained-root", str(directory / (kind + "_trained")),
                          "--random-root", str(directory / (kind + "_random")), "--out-dir", str(directory / (kind + "_analysis"))]
                execute(report, directory / (kind + "_report.log"))
            command = [sys.executable, "scripts/server/run_v2_review_controls.py",
                "--main-checkout", str(args.main_checkout.resolve()), "--out-root", str(directory / "biological"),
                "--fold", "fold_a", "--seed", str(seed), "--contexts", "1hop", "--validation-only",
                "--primary-features-only", "--extraction-candidate-policy", "manuscript",
                "--topology-control-cache", str(args.topology_control_cache.resolve()),
                "--probe-gpus", *map(str, args.gpus), "--models", *checkpoints, "T_Q", "T_Rq", "Rt_Q", "Rt_Rq"]
            for model, checkpoint in checkpoints.items():
                command += ["--candidate-checkpoint", f"{model}={checkpoint}"]
            for name, t, q in [("T_Q", "v2", "nt"), ("T_Rq", "v2", "nt_random"),
                                ("Rt_Q", "v2_random", "nt"), ("Rt_Rq", "v2_random", "nt_random")]:
                command += ["--candidate-checkpoint", f"{name}={checkpoints[t]}",
                            "--companion-checkpoint", f"{name}={checkpoints[q]}"]
            record.setdefault("probe_commands", {})[seed] = command
            save()
            execute(command, directory / "biological.log")
            report = [sys.executable, "-m", "tasks.transfer.frozen_branch_report",
                "--root", str(directory / "biological"), "--topology-root", str(directory / "biological"),
                "--sequence-root", str(directory / "biological"), "--out-dir", str(directory / "biological_analysis")]
            execute(report, directory / "biological_report.log")
            record["completed_stages"].append(f"seed_{seed}/biological_and_report")
            save()
        record["status"] = "complete"
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
