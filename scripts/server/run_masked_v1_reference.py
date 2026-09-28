#!/usr/bin/env python3
"""Queue a fixed v1 reference refit after the complete masked-feature replication."""

from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tasks.entex.prepare import fingerprint
from tasks.transfer.traitgym import write_json


def command(
    main_root: Path, topology: Path, root: Path, fold: str, seed: int
) -> list[str]:
    return [
        sys.executable,
        "scripts/server/run_v2_review_controls.py",
        "--main-checkout",
        str(main_root),
        "--topology-control-cache",
        str(topology),
        "--fold",
        fold,
        "--seed",
        str(seed),
        "--contexts",
        "1hop",
        "--models",
        "v1",
        "--primary-features-only",
        "--probe-max-iter",
        "4000",
        "--save-probes",
        "--extraction-candidate-policy",
        "manuscript",
        "--probe-gpus",
        "0",
        "1",
        "2",
        "3",
        "--out-root",
        str(root / fold / f"seed_{seed}"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "main-checkout", "topology-control-cache", "out-root"]:
        ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.config.read_text())
    if (
        plan["context"] != "1hop"
        or plan["reference_models"] != ["v1"]
        or plan["probe_max_iter"] != 4000
    ):
        raise ValueError("Unsupported reference protocol")
    args.out_root.mkdir(parents=True, exist_ok=False)
    state = dict(
        status="waiting",
        stage="replication_dependency",
        plan=plan,
        config=fingerprint(args.config),
        implementation=fingerprint(Path(__file__)),
        native_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        completed_jobs=[],
        commands=[],
    )
    env = dict(
        os.environ,
        PYTHONPATH="src:.",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )
    status = args.out_root / "status.json"
    write_json(status, state)
    try:
        deadline = time.monotonic() + 12 * 3600
        while True:
            dep = json.loads(
                (Path(plan["replication_root"]) / "status.json").read_text()
            )
            if dep["status"] in ["failed", "cancelled"]:
                raise ValueError(
                    "Primary replication failed; preserve its failure before supplemental work"
                )
            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            memory = {
                int(i): int(m)
                for i, m in (line.split(",") for line in raw.splitlines())
            }
            if dep["status"] == "complete" and all(memory[g] < 256 for g in range(4)):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Twelve-hour dependency/resource deadline exceeded")
            time.sleep(60)
        for fold, seed in product(plan["folds"], plan["seeds"]):
            cmd = command(
                args.main_checkout.resolve(),
                args.topology_control_cache.resolve(),
                args.out_root / "probes",
                fold,
                seed,
            )
            state.update(status="running", stage=f"{fold}/{seed}")
            state["commands"].append(cmd)
            write_json(status, state)
            with (args.out_root / f"{fold}_{seed}.log").open("w") as log:
                subprocess.run(
                    cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            state["completed_jobs"].append(dict(fold=fold, seed=seed))
            write_json(status, state)
        state["stage"] = "report"
        write_json(status, state)
        with (args.out_root / "report.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tasks.transfer.masked_v1_reference",
                    "--config",
                    str(args.config),
                    "--reference-root",
                    str(args.out_root / "probes"),
                    "--out-dir",
                    str(args.out_root / "analysis"),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        state.update(status="complete", stage="complete")
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(status, state)


if __name__ == "__main__":
    main()
