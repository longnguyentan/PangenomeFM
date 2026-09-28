#!/usr/bin/env python3
"""Run the fixed chromosome protocol after explicit completed-gate review."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from tasks.entex.prepare import fingerprint
from tasks.transfer.traitgym import verified_fingerprint, write_json


def architecture_template(template: dict, fold: dict) -> dict:
    """Transfer architecture metadata only; never transfer fitted weights."""
    if template["in_dim"] != 519:
        raise ValueError("Expected fixed sequence-conditioned input dimension")
    args = copy.deepcopy(template["args"])
    args.update(test_chrs=list(fold["test"]), val_chrs=list(fold["validation"]))
    return dict(
        args=args,
        in_dim=519,
        edge_feat_dim=0,
        closure="1hop",
        architecture_only=True,
        model_weights_included=False,
    )


def check_development(audit: dict) -> None:
    checks = audit.get("checks", [])
    if (
        audit.get("status") != "eligible_for_chromosome_replication"
        or not audit.get("all_declared_seeds_complete")
        or not audit.get("all_metrics_replayed")
        or not audit.get("all_probes_converged")
        or len(checks) != 6
        or {(r["task"], r["seed"]) for r in checks}
        != {(t, s) for t in ["sv", "ccre"] for s in [42, 314159, 20260806]}
        or any(
            not math.isfinite(r["difference"])
            or r["difference"] <= 0
            or r.get("feature") != "csht"
            or r.get("metric") != "auprc"
            or r.get("comparison") != "full_trained minus full_random"
            for r in checks
        )
    ):
        raise ValueError(
            "Completed development gate does not pass; replication is not authorized by this protocol"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in [
        "config",
        "template-checkpoint",
        "main-checkout",
        "topology-control-cache",
        "out-root",
    ]:
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument(
        "--reviewed-development-sha256",
        required=True,
        help="Explicitly reviewed completed report digest; prevents speculative automatic promotion",
    )
    ap.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 2, 3])
    ap.add_argument("--cache-dependency", type=Path)
    ap.add_argument("--wait-hours", type=float, default=6)
    args = ap.parse_args()
    if len(set(args.gpus)) != 4 or not 0 < args.wait_hours <= 12:
        raise ValueError("Require four unique GPUs and a bounded readiness deadline")
    plan = json.loads(args.config.read_text())
    gate_path = Path(plan["development_audit"])
    reviewed = verified_fingerprint(gate_path, args.reviewed_development_sha256)
    check_development(json.loads(gate_path.read_text()))
    development = json.loads(Path(plan["development_protocol"]).read_text())
    if plan["seeds"] != development["seeds"] or plan["arms"] != development["arms"]:
        raise ValueError("Replication must retain every declared seed and control")
    folds = json.loads(
        Path("configs/server_full_multicohort_20260806.json").read_text()
    )["rotating_chromosome_folds"]
    if [f["name"] for f in folds] != plan["folds"]:
        raise ValueError("Original chromosome folds changed")
    by_name = {fold["name"]: fold for fold in folds}
    if set(by_name[development["fold"]]["validation"]) != set(
        by_name[plan["development_exposed_test_fold"]]["test"]
    ):
        raise ValueError(
            "Development exposure does not match the declared sensitivity fold"
        )
    args.out_root.mkdir(parents=True, exist_ok=False)
    root = args.out_root.resolve()
    record = dict(
        status="waiting",
        stage="resources",
        plan=plan,
        reviewed_development=reviewed,
        implementation=fingerprint(Path(__file__)),
        native_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        completed_jobs=[],
        commands=[],
    )
    write_json(root / "status.json", record)
    env = dict(
        os.environ,
        PYTHONPATH="src:.",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        MPLCONFIGDIR="/tmp/pfm-mpl-replication",
    )
    main_root = args.main_checkout.resolve()
    sw = main_root / "server_workspace"

    def run(command, log, gpu=None):
        with log.open("w") as handle:
            subprocess.run(
                command,
                env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu))
                if gpu is not None
                else env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )

    try:
        template = torch.load(
            args.template_checkpoint, map_location="cpu", weights_only=False
        )
        nt = json.loads(Path("configs/masked_nt_objective_20260928.json").read_text())[
            "nt_cache_sha256"
        ]
        cache = (
            Path(plan["development_root"]).parent
            / "benchmark_nt_completion/benchmark_nt.npz"
        )
        verified_fingerprint(cache, nt)
        deadline = time.monotonic() + args.wait_hours * 3600
        while True:
            ready = True
            if args.cache_dependency:
                dep = (
                    json.loads(args.cache_dependency.read_text())
                    if args.cache_dependency.exists()
                    else {}
                )
                if dep.get("status") in ["failed", "cancelled", "stopped_scope_guard"]:
                    raise ValueError("Scheduled cache dependency failed")
                ready = dep.get("status") == "complete"
            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            memory = {
                int(i): int(m) for i, m in (s.split(",") for s in raw.splitlines())
            }
            if ready and all(memory[g] < 256 for g in args.gpus):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("GPU/dependency readiness deadline exceeded")
            time.sleep(60)
        for fold in folds:
            folder = root / fold["name"]
            folder.mkdir()
            fold_plan = copy.deepcopy(development)
            fold_plan.update(
                fold=fold["name"],
                downstream="Fixed frozen chromosome-test replication; no test-guided training changes",
                replication_plan=fingerprint(args.config),
            )
            config = folder / "training_plan.json"
            write_json(config, fold_plan)
            source = folder / "architecture_only.pt"
            torch.save(architecture_template(template, fold), source)
            write_json(
                folder / "architecture_source.json",
                dict(
                    source=fingerprint(args.template_checkpoint),
                    architecture=fingerprint(source),
                    source_weights_loaded_into_encoder=False,
                    fold=fold,
                ),
            )
            for seed in plan["seeds"]:
                directory = folder / f"seed_{seed}"
                directory.mkdir()
                checkpoints = {}
                training_jobs = []
                for arm, gpu in zip(plan["arms"], args.gpus):
                    if fold["name"] == "fold_a":
                        path = (
                            Path(plan["development_root"]).resolve()
                            / f"seed_{seed}"
                            / arm
                            / "checkpoint.pt"
                        )
                    else:
                        path = directory / arm / "checkpoint.pt"
                        command = [
                            sys.executable,
                            "-u",
                            "-m",
                            "training.pretrain_masked_features",
                            "--config",
                            str(config),
                            "--template-checkpoint",
                            str(source),
                            "--manifest",
                            str(
                                sw
                                / "data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv"
                            ),
                            "--full-segments",
                            str(sw / "data/processed/hprc_r2_sv/full_segments.csv.gz"),
                            "--nt-cache",
                            str(cache.resolve()),
                            "--out-dir",
                            str(path.parent),
                            "--seed",
                            str(seed),
                            "--arm",
                            arm,
                        ]
                        training_jobs.append((command, directory / (arm + ".log"), gpu))
                        record["commands"].append(command)
                    checkpoints[arm] = str(path)
                record.update(
                    status="running", stage=f"{fold['name']}/{seed}/pretraining"
                )
                write_json(root / "status.json", record)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [pool.submit(run, *job) for job in training_jobs]
                    for future in futures:
                        future.result()
                for arm, path in checkpoints.items():
                    pre = json.loads((Path(path).parent / "status.json").read_text())
                    if (
                        pre["status"] != "complete"
                        or pre["seed"] != seed
                        or pre["arm"] != arm
                        or pre["plan"]["fold"] != fold["name"]
                        or pre["biological_labels_used"]
                        or pre["test_windows_loaded"]
                    ):
                        raise ValueError("Invalid pretraining partition or source")
                    verified_fingerprint(Path(path), pre["checkpoint"]["sha256"])
                probes = directory / "biological_test"
                command = [
                    sys.executable,
                    "scripts/server/run_v2_review_controls.py",
                    "--main-checkout",
                    str(main_root),
                    "--out-root",
                    str(probes),
                    "--fold",
                    fold["name"],
                    "--seed",
                    str(seed),
                    "--contexts",
                    "1hop",
                    "--primary-features-only",
                    "--probe-max-iter",
                    "4000",
                    "--extraction-candidate-policy",
                    "manuscript",
                    "--topology-control-cache",
                    str(args.topology_control_cache.resolve()),
                    "--probe-gpus",
                    *map(str, args.gpus),
                    "--models",
                    *plan["arms"],
                ]
                for arm, path in checkpoints.items():
                    command += ["--candidate-checkpoint", f"{arm}={path}"]
                record.update(stage=f"{fold['name']}/{seed}/test_probes")
                record["commands"].append(command)
                write_json(root / "status.json", record)
                run(command, directory / "biological_test.log")
                record["completed_jobs"].append(
                    dict(
                        fold=fold["name"],
                        seed=seed,
                        probes=str(probes),
                        checkpoints=checkpoints,
                    )
                )
                write_json(root / "status.json", record)
        record["stage"] = "report"
        write_json(root / "status.json", record)
        run(
            [
                sys.executable,
                "-m",
                "tasks.transfer.masked_replication",
                "--root",
                str(root),
                "--out-dir",
                str(root.parent / (root.name + "_analysis")),
            ],
            root / "report.log",
        )
        record.update(status="complete", stage="complete")
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(root / "status.json", record)


if __name__ == "__main__":
    main()
