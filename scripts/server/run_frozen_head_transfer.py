#!/usr/bin/env python3
"""Run the prespecified weak-head transfer diagnostic after current GPU work."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from tasks.entex.prepare import fingerprint


def compare_heads(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "task", "feature"]
    if any(f.duplicated(keys).any() for f in [old, new]):
        raise ValueError("Duplicate model/task/features")
    merged = old.merge(new, on=keys, suffixes=("_mlp", "_linear"), validate="one_to_one", how="outer", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Head comparisons differ in scope")
    for key in ["fold", "seed", "context", "n_train", "n_val", "n_test", "targets_sha256"]:
        if not merged[key + "_mlp"].eq(merged[key + "_linear"]).all():
            raise ValueError("Head comparisons changed examples or partitions")
    baseline = merged.feature.isin(["cs", "csh"]) | merged.model.eq("Q")
    if not merged.loc[baseline, "scores_sha256_mlp"].eq(merged.loc[baseline, "scores_sha256_linear"]).all():
        raise ValueError("Unchanged baseline predictions differ")
    if not np.isfinite(merged[["auprc_mlp", "auprc_linear"]]).all().all():
        raise ValueError("Undefined head comparison")
    merged["delta_linear_minus_mlp"] = merged.auprc_linear - merged.auprc_mlp
    return merged.drop(columns="_merge")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ["main-checkout", "native-checkout", "old-probes", "old-analysis", "trained-root", "random-root",
                 "wait-for", "out-root", "config"]:
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--wait-hours", type=float, default=4)
    args = ap.parse_args()
    plan = json.loads(args.config.read_text())
    old = json.loads((args.old_probes / "status.json").read_text())
    if (old["status"] != "complete" or old["probe_max_iter_override"] != 4000
            or old["evaluation_partition"] != "development_validation" or old["seed"] != 42):
        raise ValueError("Require completed uniform-budget seed-42 development probes")
    nt = {}
    for name in ["nt", "nt_random"]:
        audit = json.loads((args.old_probes / "probes" / name / "ccre/1hop/audit.json").read_text())
        nt[name] = Path(audit["checkpoint"])
        if fingerprint(nt[name])["sha256"] != audit["checkpoint_sha256"]:
            raise ValueError("Unchanged NT branch checkpoint changed")
    topology = {}
    for name, root in [("v2", args.trained_root), ("v2_random", args.random_root)]:
        paths = list((root / "bidirectional_linear/run_001").glob("ckpt_*.pt"))
        if len(paths) != 1:
            raise ValueError("Expected one existing linear-head topology checkpoint")
        payload = torch.load(paths[0], map_location="cpu", weights_only=False)
        expected = dict(seed=42, hidden_dim=48, linear_predictor=True, graph_message_direction="bidirectional",
                        node_extra_features="none", objective="junction_repair", node_structure_source="visible")
        if any(payload["args"].get(k) != v for k, v in expected.items()):
            raise ValueError("Topology checkpoint differs from the prespecified intervention")
        random = name == "v2_random"
        if bool(payload["args"].get("freeze_encoder", False)) != random:
            raise ValueError("Random/trained checkpoint attribution differs")
        if random and (not payload.get("initial_encoder_sha256")
                       or payload["initial_encoder_sha256"] != payload["final_encoder_sha256"]):
            raise ValueError("Random encoder weights changed")
        topology[name] = paths[0]
    args.out_root.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "scripts/server/run_v2_review_controls.py", "--main-checkout", str(args.main_checkout),
        "--out-root", str(args.out_root / "probes"), "--fold", "fold_a", "--seed", "42", "--contexts", "1hop",
        "--validation-only", "--probe-max-iter", "4000", "--primary-features-only", "--extraction-candidate-policy", "manuscript",
        "--topology-control-cache", old["topology_control_cache"], "--probe-gpus", "0", "1", "2", "3",
        "--models", *plan["models"]]
    for name, path in topology.items():
        command += ["--candidate-checkpoint", f"{name}={path}"]
    for name, t, q in [("T_Q", "v2", "nt"), ("T_Rq", "v2", "nt_random"),
                        ("Rt_Q", "v2_random", "nt"), ("Rt_Rq", "v2_random", "nt_random")]:
        command += ["--candidate-checkpoint", f"{name}={topology[t]}", "--companion-checkpoint", f"{name}={nt[q]}"]
    report = [sys.executable, "-m", "tasks.transfer.frozen_branch_report", "--root", str(args.out_root / "probes"),
        "--topology-root", str(args.out_root / "probes"), "--sequence-root", str(args.old_probes),
        "--out-dir", str(args.out_root / "analysis")]
    receipt = dict(status="waiting", plan=plan, config=fingerprint(args.config), implementation=fingerprint(Path(__file__)),
        native_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.native_checkout, text=True).strip(),
        commands=[command, report], waiting_for=str(args.wait_for),
        checkpoints={k: fingerprint(v) for k, v in {**nt, **topology}.items()})

    def save():
        path = args.out_root / "status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(path)

    save()
    env = dict(os.environ, PYTHONPATH="src:.", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", MPLCONFIGDIR="/tmp/pfm-mpl")
    try:
        deadline = time.monotonic() + args.wait_hours * 3600
        while True:
            dependency = json.loads(args.wait_for.read_text())
            if dependency["status"] == "complete":
                break
            if dependency["status"] == "failed":
                raise RuntimeError("Dependency failed; inspect its receipt before launching")
            if time.monotonic() > deadline:
                raise TimeoutError("Bounded dependency wait expired")
            time.sleep(30)
        receipt["status"] = "running"
        save()
        for cmd, name in [(command, "probes"), (report, "report")]:
            with (args.out_root / (name + ".log")).open("w") as log:
                subprocess.run(cmd, cwd=args.native_checkout, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        previous = pd.read_csv(args.old_analysis / "audited_per_run.csv", float_precision="round_trip")
        current = pd.read_csv(args.out_root / "analysis/audited_per_run.csv", float_precision="round_trip")
        comparison = compare_heads(previous, current)
        comparison.to_csv(args.out_root / "analysis/head_comparison.csv", index=False)
        gate = json.loads((args.out_root / "analysis/development_gate.json").read_text())
        gains = comparison.loc[comparison.model.eq("T_Q") & comparison.feature.eq("csht")]
        receipt.update(status="complete", numerical_gate=gate.get("all_probes_converged"),
            control_gate=gate["status"], improves_both_tasks=bool(len(gains) == 2 and gains.delta_linear_minus_mlp.gt(0).all()),
            interpretation="Development diagnostic only; an unfavorable result is retained as a complete experiment")
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
