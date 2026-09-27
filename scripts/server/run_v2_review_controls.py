#!/usr/bin/env python3
"""Run frozen v1 versus handcrafted/random controls in a separate output root.

Fold A is an exploratory development experiment, not an untouched final test.
This runner does not select or train a v2 model. Existing probes own all feature
joins, chromosome partitions, calibration, metrics and output validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main-checkout", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--fold", default="fold_a")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--contexts", nargs="+", choices=["strict", "1hop"], default=["strict", "1hop"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--topology-control-cache", type=Path)
    args = ap.parse_args()
    out = args.out_root.resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output root: {out}")
    main_root = args.main_checkout.resolve()
    sw = main_root / "server_workspace"
    graph = sw / "data/processed/hprc_r2_sv/full_segments.csv.gz"
    config = json.loads(Path("configs/entex_v1.json").read_text())
    with graph.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != config["full_segments_sha256"]:
        raise ValueError("Original manuscript graph checksum mismatch")
    fold_config = json.loads(Path("configs/server_full_multicohort_20260806.json").read_text())
    fold = next(f for f in fold_config["rotating_chromosome_folds"] if f["name"] == args.fold)
    nt = sw / "results/frozen_sequence_fm_cache_20260815/hprc_target_union_sequence_fm.npz"
    topology = args.topology_control_cache.resolve() if args.topology_control_cache else out / "topology_control.npz"
    commands = []
    topology_provenance = None
    if args.topology_control_cache:
        topology_provenance = json.loads(Path(str(topology) + ".audit.json").read_text())
        if (not topology.is_file() or topology_provenance.get("status") != "complete"
                or topology_provenance.get("processing_version") != 2
                or topology_provenance.get("full_segments_sha256") != digest
                or topology_provenance.get("downstream_label_access") != "none"):
            raise ValueError("Shared H cache provenance does not match the reviewed graph/protocol")
        with topology.open("rb") as handle:
            topology_provenance["cache_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    else:
        commands.append([sys.executable, "-m", "tasks.transfer.topology_controls",
                         "--full-segments", str(graph), "--full-links", str(graph.with_name("full_links.csv.gz")),
                         "--target-cache", str(nt), "--out", str(topology)])
    for context in args.contexts:
        root = sw / f"results/full_multicohort_server_20260806/rotating_folds/hprc_r2/{args.fold}/seed_{args.seed}/{context}"
        matches = sorted(root.glob(f"run_*/ckpt_{context}__*.pt"))
        if len(matches) != 1:
            raise ValueError(f"Expected one canonical checkpoint in {root}; found {len(matches)}")
        trained = matches[0]
        random = out / f"random_init/{context}.pt"
        commands.append([sys.executable, "scripts/make_random_init_checkpoint.py", "--checkpoint",
                         str(trained), "--out", str(random), "--seed", str(args.seed)])
        for model, checkpoint in (("v1", trained), ("random", random)):
            for task in ("sv", "ccre"):
                command = [sys.executable, f"scripts/server/run_{task}_frozen_probe_fold.py",
                           "--checkpoint", str(checkpoint), "--manifest",
                           str(sw / "data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv"),
                           "--full-segments", str(graph), "--fold", args.fold,
                           "--seed", str(args.seed), "--closure", context, "--device", args.device,
                           "--external-sequence-cache", str(nt), "--minimum-external-coverage", "1.0",
                           "--topology-control-cache", str(topology),
                           "--out-dir", str(out / "probes" / model / task / context),
                           "--test-chrs", *fold["test"], "--val-chrs", *fold["validation"]]
                if task == "sv":
                    command += ["--examples", str(sw / "data/processed/hgsvc3_sv_breakpoint_examples_20260809/sv_breakpoint_examples.csv.gz"),
                                "--feature-cache", str(sw / "data/processed/hprc_r2_hgsvc3_sv_features_20260809.npz")]
                else:
                    command += ["--node-labels", str(sw / "data/downstream/ccre/hprc_r2_screen_v4/node_labels.csv.gz"),
                                "--feature-cache", str(sw / "data/processed/hprc_r2_ccre_screen_v4_features.npz")]
                commands.append(command)
    out.mkdir(parents=True)
    plan = {"status": "planned", "scope": "exploratory development control; no v2 promotion",
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "graph_sha256": digest, "fold": fold, "seed": args.seed,
            "random_initialization_seed": args.seed, "encoders_frozen": True,
            "topology_control_cache": str(topology), "topology_control_provenance": topology_provenance,
            "random_outputs_T_column_means": "R; random encoder, never trained topology",
            "commands": commands, "completed_commands": 0}
    status_path = out / "status.json"

    def save() -> None:
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(plan, indent=2) + "\n")
        temporary.replace(status_path)

    save()
    env = dict(os.environ, PYTHONPATH="src:.", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    try:
        for index, command in enumerate(commands):
            plan.update(status="running", active_command=index)
            save()
            with (out / f"command_{index:02d}.log").open("w") as handle:
                subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
            plan["completed_commands"] = index + 1
            save()
    except Exception as error:
        plan.update(status="failed", error=str(error))
        save()
        raise
    plan.update(status="complete")
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
