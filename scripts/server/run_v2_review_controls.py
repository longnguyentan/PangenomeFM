#!/usr/bin/env python3
"""Run frozen v1 versus handcrafted/random controls in a separate output root.

Fold A is an exploratory development experiment, not an untouched final test.
This runner does not select or train a v2 model. Existing probes own all feature
joins, chromosome partitions, calibration, metrics and output validation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
from threading import Lock


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main-checkout", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--fold", default="fold_a")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--contexts", nargs="+", choices=["strict", "1hop"], default=["strict", "1hop"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--topology-control-cache", type=Path)
    ap.add_argument("--validation-only", action="store_true")
    ap.add_argument("--probe-max-iter", type=int, help="Uniform convergence sensitivity for every probe")
    ap.add_argument("--primary-features-only", action="store_true",
                    help="Use C+S, C+S+T, C+S+H, C+S+H+T for the bounded development comparison")
    ap.add_argument("--candidate-checkpoint", action="append", default=[], metavar="NAME=PATH",
                    help="Additional explicitly named checkpoints; requires one context")
    ap.add_argument("--models", nargs="+", help="Explicit subset; default runs every provided model")
    ap.add_argument('--companion-checkpoint', action='append', default=[], metavar='MODEL=PATH',
                    help='Append one independently frozen branch to an explicitly provided candidate')
    ap.add_argument("--extraction-candidate-policy", choices=["checkpoint", "manuscript"], default="checkpoint")
    ap.add_argument("--probe-gpus", type=int, nargs="+",
                    help="Run independent probes concurrently, one at a time per listed GPU")
    args = ap.parse_args()
    if args.probe_gpus and (len(set(args.probe_gpus)) != len(args.probe_gpus) or args.device != "cuda"):
        raise ValueError("Parallel probe GPUs must be unique and use --device cuda")
    candidates = []
    for specification in args.candidate_checkpoint:
        name, path = specification.split("=", 1)
        if not name.replace('_', '').replace('-', '').isalnum() or name in {'v1', 'random'}:
            raise ValueError("Use a unique simple candidate name distinct from v1/random")
        candidates.append((name, Path(path).resolve()))
        if not candidates[-1][1].is_file():
            raise FileNotFoundError(candidates[-1][1])
    if len({name for name, _ in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate checkpoint names")
    if candidates and len(args.contexts) != 1:
        raise ValueError("Candidate checkpoints require one explicit context")
    companions = {}
    for spec in args.companion_checkpoint:
        name, path = spec.split('=', 1)
        path = Path(path).resolve()
        if name in companions or name not in dict(candidates) or not path.is_file():
            raise ValueError('Companion must be one existing checkpoint for a declared candidate')
        if path == dict(candidates)[name]:
            raise ValueError('Duplicate branch does not add independent information')
        companions[name] = path
    selected_models = args.models or ["v1", "random", *[name for name, _ in candidates]]
    if len(set(selected_models)) != len(selected_models) or set(selected_models) - {"v1", "random", *[name for name, _ in candidates]}:
        raise ValueError("Unknown or duplicate model selection")
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
        if "random" in selected_models:
            commands.append([sys.executable, "scripts/make_random_init_checkpoint.py", "--checkpoint",
                             str(trained), "--out", str(random), "--seed", str(args.seed)])
        for model, checkpoint in [("v1", trained), ("random", random), *candidates]:
            if model not in selected_models:
                continue
            for task in ("sv", "ccre"):
                command = [sys.executable, f"scripts/server/run_{task}_frozen_probe_fold.py",
                           "--checkpoint", str(checkpoint), "--manifest",
                           str(sw / "data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv"),
                           "--full-segments", str(graph), "--fold", args.fold,
                           "--seed", str(args.seed), "--closure", context, "--device", args.device,
                           "--external-sequence-cache", str(nt), "--minimum-external-coverage", "1.0",
                           "--topology-control-cache", str(topology),
                           "--extraction-candidate-policy", args.extraction_candidate_policy,
                           "--out-dir", str(out / "probes" / model / task / context),
                           "--test-chrs", *fold["test"], "--val-chrs", *fold["validation"]]
                if task == "sv":
                    command += ["--examples", str(sw / "data/processed/hgsvc3_sv_breakpoint_examples_20260809/sv_breakpoint_examples.csv.gz"),
                                "--feature-cache", str(sw / "data/processed/hprc_r2_hgsvc3_sv_features_20260809.npz")]
                else:
                    command += ["--node-labels", str(sw / "data/downstream/ccre/hprc_r2_screen_v4/node_labels.csv.gz"),
                                "--feature-cache", str(sw / "data/processed/hprc_r2_ccre_screen_v4_features.npz")]
                if args.validation_only:
                    command.append("--validation-only")
                if args.probe_max_iter is not None:
                    command += ["--probe-max-iter", str(args.probe_max_iter)]
                if model in companions:
                    command += ['--companion-checkpoint', str(companions[model])]
                if args.primary_features_only:
                    suffix = "_pair" if task == "sv" else ""
                    primary = ["coordinate_plus_frozen_sequence_fm",
                               "coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm",
                               "coordinate_plus_frozen_sequence_fm_plus_topology_control",
                               "coordinate_plus_frozen_sequence_fm_plus_topology_control_plus_frozen_pangenomefm"]
                    command += ["--feature-sets", *[name + suffix for name in primary]]
                commands.append(command)
    out.mkdir(parents=True)
    plan = {"status": "planned", "scope": "exploratory development control; no v2 promotion",
            "probe_max_iter_override": args.probe_max_iter,
            "companion_checkpoints": {k: str(v) for k, v in companions.items()},
            "models": selected_models, "extraction_candidate_policy": args.extraction_candidate_policy,
            "evaluation_partition": "development_validation" if args.validation_only else "test",
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
    lock = Lock()
    completed = set()
    available = Queue()
    for gpu in args.probe_gpus or []:
        available.put(gpu)

    def execute(index: int, command: list[str], parallel: bool = False) -> None:
        gpu = available.get() if parallel else None
        child_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu)) if parallel else env
        with lock:
            plan.update(status="running", active_command=index)
            plan.setdefault("gpu_assignments", {})[str(index)] = gpu
            save()
        try:
            with (out / f"command_{index:02d}.log").open("w") as handle:
                subprocess.run(command, env=child_env, stdout=handle, stderr=subprocess.STDOUT, check=True)
            with lock:
                completed.add(index)
                plan.update(completed_commands=len(completed), completed_command_indices=sorted(completed))
                save()
        finally:
            if parallel:
                available.put(gpu)

    try:
        if args.probe_gpus:
            probes = []
            for index, command in enumerate(commands):
                if command[1] in {"scripts/server/run_ccre_frozen_probe_fold.py", "scripts/server/run_sv_frozen_probe_fold.py"}:
                    probes.append((index, command))
                else:
                    execute(index, command)
            with ThreadPoolExecutor(max_workers=len(args.probe_gpus)) as pool:
                futures = [pool.submit(execute, index, command, True) for index, command in probes]
                for future in futures:
                    future.result()
        else:
            for index, command in enumerate(commands):
                execute(index, command)
    except Exception as error:
        plan.update(status="failed", error=str(error))
        save()
        raise
    plan.update(status="complete")
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
