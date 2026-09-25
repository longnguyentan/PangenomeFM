"""Evaluate a scaling checkpoint frozen using the original biological runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.server.run_ccre_frozen_probe_matrix import build_jobs, checkpoint_for
from tasks.entex.prepare import fingerprint
from tasks.entex.probe import FEATURES


def commands(args: argparse.Namespace, job, checkpoint: Path) -> dict[str, list[str]]:
    root = args.resource_root
    graph = root / "data/processed/hprc_r2_sv/full_segments.csv.gz"
    manifest = root / "data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv"
    features = root / "data/processed/hprc_r2_ccre_screen_v4_features.npz"
    sequence = (
        root
        / "results/frozen_sequence_fm_cache_20260815/hprc_target_union_sequence_fm.npz"
    )
    base = [
        "--checkpoint",
        str(checkpoint),
        "--manifest",
        str(manifest),
        "--full-segments",
        str(graph),
        "--fold",
        job.fold,
        "--test-chrs",
        *job.test,
        "--val-chrs",
        *job.validation,
        "--closure",
        job.closure,
        "--device",
        args.device,
        "--seed",
        str(job.seed),
        "--external-sequence-cache",
        str(sequence),
        "--canonical-conflict-policy",
        "exclude",
    ]
    suffix = Path(job.fold) / f"seed_{job.seed}" / job.closure
    results = {}
    for task in ["sv", "ccre"]:
        out = args.out_root / task / suffix
        command = [
            sys.executable,
            f"scripts/server/run_{task}_frozen_probe_fold.py",
            *base,
            "--out-dir",
            str(out),
        ]
        if task == "sv":
            command += [
                "--examples",
                str(
                    root
                    / "data/processed/hgsvc3_sv_breakpoint_examples_20260809/sv_breakpoint_examples.csv.gz"
                ),
                "--feature-cache",
                str(root / "data/processed/hprc_r2_hgsvc3_sv_features_20260809.npz"),
                "--feature-sets",
                *(f + "_pair" for f in FEATURES),
            ]
        else:
            command += [
                "--node-labels",
                str(root / "data/downstream/ccre/hprc_r2_screen_v4/node_labels.csv.gz"),
                "--feature-cache",
                str(features),
                "--feature-sets",
                *FEATURES,
            ]
        results[task] = command
    prepared = args.entex_root / "p2"
    results["ctcf"] = [
        sys.executable,
        "-m",
        "tasks.entex.probe",
        "--task",
        "p2",
        "--subtask",
        "ctcf",
        "--measurements",
        str(prepared / "ctcf_measurements.parquet"),
        "--loci",
        str(prepared / "ctcf_loci.parquet"),
        "--mapping-dir",
        str(prepared / "mapping_ctcf"),
        "--full-segments",
        str(graph),
        "--manifest",
        str(manifest),
        "--feature-cache",
        str(features),
        "--sequence-cache",
        str(sequence),
        "--results-root",
        str(args.checkpoint_root),
        "--out-root",
        str(args.out_root / "ctcf"),
        "--folds",
        job.fold,
        "--seeds",
        str(job.seed),
        "--contexts",
        job.closure,
        "--device",
        args.device,
        "--topology-cache-root",
        str(args.out_root / "topology_cache"),
        "--cache-all-reference-targets",
    ]
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--resource-root", type=Path, default=Path("server_workspace"))
    ap.add_argument("--entex-root", type=Path, default=Path("data/entex/v1"))
    ap.add_argument("--fold", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--context", choices=["strict", "1hop"], required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--tasks",
        nargs="+",
        choices=["sv", "ccre", "ctcf"],
        default=["sv", "ccre", "ctcf"],
    )
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(Path("configs/entex_v1.json").read_text())
    job = next(
        j
        for j in build_jobs(json.loads(Path(cfg["manuscript_config"]).read_text()))
        if (j.fold, j.seed, j.closure) == (args.fold, args.seed, args.context)
    )
    checkpoint = checkpoint_for(args.checkpoint_root, job)
    graph = args.resource_root / "data/processed/hprc_r2_sv/full_segments.csv.gz"
    if fingerprint(graph)["sha256"] != cfg["full_segments_sha256"]:
        raise ValueError("Scaling evaluation graph differs from canonical resource")
    plan = commands(args, job, checkpoint)
    suffix = Path(job.fold) / f"seed_{job.seed}" / job.closure
    logs = args.out_root / "execution" / suffix
    logs.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        out = args.out_root / task / suffix
        if (out / "audit.json").exists():
            audit = json.loads((out / "audit.json").read_text())
            if task == "ctcf":
                actual = audit.get("checkpoint", {}).get("sha256")
            else:
                actual = audit.get("checkpoint_sha256")
            if (
                audit.get("status") != "complete"
                or actual != fingerprint(checkpoint)["sha256"]
            ):
                raise ValueError(
                    "Existing frozen evaluation is incomplete or uses another checkpoint"
                )
            print(task, "verified existing", flush=True)
            continue
        metadata = dict(
            task=task,
            command=plan[task],
            checkpoint=fingerprint(checkpoint),
            biological_encoder_training=False,
            status="planned",
        )
        status = logs / f"{task}.json"
        status.write_text(json.dumps(metadata, indent=2) + "\n")
        print(" ".join(plan[task]), flush=True)
        if not args.execute:
            continue
        with (logs / f"{task}.log").open("w") as log:
            process = subprocess.run(
                plan[task], stdout=log, stderr=subprocess.STDOUT, check=False
            )
        metadata.update(
            status="complete" if process.returncode == 0 else "failed",
            return_code=process.returncode,
        )
        status.write_text(json.dumps(metadata, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(f"Frozen {task} evaluation failed; inspect {logs}")


if __name__ == "__main__":
    main()
