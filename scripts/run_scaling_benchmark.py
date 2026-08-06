#!/usr/bin/env python3
"""Run one-epoch CPU scaling probes on increasing graph-slice counts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import shutil
from pathlib import Path

import pandas as pd
import psutil

from evaluation.splits import normalize_chrom


def gpu_memory_mib(pid: int) -> int | None:
    """Return NVIDIA memory attributed to a process, when nvidia-smi is available."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    process = subprocess.run(
        [
            executable,
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        return None
    total = 0
    found = False
    for line in process.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) >= 2 and fields[0] == str(pid):
            try:
                total += int(fields[1])
                found = True
            except ValueError:
                pass
    return total if found else None


def run(args: argparse.Namespace) -> dict[str, object]:
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["closure"] == args.closure].copy()
    normalized = manifest["target_sn"].astype(str).map(normalize_chrom)
    train_chromosomes = {normalize_chrom(value) for value in args.train_chromosomes}
    train = manifest[normalized.isin(train_chromosomes)]
    validation = manifest[normalized == normalize_chrom(args.validation_chromosome)].head(
        args.fixed_evaluation_slices
    )
    test = manifest[normalized == normalize_chrom(args.test_chromosome)].head(
        args.fixed_evaluation_slices
    )
    if len(validation) < args.fixed_evaluation_slices or len(test) < args.fixed_evaluation_slices:
        raise ValueError("Insufficient fixed validation/test slices")
    if max(args.train_counts) > len(train):
        raise ValueError(f"Requested {max(args.train_counts)} training slices but only {len(train)} exist")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.source_root)
    for train_count in args.train_counts:
        subset = pd.concat([train.head(train_count), validation, test], ignore_index=True)
        subset_dir = args.out_dir / f"train_{train_count:03d}"
        subset_dir.mkdir(parents=True, exist_ok=True)
        subset_manifest = subset_dir / "manifest.csv"
        subset.to_csv(subset_manifest, index=False)
        result_dir = subset_dir / "results"
        command = [
            sys.executable,
            "-m",
            "training.pretrain",
            "--manifest",
            str(subset_manifest),
            "--full_segments",
            str(args.full_segments),
            "--out_dir",
            str(result_dir),
            "--closures",
            args.closure,
            "--hidden_dim",
            str(args.hidden_dim),
            "--n_heads",
            str(args.n_heads),
            "--n_layers",
            str(args.n_layers),
            "--epochs",
            str(args.epochs),
            "--patience",
            "1",
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--dual_stream",
            "--multiscale_rope",
            "--orientation_rope",
            "--adaptive_window",
            "--focal_loss",
            "--drop_edge",
            "--mask_query_edges",
            "--batch_size",
            str(args.batch_size),
            "--test_chrs",
            args.test_chromosome,
            "--val_chrs",
            args.validation_chromosome,
        ]
        if args.lazy_tensorize:
            command.append("--lazy_tensorize")
        start = time.perf_counter()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        monitored = psutil.Process(process.pid)
        maximum_resident_bytes = 0
        maximum_gpu_memory_mib = 0
        polls = 0
        while process.poll() is None:
            try:
                resident = monitored.memory_info().rss
                maximum_resident_bytes = max(maximum_resident_bytes, resident)
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                pass
            if args.device.startswith("cuda") and polls % 5 == 0:
                used_mib = gpu_memory_mib(process.pid)
                maximum_gpu_memory_mib = max(maximum_gpu_memory_mib, used_mib or 0)
            polls += 1
            time.sleep(0.1)
        stdout, stderr = process.communicate()
        wall_seconds = time.perf_counter() - start
        (subset_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (subset_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        rows.append(
            {
                "train_slices": train_count,
                "validation_slices": len(validation),
                "test_slices": len(test),
                "total_slices": len(subset),
                "visible_nodes_sum": int(subset["n_segments"].sum()),
                "visible_link_rows_sum": int(subset["n_links"].sum()),
                "wall_seconds": wall_seconds,
                "seconds_per_epoch": wall_seconds / args.epochs,
                "maximum_resident_bytes": maximum_resident_bytes or None,
                "maximum_gpu_memory_mib": maximum_gpu_memory_mib or None,
                "returncode": process.returncode,
                "command": " ".join(command),
                "subset_manifest": str(subset_manifest),
                "result_dir": str(result_dir),
            }
        )
        if process.returncode != 0:
            break

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "scaling_benchmark.csv", index=False)
    successful = frame[frame["returncode"] == 0]
    summary = {
        "device": args.device,
        "epochs_per_run": args.epochs,
        "closure": args.closure,
        "successful_runs": int(len(successful)),
        "requested_runs": len(args.train_counts),
        "benchmark_model_dimensions": {
            "hidden_dim": args.hidden_dim,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
        },
        "note": "Fixed validation/test slices are included in each wall-time measurement.",
    }
    if not successful.empty:
        largest = successful.sort_values("train_slices").iloc[-1]
        full_slices = int(len(manifest))
        summary["rough_full_run_projection"] = {
            "full_manifest_rows_all_closures": full_slices,
            "largest_measured_train_slices": int(largest["train_slices"]),
            "largest_measured_seconds_per_epoch": float(largest["seconds_per_epoch"]),
            "requested_full_epochs": args.project_epochs,
            "linear_projection_seconds": float(
                largest["seconds_per_epoch"]
                * (full_slices / int(largest["total_slices"]))
                * args.project_epochs
            ),
            "warning": "Planning estimate only; validate with a multi-epoch pilot because graph sizes and early stopping are nonlinear.",
        }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--full-segments", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--source-root", default=Path("src"), type=Path)
    parser.add_argument("--closure", default="1hop")
    parser.add_argument("--train-chromosomes", nargs="+", default=["id=CHM13|chr19", "id=CHM13|chr21"])
    parser.add_argument("--validation-chromosome", default="id=CHM13|chr22")
    parser.add_argument("--test-chromosome", default="id=CHM13|chrY")
    parser.add_argument("--fixed-evaluation-slices", type=int, default=2)
    parser.add_argument("--train-counts", nargs="+", type=int, default=[2, 5, 10, 20])
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--project-epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lazy-tensorize", action="store_true")
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
