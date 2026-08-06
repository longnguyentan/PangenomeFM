#!/usr/bin/env python3
"""Capture server resources and fail early on unsafe full-run conditions."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from download_full_data import load_manifest, select_profile


REPO_ROOT = Path(__file__).resolve().parents[2]
MINIMUM_FREE_GIB = {
    "sv-core": 250,
    "analysis-full": 500,
    "archive-full-resolution": 1500,
}


def command_version(command: str, arguments: list[str]) -> dict[str, object]:
    executable = shutil.which(command)
    if executable is None:
        return {"available": False, "path": None, "version": None}
    process = subprocess.run(
        [executable, *arguments], capture_output=True, text=True, check=False
    )
    first_line = (process.stdout or process.stderr).strip().splitlines()
    return {
        "available": process.returncode == 0,
        "path": executable,
        "version": first_line[0] if first_line else None,
        "returncode": process.returncode,
    }


def gpu_inventory() -> list[dict[str, object]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    fields = "index,name,memory.total,driver_version"
    process = subprocess.run(
        [executable, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        return []
    rows = []
    for line in process.stdout.strip().splitlines():
        index, name, memory_mib, driver = [item.strip() for item in line.split(",", 3)]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_mib": int(memory_mib),
                "driver_version": driver,
            }
        )
    return rows


def check_urls(resources, limit: int | None) -> list[dict[str, object]]:
    rows = []
    for resource in resources[:limit]:
        process = subprocess.run(
            [
                "curl", "--location", "--head", "--fail", "--silent",
                "--show-error", "--max-time", "30", resource.url,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        rows.append(
            {
                "resource_id": resource.resource_id,
                "url": resource.url,
                "reachable": process.returncode == 0,
                "returncode": process.returncode,
                "error": process.stderr.strip() or None,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "configs/server_full_data_manifest.tsv",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(MINIMUM_FREE_GIB),
        default="analysis-full",
    )
    parser.add_argument("--check-urls", action="store_true")
    parser.add_argument("--url-limit", type=int)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    resources = select_profile(load_manifest(args.manifest), args.profile)
    known_download_bytes = sum(resource.expected_bytes or 0 for resource in resources)
    unknown_size_resources = [
        resource.resource_id for resource in resources if resource.expected_bytes is None
    ]
    disk = shutil.disk_usage(data_root)
    gpus = gpu_inventory()
    tools = {
        "python": command_version(sys.executable, ["--version"]),
        "curl": command_version("curl", ["--version"]),
        "tmux": command_version("tmux", ["-V"]),
        "vg": command_version("vg", ["version"]),
        "zstd": command_version("zstd", ["--version"]),
        "git": command_version("git", ["--version"]),
    }
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
    except ImportError:
        torch_info = {"available": False}

    minimum_free_gib = MINIMUM_FREE_GIB[args.profile]
    warnings = []
    failures = []
    free_gib = disk.free / 2**30
    if free_gib < minimum_free_gib:
        failures.append(
            f"free disk {free_gib:.1f} GiB is below the {minimum_free_gib} GiB "
            f"minimum for profile {args.profile}"
        )
    if psutil.virtual_memory().total < 64 * 2**30:
        warnings.append("Less than 64 GiB RAM; reduce PREP_JOBS to 1 and benchmark memory first.")
    if (os.cpu_count() or 1) < 16:
        warnings.append("Fewer than 16 logical CPUs; parallel preparation will be I/O/CPU limited.")
    if not gpus:
        warnings.append("No NVIDIA GPU detected; full training is implemented but CPU-only execution is not practical.")
    elif max(int(gpu["memory_mib"]) for gpu in gpus) < 24 * 1024:
        warnings.append("No GPU has at least 24 GiB VRAM; lower candidate batch size after a measured pilot.")
    for required_tool in ("curl", "tmux", "git"):
        if not tools[required_tool]["available"]:
            failures.append(f"required tool is unavailable: {required_tool}")
    if not tools["vg"]["available"]:
        warnings.append("vg is unavailable; SV-GFA training can run, but GBZ path inventories cannot.")

    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "profile": args.profile,
        "data_root": str(data_root),
        "results_root": str(results_root),
        "resources": len(resources),
        "known_download_bytes": known_download_bytes,
        "unknown_size_resources": unknown_size_resources,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "minimum_free_gib": minimum_free_gib,
        },
        "cpu": {
            "logical_count": os.cpu_count(),
            "physical_count": psutil.cpu_count(logical=False),
        },
        "memory": {
            "total_bytes": psutil.virtual_memory().total,
            "available_bytes": psutil.virtual_memory().available,
        },
        "gpus": gpus,
        "torch": torch_info,
        "tools": tools,
        "url_checks": check_urls(resources, args.url_limit) if args.check_urls else [],
        "warnings": warnings,
        "failures": failures,
        "status": "failed" if failures else ("warning" if warnings else "ready"),
        "resource_guidance": {
            "preprocessing_cpu_threads": "16-32 logical CPUs across 2-4 independent datasets",
            "preprocessing_ram": "64 GiB minimum; 128 GiB recommended",
            "training_gpu": "one 24+ GiB NVIDIA GPU per concurrent run; benchmark candidate batch size before queueing",
            "training_runtime": "measured by the included scaling/pilot stage; do not extrapolate from the local CPU smoke test",
            "checkpoint_and_prediction_reserve": "at least 200 GiB within the profile-specific free-space minimum",
        },
    }
    output = results_root / "server_preflight.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, indent=2))
    print(f"Preflight report: {output}")
    return 2 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
