#!/usr/bin/env python3
"""Capture reproducibility and accelerator information as machine-readable JSON."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "networkx",
    "matplotlib",
    "seaborn",
    "psutil",
    "pyarrow",
    "pytest",
]


def _command(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    distributions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            distributions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            distributions[package] = None

    accelerators: dict[str, Any] = {
        "torch_importable": False,
        "cuda_available": False,
        "mps_built": False,
        "mps_available": False,
    }
    try:
        torch = importlib.import_module("torch")
        accelerators.update(
            {
                "torch_importable": True,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "mps_built": bool(torch.backends.mps.is_built()),
                "mps_available": bool(torch.backends.mps.is_available()),
            }
        )
    except ImportError:
        pass

    memory_bytes = None
    try:
        psutil = importlib.import_module("psutil")
        memory_bytes = int(psutil.virtual_memory().total)
    except ImportError:
        pass
    disk = shutil.disk_usage(Path.cwd())
    payload = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": __import__("os").cpu_count(),
        "physical_memory_bytes": memory_bytes,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "packages": distributions,
        "accelerators": accelerators,
        "external_tools": {
            name: {"path": shutil.which(name), "version": _command([name, "--version"])}
            for name in ["vg", "odgi", "bcftools", "samtools", "bedtools"]
        },
        "git": {
            "commit": _command(["git", "rev-parse", "HEAD"]),
            "branch": _command(["git", "branch", "--show-current"]),
            "status_porcelain_lines": len(
                (_command(["git", "status", "--porcelain"]) or "").splitlines()
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
