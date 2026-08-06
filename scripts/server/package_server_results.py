#!/usr/bin/env python3
"""Create a checksummed result bundle for transfer back from the server."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def expand(value):
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, str):
        item = os.path.expandvars(value)
        if "${" in item:
            raise ValueError(f"Unresolved environment variable: {value}")
        return item
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/server_full_multicohort_20260806.json")
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--exclude-predictions", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = expand(json.loads(config_path.read_text(encoding="utf-8")))
    root = Path(config["outputs"]["root"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    queue_phases = ["final", "folds", "transfer", "release"]
    queue_status = {}
    for phase in queue_phases:
        path = root / f"gpu_queue_{phase}_summary.json"
        if not path.exists():
            queue_status[phase] = "missing"
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        queue_status[phase] = {
            "jobs_requested": payload.get("jobs_requested"),
            "jobs_recorded": payload.get("jobs_recorded"),
            "failures": payload.get("failures"),
        }
    incomplete = [
        phase
        for phase, status in queue_status.items()
        if status == "missing"
        or int(status.get("failures", 1)) != 0
        or status.get("jobs_requested") != status.get("jobs_recorded")
    ]
    if args.require_complete and incomplete:
        raise RuntimeError(f"Cannot package as complete; incomplete phases: {incomplete}")

    snapshot = root / "server_handoff" / "code_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    sources = [
        *sorted((REPO_ROOT / "configs").glob("server_*")),
        config_path,
        REPO_ROOT / "environment-server.yml",
        REPO_ROOT / "docs/SERVER_FULL_DATA_EXECUTION.md",
    ]
    sources.extend(sorted((REPO_ROOT / "scripts/server").glob("*")))
    for source in sources:
        if source.is_file():
            relative = source.relative_to(REPO_ROOT)
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    inventory_rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "result_file_inventory" in path.name:
            continue
        if not args.include_checkpoints and path.suffix == ".pt":
            continue
        if args.exclude_predictions and "predictions" in path.name:
            continue
        inventory_rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    handoff = root / "server_handoff"
    handoff.mkdir(parents=True, exist_ok=True)
    inventory_path = handoff / "result_file_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(inventory_rows)
    manifest = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "analysis_id": config["analysis_id"],
        "results_root": str(root),
        "files": len(inventory_rows),
        "bytes": sum(int(row["bytes"]) for row in inventory_rows),
        "include_checkpoints": args.include_checkpoints,
        "include_predictions": not args.exclude_predictions,
        "queue_status": queue_status,
        "incomplete_phases": incomplete,
    }
    (handoff / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output:
        output = args.output.resolve()
    elif shutil.which("zstd"):
        output = root.parent / f"{root.name}_{timestamp}.tar.zst"
    else:
        output = root.parent / f"{root.name}_{timestamp}.tar.gz"
    command = ["tar"]
    if output.suffix == ".zst":
        command.extend(["--zstd", "-cf", str(output)])
    else:
        command.extend(["-czf", str(output)])
    if not args.include_checkpoints:
        command.append("--exclude=*.pt")
    if args.exclude_predictions:
        command.append("--exclude=*predictions*")
    command.extend(["-C", str(root.parent), root.name])
    subprocess.run(command, check=True)
    bundle_sha = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{bundle_sha}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({**manifest, "bundle": str(output), "bundle_sha256": bundle_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
