#!/usr/bin/env python3
"""Resumable, checksummed downloader for the server pangenome resource matrix.

The script never overwrites a completed file whose size or checksum disagrees
with the pinned manifest. Downloads go to ``*.part`` and are atomically renamed
only after validation. A machine-readable ledger records every verification.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class Resource:
    resource_id: str
    profiles: tuple[str, ...]
    dataset: str
    release: str
    role: str
    url: str
    relative_path: str
    expected_bytes: int | None
    checksum_type: str | None
    checksum: str | None
    compression: str
    required: bool
    notes: str


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_manifest(path: Path) -> list[Resource]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    resources: list[Resource] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        resource_id = row["resource_id"].strip()
        if not resource_id or resource_id in seen:
            raise ValueError(f"Invalid/duplicate resource_id at {path}:{line_number}")
        seen.add(resource_id)
        checksum_type = row["checksum_type"].strip().lower() or None
        checksum = row["checksum"].strip().lower() or None
        if bool(checksum_type) != bool(checksum):
            raise ValueError(f"Incomplete checksum at {path}:{line_number}")
        if checksum_type not in {None, "md5", "sha256"}:
            raise ValueError(f"Unsupported checksum type: {checksum_type}")
        resources.append(
            Resource(
                resource_id=resource_id,
                profiles=tuple(
                    value.strip()
                    for value in row["profiles"].split(",")
                    if value.strip()
                ),
                dataset=row["dataset"].strip(),
                release=row["release"].strip(),
                role=row["role"].strip(),
                url=row["url"].strip(),
                relative_path=row["relative_path"].strip(),
                expected_bytes=(
                    int(row["expected_bytes"])
                    if row["expected_bytes"].strip()
                    else None
                ),
                checksum_type=checksum_type,
                checksum=checksum,
                compression=row["compression"].strip().lower(),
                required=_truthy(row["required"]),
                notes=row["notes"].strip(),
            )
        )
    return resources


def select_profile(resources: list[Resource], profile: str) -> list[Resource]:
    selected = [resource for resource in resources if profile in resource.profiles]
    if not selected:
        profiles = sorted({profile for resource in resources for profile in resource.profiles})
        raise ValueError(f"Unknown/empty profile {profile!r}; choose from {profiles}")
    return selected


def hash_file(path: Path, *, need_md5: bool) -> dict[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5() if need_md5 else None  # noqa: S324 - release integrity checksum
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            sha256.update(chunk)
            if md5 is not None:
                md5.update(chunk)
    result = {"sha256": sha256.hexdigest()}
    if md5 is not None:
        result["md5"] = md5.hexdigest()
    return result


def verify(resource: Resource, path: Path, *, gzip_test: bool) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if resource.expected_bytes is not None and size != resource.expected_bytes:
        raise ValueError(
            f"{resource.resource_id}: expected {resource.expected_bytes} bytes, found {size}"
        )
    hashes = hash_file(path, need_md5=resource.checksum_type == "md5")
    if resource.checksum_type and hashes[resource.checksum_type] != resource.checksum:
        raise ValueError(
            f"{resource.resource_id}: {resource.checksum_type} mismatch; "
            f"expected {resource.checksum}, found {hashes[resource.checksum_type]}"
        )
    if gzip_test and resource.compression == "gzip":
        # gzip.open catches truncation and CRC disagreement without creating an
        # expanded copy. Reading in Python also works where `gzip` is absent.
        with gzip.open(path, "rb") as handle:
            while handle.read(16 * 1024 * 1024):
                pass
    stat = path.stat()
    return {
        "resource_id": resource.resource_id,
        "dataset": resource.dataset,
        "release": resource.release,
        "url": resource.url,
        "path": str(path),
        "bytes": size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashes["sha256"],
        "manifest_checksum_type": resource.checksum_type,
        "manifest_checksum": resource.checksum,
        "gzip_integrity_checked": bool(gzip_test and resource.compression == "gzip"),
        "status": "verified",
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def download_one(
    resource: Resource,
    *,
    data_root: Path,
    verify_only: bool,
    gzip_test: bool,
) -> dict[str, object]:
    destination = data_root / resource.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with PRINT_LOCK:
            print(f"[verify] {resource.resource_id}: {destination}", flush=True)
        return verify(resource, destination, gzip_test=gzip_test)
    if verify_only:
        raise FileNotFoundError(f"Missing required resource: {destination}")

    partial = destination.with_name(destination.name + ".part")
    command = [
        "curl",
        "--location",
        "--fail",
        "--show-error",
        "--retry",
        "8",
        "--retry-delay",
        "5",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        resource.url,
    ]
    with PRINT_LOCK:
        resumed = partial.stat().st_size if partial.exists() else 0
        print(
            f"[download] {resource.resource_id}: resume={resumed:,} -> {destination}",
            flush=True,
        )
    process = subprocess.run(command, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"curl failed ({process.returncode}) for {resource.resource_id}; "
            f"the resumable partial file remains at {partial}"
        )
    record = verify(resource, partial, gzip_test=gzip_test)
    partial.replace(destination)
    record["path"] = str(destination)
    with PRINT_LOCK:
        print(f"[complete] {resource.resource_id}: {destination}", flush=True)
    return record


def write_ledger(
    records: list[dict[str, object]], *, data_root: Path, manifest: Path, profile: str
) -> None:
    records = sorted(records, key=lambda row: str(row["resource_id"]))
    payload = {
        "schema_version": 1,
        "profile": profile,
        "manifest": str(manifest.resolve()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records": records,
    }
    json_path = data_root / "download_ledger.json"
    temp = json_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(json_path)
    csv_path = data_root / "download_ledger.csv"
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def print_resources(resources: list[Resource], data_root: Path) -> None:
    total = sum(resource.expected_bytes or 0 for resource in resources)
    unknown = sum(resource.expected_bytes is None for resource in resources)
    print("resource_id\tdataset\texpected_GiB\tdestination")
    for resource in resources:
        gib = "unknown" if resource.expected_bytes is None else f"{resource.expected_bytes / 2**30:.3f}"
        print(f"{resource.resource_id}\t{resource.dataset}\t{gib}\t{data_root / resource.relative_path}")
    print(f"Known total: {total / 2**30:.2f} GiB; unknown-size resources: {unknown}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "configs/server_full_data_manifest.tsv",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=["sv-core", "analysis-full", "archive-full-resolution"],
        default="analysis-full",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--skip-gzip-test",
        action="store_true",
        help="Skip decompression/CRC validation (not recommended for final runs).",
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=25.0,
        help="Free space that must remain after known outstanding downloads.",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    resources = select_profile(load_manifest(args.manifest), args.profile)
    data_root = args.data_root.expanduser().resolve()
    print_resources(resources, data_root)
    if args.list or args.dry_run:
        return 0
    if shutil.which("curl") is None and not args.verify_only:
        raise RuntimeError("curl is required for resumable downloads")
    data_root.mkdir(parents=True, exist_ok=True)
    outstanding = sum(
        resource.expected_bytes or 0
        for resource in resources
        if not (data_root / resource.relative_path).exists()
    )
    free = shutil.disk_usage(data_root).free
    required = outstanding + int(args.reserve_gib * 2**30)
    if free < required:
        raise RuntimeError(
            f"Insufficient free space: {free / 2**30:.1f} GiB free, "
            f"{required / 2**30:.1f} GiB required for known downloads plus reserve"
        )

    records: list[dict[str, object]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                download_one,
                resource,
                data_root=data_root,
                verify_only=args.verify_only,
                gzip_test=not args.skip_gzip_test,
            ): resource
            for resource in resources
        }
        for future in as_completed(futures):
            resource = futures[future]
            try:
                records.append(future.result())
            except Exception as error:  # keep independent downloads running
                message = f"{resource.resource_id}: {error}"
                failures.append(message)
                with PRINT_LOCK:
                    print(f"[failed] {message}", file=sys.stderr, flush=True)
    write_ledger(records, data_root=data_root, manifest=args.manifest, profile=args.profile)
    if failures:
        print("Download/verification failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"Verified {len(records)} resources; ledger: {data_root / 'download_ledger.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
