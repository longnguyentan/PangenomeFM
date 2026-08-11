#!/usr/bin/env python3
"""Build a minimal, GBZ-compatible segment index from a full GFA.

Path materialization needs only the original GFA segment name, a deterministic
integer identifier, and segment length. Running the general graph parser on a
large small-variant GFA would also serialize sequences and links that this task
never reads. This streaming utility writes only ``id,name,LN`` and audits
malformed, duplicate, zero-length, or length-inconsistent segments.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import TextIO


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def open_deterministic_gzip_text(path: Path) -> TextIO:
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def segment_length(sequence: str, tags: list[str]) -> tuple[int, bool]:
    tagged_length: int | None = None
    for tag in tags:
        if tag.startswith("LN:i:"):
            try:
                tagged_length = int(tag[5:])
            except ValueError:
                tagged_length = None
            break
    sequence_length = None if sequence == "*" else len(sequence)
    if tagged_length is not None:
        return tagged_length, sequence_length is not None and tagged_length != sequence_length
    return sequence_length or 0, False


def build_index(
    *,
    gfa: Path,
    output: Path,
    audit: Path,
    max_segments: int | None = None,
) -> dict[str, object]:
    if output.exists() or audit.exists():
        raise FileExistsError("Refusing to overwrite an existing index or audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    started = time.monotonic()
    previous_numeric_name: int | None = None
    all_names_numeric = True
    strictly_increasing_numeric_names = True
    counters = {
        "lines_read": 0,
        "segments": 0,
        "paths": 0,
        "walks": 0,
        "malformed_segments": 0,
        "adjacent_duplicate_segment_names": 0,
        "zero_length_segments": 0,
        "sequence_ln_disagreements": 0,
    }
    try:
        with open_text(gfa) as source, open_deterministic_gzip_text(temporary) as target:
            writer = csv.DictWriter(target, fieldnames=["id", "name", "LN"])
            writer.writeheader()
            for line in source:
                counters["lines_read"] += 1
                if line.startswith("P\t"):
                    counters["paths"] += 1
                    continue
                if line.startswith("W\t"):
                    counters["walks"] += 1
                    continue
                if not line.startswith("S\t"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3 or not fields[1]:
                    counters["malformed_segments"] += 1
                    continue
                name = fields[1]
                try:
                    numeric_name = int(name)
                except ValueError:
                    all_names_numeric = False
                    numeric_name = None
                if numeric_name is not None and previous_numeric_name is not None:
                    if numeric_name == previous_numeric_name:
                        counters["adjacent_duplicate_segment_names"] += 1
                        continue
                    if numeric_name < previous_numeric_name:
                        strictly_increasing_numeric_names = False
                if numeric_name is not None:
                    previous_numeric_name = numeric_name
                length, disagrees = segment_length(fields[2], fields[3:])
                counters["zero_length_segments"] += int(length <= 0)
                counters["sequence_ln_disagreements"] += int(disagrees)
                writer.writerow({"id": counters["segments"], "name": name, "LN": length})
                counters["segments"] += 1
                if counters["segments"] % 1_000_000 == 0:
                    print(f"[segment-index] {counters['segments']:,} segments", flush=True)
                if max_segments is not None and counters["segments"] >= max_segments:
                    break
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    success = {
        "has_segments": counters["segments"] > 0,
        "no_malformed_segments": counters["malformed_segments"] == 0,
        # Official Minigraph-Cactus GFAs use numeric segment identifiers in
        # increasing S-record order.  Auditing that invariant proves
        # uniqueness without retaining tens of millions of Python strings.
        "unique_segment_names": (
            all_names_numeric
            and strictly_increasing_numeric_names
            and counters["adjacent_duplicate_segment_names"] == 0
        ),
        "positive_lengths": counters["zero_length_segments"] == 0,
        "consistent_lengths": counters["sequence_ln_disagreements"] == 0,
    }
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "complete" if all(success.values()) else "audit_warning",
        "gfa": str(gfa.resolve()),
        "gfa_bytes": gfa.stat().st_size,
        "gfa_sha256": sha256_file(gfa),
        "output": str(output.resolve()),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "max_segments": max_segments,
        "counters": counters,
        "all_segment_names_numeric": all_names_numeric,
        "strictly_increasing_numeric_segment_names": strictly_increasing_numeric_names,
        "success_criteria": success,
        "wall_seconds": time.monotonic() - started,
        "scope": "minimal id/name/LN index; sequences and links intentionally omitted",
    }
    audit.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gfa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--max-segments", type=int)
    args = parser.parse_args()
    result = build_index(
        gfa=args.gfa,
        output=args.output,
        audit=args.audit,
        max_segments=args.max_segments,
    )
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
