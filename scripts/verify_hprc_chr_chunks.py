#!/usr/bin/env python3
"""Verify that bounded GFA chunks cover every exact path in a path manifest."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import TextIO


SLICE_SUFFIX = re.compile(r"\[(\d+)-(\d+)\]$")


def _open(path: Path) -> TextIO:
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open()


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for left, right in sorted(intervals):
        if not merged or left > merged[-1][1]:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    return merged


def verify(
    path_list: Path, chunks_dir: Path, output: Path
) -> dict[str, object]:
    expected = {
        line.strip()
        for line in path_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    unsliced: set[str] = set()
    chunk_hits: dict[str, list[str]] = {}
    failures: list[str] = []
    chunks = sorted(chunks_dir.glob("chunk*.gfa.gz"))
    for chunk in chunks:
        matched = [name for name in expected if name in chunk.name]
        if not matched:
            failures.append(f"cannot map chunk filename to manifest: {chunk.name}")
            break
        longest = max(map(len, matched))
        active = {name for name in matched if len(name) == longest}
        if len(active) != 1:
            failures.append(f"ambiguous chunk target: {chunk.name}")
            break
        hits: set[str] = set()
        with _open(chunk) as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if not fields:
                    continue
                if fields[0] == "P" and len(fields) >= 3:
                    match = SLICE_SUFFIX.search(fields[1])
                    source = fields[1][: match.start()] if match else fields[1]
                    if source not in active:
                        continue
                    hits.add(source)
                    if match:
                        intervals[source].append(
                            (int(match.group(1)), int(match.group(2)))
                        )
                    else:
                        unsliced.add(source)
                elif fields[0] == "W" and len(fields) >= 7:
                    walk_source = f"{fields[1]}#{fields[2]}#{fields[3]}"
                    # GFA W records do not carry the GBZ phase-block suffix
                    # present in path metadata (for example, trailing "#0").
                    # The chunk filename identifies the sole extraction
                    # target, so restore that suffix only when its W base
                    # matches exactly.
                    targets = [
                        name
                        for name in active
                        if name == walk_source
                        or (
                            name.endswith("#0")
                            and name.rsplit("#", 1)[0] == walk_source
                        )
                    ]
                    if len(targets) == 1:
                        target = targets[0]
                        hits.add(target)
                        intervals[target].append(
                            (int(fields[4]), int(fields[5]))
                        )
        if hits:
            chunk_hits[chunk.name] = sorted(hits)

    per_path: dict[str, object] = {}
    for source in sorted(expected):
        merged = _merge(intervals.get(source, []))
        gaps = [
            [merged[idx - 1][1], merged[idx][0]]
            for idx in range(1, len(merged))
            if merged[idx][0] > merged[idx - 1][1]
        ]
        present = source in unsliced or bool(merged)
        if not present:
            failures.append(f"missing path {source}")
        if (
            source not in unsliced
            and merged
            and merged[0][0] != 0
            and source.count("#") >= 3
        ):
            failures.append(f"fragmented path does not start at zero: {source}")
        if gaps and source not in unsliced:
            failures.append(f"path has uncovered gaps: {source}")
        per_path[source] = {
            "present": present,
            "unsliced_record": source in unsliced,
            "merged_relative_intervals": merged,
            "gaps": gaps,
        }
    summary = {
        "task": "verify_exact_bounded_hprc_paths",
        "path_list": str(path_list),
        "chunks_dir": str(chunks_dir),
        "n_expected_paths": len(expected),
        "n_gfa_chunks": len(chunks),
        "n_chunks_with_target_path": len(chunk_hits),
        "failures": failures,
        "paths": per_path,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        raise ValueError("; ".join(failures))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-list", type=Path, required=True)
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.path_list, args.chunks_dir, args.out), indent=2))


if __name__ == "__main__":
    main()
