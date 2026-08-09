#!/usr/bin/env python3
"""Rematerialize a GFA as the union of retained embedded sample paths.

Deleting sample metadata from a GBWT/GBZ does not remove nodes or edges that
were contributed only by those assemblies.  For GFA files with W/P records,
this script performs the stronger operation needed for a shared-donor-excluded
stress test:

1. stream all retained W/P walks and collect their oriented path edges;
2. write only segments visited by those paths and links traversed by them;
3. omit path records from the derived training GFA while preserving headers.

An edge and its reverse-complement traversal are treated as the same bidirected
GFA link.  The audit records selected/excluded samples and requires every
retained link endpoint to be retained.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, TextIO


def open_text(path: Path) -> TextIO:
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" else path.open("r", encoding="utf-8", errors="replace")


def open_deterministic_output(path: Path) -> TextIO:
    if path.suffix != ".gz":
        return path.open("w", encoding="utf-8", newline="")
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def flip(orientation: str) -> str:
    if orientation == "+":
        return "-"
    if orientation == "-":
        return "+"
    raise ValueError(f"Invalid orientation: {orientation!r}")


def canonical_edge(
    left: str, left_orientation: str, right: str, right_orientation: str
) -> tuple[str, str, str, str]:
    direct = (left, left_orientation, right, right_orientation)
    reverse = (right, flip(right_orientation), left, flip(left_orientation))
    return min(direct, reverse)


def iter_walk(walk: str) -> Iterable[tuple[str, str]]:
    index = 0
    while index < len(walk):
        marker = walk[index]
        if marker not in "><":
            raise ValueError(f"Invalid W-line walk near {walk[index:index + 40]!r}")
        end = index + 1
        while end < len(walk) and walk[end] not in "><":
            end += 1
        name = walk[index + 1 : end]
        if not name:
            raise ValueError("Empty W-line segment")
        yield name, "+" if marker == ">" else "-"
        index = end


def iter_p_segments(value: str) -> Iterable[tuple[str, str]]:
    for token in value.split(","):
        token = token.strip()
        if len(token) < 2 or token[-1] not in "+-":
            raise ValueError(f"Invalid P-line segment token: {token!r}")
        yield token[:-1], token[-1]


def add_path_topology(
    handles: Iterable[tuple[str, str]],
    nodes: set[str],
    edges: set[tuple[str, str, str, str]],
) -> tuple[int, int]:
    handle_count = 0
    edge_count = 0
    previous: tuple[str, str] | None = None
    for current in handles:
        nodes.add(current[0])
        handle_count += 1
        if previous is not None:
            edges.add(canonical_edge(previous[0], previous[1], current[0], current[1]))
            edge_count += 1
        previous = current
    return handle_count, edge_count


def sample_from_p_name(path_name: str) -> str:
    return path_name.split("#", 1)[0] if "#" in path_name else path_name


def collect_retained_topology(
    source_gfa: Path,
    *,
    exclude_samples: set[str],
    include_samples: set[str] | None,
    include_p_paths: bool,
) -> tuple[set[str], set[tuple[str, str, str, str]], dict[str, object]]:
    nodes: set[str] = set()
    edges: set[tuple[str, str, str, str]] = set()
    selected_samples: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    with open_text(source_gfa) as handle:
        for line in handle:
            if line.startswith("W\t"):
                counters["w_records"] += 1
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 7:
                    counters["malformed_path_records"] += 1
                    continue
                sample = fields[1]
                if sample in exclude_samples or (include_samples is not None and sample not in include_samples):
                    excluded_counts[sample] += 1
                    continue
                try:
                    handles, traversals = add_path_topology(iter_walk(fields[6]), nodes, edges)
                except ValueError:
                    counters["malformed_path_records"] += 1
                    continue
                selected_samples[sample] += 1
                counters["selected_w_records"] += 1
                counters["selected_handles"] += handles
                counters["selected_edge_traversals"] += traversals
            elif include_p_paths and line.startswith("P\t"):
                counters["p_records"] += 1
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3:
                    counters["malformed_path_records"] += 1
                    continue
                sample = sample_from_p_name(fields[1])
                if sample in exclude_samples or (include_samples is not None and sample not in include_samples):
                    excluded_counts[sample] += 1
                    continue
                try:
                    handles, traversals = add_path_topology(iter_p_segments(fields[2]), nodes, edges)
                except ValueError:
                    counters["malformed_path_records"] += 1
                    continue
                selected_samples[sample] += 1
                counters["selected_p_records"] += 1
                counters["selected_handles"] += handles
                counters["selected_edge_traversals"] += traversals
    summary = {
        "counters": {str(key): int(value) for key, value in counters.items()},
        "selected_samples": sorted(selected_samples),
        "selected_path_records_by_sample": dict(sorted(selected_samples.items())),
        "excluded_path_records_by_sample": dict(sorted(excluded_counts.items())),
        "retained_unique_nodes_from_paths": len(nodes),
        "retained_unique_bidirected_edges_from_paths": len(edges),
    }
    return nodes, edges, summary


def filter_gfa(
    source_gfa: Path,
    output_gfa: Path,
    *,
    retained_nodes: set[str],
    retained_edges: set[tuple[str, str, str, str]],
) -> dict[str, int]:
    counters: Counter[str] = Counter()
    written_nodes: set[str] = set()
    dangling_links = 0
    with open_text(source_gfa) as source, open_deterministic_output(output_gfa) as output:
        for line in source:
            record_type = line[:1]
            counters[f"input_{record_type or 'blank'}"] += 1
            if record_type == "H":
                output.write(line)
                counters["written_H"] += 1
            elif record_type == "S":
                fields = line.split("\t", 2)
                if len(fields) >= 2 and fields[1] in retained_nodes:
                    output.write(line)
                    written_nodes.add(fields[1])
                    counters["written_S"] += 1
            elif record_type == "L":
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    counters["malformed_L"] += 1
                    continue
                edge = canonical_edge(fields[1], fields[2], fields[3], fields[4])
                if edge in retained_edges:
                    if fields[1] not in retained_nodes or fields[3] not in retained_nodes:
                        dangling_links += 1
                        continue
                    output.write(line)
                    counters["written_L"] += 1
            elif record_type in {"W", "P"}:
                counters["omitted_path_records"] += 1
            else:
                counters["omitted_other_records"] += 1
    counters["retained_nodes_missing_source_segment"] = len(retained_nodes - written_nodes)
    counters["dangling_links_suppressed"] = dangling_links
    return {str(key): int(value) for key, value in counters.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(
    *,
    source_gfa: Path,
    output_gfa: Path,
    audit_path: Path,
    exclude_samples: set[str],
    include_samples: set[str] | None,
    include_p_paths: bool,
) -> dict[str, object]:
    if output_gfa.exists() or audit_path.exists():
        raise FileExistsError("Refusing to overwrite an existing graph or audit")
    output_gfa.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    nodes, edges, path_summary = collect_retained_topology(
        source_gfa,
        exclude_samples=exclude_samples,
        include_samples=include_samples,
        include_p_paths=include_p_paths,
    )
    filter_summary = filter_gfa(
        source_gfa,
        output_gfa,
        retained_nodes=nodes,
        retained_edges=edges,
    )
    success = {
        "has_embedded_paths": bool(path_summary["counters"].get("selected_w_records", 0) or path_summary["counters"].get("selected_p_records", 0)),
        "has_nodes": filter_summary.get("written_S", 0) > 0,
        "has_links": filter_summary.get("written_L", 0) > 0,
        "all_path_nodes_present_in_source": filter_summary.get("retained_nodes_missing_source_segment", 0) == 0,
        "no_dangling_links": filter_summary.get("dangling_links_suppressed", 0) == 0,
        "all_requested_exclusions_observed": all(
            sample in path_summary["excluded_path_records_by_sample"] for sample in exclude_samples
        ),
    }
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "complete" if all(success.values()) else "audit_warning",
        "source_gfa": str(source_gfa.resolve()),
        "source_gfa_bytes": source_gfa.stat().st_size,
        "source_gfa_sha256": sha256_file(source_gfa),
        "output_gfa": str(output_gfa.resolve()),
        "output_gfa_bytes": output_gfa.stat().st_size,
        "output_gfa_sha256": sha256_file(output_gfa),
        "excluded_samples": sorted(exclude_samples),
        "explicit_included_samples": sorted(include_samples) if include_samples else None,
        "include_p_paths": include_p_paths,
        "path_scan": path_summary,
        "gfa_filter": filter_summary,
        "success_criteria": success,
        "wall_seconds": time.monotonic() - started,
        "interpretation": (
            "The derived graph is the union of links traversed by retained embedded paths. "
            "Unlike deleting GBWT metadata, it removes nodes/links unique to excluded paths."
        ),
    }
    audit_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gfa", type=Path, required=True)
    parser.add_argument("--output-gfa", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--exclude-samples", nargs="*", default=[])
    parser.add_argument("--include-samples-file", type=Path)
    parser.add_argument("--exclude-p-paths", action="store_true")
    args = parser.parse_args()
    include_samples = None
    if args.include_samples_file:
        include_samples = {
            line.strip()
            for line in args.include_samples_file.read_text().splitlines()
            if line.strip()
        }
    payload = materialize(
        source_gfa=args.source_gfa,
        output_gfa=args.output_gfa,
        audit_path=args.audit,
        exclude_samples=set(args.exclude_samples),
        include_samples=include_samples,
        include_p_paths=not args.exclude_p_paths,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
