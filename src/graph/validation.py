"""Streaming validation for cleaned pangenome graph tables.

The project uses CSV/CSV.GZ tables derived from GFA/rGFA graphs.  This module
checks those tables without loading nucleotide sequence columns into a large
Pandas object.  It deliberately distinguishes fatal schema/endpoint problems
from descriptive warnings such as multiple chromosome components.

Example
-------
PYTHONPATH=src python -m graph.validation \
    --segments data/hprc/full_segments.csv \
    --links data/hprc/full_links.csv \
    --out-dir results/validation/hprc \
    --label hprc_r2 --max-node-length 1024 --sequence-sample-modulus 10
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import IO, Any

import numpy as np


SEGMENT_REQUIRED = {"id", "name", "seq", "LN", "SN", "SO", "SR"}
LINK_REQUIRED = {"from_seg", "from_orient", "to_seg", "to_orient", "overlap"}
VALID_ORIENTATIONS = {"+", "-"}

csv.field_size_limit(sys.maxsize)


def _open_text(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _hash_bytes(value: str) -> bytes:
    return hashlib.blake2b(value.encode("ascii", errors="replace"), digest_size=16).digest()


_RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _canonical_sequence_hash(sequence: str) -> bytes:
    reverse_complement = sequence.translate(_RC)[::-1]
    return _hash_bytes(min(sequence.upper(), reverse_complement.upper()))


def _quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "p95": None, "p99": None, "max": None}
    arr = np.asarray(values, dtype=np.int64)
    return {
        "min": int(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": int(arr.max()),
    }


def _donor_haplotype(sn: str) -> tuple[str | None, str | None]:
    """Parse common PanSN and HGSVC ``id=sample.hap|contig`` names."""
    token = str(sn)
    if token.startswith("id="):
        token = token[3:].split("|", 1)[0]
    else:
        token = token.split("#", 1)[0]
    if token in {"GRCh38", "CHM13", "SN", ""}:
        return None, None
    if token.endswith((".1", ".2")):
        return token[:-2], token[-1]
    if "#" in sn:
        parts = sn.split("#")
        if len(parts) > 1 and parts[1] in {"1", "2"}:
            return parts[0], parts[1]
    return token, None


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = bytearray(size)

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _issue(
    rows: list[dict[str, Any]],
    issue_type: str,
    count: int,
    severity: str,
    details: str,
) -> None:
    rows.append(
        {
            "issue_type": issue_type,
            "count": int(count),
            "severity": severity,
            "details": details,
        }
    )


def validate_graph_tables(
    *,
    segments_path: str | Path,
    links_path: str | Path,
    label: str,
    max_node_length: int = 1024,
    sequence_sample_modulus: int = 1,
    duplicate_sequence_min_length: int = 31,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one cleaned graph and return a summary plus issue rows.

    ``sequence_sample_modulus`` deterministically audits approximately one in
    N segment sequences by segment-name hash.  Use 1 for an exhaustive audit
    and 0 to skip sequence-duplicate checks.  Length and LN/sequence agreement
    are always checked for every segment.
    """
    segments_path = Path(segments_path)
    links_path = Path(links_path)
    if max_node_length < 1:
        raise ValueError("max_node_length must be positive")
    if sequence_sample_modulus < 0:
        raise ValueError("sequence_sample_modulus cannot be negative")

    issues: list[dict[str, Any]] = []
    segment_names: dict[str, int] = {}
    segment_lengths: list[int] = []
    segment_length_by_index: list[int | None] = []
    segment_id_seen: set[str] = set()
    sn_seen: set[str] = set()
    donors: set[str] = set()
    haplotypes: set[tuple[str, str]] = set()
    exact_sequence_hashes: Counter[bytes] = Counter()
    canonical_sequence_hashes: Counter[bytes] = Counter()
    stats: dict[str, Any] = {
        "label": label,
        "segments_path": str(segments_path),
        "links_path": str(links_path),
        "max_node_length_policy": int(max_node_length),
        "sequence_sample_modulus": int(sequence_sample_modulus),
        "duplicate_sequence_min_length": int(duplicate_sequence_min_length),
        "segments": 0,
        "links": 0,
    }

    duplicate_names = 0
    duplicate_ids = 0
    invalid_ln = 0
    sequence_length_mismatches = 0
    unknown_sequences = 0
    negative_coordinates = 0
    over_limit = 0
    audited_sequences = 0

    with _open_text(segments_path) as handle:
        reader = csv.DictReader(handle)
        missing = SEGMENT_REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Segment table missing columns: {sorted(missing)}")
        for row in reader:
            index = stats["segments"]
            stats["segments"] += 1
            name = str(row.get("name", ""))
            seg_id = str(row.get("id", ""))
            sequence = str(row.get("seq", ""))
            length = _as_int(row.get("LN"))
            coordinate = _as_int(row.get("SO"))
            sn = str(row.get("SN", ""))

            if name in segment_names:
                duplicate_names += 1
            else:
                segment_names[name] = index
            if seg_id in segment_id_seen:
                duplicate_ids += 1
            segment_id_seen.add(seg_id)
            sn_seen.add(sn)
            donor, haplotype = _donor_haplotype(sn)
            if donor:
                donors.add(donor)
            if donor and haplotype:
                haplotypes.add((donor, haplotype))

            if length is None or length < 0:
                invalid_ln += 1
                segment_length_by_index.append(None)
            else:
                segment_lengths.append(length)
                segment_length_by_index.append(length)
                over_limit += int(length > max_node_length)
                if sequence in {"", "*"}:
                    unknown_sequences += 1
                elif len(sequence) != length:
                    sequence_length_mismatches += 1
            if coordinate is None or coordinate < 0:
                negative_coordinates += 1

            if (
                sequence_sample_modulus > 0
                and sequence not in {"", "*"}
                and (length or 0) >= duplicate_sequence_min_length
                and int.from_bytes(_hash_bytes(name)[:8], "little") % sequence_sample_modulus == 0
            ):
                audited_sequences += 1
                exact_sequence_hashes[_hash_bytes(sequence.upper())] += 1
                canonical_sequence_hashes[_canonical_sequence_hash(sequence)] += 1

    n_segments = int(stats["segments"])
    union_find = _UnionFind(n_segments)
    degrees = np.zeros(n_segments, dtype=np.int64)
    duplicate_link_keys: set[tuple[str, str, str, str, str]] = set()
    duplicate_links = 0
    missing_endpoints = 0
    invalid_orientations = 0
    self_links = 0
    endpoint_length_mismatches = 0
    missing_examples: list[str] = []

    with _open_text(links_path) as handle:
        reader = csv.DictReader(handle)
        missing = LINK_REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Link table missing columns: {sorted(missing)}")
        for row in reader:
            stats["links"] += 1
            source = str(row.get("from_seg", ""))
            target = str(row.get("to_seg", ""))
            source_orientation = str(row.get("from_orient", ""))
            target_orientation = str(row.get("to_orient", ""))
            overlap = str(row.get("overlap", ""))
            key = (source, source_orientation, target, target_orientation, overlap)
            if key in duplicate_link_keys:
                duplicate_links += 1
            else:
                duplicate_link_keys.add(key)
            invalid_orientations += int(source_orientation not in VALID_ORIENTATIONS)
            invalid_orientations += int(target_orientation not in VALID_ORIENTATIONS)
            self_links += int(source == target)

            source_index = segment_names.get(source)
            target_index = segment_names.get(target)
            if source_index is None or target_index is None:
                missing_endpoints += 1
                if len(missing_examples) < 10:
                    missing_examples.append(f"{source}{source_orientation}->{target}{target_orientation}")
                continue
            union_find.union(source_index, target_index)
            degrees[source_index] += 1
            degrees[target_index] += 1
            declared_l1 = _as_int(row.get("L1"))
            declared_l2 = _as_int(row.get("L2"))
            source_length = segment_length_by_index[source_index]
            target_length = segment_length_by_index[target_index]
            if declared_l1 is not None and source_length is not None and declared_l1 != source_length:
                endpoint_length_mismatches += 1
            if declared_l2 is not None and target_length is not None and declared_l2 != target_length:
                endpoint_length_mismatches += 1

    component_sizes = Counter(union_find.find(i) for i in range(n_segments))
    exact_duplicate_records = sum(count - 1 for count in exact_sequence_hashes.values() if count > 1)
    rc_duplicate_records = sum(count - 1 for count in canonical_sequence_hashes.values() if count > 1)
    stats.update(
        {
            "unique_segment_names": len(segment_names),
            "unique_sequence_names": len(sn_seen),
            "donors": len(donors),
            "phased_haplotypes": len(haplotypes),
            "node_length": _quantiles(segment_lengths),
            "nodes_over_length_policy": int(over_limit),
            "nodes_over_length_policy_fraction": float(over_limit / n_segments) if n_segments else math.nan,
            "unknown_sequences": int(unknown_sequences),
            "sequence_length_mismatches": int(sequence_length_mismatches),
            "negative_or_invalid_coordinates": int(negative_coordinates),
            "duplicate_segment_names": int(duplicate_names),
            "duplicate_segment_ids": int(duplicate_ids),
            "missing_link_endpoints": int(missing_endpoints),
            "invalid_orientation_fields": int(invalid_orientations),
            "duplicate_link_rows": int(duplicate_links),
            "self_links": int(self_links),
            "endpoint_length_mismatches": int(endpoint_length_mismatches),
            "components": len(component_sizes),
            "largest_component_nodes": max(component_sizes.values(), default=0),
            "isolated_nodes": int(np.sum(degrees == 0)),
            "branching_nodes_degree_gt_2": int(np.sum(degrees > 2)),
            "degree": _quantiles(degrees.astype(int).tolist()),
            "sequence_duplicate_audit": {
                "audited_sequences": int(audited_sequences),
                "exact_duplicate_records": int(exact_duplicate_records),
                "exact_duplicate_groups": int(sum(v > 1 for v in exact_sequence_hashes.values())),
                "canonical_reverse_complement_duplicate_records": int(rc_duplicate_records),
                "canonical_reverse_complement_duplicate_groups": int(
                    sum(v > 1 for v in canonical_sequence_hashes.values())
                ),
            },
        }
    )

    checks = [
        ("duplicate_segment_names", duplicate_names, "error", "Segment names must be unique."),
        ("duplicate_segment_ids", duplicate_ids, "error", "Segment IDs should be unique."),
        ("invalid_lengths", invalid_ln, "error", "LN must be a non-negative integer."),
        (
            "sequence_length_mismatch",
            sequence_length_mismatches,
            "error",
            "Declared LN differs from materialized sequence length.",
        ),
        (
            "negative_or_invalid_coordinates",
            negative_coordinates,
            "error",
            "SO must be a non-negative integer.",
        ),
        ("missing_link_endpoints", missing_endpoints, "error", "; ".join(missing_examples)),
        (
            "invalid_orientation_fields",
            invalid_orientations,
            "error",
            "Only '+' and '-' are valid endpoint orientations.",
        ),
        ("duplicate_link_rows", duplicate_links, "warning", "Exact oriented link rows are repeated."),
        (
            "endpoint_length_mismatches",
            endpoint_length_mismatches,
            "warning",
            "Optional L1/L2 values differ from endpoint LN.",
        ),
        (
            "nodes_over_configured_length",
            over_limit,
            "warning",
            f"Nodes exceed the configured {max_node_length}-bp preprocessing threshold.",
        ),
        (
            "disconnected_components",
            max(len(component_sizes) - 1, 0),
            "info",
            "Multiple chromosomes/contigs normally create multiple components.",
        ),
        ("isolated_nodes", int(np.sum(degrees == 0)), "warning", "Segments are not incident to any link."),
        (
            "sampled_exact_sequence_duplicates",
            exact_duplicate_records,
            "info",
            "Exact duplicates among deterministically audited sequences.",
        ),
        (
            "sampled_reverse_complement_duplicates",
            max(rc_duplicate_records - exact_duplicate_records, 0),
            "info",
            "Additional canonical sequence/reverse-complement duplicates.",
        ),
    ]
    for issue_type, count, severity, details in checks:
        _issue(issues, issue_type, int(count), severity, details)

    fatal_count = sum(row["count"] for row in issues if row["severity"] == "error")
    stats["validation_status"] = "pass" if fatal_count == 0 else "fail"
    stats["fatal_issue_count"] = int(fatal_count)
    return stats, issues


def validate_path_metadata(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate a compact path/walk inventory produced by ``index-paths``.

    GFA ``W`` records may split one logical path into multiple coordinate
    intervals, so repeated path names are descriptive rather than fatal when
    path_start/path_end are present.  Exact repeated records remain errors.
    """
    path = Path(path)
    delimiter = "\t" if path.suffix == ".tsv" else ","
    path_ids: set[str] = set()
    path_records: set[tuple[str, str, str]] = set()
    duplicate_ids = 0
    duplicate_records = 0
    invalid_step_counts = 0
    invalid_haplotypes = 0
    rows = 0
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = set(reader.fieldnames or [])
        id_column = next(
            (candidate for candidate in ["path_name", "name", "#NAME"] if candidate in fields),
            None,
        )
        if id_column is None:
            raise ValueError("Path metadata must contain path_name or name")
        has_intervals = {"path_start", "path_end"}.issubset(fields)
        for row in reader:
            rows += 1
            path_id = str(row.get(id_column, ""))
            if path_id in path_ids:
                duplicate_ids += 1
            path_ids.add(path_id)
            record_key = (
                path_id,
                str(row.get("path_start", "")),
                str(row.get("path_end", "")),
            )
            if record_key in path_records:
                duplicate_records += 1
            path_records.add(record_key)
            step_count_column = "step_count" if "step_count" in row else "STEP_COUNT" if "STEP_COUNT" in row else None
            if step_count_column:
                value = _as_int(row.get(step_count_column))
                invalid_step_counts += int(value is None or value < 0)
            haplotype_column = "haplotype" if "haplotype" in row else "HAPLOTYPE" if "HAPLOTYPE" in row else None
            if haplotype_column and str(row.get(haplotype_column, "")) not in {"", "0", "1", "2"}:
                invalid_haplotypes += 1
    issues = [
        {
            "issue_type": "duplicate_path_identifiers",
            "count": duplicate_ids,
            "severity": "warning" if has_intervals else "error",
            "details": (
                "Repeated logical path names across W-record coordinate intervals."
                if has_intervals
                else "Path identifiers must be unique when no interval fields distinguish records."
            ),
        },
        {
            "issue_type": "duplicate_path_records",
            "count": duplicate_records,
            "severity": "error",
            "details": "Repeated path name plus path_start/path_end record.",
        },
        {
            "issue_type": "invalid_path_step_counts",
            "count": invalid_step_counts,
            "severity": "error",
            "details": "step_count must be non-negative when present.",
        },
        {
            "issue_type": "noncanonical_haplotype_labels",
            "count": invalid_haplotypes,
            "severity": "warning",
            "details": "Observed haplotype labels outside 0/1/2.",
        },
    ]
    summary = {
        "path": str(path),
        "rows": rows,
        "unique_path_identifiers": len(path_ids),
        "unique_path_records": len(path_records),
        "validation_status": "pass"
        if not any(row["count"] and row["severity"] == "error" for row in issues)
        else "fail",
    }
    return summary, issues


def write_validation_report(
    *,
    summary: dict[str, Any],
    issues: list[dict[str, Any]],
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (out_dir / "validation_issues.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["issue_type", "count", "severity", "details"])
        writer.writeheader()
        writer.writerows(issues)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--links", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-node-length", type=int, default=1024)
    parser.add_argument(
        "--sequence-sample-modulus",
        type=int,
        default=1,
        help="Audit sequences whose deterministic name hash is 0 mod N; 1=all, 0=skip.",
    )
    parser.add_argument("--duplicate-sequence-min-length", type=int, default=31)
    parser.add_argument("--path-metadata", default=None)
    args = parser.parse_args()

    summary, issues = validate_graph_tables(
        segments_path=args.segments,
        links_path=args.links,
        label=args.label,
        max_node_length=args.max_node_length,
        sequence_sample_modulus=args.sequence_sample_modulus,
        duplicate_sequence_min_length=args.duplicate_sequence_min_length,
    )
    if args.path_metadata:
        path_summary, path_issues = validate_path_metadata(args.path_metadata)
        summary["path_metadata"] = path_summary
        issues.extend(path_issues)
        if path_summary["validation_status"] == "fail":
            summary["validation_status"] = "fail"
    write_validation_report(summary=summary, issues=issues, out_dir=args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
