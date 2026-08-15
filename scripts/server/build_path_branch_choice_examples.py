#!/usr/bin/env python3
"""Build donor-disjoint focal path branch-choice candidate groups.

For each observed path transition ``u -> v``, negatives are other aggregate
graph edges leaving the same oriented source handle ``u``.  They are negatives
for that focal transition only, not claims that the edge is globally absent or
biologically invalid.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Iterable, TextIO

import pandas as pd

from scripts.server.materialize_path_resolved_examples import (
    open_deterministic_gzip_text,
    stable_rank,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def flip(orientation: str) -> str:
    if orientation == "+":
        return "-"
    if orientation == "-":
        return "+"
    raise ValueError(f"Invalid orientation: {orientation!r}")


def normalized_segment(value: object) -> str:
    text = str(value)
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def handle_oid(segment: str, orientation: str) -> int | None:
    if not segment.isdigit():
        return None
    return 2 * int(segment) + int(orientation == "-")


def directed_edges(fields: list[str]) -> Iterable[
    tuple[tuple[str, str], tuple[str, str]]
]:
    """Yield forward and reverse-complement traversals for one GFA L row."""

    if len(fields) < 5 or fields[0] != "L":
        return
    left = (fields[1], fields[2])
    right = (fields[3], fields[4])
    if left[1] not in {"+", "-"} or right[1] not in {"+", "-"}:
        return
    yield left, right
    yield (right[0], flip(right[1])), (left[0], flip(left[1]))


def select_positives(path: Path, max_positive_examples: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="infer")
    required = {
        "dataset",
        "donor_split",
        "sample",
        "haplotype",
        "locus",
        "chromosome",
        "path_name",
        "window_start",
        "window_end",
        "path_edge_index",
        "path_position",
        "u_segment",
        "u_orientation",
        "v_segment",
        "v_orientation",
        "deterministic_rank",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Positive path edges miss columns: {sorted(missing)}")
    frame = frame.copy()
    frame["u_segment"] = frame["u_segment"].map(normalized_segment)
    frame["v_segment"] = frame["v_segment"].map(normalized_segment)
    if not frame["u_orientation"].isin({"+", "-"}).all() or not frame[
        "v_orientation"
    ].isin({"+", "-"}).all():
        raise ValueError("Positive path edges contain invalid orientations")
    split_counts = frame.groupby("sample")["donor_split"].nunique()
    if (split_counts > 1).any():
        raise ValueError("A donor occurs in more than one inherited split")
    identity = ["path_name", "path_edge_index", "u_segment", "u_orientation"]
    if frame.duplicated(identity).any():
        raise ValueError("Positive transition identifiers are not unique")
    frame = frame.sort_values(
        ["deterministic_rank", "path_name", "path_edge_index"], kind="stable"
    ).reset_index(drop=True)
    if max_positive_examples is not None:
        frame = frame.head(max_positive_examples).copy()
    return frame


def collect_adjacency(
    *,
    gfa: Path,
    positive_targets: dict[tuple[str, str], set[tuple[str, str]]],
    alternatives_per_positive: int,
) -> tuple[dict[tuple[str, str], list[tuple[str, str]]], dict[str, int]]:
    sources = set(positive_targets)
    neighbors: dict[tuple[str, str], set[tuple[str, str]]] = {}
    counters = {
        "gfa_lines": 0,
        "gfa_link_rows": 0,
        "directed_traversals": 0,
        "directed_traversals_from_requested_sources": 0,
    }
    with open_text(gfa) as handle:
        for line in handle:
            counters["gfa_lines"] += 1
            if not line.startswith("L\t"):
                continue
            counters["gfa_link_rows"] += 1
            fields = line.rstrip("\n").split("\t")
            for source, target in directed_edges(fields):
                counters["directed_traversals"] += 1
                if source not in sources:
                    continue
                counters["directed_traversals_from_requested_sources"] += 1
                values = neighbors.setdefault(source, set())
                values.add(target)
                # Keep a deterministic bounded superset.  The additional
                # positive-target allowance guarantees enough candidates
                # remain after excluding a focal observed target.
                limit = alternatives_per_positive + len(positive_targets[source])
                if len(values) > max(2 * limit, 16):
                    preserved = values & positive_targets[source]
                    values_sorted = sorted(
                        values - preserved,
                        key=lambda target_value: (
                            stable_rank(*source, *target_value),
                            target_value,
                        ),
                    )
                    neighbors[source] = preserved | set(
                        values_sorted[: max(limit - len(preserved), 0)]
                    )
    result: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for source, values in neighbors.items():
        limit = alternatives_per_positive + len(positive_targets[source])
        preserved = values & positive_targets[source]
        ranked_other = sorted(
            values - preserved,
            key=lambda target: (stable_rank(*source, *target), target),
        )[: max(limit - len(preserved), 0)]
        result[source] = sorted(preserved) + ranked_other
    return result, counters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-edges", type=Path, required=True)
    parser.add_argument("--gfa", type=Path, required=True)
    parser.add_argument("--graph-audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alternatives-per-positive", type=int, default=5)
    parser.add_argument("--max-positive-examples", type=int)
    args = parser.parse_args()
    if args.alternatives_per_positive < 1:
        parser.error("--alternatives-per-positive must be positive")
    if args.max_positive_examples is not None and args.max_positive_examples < 1:
        parser.error("--max-positive-examples must be positive")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.out_dir}")

    graph_audit = json.loads(args.graph_audit.read_text(encoding="utf-8"))
    if graph_audit.get("status") != "complete":
        raise ValueError("Graph segment-index audit is not complete")
    expected_gfa_sha = graph_audit.get("gfa_sha256")
    actual_gfa_sha = sha256_file(args.gfa)
    if expected_gfa_sha != actual_gfa_sha:
        raise ValueError(
            f"GFA checksum differs from graph audit: {actual_gfa_sha} != {expected_gfa_sha}"
        )

    started = time.monotonic()
    positives = select_positives(args.positive_edges, args.max_positive_examples)
    positive_targets: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for row in positives.itertuples(index=False):
        source = (str(row.u_segment), str(row.u_orientation))
        positive_targets.setdefault(source, set()).add(
            (str(row.v_segment), str(row.v_orientation))
        )
    adjacency, graph_counters = collect_adjacency(
        gfa=args.gfa,
        positive_targets=positive_targets,
        alternatives_per_positive=args.alternatives_per_positive,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "branch_choice_candidates.csv.gz"
    fieldnames = [
        "group_id",
        "dataset",
        "donor_split",
        "sample",
        "haplotype",
        "locus",
        "chromosome",
        "path_name",
        "window_start",
        "window_end",
        "path_edge_index",
        "path_position",
        "u_segment",
        "u_orientation",
        "u_oid",
        "candidate_v_segment",
        "candidate_v_orientation",
        "candidate_v_oid",
        "label",
        "candidate_rank",
    ]
    candidate_groups = 0
    candidate_rows = 0
    negative_rows = 0
    no_alternative = 0
    observed_edge_absent_from_gfa = 0
    split_group_counts: dict[str, int] = {}
    with open_deterministic_gzip_text(output) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in positives.itertuples(index=False):
            source = (str(row.u_segment), str(row.u_orientation))
            observed = (str(row.v_segment), str(row.v_orientation))
            source_neighbors = adjacency.get(source, [])
            if observed not in source_neighbors:
                observed_edge_absent_from_gfa += 1
            alternatives = [value for value in source_neighbors if value != observed][
                : args.alternatives_per_positive
            ]
            if not alternatives:
                no_alternative += 1
                continue
            group_id = hashlib.blake2b(
                (
                    f"{row.path_name}\x1f{row.path_edge_index}\x1f"
                    f"{source[0]}\x1f{source[1]}\x1f{observed[0]}\x1f{observed[1]}"
                ).encode("utf-8"),
                digest_size=16,
            ).hexdigest()
            common = {
                "group_id": group_id,
                "dataset": row.dataset,
                "donor_split": row.donor_split,
                "sample": row.sample,
                "haplotype": row.haplotype,
                "locus": row.locus,
                "chromosome": row.chromosome,
                "path_name": row.path_name,
                "window_start": row.window_start,
                "window_end": row.window_end,
                "path_edge_index": row.path_edge_index,
                "path_position": row.path_position,
                "u_segment": source[0],
                "u_orientation": source[1],
                "u_oid": handle_oid(*source),
            }
            candidates = [(observed, 1), *[(value, 0) for value in alternatives]]
            for candidate_rank, (target, label) in enumerate(candidates):
                writer.writerow(
                    {
                        **common,
                        "candidate_v_segment": target[0],
                        "candidate_v_orientation": target[1],
                        "candidate_v_oid": handle_oid(*target),
                        "label": label,
                        "candidate_rank": candidate_rank,
                    }
                )
                candidate_rows += 1
                negative_rows += int(label == 0)
            candidate_groups += 1
            split = str(row.donor_split)
            split_group_counts[split] = split_group_counts.get(split, 0) + 1

    status = "complete"
    if candidate_groups == 0 or observed_edge_absent_from_gfa:
        status = "audit_warning"
    audit = {
        "schema_version": 1,
        "status": status,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "positive_edges": str(args.positive_edges.resolve()),
        "positive_edges_sha256": sha256_file(args.positive_edges),
        "gfa": str(args.gfa.resolve()),
        "gfa_sha256": actual_gfa_sha,
        "graph_audit": str(args.graph_audit.resolve()),
        "graph_audit_sha256": sha256_file(args.graph_audit),
        "selected_positive_rows": int(len(positives)),
        "unique_oriented_sources": int(len(positive_targets)),
        "candidate_groups": int(candidate_groups),
        "candidate_rows": int(candidate_rows),
        "negative_rows": int(negative_rows),
        "positive_rows_without_alternative": int(no_alternative),
        "observed_edges_absent_from_gfa": int(observed_edge_absent_from_gfa),
        "alternatives_per_positive": int(args.alternatives_per_positive),
        "donor_split_group_counts": split_group_counts,
        "graph_counters": graph_counters,
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "label_definition": "one observed next handle versus other aggregate-graph outgoing handles from the identical oriented source",
        "negative_scope": "focal transition only; a negative candidate may be valid on another path or at another occurrence",
        "leakage_policy": "donor_split is inherited unchanged; each donor was validated to occur in one split",
        "embedding_gate": "a PangenomeFM branch-choice probe must not run until candidate node identities are mapped to an audited frozen-embedding universe",
        "wall_seconds": time.monotonic() - started,
    }
    (args.out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
