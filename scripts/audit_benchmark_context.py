#!/usr/bin/env python3
"""Audit graph-context exposure, interval matching, and split duplication.

This is intentionally independent of model checkpoints.  It reads a benchmark
manifest and its materialized slices, then writes manuscript-source CSV/JSON
files describing exactly how much graph/sequence context each task exposes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


_RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _canonical_hash(sequence: str) -> str:
    sequence = sequence.upper()
    reverse_complement = sequence.translate(_RC)[::-1]
    return hashlib.blake2b(min(sequence, reverse_complement).encode(), digest_size=16).hexdigest()


def _resolve(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    candidate = manifest_path.parent / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve {path_value!r} from {manifest_path}")


def _chromosome(target_sn: str) -> str:
    matches = re.findall(r"chr(?:[0-9]+|X|Y|M|MT)", str(target_sn), flags=re.IGNORECASE)
    return matches[-1] if matches else str(target_sn)


def _split_for_chromosome(chromosome: str, test: set[str], validation: set[str]) -> str:
    if chromosome in test:
        return "test"
    if chromosome in validation:
        return "validation"
    return "train"


def _interval_from_row(row: pd.Series, meta: dict[str, object]) -> tuple[int, int]:
    start = row.get("start", meta.get("start"))
    end = row.get("end", meta.get("end"))
    if pd.isna(start) or pd.isna(end):
        raise ValueError(f"Slice {row.get('slice_id', row.get('name'))} lacks start/end")
    return int(start), int(end)


def _graph_statistics(links: pd.DataFrame, seed: int) -> dict[str, float | int]:
    graph = nx.Graph()
    graph.add_edges_from(zip(links["from_seg"].astype(str), links["to_seg"].astype(str)))
    if graph.number_of_nodes() == 0:
        return {
            "visible_graph_nodes_incident": 0,
            "visible_graph_edges": 0,
            "connected_components": 0,
            "largest_component_fraction": np.nan,
            "branching_fraction": np.nan,
            "estimated_largest_component_diameter": np.nan,
            "estimated_graph_radius": np.nan,
        }
    components = list(nx.connected_components(graph))
    largest = max(components, key=len)
    largest_subgraph = graph.subgraph(largest)
    if len(largest) == 1:
        diameter = 0.0
    else:
        diameter = float(nx.approximation.diameter(largest_subgraph, seed=seed))
    degrees = np.fromiter((degree for _, degree in graph.degree()), dtype=float)
    return {
        "visible_graph_nodes_incident": int(graph.number_of_nodes()),
        "visible_graph_edges": int(graph.number_of_edges()),
        "connected_components": int(len(components)),
        "largest_component_fraction": float(len(largest) / graph.number_of_nodes()),
        "branching_fraction": float(np.mean(degrees > 2)),
        "estimated_largest_component_diameter": diameter,
        "estimated_graph_radius": diameter / 2.0,
    }


def audit_manifest(
    manifest_path: Path,
    out_dir: Path,
    test_chromosomes: set[str],
    validation_chromosomes: set[str],
    min_duplicate_length: int,
    seed: int,
) -> dict[str, object]:
    manifest = pd.read_csv(manifest_path)
    if "slice_id" not in manifest:
        manifest["slice_id"] = manifest["name"]
    duplicate_slice_ids = int(manifest["slice_id"].duplicated().sum())

    context_rows: list[dict[str, object]] = []
    intervals: list[dict[str, object]] = []
    sequence_hashes: dict[str, set[str]] = defaultdict(set)
    segment_names: dict[str, set[str]] = defaultdict(set)

    for row_index, row in manifest.iterrows():
        meta_path = _resolve(str(row["meta_path"]), manifest_path)
        segments_path = _resolve(str(row["segments_path"]), manifest_path)
        links_path = _resolve(str(row["links_path"]), manifest_path)
        edge_path = _resolve(str(row["edge_pred_path"]), manifest_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        segments = pd.read_csv(segments_path, compression="infer")
        links = pd.read_csv(links_path, compression="infer")
        edge_targets = pd.read_csv(edge_path, compression="infer")

        start, end = _interval_from_row(row, meta)
        target_sn = str(row["target_sn"])
        chromosome = _chromosome(target_sn)
        split = _split_for_chromosome(chromosome, test_chromosomes, validation_chromosomes)
        closure = str(row["closure"])
        context_regime = str(
            row.get(
                "context_regime",
                "core-node-induced-subgraph"
                if closure == "strict"
                else "endpoint-expanded-induced-subgraph",
            )
        )

        lengths = pd.to_numeric(segments["LN"], errors="coerce")
        positive_count = int((edge_targets["label"] == 1).sum())
        negative_count = int((edge_targets["label"] == 0).sum())
        candidate_count = int(len(edge_targets))
        default_test_count = int(candidate_count * 0.2)
        default_validation_count = int(candidate_count * 0.1)
        default_training_count = (
            candidate_count - default_test_count - default_validation_count
        )
        graph_stats = _graph_statistics(links, seed + row_index)
        context_rows.append(
            {
                "slice_id": row["slice_id"],
                "target_sn": target_sn,
                "chromosome": chromosome,
                "split": split,
                "closure_legacy_label": closure,
                "context_regime": context_regime,
                "start": start,
                "end": end,
                "target_interval_bp": end - start,
                "visible_nodes": int(len(segments)),
                "visible_link_rows": int(len(links)),
                "materialized_sequence_bp": int(lengths.fillna(0).sum()),
                "mean_materialized_node_bp": float(lengths.mean()),
                "maximum_materialized_node_bp": float(lengths.max()),
                "candidate_targets": candidate_count,
                "positive_targets": positive_count,
                "negative_targets": negative_count,
                "default_training_candidates": default_training_count,
                "default_validation_candidates": default_validation_count,
                "default_test_candidates": default_test_count,
                "default_loader_eligible": default_training_count >= 10,
                "positive_to_negative_ratio": (
                    positive_count / negative_count if negative_count else np.nan
                ),
                "positive_count_equals_visible_link_rows": positive_count == len(links),
                "target_edges_visible_in_unmasked_graph": True,
                **graph_stats,
            }
        )
        intervals.append(
            {
                "slice_id": row["slice_id"],
                "target_sn": target_sn,
                "closure": closure,
                "start": start,
                "end": end,
            }
        )

        for segment in segments.itertuples(index=False):
            name = str(getattr(segment, "name"))
            sequence = str(getattr(segment, "seq"))
            segment_names[split].add(name)
            if sequence not in {"", "*", "nan"} and len(sequence) >= min_duplicate_length:
                sequence_hashes[split].add(_canonical_hash(sequence))

    context = pd.DataFrame(context_rows)
    interval_frame = pd.DataFrame(intervals)
    overlap_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for (target_sn, closure), group in interval_frame.groupby(["target_sn", "closure"]):
        ordered = sorted(zip(group["start"], group["end"], group["slice_id"]))
        for (start_a, end_a, slice_a), (start_b, end_b, slice_b) in combinations(ordered, 2):
            overlap = max(0, min(end_a, end_b) - max(start_a, start_b))
            union = max(end_a, end_b) - min(start_a, start_b)
            overlap_rows.append(
                {
                    "target_sn": target_sn,
                    "closure": closure,
                    "slice_a": slice_a,
                    "slice_b": slice_b,
                    "overlap_bp": overlap,
                    "interval_jaccard": overlap / union if union else np.nan,
                }
            )
        merged: list[list[int]] = []
        for start, end, _ in ordered:
            if not merged or start >= merged[-1][1]:
                merged.append([int(start), int(end)])
            else:
                merged[-1][1] = max(merged[-1][1], int(end))
        nominal = int(sum(end - start for start, end, _ in ordered))
        unique = int(sum(end - start for start, end in merged))
        coverage_rows.append(
            {
                "target_sn": target_sn,
                "closure": closure,
                "windows": len(ordered),
                "nominal_bp": nominal,
                "unique_bp": unique,
                "redundancy_ratio": nominal / unique if unique else np.nan,
            }
        )

    match_rows: list[dict[str, object]] = []
    for target_sn, group in interval_frame.groupby("target_sn"):
        by_closure = {
            closure: set(zip(values["start"].astype(int), values["end"].astype(int)))
            for closure, values in group.groupby("closure")
        }
        closures = sorted(by_closure)
        for left, right in combinations(closures, 2):
            shared = by_closure[left] & by_closure[right]
            union = by_closure[left] | by_closure[right]
            match_rows.append(
                {
                    "target_sn": target_sn,
                    "closure_a": left,
                    "closure_b": right,
                    "intervals_a": len(by_closure[left]),
                    "intervals_b": len(by_closure[right]),
                    "exactly_matched_intervals": len(shared),
                    "interval_set_jaccard": len(shared) / len(union) if union else np.nan,
                }
            )

    leakage_rows: list[dict[str, object]] = []
    for left, right in combinations(["train", "validation", "test"], 2):
        seq_shared = sequence_hashes[left] & sequence_hashes[right]
        name_shared = segment_names[left] & segment_names[right]
        leakage_rows.append(
            {
                "split_a": left,
                "split_b": right,
                "segment_names_a": len(segment_names[left]),
                "segment_names_b": len(segment_names[right]),
                "shared_segment_names": len(name_shared),
                "canonical_sequence_hashes_a": len(sequence_hashes[left]),
                "canonical_sequence_hashes_b": len(sequence_hashes[right]),
                "shared_canonical_sequence_hashes": len(seq_shared),
                "fraction_split_b_sequence_hashes_shared": (
                    len(seq_shared) / len(sequence_hashes[right]) if sequence_hashes[right] else np.nan
                ),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    context.to_csv(out_dir / "context_statistics.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(out_dir / "window_overlap_pairs.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(out_dir / "window_coverage.csv", index=False)
    pd.DataFrame(match_rows).to_csv(out_dir / "closure_interval_matching.csv", index=False)
    pd.DataFrame(leakage_rows).to_csv(out_dir / "split_duplicate_audit.csv", index=False)

    aggregate = (
        context.groupby(["closure_legacy_label", "context_regime"], dropna=False)
        .agg(
            slices=("slice_id", "size"),
            mean_visible_nodes=("visible_nodes", "mean"),
            mean_visible_edges=("visible_link_rows", "mean"),
            mean_materialized_sequence_bp=("materialized_sequence_bp", "mean"),
            mean_estimated_graph_radius=("estimated_graph_radius", "mean"),
            mean_candidate_targets=("candidate_targets", "mean"),
            loader_eligible_slices=("default_loader_eligible", "sum"),
            all_positive_counts_equal_visible_links=(
                "positive_count_equals_visible_link_rows",
                "all",
            ),
        )
        .reset_index()
    )
    aggregate.to_csv(out_dir / "context_summary_by_regime.csv", index=False)
    summary = {
        "manifest": str(manifest_path),
        "slices": int(len(context)),
        "duplicate_slice_ids": duplicate_slice_ids,
        "test_chromosomes": sorted(test_chromosomes),
        "validation_chromosomes": sorted(validation_chromosomes),
        "all_target_edges_visible_before_query_masking": bool(
            context["target_edges_visible_in_unmasked_graph"].all()
        ),
        "all_positive_counts_equal_visible_link_rows": bool(
            context["positive_count_equals_visible_link_rows"].all()
        ),
        "default_loader_eligible_slices": int(
            context["default_loader_eligible"].sum()
        ),
        "default_loader_ineligible_slices": int(
            (~context["default_loader_eligible"]).sum()
        ),
        "exactly_matched_interval_pairs": int(
            pd.DataFrame(match_rows).get("exactly_matched_intervals", pd.Series(dtype=int)).sum()
        ),
        "outputs": sorted(path.name for path in out_dir.glob("*.csv")),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--test-chromosomes", nargs="+", default=["chr1", "chr8", "chr19", "chrY"])
    parser.add_argument("--validation-chromosomes", nargs="+", default=["chr16"])
    parser.add_argument("--min-duplicate-length", type=int, default=31)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = audit_manifest(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        test_chromosomes=set(args.test_chromosomes),
        validation_chromosomes=set(args.validation_chromosomes),
        min_duplicate_length=args.min_duplicate_length,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
