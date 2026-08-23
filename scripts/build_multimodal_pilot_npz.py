#!/usr/bin/env python3
"""Assemble an audited NPZ pilot for ``training.multimodal_pretrain``.

The builder joins frozen topology profiles, raw sequence tokens or frozen
sequence profiles, canonical edge examples, and optional path/branch/anchor
tables by stable node ID.  It writes every dropped example and never silently
changes a label or split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.multimodal_pangenome import MASK_TOKEN, PAD_TOKEN
from training.multimodal_pretrain import validate_arrays


SPLITS = {"train": 0, "validation": 1, "val": 1, "test": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t" if path.suffix in {".tsv", ".tab"} else ",", compression="infer")


def load_profiles(path: Path, value_key: str) -> tuple[np.ndarray, np.ndarray]:
    packed = np.load(path, allow_pickle=False)
    if not {"ids", value_key}.issubset(packed.files):
        raise ValueError(f"{path} requires arrays ids and {value_key}")
    ids = packed["ids"].astype(str)
    values = packed[value_key]
    if len(ids) != len(values) or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path} IDs must be unique and aligned to {value_key}")
    if value_key == "embeddings" and (values.ndim != 2 or not np.isfinite(values).all()):
        raise ValueError(f"{path} embeddings must be finite [nodes, dimensions]")
    if value_key == "tokens" and (values.ndim != 2 or values.min() < PAD_TOKEN or values.max() > MASK_TOKEN):
        raise ValueError(f"{path} tokens must be [nodes, length] using vocabulary 0..{MASK_TOKEN}")
    return ids, values


def split_codes(values: pd.Series, name: str) -> np.ndarray:
    normalized = values.astype(str).str.lower().map(SPLITS)
    if normalized.isna().any():
        invalid = sorted(values[normalized.isna()].astype(str).unique())
        raise ValueError(f"{name} contains invalid split labels: {invalid}")
    return normalized.to_numpy(np.int64)


def masked_tokens(
    tokens: np.ndarray, node_split: np.ndarray, rate: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < rate < 1:
        raise ValueError("mask rate must be between zero and one")
    rng = np.random.default_rng(seed)
    masked = tokens.astype(np.int64, copy=True)
    labels = np.full_like(masked, -100)
    for node in range(len(masked)):
        positions = np.flatnonzero(masked[node] != PAD_TOKEN)
        if not len(positions):
            continue
        count = max(1, int(round(rate * len(positions))))
        selected = rng.choice(positions, size=min(count, len(positions)), replace=False)
        labels[node, selected] = masked[node, selected]
        masked[node, selected] = MASK_TOKEN
    return masked, labels


def factorize(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    categories = sorted(values.fillna("UNK").astype(str).unique())
    lookup = {value: index for index, value in enumerate(categories)}
    return values.fillna("UNK").astype(str).map(lookup).to_numpy(np.int64), categories


def build_paths(
    table: pd.DataFrame,
    *,
    lookup: dict[str, int],
    path_id_column: str,
    node_id_column: str,
    position_column: str,
    split_column: str,
    max_path_length: int,
    min_subpath_nodes: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    required = {path_id_column, node_id_column, position_column, split_column}
    missing = required - set(table)
    if missing:
        raise ValueError(f"Path membership table is missing: {sorted(missing)}")
    sequences: list[list[int]] = []
    pairs: list[tuple[int, int, int, int]] = []
    failures: list[dict[str, Any]] = []
    for path_id, group in table.groupby(path_id_column, sort=True):
        splits = group[split_column].astype(str).str.lower().unique()
        if len(splits) != 1 or splits[0] not in SPLITS:
            failures.append({"kind": "path", "id": str(path_id), "reason": "ambiguous_or_invalid_split"})
            continue
        ordered_ids = group.sort_values(position_column, kind="stable")[node_id_column].astype(str).tolist()
        if any(value not in lookup for value in ordered_ids):
            failures.append({"kind": "path", "id": str(path_id), "reason": "node_embedding_missing"})
            continue
        nodes = [lookup[value] for value in ordered_ids]
        if len(nodes) < 2 * min_subpath_nodes:
            failures.append({"kind": "path", "id": str(path_id), "reason": "too_short"})
            continue
        midpoint = len(nodes) // 2
        left = nodes[max(0, midpoint - max_path_length) : midpoint]
        right = nodes[midpoint : midpoint + max_path_length]
        left_index = len(sequences)
        sequences.extend([left, right])
        split = SPLITS[splits[0]]
        pairs.extend([(left_index, left_index + 1, 1, split), (left_index + 1, left_index, 0, split)])
    if not sequences:
        return {}, failures
    width = max(map(len, sequences))
    path_nodes = np.full((len(sequences), width), -1, dtype=np.int64)
    for index, sequence in enumerate(sequences):
        path_nodes[index, : len(sequence)] = sequence
    return {
        "path_nodes": path_nodes,
        "path_pair_left": np.array([row[0] for row in pairs], dtype=np.int64),
        "path_pair_right": np.array([row[1] for row in pairs], dtype=np.int64),
        "path_order_labels": np.array([row[2] for row in pairs], dtype=np.float32),
        "path_order_split": np.array([row[3] for row in pairs], dtype=np.int64),
    }, failures


def build_branches(
    table: pd.DataFrame,
    *,
    lookup: dict[str, int],
    group_column: str,
    source_column: str,
    candidate_column: str,
    label_column: str,
    split_column: str,
    choices: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    required = {group_column, source_column, candidate_column, label_column, split_column}
    missing = required - set(table)
    if missing:
        raise ValueError(f"Branch table is missing: {sorted(missing)}")
    sources: list[int] = []
    candidates: list[list[int]] = []
    labels: list[int] = []
    splits: list[int] = []
    failures: list[dict[str, Any]] = []
    for group_id, group in table.groupby(group_column, sort=True):
        source_ids = group[source_column].astype(str).unique()
        split_values = group[split_column].astype(str).str.lower().unique()
        positives = group.loc[pd.to_numeric(group[label_column], errors="coerce").eq(1)]
        negatives = group.loc[pd.to_numeric(group[label_column], errors="coerce").eq(0)]
        if len(source_ids) != 1 or len(split_values) != 1 or split_values[0] not in SPLITS:
            failures.append({"kind": "branch", "id": str(group_id), "reason": "ambiguous_source_or_split"})
            continue
        if len(positives) != 1 or len(negatives) < choices - 1:
            failures.append({"kind": "branch", "id": str(group_id), "reason": "requires_one_positive_and_enough_negatives"})
            continue
        selected = pd.concat([positives, negatives.head(choices - 1)], ignore_index=True)
        node_ids = [source_ids[0], *selected[candidate_column].astype(str).tolist()]
        if any(value not in lookup for value in node_ids):
            failures.append({"kind": "branch", "id": str(group_id), "reason": "node_embedding_missing"})
            continue
        sources.append(lookup[source_ids[0]])
        candidates.append([lookup[value] for value in selected[candidate_column].astype(str)])
        labels.append(0)
        splits.append(SPLITS[split_values[0]])
    if not sources:
        return {}, failures
    return {
        "branch_sources": np.asarray(sources, dtype=np.int64),
        "branch_candidates": np.asarray(candidates, dtype=np.int64),
        "branch_labels": np.asarray(labels, dtype=np.int64),
        "branch_split": np.asarray(splits, dtype=np.int64),
    }, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-npz", type=Path, required=True)
    parser.add_argument("--sequence-npz", type=Path, required=True)
    parser.add_argument("--sequence-key", choices=["tokens", "embeddings"], required=True)
    parser.add_argument("--edge-table", type=Path, required=True)
    parser.add_argument("--node-table", type=Path)
    parser.add_argument("--path-membership", type=Path)
    parser.add_argument("--branch-table", type=Path)
    parser.add_argument("--cross-graph-anchors", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--node-id-column", default="node_id")
    parser.add_argument("--edge-left-column", default="source_segment_name")
    parser.add_argument("--edge-right-column", default="destination_segment_name")
    parser.add_argument("--edge-label-column", default="label")
    parser.add_argument("--edge-split-column", default="candidate_partition")
    parser.add_argument("--coordinate-columns", nargs="*", default=[])
    parser.add_argument("--population-column")
    parser.add_argument("--haplotype-column")
    parser.add_argument("--node-split-column")
    parser.add_argument("--path-id-column", default="path_id")
    parser.add_argument("--path-node-column", default="node_id")
    parser.add_argument("--path-position-column", default="position")
    parser.add_argument("--path-split-column", default="split")
    parser.add_argument("--max-path-length", type=int, default=128)
    parser.add_argument("--min-subpath-nodes", type=int, default=2)
    parser.add_argument("--branch-group-column", default="group_id")
    parser.add_argument("--branch-source-column", default="u_oid")
    parser.add_argument("--branch-candidate-column", default="candidate_v_oid")
    parser.add_argument("--branch-label-column", default="label")
    parser.add_argument("--branch-split-column", default="donor_split")
    parser.add_argument("--branch-choices", type=int, default=4)
    parser.add_argument("--anchor-left-column", default="left_node_id")
    parser.add_argument("--anchor-right-column", default="right_node_id")
    parser.add_argument("--anchor-split-column", default="split")
    parser.add_argument("--mask-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    topology_ids, topology = load_profiles(args.topology_npz, "embeddings")
    sequence_ids, sequence = load_profiles(args.sequence_npz, args.sequence_key)
    topology_lookup = {value: index for index, value in enumerate(topology_ids)}
    sequence_lookup = {value: index for index, value in enumerate(sequence_ids)}
    common = sorted(set(topology_lookup) & set(sequence_lookup))
    if not common:
        raise ValueError("Topology and sequence profiles share no node IDs")
    if args.require_complete and (len(common) != len(topology_ids) or len(common) != len(sequence_ids)):
        raise ValueError("Topology and sequence node universes differ under --require-complete")
    lookup = {value: index for index, value in enumerate(common)}
    arrays: dict[str, np.ndarray] = {
        "topology_embeddings": np.stack([topology[topology_lookup[value]] for value in common]).astype(np.float32)
    }
    if args.sequence_key == "tokens":
        raw_tokens = np.stack([sequence[sequence_lookup[value]] for value in common]).astype(np.int64)
        node_split = np.zeros(len(common), dtype=np.int64)
        arrays["sequence_tokens"], arrays["masked_token_labels"] = masked_tokens(
            raw_tokens, node_split, args.mask_rate, args.seed
        )
    else:
        arrays["sequence_features"] = np.stack([sequence[sequence_lookup[value]] for value in common]).astype(np.float32)
        node_split = np.zeros(len(common), dtype=np.int64)

    vocabularies: dict[str, list[str]] = {}
    if args.node_table:
        metadata = read_table(args.node_table)
        if args.node_id_column not in metadata or metadata[args.node_id_column].astype(str).duplicated().any():
            raise ValueError("Node metadata IDs must be present and unique")
        metadata = metadata.assign(_node_id=metadata[args.node_id_column].astype(str)).set_index("_node_id")
        if args.require_complete and not set(common).issubset(metadata.index):
            raise ValueError("Node metadata does not cover the complete aligned universe")
        metadata = metadata.reindex(common)
        if args.coordinate_columns:
            coordinates = metadata[args.coordinate_columns].apply(pd.to_numeric, errors="coerce")
            if coordinates.isna().any().any():
                raise ValueError("Coordinate columns contain missing or non-numeric values")
            arrays["coordinates"] = coordinates.to_numpy(np.float32)
        if args.population_column:
            arrays["population_ids"], vocabularies["population"] = factorize(metadata[args.population_column])
        if args.haplotype_column:
            arrays["haplotype_ids"], vocabularies["haplotype"] = factorize(metadata[args.haplotype_column])
        if args.node_split_column:
            node_split = split_codes(metadata[args.node_split_column], args.node_split_column)
            arrays["node_split"] = node_split
            if "sequence_tokens" in arrays:
                raw_tokens = np.stack([sequence[sequence_lookup[value]] for value in common]).astype(np.int64)
                arrays["sequence_tokens"], arrays["masked_token_labels"] = masked_tokens(
                    raw_tokens, node_split, args.mask_rate, args.seed
                )

    failures: list[dict[str, Any]] = []
    edges = read_table(args.edge_table)
    required = {args.edge_left_column, args.edge_right_column, args.edge_label_column, args.edge_split_column}
    missing = required - set(edges)
    if missing:
        raise ValueError(f"Edge table is missing: {sorted(missing)}")
    valid = edges[args.edge_left_column].astype(str).isin(lookup) & edges[args.edge_right_column].astype(str).isin(lookup)
    for index in edges.index[~valid]:
        failures.append({"kind": "edge", "id": str(index), "reason": "node_embedding_missing"})
    if args.require_complete and not valid.all():
        raise ValueError(f"Only {int(valid.sum())}/{len(edges)} edges map to aligned profiles")
    edges = edges.loc[valid].copy()
    labels = pd.to_numeric(edges[args.edge_label_column], errors="coerce")
    if not labels.isin([0, 1]).all():
        raise ValueError("Edge labels must be binary")
    arrays.update(
        {
            "edge_u": edges[args.edge_left_column].astype(str).map(lookup).to_numpy(np.int64),
            "edge_v": edges[args.edge_right_column].astype(str).map(lookup).to_numpy(np.int64),
            "edge_labels": labels.to_numpy(np.float32),
            "edge_split": split_codes(edges[args.edge_split_column], args.edge_split_column),
        }
    )

    if args.path_membership:
        path_arrays, path_failures = build_paths(
            read_table(args.path_membership),
            lookup=lookup,
            path_id_column=args.path_id_column,
            node_id_column=args.path_node_column,
            position_column=args.path_position_column,
            split_column=args.path_split_column,
            max_path_length=args.max_path_length,
            min_subpath_nodes=args.min_subpath_nodes,
        )
        arrays.update(path_arrays)
        failures.extend(path_failures)
    if args.branch_table:
        branch_arrays, branch_failures = build_branches(
            read_table(args.branch_table),
            lookup=lookup,
            group_column=args.branch_group_column,
            source_column=args.branch_source_column,
            candidate_column=args.branch_candidate_column,
            label_column=args.branch_label_column,
            split_column=args.branch_split_column,
            choices=args.branch_choices,
        )
        arrays.update(branch_arrays)
        failures.extend(branch_failures)
    if args.cross_graph_anchors:
        anchors = read_table(args.cross_graph_anchors)
        required = {args.anchor_left_column, args.anchor_right_column, args.anchor_split_column}
        missing = required - set(anchors)
        if missing:
            raise ValueError(f"Anchor table is missing: {sorted(missing)}")
        valid = anchors[args.anchor_left_column].astype(str).isin(lookup) & anchors[args.anchor_right_column].astype(str).isin(lookup)
        for index in anchors.index[~valid]:
            failures.append({"kind": "anchor", "id": str(index), "reason": "node_embedding_missing"})
        anchors = anchors.loc[valid]
        arrays.update(
            {
                "cross_graph_left_nodes": anchors[args.anchor_left_column].astype(str).map(lookup).to_numpy(np.int64),
                "cross_graph_right_nodes": anchors[args.anchor_right_column].astype(str).map(lookup).to_numpy(np.int64),
                "cross_graph_split": split_codes(anchors[args.anchor_split_column], args.anchor_split_column),
            }
        )

    validate_arrays(arrays)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "multimodal_pilot.npz", **arrays)
    pd.DataFrame(failures, columns=["kind", "id", "reason"]).to_csv(
        args.out_dir / "conversion_failures.tsv", sep="\t", index=False
    )
    audit = {
        "status": "complete",
        "aligned_nodes": len(common),
        "topology_nodes": len(topology_ids),
        "sequence_nodes": len(sequence_ids),
        "edge_rows": len(edges),
        "failure_rows": len(failures),
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
        "vocabularies": vocabularies,
        "inputs": {
            "topology_npz": {"path": str(args.topology_npz.resolve()), "sha256": sha256_file(args.topology_npz)},
            "sequence_npz": {"path": str(args.sequence_npz.resolve()), "sha256": sha256_file(args.sequence_npz)},
            "edge_table": {"path": str(args.edge_table.resolve()), "sha256": sha256_file(args.edge_table)},
        },
        "output": str((args.out_dir / "multimodal_pilot.npz").resolve()),
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

