#!/usr/bin/env python3
"""Measure how ENCODE cCRE intervals span GRCh38 graph nodes and path links."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _segment_number(values: pd.Series) -> np.ndarray:
    parsed = pd.to_numeric(values.astype(str).str.extract(r"^s([0-9]+)$", expand=False), errors="coerce")
    return parsed.fillna(0).to_numpy(np.int64) - 1


def analyze(
    *,
    ccre_bed: Path,
    node_labels: Path,
    full_links: Path,
    out_dir: Path,
) -> dict[str, object]:
    labels = pd.read_csv(node_labels, compression="infer")
    labels = labels[["segid", "chrom", "SO", "LN"]].copy()
    labels["end"] = labels["SO"] + labels["LN"]
    labels = labels.sort_values(["chrom", "SO", "end", "segid"]).reset_index(drop=True)

    maximum_segid = int(labels["segid"].max())
    rank_by_segid = np.full(maximum_segid + 1, -1, dtype=np.int32)
    chromosome_by_segid = np.full(maximum_segid + 1, -1, dtype=np.int16)
    chromosome_codes = {chromosome: code for code, chromosome in enumerate(sorted(labels["chrom"].unique()))}
    transitions: dict[str, np.ndarray] = {}
    for chromosome, group in labels.groupby("chrom", sort=False):
        segids = group["segid"].to_numpy(np.int64)
        ranks = np.arange(len(group), dtype=np.int32)
        rank_by_segid[segids] = ranks
        chromosome_by_segid[segids] = chromosome_codes[chromosome]
        transitions[chromosome] = np.zeros(max(len(group) - 1, 0), dtype=bool)

    link_rows = 0
    parsed_link_rows = 0
    reference_path_transitions_found = 0
    for chunk in pd.read_csv(
        full_links,
        usecols=["from_seg", "to_seg"],
        chunksize=250_000,
    ):
        link_rows += len(chunk)
        source = _segment_number(chunk["from_seg"])
        target = _segment_number(chunk["to_seg"])
        valid = (
            (source >= 0)
            & (target >= 0)
            & (source <= maximum_segid)
            & (target <= maximum_segid)
        )
        source = source[valid]
        target = target[valid]
        source_rank = rank_by_segid[source]
        target_rank = rank_by_segid[target]
        source_chromosome = chromosome_by_segid[source]
        target_chromosome = chromosome_by_segid[target]
        adjacent = (
            (source_rank >= 0)
            & (target_rank >= 0)
            & (source_chromosome == target_chromosome)
            & (np.abs(source_rank - target_rank) == 1)
        )
        parsed_link_rows += int(valid.sum())
        for code in np.unique(source_chromosome[adjacent]):
            chromosome = next(name for name, value in chromosome_codes.items() if value == code)
            positions = np.minimum(source_rank[adjacent & (source_chromosome == code)], target_rank[adjacent & (source_chromosome == code)])
            before = int(transitions[chromosome][positions].sum())
            transitions[chromosome][positions] = True
            reference_path_transitions_found += int(transitions[chromosome][positions].sum()) - before

    ccre = pd.read_csv(
        ccre_bed,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5],
        names=["chrom", "start", "end", "encode_id", "screen_id", "ccre_class"],
    )
    ccre["length_bp"] = ccre["end"] - ccre["start"]
    ccre["graph_nodes_overlapped"] = 0
    ccre["reference_path_edges_crossed"] = 0
    ccre["maximum_reference_path_graph_distance"] = 0
    ccre["node_span_status"] = "chromosome_absent_from_graph_labels"

    for chromosome, indices in ccre.groupby("chrom").groups.items():
        chromosome_nodes = labels[labels["chrom"] == chromosome]
        if chromosome_nodes.empty:
            continue
        node_start = chromosome_nodes["SO"].to_numpy(np.int64)
        node_end = chromosome_nodes["end"].to_numpy(np.int64)
        starts = ccre.loc[indices, "start"].to_numpy(np.int64)
        ends = ccre.loc[indices, "end"].to_numpy(np.int64)
        left = np.searchsorted(node_end, starts, side="right")
        right = np.searchsorted(node_start, ends, side="left")
        node_count = np.maximum(right - left, 0)

        transition_prefix = np.r_[0, np.cumsum(transitions[chromosome], dtype=np.int64)]
        edge_count = np.zeros(len(node_count), dtype=np.int64)
        spans_edges = node_count >= 2
        edge_count[spans_edges] = (
            transition_prefix[right[spans_edges] - 1] - transition_prefix[left[spans_edges]]
        )
        ccre.loc[indices, "graph_nodes_overlapped"] = node_count
        ccre.loc[indices, "reference_path_edges_crossed"] = edge_count
        ccre.loc[indices, "maximum_reference_path_graph_distance"] = edge_count
        ccre.loc[indices, "node_span_status"] = np.where(node_count > 0, "measured", "no_node_overlap")

    ccre["sequence_complexity"] = np.nan
    ccre["sequence_complexity_status"] = "not_computed_no_indexed_GRCh38_fasta"
    ccre["length_bin"] = pd.cut(
        ccre["length_bp"],
        bins=[-np.inf, 200, 500, 1000, np.inf],
        labels=["<=200", "201-500", "501-1000", ">1000"],
    ).astype(str)
    ccre["node_span_bin"] = pd.cut(
        ccre["graph_nodes_overlapped"],
        bins=[-np.inf, 0, 1, 2, 5, np.inf],
        labels=["0", "1", "2", "3-5", ">=6"],
    ).astype(str)

    out_dir.mkdir(parents=True, exist_ok=True)
    ccre.to_csv(out_dir / "ccre_graph_span.csv.gz", index=False, compression="gzip")
    aggregate = (
        ccre.groupby(["ccre_class", "length_bin", "node_span_bin"], observed=True)
        .agg(
            ccre_count=("encode_id", "size"),
            mean_length_bp=("length_bp", "mean"),
            mean_nodes_overlapped=("graph_nodes_overlapped", "mean"),
            mean_reference_path_edges_crossed=("reference_path_edges_crossed", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(out_dir / "ccre_graph_span_strata.csv", index=False)
    chromosome_summary = (
        ccre.groupby("chrom")
        .agg(
            ccre_count=("encode_id", "size"),
            median_length_bp=("length_bp", "median"),
            mean_nodes_overlapped=("graph_nodes_overlapped", "mean"),
            fraction_spanning_multiple_nodes=("graph_nodes_overlapped", lambda values: float(np.mean(values > 1))),
            fraction_with_measured_reference_edge=("reference_path_edges_crossed", lambda values: float(np.mean(values > 0))),
        )
        .reset_index()
    )
    chromosome_summary.to_csv(out_dir / "ccre_graph_span_by_chromosome.csv", index=False)
    summary = {
        "ccres": int(len(ccre)),
        "ccre_classes": ccre["ccre_class"].value_counts().to_dict(),
        "reference_graph_nodes": int(len(labels)),
        "graph_link_rows_scanned": int(link_rows),
        "parseable_s_numeric_link_rows": int(parsed_link_rows),
        "reference_path_transitions_found": int(reference_path_transitions_found),
        "ccres_with_node_overlap": int((ccre["graph_nodes_overlapped"] > 0).sum()),
        "ccres_spanning_multiple_nodes": int((ccre["graph_nodes_overlapped"] > 1).sum()),
        "fraction_spanning_multiple_nodes": float(np.mean(ccre["graph_nodes_overlapped"] > 1)),
        "sequence_complexity_status": "blocked: no indexed GRCh38 FASTA is present locally",
        "edge_definition": "observed graph links between consecutive GRCh38 path nodes overlapped by the cCRE",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccre-bed", type=Path, required=True)
    parser.add_argument("--node-labels", type=Path, required=True)
    parser.add_argument("--full-links", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
