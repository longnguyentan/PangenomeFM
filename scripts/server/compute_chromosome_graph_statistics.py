#!/usr/bin/env python3
"""Compute comparable chromosome-level graph statistics for every dataset."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.splits import normalize_chrom


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL = {f"chr{index}" for index in range(1, 23)} | {"chrX", "chrY"}


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


def statistics(dataset: str, segments_path: Path, links_path: Path, chunksize: int):
    segments = pd.read_csv(
        segments_path,
        usecols=lambda column: column in {"name", "id", "SN", "SO", "LN"},
        compression="infer",
    )
    identifier = "name" if "name" in segments else "id"
    segments["chromosome"] = segments["SN"].astype(str).map(normalize_chrom)
    segments = segments[segments["chromosome"].isin(CANONICAL)].copy()
    segments["LN"] = pd.to_numeric(segments["LN"], errors="coerce").fillna(0)
    segments["SO"] = pd.to_numeric(segments["SO"], errors="coerce")
    mapping = dict(zip(segments[identifier].astype(str), segments["chromosome"]))

    grouped = segments.groupby("chromosome", observed=True)
    rows = grouped["LN"].agg(
        nodes="size", sequence_bp="sum", mean_node_length="mean",
        median_node_length="median", max_node_length="max",
    ).reset_index()
    spans = grouped.apply(
        lambda frame: float(
            np.nanmax(frame["SO"] + frame["LN"]) - np.nanmin(frame["SO"])
        )
        if frame["SO"].notna().any()
        else np.nan,
        include_groups=False,
    ).rename("coordinate_span_bp")
    rows = rows.merge(spans.reset_index(), on="chromosome", how="left")

    within_counts: dict[str, int] = {}
    incident_counts: dict[str, int] = {}
    cross_edges = 0
    missing_endpoints = 0
    total_edges = 0
    for chunk in pd.read_csv(
        links_path,
        usecols=["from_seg", "to_seg"],
        compression="infer",
        chunksize=chunksize,
    ):
        total_edges += len(chunk)
        left = chunk["from_seg"].astype(str).map(mapping)
        right = chunk["to_seg"].astype(str).map(mapping)
        missing_endpoints += int((left.isna() | right.isna()).sum())
        same = left.notna() & right.notna() & (left == right)
        cross_edges += int((left.notna() & right.notna() & (left != right)).sum())
        for chrom, count in left[same].value_counts().items():
            within_counts[str(chrom)] = within_counts.get(str(chrom), 0) + int(count)
        for chrom, count in pd.concat([left[left.notna()], right[right.notna()]]).value_counts().items():
            incident_counts[str(chrom)] = incident_counts.get(str(chrom), 0) + int(count)

    rows["within_chromosome_edges"] = rows["chromosome"].map(within_counts).fillna(0).astype(int)
    rows["incident_edge_endpoints"] = rows["chromosome"].map(incident_counts).fillna(0).astype(int)
    rows["mean_incident_degree"] = rows["incident_edge_endpoints"] / rows["nodes"].clip(lower=1)
    rows["edge_node_ratio"] = rows["within_chromosome_edges"] / rows["nodes"].clip(lower=1)
    rows["nodes_per_megabase"] = rows["nodes"] / (rows["coordinate_span_bp"] / 1e6)
    rows["edges_per_megabase"] = rows["within_chromosome_edges"] / (rows["coordinate_span_bp"] / 1e6)
    rows.insert(0, "dataset", dataset)
    summary = {
        "dataset": dataset,
        "segments_path": str(segments_path),
        "links_path": str(links_path),
        "canonical_nodes": int(rows["nodes"].sum()),
        "links_total": total_edges,
        "cross_chromosome_edges": cross_edges,
        "link_rows_with_missing_or_noncanonical_endpoint": missing_endpoints,
        "chromosomes": int(len(rows)),
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/server_full_multicohort_20260806.json")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--out-dir")
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = expand(json.loads(config_path.read_text(encoding="utf-8")))
    selected = args.datasets or list(config["datasets"])
    out_dir = Path(args.out_dir or (Path(config["outputs"]["root"]) / "chromosome_graph_statistics"))
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    summaries = []
    for name in selected:
        dataset = config["datasets"][name]
        frame, summary = statistics(
            name, Path(dataset["segments"]), Path(dataset["links"]), args.chunksize
        )
        frame.to_csv(out_dir / f"{name}_chromosome_graph_statistics.csv", index=False)
        frames.append(frame)
        summaries.append(summary)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "chromosome_graph_statistics.csv", index=False)
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "datasets": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(combined.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
