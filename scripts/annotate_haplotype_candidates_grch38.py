#!/usr/bin/env python3
"""Project high-effect chr8 haplotype pairs to shared GRCh38 nodes and cCREs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _node_set(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {
        f"s{token}" if token.isdigit() else token
        for token in str(value).split(",")
        if token
    }


def annotate(
    *,
    pairs_path: str | Path,
    segments_path: str | Path,
    ccre_bed: str | Path,
    out_dir: str | Path,
    top_per_donor: int = 20,
) -> dict[str, object]:
    columns = [
        "donor_id",
        "h1_contig",
        "h2_contig",
        "h1_window_start",
        "h2_window_start",
        "h1_embedding_id",
        "h2_embedding_id",
        "shared_nodes",
        "methylation_delta",
        "node_jaccard",
        "h1_repeat_fraction",
        "h2_repeat_fraction",
        "h1_cpgs",
        "h2_cpgs",
        "h1_depth",
        "h2_depth",
    ]
    pairs = pd.read_csv(
        pairs_path,
        compression="infer",
        usecols=columns,
        low_memory=False,
    ).reset_index(names="row_index")
    pairs["absolute_methylation_delta"] = pairs["methylation_delta"].abs()
    pairs["graph_divergence"] = 1.0 - pairs["node_jaccard"]
    candidates = (
        pairs.sort_values(
            ["absolute_methylation_delta", "graph_divergence"],
            ascending=False,
        )
        .groupby("donor_id", sort=True)
        .head(top_per_donor)
        .copy()
    )
    candidate_nodes = [_node_set(value) for value in candidates["shared_nodes"]]
    requested_nodes = set().union(*candidate_nodes)

    coordinates: dict[str, tuple[int, int]] = {}
    for chunk in pd.read_csv(
        segments_path,
        usecols=["name", "LN", "SN", "SO", "SR"],
        chunksize=100_000,
    ):
        selected = chunk[
            chunk["name"].astype(str).isin(requested_nodes)
            & chunk["SN"].eq("GRCh38#0#chr8")
            & chunk["SR"].eq(0)
        ]
        for row in selected.itertuples(index=False):
            coordinates[str(row.name)] = (int(row.SO), int(row.SO) + int(row.LN))

    ccre = pd.read_csv(
        ccre_bed,
        sep="\t",
        header=None,
        names=["chromosome", "start", "end", "ccre_id", "encode_id", "class"],
        usecols=range(6),
    )
    ccre = ccre[ccre["chromosome"].eq("chr8")].sort_values("start")
    ccre_starts = ccre["start"].to_numpy()
    annotation_rows = []
    for nodes in candidate_nodes:
        spans = [coordinates[node] for node in nodes if node in coordinates]
        if spans:
            start = min(span[0] for span in spans)
            end = max(span[1] for span in spans)
            upper = int(np.searchsorted(ccre_starts, end, side="left"))
            overlaps = ccre.iloc[:upper]
            overlaps = overlaps[overlaps["end"] > start]
        else:
            start = end = np.nan
            overlaps = ccre.iloc[0:0]
        annotation_rows.append(
            {
                "grch38_chromosome": "chr8" if spans else "",
                "grch38_anchor_start": start,
                "grch38_anchor_end": end,
                "n_shared_nodes_with_grch38_coordinate": len(spans),
                "n_overlapping_encode_ccres": int(len(overlaps)),
                "encode_ccre_classes": ";".join(
                    sorted(overlaps["class"].astype(str).unique())
                ),
                "encode_ccre_ids": ";".join(
                    overlaps["ccre_id"].astype(str).head(20)
                ),
            }
        )
    candidates = pd.concat(
        [
            candidates.drop(columns=["shared_nodes"]).reset_index(drop=True),
            pd.DataFrame(annotation_rows),
        ],
        axis=1,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "high_effect_candidates_grch38_ccre.csv"
    candidates.to_csv(output, index=False)
    nonempty_classes = candidates.loc[
        candidates["encode_ccre_classes"].fillna("").ne(""),
        "encode_ccre_classes",
    ].astype(str)
    class_counts = (
        nonempty_classes.str.get_dummies(sep=";").sum().to_dict()
        if len(nonempty_classes)
        else {}
    )
    summary = {
        "task": "high_effect_haplotype_candidate_reference_annotation",
        "pairs_path": str(pairs_path),
        "segments_path": str(segments_path),
        "ccre_bed": str(ccre_bed),
        "n_candidates": int(len(candidates)),
        "top_per_donor": int(top_per_donor),
        "n_with_grch38_anchor": int(
            candidates["grch38_anchor_start"].notna().sum()
        ),
        "n_overlapping_encode_ccre": int(
            candidates["n_overlapping_encode_ccres"].gt(0).sum()
        ),
        "ccre_class_counts": class_counts,
        "output": str(output),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--ccre-bed", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-per-donor", type=int, default=20)
    args = parser.parse_args()
    annotate(
        pairs_path=args.pairs,
        segments_path=args.segments,
        ccre_bed=args.ccre_bed,
        out_dir=args.out_dir,
        top_per_donor=args.top_per_donor,
    )


if __name__ == "__main__":
    main()
