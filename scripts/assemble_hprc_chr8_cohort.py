#!/usr/bin/env python3
"""Combine donor pair tables, attach population metadata, and annotate context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tasks.haplotype.annotations import annotate_paired_windows


def _read_pairs(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="infer")
    if "anchor_method" in frame:
        return frame
    required = {"shared_nodes", "shared_node_bp"}
    if not required.issubset(frame):
        raise ValueError(
            f"{path} lacks anchor_method and shared-node evidence."
        )
    valid = (
        frame["shared_nodes"].fillna(0).astype(float).ge(2)
        & frame["shared_node_bp"].fillna(0).astype(float).ge(500)
    )
    if not valid.all():
        raise ValueError(
            f"{path} has legacy rows below shared-node anchor thresholds."
        )
    frame["anchor_method"] = "shared_graph_nodes"
    frame["anchor_method_provenance"] = (
        "inferred_from_shared_nodes_ge_2_and_shared_bp_ge_500"
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--epigenome-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-missing-annotations", action="store_true")
    args = parser.parse_args()
    frames = [_read_pairs(path) for path in args.pairs]
    for frame in frames:
        if "anchor_method_provenance" not in frame:
            frame["anchor_method_provenance"] = "reported"
    combined = pd.concat(frames, ignore_index=True)
    cohort = (
        pd.read_csv(args.cohort, sep="\t")
        .drop_duplicates("donor_id")
        [["donor_id", "population", "super_population"]]
    )
    combined = combined.merge(
        cohort, on="donor_id", how="left", validate="many_to_one"
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_name("paired_windows_unannotated.csv.gz")
    combined.to_csv(raw_path, index=False, compression="gzip")
    annotation_summary = annotate_paired_windows(
        pairs_path=raw_path,
        epigenome_root=args.epigenome_root,
        out_path=out_path,
        strict=not args.allow_missing_annotations,
    )
    summary = {
        "n_pairs": len(combined),
        "n_donors": int(combined["donor_id"].nunique()),
        "population_counts": combined.groupby("super_population")[
            "donor_id"
        ].nunique().to_dict(),
        "annotation": annotation_summary,
    }
    (out_path.parent / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
