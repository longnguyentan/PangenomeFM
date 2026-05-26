"""
Map ENCODE cCRE BED intervals to graph nodes on GRCh38 walks.

Output
------
outputs/.../ccre/node_labels.csv.gz    one row per GRCh38-walk segment
outputs/.../ccre/class_by_chrom.csv    chrom x ccre_class crosstab
outputs/.../ccre/summary.json          dataset-level counts and run args

Usage
-----
python -m tasks.ccre.label_nodes \
    --full_segments data/hprc/full_segments.csv \
    --ccre_bed      data/encode/GRCh38-human-cCREs.bed \
    --out_dir       data/hprc/ccre
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from graph.io import read_segments_csv
from graph.slicing import build_global_index
from utils.versioning import resolve_run_dir
from tasks.ccre.encoding import (
    parse_encode_bed,
    map_ccre_to_ref_nodes,
    summarize_label_table,
    CCRE_CLASSES,
    CCRE_CLASS_TO_IDX,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full_segments",
        required=True,
        help="Path to full_segments.csv (HPRC graph segments)",
    )
    ap.add_argument(
        "--ccre_bed", required=True, help="Path to ENCODE GRCh38 cCRE 6-col BED file"
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Base output directory (will be versioned to run_NNN/)",
    )
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = resolve_run_dir(Path(args.out_dir))
    print(f"[10] output: {out_dir}")

    # 1. Load segments and build global index (must match 06b's indexing)
    print(f"[10] Loading segments from {args.full_segments}")
    segments = read_segments_csv(args.full_segments)
    seg_index, seg_u = build_global_index(segments)
    print(f"[10] seg_index size = {len(seg_index):,}")

    # 2. Parse ENCODE BED
    print(f"[10] Parsing BED from {args.ccre_bed}")
    ccre_df = parse_encode_bed(args.ccre_bed)
    print(f"[10] cCREs loaded: {len(ccre_df):,}")
    print(f"[10] class distribution:")
    print(ccre_df["cCRE_class"].value_counts().to_string())

    # 3. Sort-merge overlap against GRCh38 walks
    print(f"[10] Computing cCRE -> segment overlaps ...")
    node_labels = map_ccre_to_ref_nodes(ccre_df, seg_u, seg_index, verbose=args.verbose)

    # 4. Add integer label column (what the trainer will consume)
    node_labels["ccre_label"] = (
        node_labels["ccre_class"].map(CCRE_CLASS_TO_IDX).astype(np.int32)
    )

    # 5. Save
    labels_path = out_dir / "node_labels.csv.gz"
    node_labels.to_csv(labels_path, index=False, compression="gzip")
    print(f"[10] Saved {len(node_labels):,} rows -> {labels_path}")

    ct = summarize_label_table(node_labels)
    ct_path = out_dir / "class_by_chrom.csv"
    ct.to_csv(ct_path)
    print(f"[10] Saved crosstab -> {ct_path}")

    # 6. Summary
    total_by_class = (
        node_labels["ccre_class"].value_counts().reindex(CCRE_CLASSES, fill_value=0)
    )
    summary = {
        "args": vars(args),
        "n_segments_total": int(len(seg_u)),
        "n_grch38_walk_segments": int(len(node_labels)),
        "n_ccres_in_bed": int(len(ccre_df)),
        "total_by_class": {k: int(v) for k, v in total_by_class.items()},
        "frac_labeled_nonbg": float((node_labels["ccre_class"] != "background").mean()),
        "chroms_covered": sorted(node_labels["chrom"].unique().tolist()),
    }
    sum_path = out_dir / "summary.json"
    sum_path.write_text(json.dumps(summary, indent=2))
    print(f"[10] Saved summary -> {sum_path}")

    # Readable stdout summary
    print("\n[10] === SUMMARY ===")
    print(f"  GRCh38 walk segments labeled: {len(node_labels):,}")
    print(f"  Non-background fraction:      {summary['frac_labeled_nonbg']:.3f}")
    print(f"  Per-class totals:")
    for cls in CCRE_CLASSES:
        print(f"    {cls:12s} {summary['total_by_class'][cls]:>10,}")


if __name__ == "__main__":
    main()
