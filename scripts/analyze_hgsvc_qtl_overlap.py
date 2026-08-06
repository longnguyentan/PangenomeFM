#!/usr/bin/env python3
"""Run a resource-aware HGSVC molecular-QTL overlap pilot on graph windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.qtl_overlap import (
    exact_spearman_permutation,
    load_context_windows,
    stream_qtl_overlaps,
)


def _parse_qtl(value: str) -> tuple[str, Path]:
    try:
        qtl_type, path = value.split("=", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("QTL input must be TYPE=PATH") from exc
    return qtl_type, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-metrics", action="append", type=Path, required=True)
    parser.add_argument("--qtl", action="append", type=_parse_qtl, required=True)
    parser.add_argument("--chromosome", default="chr8")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--p-threshold", type=float, default=5e-8)
    parser.add_argument("--empirical-threshold", type=float, default=0.05)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        parser.error(f"Output directory is not empty: {args.output_dir}; use --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    windows = load_context_windows(args.context_metrics, args.chromosome)
    windows.to_csv(args.output_dir / "context_windows.csv", index=False)

    overlap_frames: list[pd.DataFrame] = []
    top_frames: list[pd.DataFrame] = []
    processed: dict[str, int] = {}
    for qtl_type, qtl_path in args.qtl:
        overlaps, top_rows, rows_processed = stream_qtl_overlaps(
            qtl_path,
            windows,
            qtl_type=qtl_type,
            chunksize=args.chunksize,
            p_threshold=args.p_threshold,
            empirical_threshold=args.empirical_threshold,
            top_n=args.top_n,
        )
        overlap_frames.append(overlaps)
        if not top_rows.empty:
            top_frames.append(top_rows)
        processed[qtl_type] = rows_processed

    overlap = pd.concat(overlap_frames, ignore_index=True).merge(
        windows, on="interval_id", how="left", validate="many_to_one"
    )
    overlap.to_csv(args.output_dir / "region_qtl_overlap.csv", index=False)
    if top_frames:
        pd.concat(top_frames, ignore_index=True).to_csv(
            args.output_dir / "qtl_top_associations.csv.gz", index=False
        )

    group_summary = (
        overlap.groupby(["qtl_type", "context_recovery_group"], as_index=False)
        .agg(
            n_windows=("interval_id", "nunique"),
            mean_context_auroc_gain=("context_auroc_gain", "mean"),
            total_significant_variant_positions=(
                "genomewide_significant_variant_positions",
                "sum",
            ),
            mean_significant_variant_positions_per_mb=(
                "significant_variant_positions_per_mb",
                "mean",
            ),
            median_neg_log10_min_p=("neg_log10_min_p", "median"),
        )
    )
    group_summary.to_csv(args.output_dir / "qtl_group_summary.csv", index=False)

    correlations: list[dict[str, object]] = []
    for qtl_type, frame in overlap.groupby("qtl_type"):
        for outcome in (
            "significant_variant_positions_per_mb",
            "neg_log10_min_p",
            "empirical_significant_features",
        ):
            rho, p_value, n = exact_spearman_permutation(
                frame["context_auroc_gain"], frame[outcome]
            )
            correlations.append(
                {
                    "qtl_type": qtl_type,
                    "predictor": "context_auroc_gain",
                    "outcome": outcome,
                    "spearman_rho": rho,
                    "exact_permutation_p_value": p_value,
                    "n_windows": n,
                }
            )
    correlation_frame = pd.DataFrame(correlations)
    correlation_frame.to_csv(args.output_dir / "qtl_correlations.csv", index=False)

    summary = {
        "analysis": "descriptive chromosome-level direct overlap pilot",
        "chromosome": args.chromosome,
        "coordinate_rule": "graph window [start,end) vs one-based QTL POS: start < POS <= end",
        "n_windows": int(windows["interval_id"].nunique()),
        "n_context_seeds": int(windows["n_seeds"].min()),
        "qtl_rows_processed": processed,
        "p_threshold": args.p_threshold,
        "empirical_feature_threshold": args.empirical_threshold,
        "limitations": [
            "Only preselected 50-kb chromosome 8 graph windows were tested.",
            "The number of windows is too small for an enrichment or colocalization claim.",
            "Overlap does not establish that the graph-model score is functionally predictive.",
            "Empirical feature p-values correct within-feature variant scans, not all reported rows.",
        ],
        "outputs": [
            "context_windows.csv",
            "region_qtl_overlap.csv",
            "qtl_top_associations.csv.gz",
            "qtl_group_summary.csv",
            "qtl_correlations.csv",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

