#!/usr/bin/env python3
"""Stratify the exact matched cCRE prediction universe by graph-node properties."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _metrics(frame: pd.DataFrame, score_column: str) -> dict[str, float | int]:
    y_true = frame["y_true"].to_numpy()
    score = frame[score_column].to_numpy()
    return {
        "n": len(frame),
        "positives": int(y_true.sum()),
        "positive_fraction": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, score)) if np.unique(y_true).size == 2 else np.nan,
        "auprc": float(average_precision_score(y_true, score)) if np.unique(y_true).size == 2 else np.nan,
        "brier": float(brier_score_loss(y_true, score)),
    }


def analyze(predictions: Path, node_labels: Path, out_dir: Path) -> dict[str, object]:
    prediction = pd.read_csv(predictions, compression="infer")
    labels = pd.read_csv(node_labels, compression="infer")
    annotated = prediction.merge(labels, on="segid", how="left", validate="one_to_one")
    if annotated["LN"].isna().any():
        raise ValueError("Some matched prediction segids are absent from node labels")
    score_columns = [
        column
        for column in ["frozen_gat", "sequence", "serialized", "structural"]
        if column in annotated
    ]
    annotated["node_length_bin"] = pd.cut(
        annotated["LN"],
        bins=[-np.inf, 10, 50, 200, 1000, np.inf],
        labels=["<=10", "11-50", "51-200", "201-1000", ">1000"],
    ).astype(str)
    annotated["ccres_touching_bin"] = pd.cut(
        annotated["n_ccres_touching"],
        bins=[-np.inf, 0, 1, np.inf],
        labels=["0", "1", ">=2"],
    ).astype(str)
    annotated["ccre_overlap_fraction"] = np.minimum(
        1.0, annotated["overlap_bp"] / annotated["LN"].clip(lower=1)
    )
    annotated["ccre_overlap_fraction_bin"] = pd.cut(
        annotated["ccre_overlap_fraction"],
        bins=[-np.inf, 0, 0.25, 0.75, np.inf],
        labels=["0", "(0,0.25]", "(0.25,0.75]", "(0.75,1]"],
    ).astype(str)

    rows: list[dict[str, object]] = []
    strata = {
        "chromosome": "chrom",
        "ccre_class": "ccre_class",
        "node_length": "node_length_bin",
        "ccres_touching": "ccres_touching_bin",
        "overlap_fraction": "ccre_overlap_fraction_bin",
        "genomic_block": "genomic_block",
    }
    for stratum_name, column in strata.items():
        for stratum_value, group in annotated.groupby(column, dropna=False):
            for score_column in score_columns:
                rows.append(
                    {
                        "stratum": stratum_name,
                        "stratum_value": str(stratum_value),
                        "model": score_column,
                        **_metrics(group, score_column),
                    }
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(out_dir / "matched_predictions_annotated.csv.gz", index=False, compression="gzip")
    pd.DataFrame(rows).to_csv(out_dir / "stratified_metrics.csv", index=False)
    summary = {
        "matched_nodes": len(annotated),
        "chromosomes": annotated["chrom"].value_counts().to_dict(),
        "genomic_blocks": int(annotated["genomic_block"].nunique()),
        "models": score_columns,
        "important_design_limit": (
            "The available matched cCRE predictions do not include paired core-node and "
            "endpoint-expanded context scores, so expanded-only recovery cannot be estimated."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--node-labels", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.predictions, args.node_labels, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
