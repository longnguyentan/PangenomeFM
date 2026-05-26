"""
Binary cCRE node/region classification.

Task v1:
    input: graph segment/node features
    output: whether the segment overlaps any ENCODE cCRE

This module intentionally starts with a transparent structural-feature baseline.
Frozen/fine-tuned encoder variants should use the same split and metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from tasks.ccre.baselines import _norm_chrom, build_per_node_structural_features
from tasks.ccre.encoding import CCRE_CLASS_TO_IDX
from graph.features import build_oid_metadata_from_segments
from graph.io import read_links_csv, read_segments_csv
from graph.neg_sampling import oriented_ids_from_links
from graph.slicing import build_global_index
from utils.versioning import resolve_run_dir


def _global_degree_map(links: pd.DataFrame, seg_index: pd.Index) -> dict[int, int]:
    u, v = oriented_ids_from_links(links, seg_index)
    deg_map: dict[int, int] = {}
    for oid in u.tolist():
        deg_map[int(oid)] = deg_map.get(int(oid), 0) + 1
    for oid in v.tolist():
        deg_map[int(oid)] = deg_map.get(int(oid), 0) + 1
    return deg_map


def _choose_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    if len(np.unique(y_val)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    if len(thresholds) == 0:
        return 0.5
    f1 = (2 * precision[:-1] * recall[:-1]) / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(np.int64)
    out = {
        "positive_fraction": float(np.mean(y)),
        "threshold": float(threshold),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def run_binary_baseline(
    *,
    full_segments: str | Path,
    full_links: str | Path,
    node_labels: str | Path,
    out_dir: str | Path,
    test_chrs: list[str],
    val_chr: str,
    seed: int = 42,
    negative_train_fraction: float = 1.0,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    out_dir = resolve_run_dir(Path(out_dir))

    test_chrs_norm = {_norm_chrom(c) for c in test_chrs}
    val_chr_norm = _norm_chrom(val_chr)
    if val_chr_norm in test_chrs_norm:
        print(
            "[ccre-binary] WARNING: validation chromosome overlaps test set. "
            "Use disjoint chromosomes for paper runs."
        )

    segments = read_segments_csv(full_segments)
    links = read_links_csv(full_links)
    seg_index, _ = build_global_index(segments)
    md = build_oid_metadata_from_segments(segments, seg_index)
    deg_map = _global_degree_map(links, seg_index)

    labels = pd.read_csv(node_labels, compression="infer")
    labels["chrom"] = labels["chrom"].astype(str)

    segids = labels["segid"].to_numpy(np.int64)
    X = build_per_node_structural_features(segids, md, deg_map, drop_is_grch38=True)
    bg_idx = CCRE_CLASS_TO_IDX["background"]
    y = (labels["ccre_label"].to_numpy(np.int64) != bg_idx).astype(np.int64)
    chrom = labels["chrom"].to_numpy()

    is_test = np.isin(chrom, list(test_chrs_norm))
    is_val = chrom == val_chr_norm
    is_train = ~(is_test | is_val)

    X_train, y_train = X[is_train], y[is_train]
    X_val, y_val = X[is_val], y[is_val]
    X_test, y_test = X[is_test], y[is_test]
    chrom_test = chrom[is_test]

    if negative_train_fraction < 1.0:
        neg = np.where(y_train == 0)[0]
        keep_neg_n = int(len(neg) * negative_train_fraction)
        drop_neg = rng.choice(neg, size=len(neg) - keep_neg_n, replace=False)
        keep = np.ones(len(y_train), dtype=bool)
        keep[drop_neg] = False
        X_train, y_train = X_train[keep], y_train[keep]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X_train_s, y_train)

    p_val = clf.predict_proba(X_val_s)[:, 1] if len(X_val_s) else np.array([])
    threshold = _choose_threshold(y_val, p_val) if len(p_val) else 0.5
    p_test = clf.predict_proba(X_test_s)[:, 1]

    summary: dict[str, object] = {
        "task": "binary_ccre_node_classification",
        "positive_label": "any_non_background_ccre",
        "seed": seed,
        "test_chrs": sorted(test_chrs_norm),
        "val_chr": val_chr_norm,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train_positive_fraction": float(np.mean(y_train)) if len(y_train) else 0.0,
        "val_positive_fraction": float(np.mean(y_val)) if len(y_val) else 0.0,
        "test_metrics": _metrics(y_test, p_test, threshold),
    }
    if len(p_val):
        summary["val_metrics"] = _metrics(y_val, p_val, threshold)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "chrom": chrom_test,
            "y_true": y_test,
            "p_ccre": p_test,
            "y_pred": (p_test >= threshold).astype(np.int64),
        }
    ).to_csv(out_dir / "test_predictions.csv.gz", index=False, compression="gzip")
    print(json.dumps(summary["test_metrics"], indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Binary cCRE structural baseline.")
    ap.add_argument("--full_segments", required=True)
    ap.add_argument("--full_links", required=True)
    ap.add_argument("--node_labels", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_chrs", nargs="+", required=True)
    ap.add_argument("--val_chr", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negative_train_fraction", type=float, default=1.0)
    args = ap.parse_args()

    run_binary_baseline(
        full_segments=args.full_segments,
        full_links=args.full_links,
        node_labels=args.node_labels,
        out_dir=args.out_dir,
        test_chrs=args.test_chrs,
        val_chr=args.val_chr,
        seed=args.seed,
        negative_train_fraction=args.negative_train_fraction,
    )


if __name__ == "__main__":
    main()

