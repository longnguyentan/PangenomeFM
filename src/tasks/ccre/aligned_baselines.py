"""Aligned cCRE baselines for all-node and benchmark-window evaluation.

This module is the main runner for Prof's downstream-comparison requests:

* binary cCRE vs background
* reduced 3/4/5-class cCRE grouping
* category-specific binary tasks such as enhancer-like vs background
* linear-reference-only, graph-feature, structural, and linearized-graph inputs
* logistic regression, MLP, and random-forest baselines
* all-node or benchmark-window-restricted evaluation

The benchmark-window mode lets logistic/MLP baselines use the same node
universe as the current GAT cCRE experiments.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from graph.features import build_oid_metadata_from_segments
from graph.io import read_links_csv, read_segments_csv
from graph.neg_sampling import oriented_ids_from_links
from graph.slicing import build_global_index
from tasks.ccre.baselines import _norm_chrom, build_per_node_structural_features
from tasks.ccre.binary import _choose_threshold
from tasks.ccre.label_groups import (
    canonical_scheme,
    category_binary_labels,
    class_names_for_scheme,
    count_labels,
    describe_scheme,
    map_label_indices,
)
from utils.versioning import resolve_run_dir

csv.field_size_limit(sys.maxsize)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open("r", newline="")


def _segid_from_slice_id(raw: str) -> int | None:
    if raw.startswith("s") and raw[1:].isdigit():
        return int(raw[1:])
    if raw.isdigit():
        return int(raw)
    return None


def _resolve_path(path: str | Path, root: Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    p2 = root / p
    if p2.exists():
        return p2
    return p


def _global_degree_map(links: pd.DataFrame, seg_index: pd.Index) -> dict[int, int]:
    u, v = oriented_ids_from_links(links, seg_index)
    deg_map: dict[int, int] = {}
    for oid in u.tolist():
        deg_map[int(oid)] = deg_map.get(int(oid), 0) + 1
    for oid in v.tolist():
        deg_map[int(oid)] = deg_map.get(int(oid), 0) + 1
    return deg_map


def _covered_segids_from_manifest(
    manifest: Path,
    *,
    root: Path,
    closures: set[str],
    split: str,
    test_chrs: set[str],
    val_chrs: set[str],
    labeled_segids: set[int],
) -> set[int]:
    covered: set[int] = set()
    manifest_df = pd.read_csv(manifest)
    for _, row in manifest_df.iterrows():
        target = str(row["target_sn"])
        target_split = "test" if target in test_chrs else "val" if target in val_chrs else "train"
        if split != "all" and target_split != split:
            continue
        if str(row["closure"]) not in closures:
            continue
        seg_path = _resolve_path(row["segments_path"], root)
        with _open_text(seg_path) as fh:
            reader = csv.DictReader(fh)
            for seg_row in reader:
                segid = _segid_from_slice_id(seg_row["id"])
                if segid is not None and segid in labeled_segids:
                    covered.add(segid)
    return covered


def _linearized_context_features(labels: pd.DataFrame, degree_by_segid: dict[int, int]) -> np.ndarray:
    df = labels[["segid", "chrom", "SO", "LN"]].copy()
    df["_orig_order"] = np.arange(len(df))
    df["degree"] = df["segid"].map(lambda x: degree_by_segid.get(int(x), 0)).astype(float)
    pieces = []
    for _, grp in df.sort_values(["chrom", "SO", "segid"]).groupby("chrom", sort=False):
        g = grp.copy()
        ln = g["LN"].astype(float)
        deg = g["degree"].astype(float)
        so = g["SO"].astype(float)
        prev_gap = (so - (so.shift(1) + ln.shift(1))).fillna(0).clip(lower=0)
        next_gap = ((so.shift(-1) - (so + ln))).fillna(0).clip(lower=0)
        g["prev_ln"] = ln.shift(1).fillna(0)
        g["next_ln"] = ln.shift(-1).fillna(0)
        g["prev_gap"] = prev_gap
        g["next_gap"] = next_gap
        g["rolling_ln_mean"] = ln.rolling(5, center=True, min_periods=1).mean()
        g["rolling_degree_mean"] = deg.rolling(5, center=True, min_periods=1).mean()
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True).sort_values("_orig_order")
    arr = out[
        [
            "prev_ln",
            "next_ln",
            "prev_gap",
            "next_gap",
            "rolling_ln_mean",
            "rolling_degree_mean",
        ]
    ].to_numpy(float)
    return np.log1p(np.maximum(arr, 0.0)).astype(np.float32)


def _feature_matrix(
    *,
    feature_set: str,
    labels: pd.DataFrame,
    full_segments: str | Path,
    full_links: str | Path,
) -> tuple[np.ndarray, list[str]]:
    segments = read_segments_csv(full_segments)
    links = read_links_csv(full_links)
    seg_index, _ = build_global_index(segments)
    md = build_oid_metadata_from_segments(segments, seg_index)
    deg_map = _global_degree_map(links, seg_index)

    segids = labels["segid"].to_numpy(np.int64)
    structural = build_per_node_structural_features(segids, md, deg_map, drop_is_grch38=True)
    degree_by_segid = {
        int(segid): deg_map.get(int(segid) * 2, 0) + deg_map.get(int(segid) * 2 + 1, 0)
        for segid in segids
    }

    if feature_set == "coordinate":
        return structural[:, :3], ["log1p_SO", "log1p_LN", "SR"]
    if feature_set == "graph":
        return structural[:, 3:4], ["log1p_degree"]
    if feature_set == "structural":
        return structural, ["log1p_SO", "log1p_LN", "SR", "log1p_degree", "orient_fwd"]
    if feature_set == "linearized_graph":
        context = _linearized_context_features(labels, degree_by_segid)
        X = np.concatenate([structural[:, :4], context], axis=1)
        names = [
            "log1p_SO",
            "log1p_LN",
            "SR",
            "log1p_degree",
            "prev_ln",
            "next_ln",
            "prev_gap",
            "next_gap",
            "rolling_ln_mean",
            "rolling_degree_mean",
        ]
        return X, names
    raise ValueError(f"Unknown feature_set={feature_set!r}")


def _build_labels(
    labels: pd.DataFrame,
    *,
    label_scheme: str,
    positive_group: str | None,
    background_only_negative: bool,
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    raw = labels["ccre_label"].to_numpy(np.int64)
    scheme = canonical_scheme(label_scheme)
    if scheme == "category_binary":
        if positive_group is None:
            raise ValueError("--positive-group is required for category_binary")
        y = category_binary_labels(
            raw,
            positive_group=positive_group,
            background_only_negative=background_only_negative,
        )
        names = ["background", positive_group]
        desc = describe_scheme("category_binary", positive_group)
    else:
        y = map_label_indices(raw, scheme=scheme)
        names = class_names_for_scheme(scheme)
        desc = describe_scheme(scheme)
    return y, names, desc


def _fit_model(method: str, seed: int):
    if method == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=800,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    if method == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                batch_size=512,
                early_stopping=True,
                max_iter=200,
                random_state=seed,
            ),
        )
    if method == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unknown method={method!r}")


def _binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(np.int64)
    out = {
        "threshold": float(threshold),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "positive_fraction": float(np.mean(y_true)),
    }
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, prob))
        out["auprc"] = float(average_precision_score(y_true, prob))
    return out


def _multiclass_metrics(y_true: np.ndarray, pred: np.ndarray, labels_order: list[int]) -> dict[str, float]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=labels_order,
        zero_division=0,
    )
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=labels_order, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", labels=labels_order, zero_division=0)),
        "per_class": [
            {
                "class_index": int(i),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for i, p, r, f, s in zip(labels_order, precision, recall, f1, support)
        ],
    }


def run_aligned_baseline(
    *,
    full_segments: str | Path,
    full_links: str | Path,
    node_labels: str | Path,
    out_dir: str | Path,
    test_chrs: list[str],
    val_chrs: list[str],
    method: str,
    feature_set: str,
    label_scheme: str,
    positive_group: str | None,
    background_only_negative: bool,
    evaluation_universe: str,
    benchmark_manifest: str | Path | None,
    closures: list[str],
    seed: int,
) -> dict[str, object]:
    root = Path.cwd()
    out_dir = resolve_run_dir(Path(out_dir))
    labels_df = pd.read_csv(node_labels, compression="infer")
    labels_df["chrom"] = labels_df["chrom"].astype(str)

    test_chr_norm = {_norm_chrom(c) for c in test_chrs}
    val_chr_norm = {_norm_chrom(c) for c in val_chrs}
    test_sn = {f"GRCh38#0#{c}" for c in test_chr_norm}
    val_sn = {f"GRCh38#0#{c}" for c in val_chr_norm}

    y_all, class_names, label_desc = _build_labels(
        labels_df,
        label_scheme=label_scheme,
        positive_group=positive_group,
        background_only_negative=background_only_negative,
    )
    valid = y_all != -100

    universe_segids: set[int] | None = None
    if evaluation_universe != "all":
        if benchmark_manifest is None:
            raise ValueError("--benchmark-manifest is required for benchmark-window evaluation")
        manifest = Path(benchmark_manifest)
        universe_segids = _covered_segids_from_manifest(
            manifest,
            root=root,
            closures=set(closures),
            split="all",
            test_chrs=test_sn,
            val_chrs=val_sn,
            labeled_segids=set(labels_df["segid"].astype(int).tolist()),
        )
        valid &= labels_df["segid"].astype(int).isin(universe_segids).to_numpy()

    labels_df = labels_df.loc[valid].reset_index(drop=True)
    y = y_all[valid]
    X, feature_names = _feature_matrix(
        feature_set=feature_set,
        labels=labels_df,
        full_segments=full_segments,
        full_links=full_links,
    )
    chrom = labels_df["chrom"].to_numpy()
    is_test = np.isin(chrom, list(test_chr_norm))
    is_val = np.isin(chrom, list(val_chr_norm))
    is_train = ~(is_test | is_val)

    X_train, y_train = X[is_train], y[is_train]
    X_val, y_val = X[is_val], y[is_val]
    X_test, y_test = X[is_test], y[is_test]
    chrom_test = chrom[is_test]
    segid_test = labels_df["segid"].to_numpy(np.int64)[is_test]

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training split has fewer than two labels after filtering.")
    if len(y_test) == 0:
        raise RuntimeError("Test split is empty after filtering.")

    model = _fit_model(method, seed)
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    pred_val = model.predict(X_val) if len(X_val) else np.array([], dtype=np.int64)

    labels_order = list(range(len(class_names)))
    summary: dict[str, object] = {
        "method": method,
        "feature_set": feature_set,
        "feature_names": feature_names,
        "label": label_desc,
        "evaluation_universe": evaluation_universe,
        "closures": closures,
        "test_chrs": sorted(test_chr_norm),
        "val_chrs": sorted(val_chr_norm),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train_label_counts": count_labels(y_train, class_names),
        "test_label_counts": count_labels(y_test, class_names),
    }

    pred_payload = {
        "segid": segid_test,
        "chrom": chrom_test,
        "y_true": y_test,
        "y_pred": pred_test,
    }

    if len(class_names) == 2 and hasattr(model, "predict_proba"):
        p_val = model.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])
        threshold = _choose_threshold(y_val, p_val) if len(p_val) else 0.5
        p_test = model.predict_proba(X_test)[:, 1]
        summary["test_metrics"] = _binary_metrics(y_test, p_test, threshold)
        if len(p_val):
            summary["val_metrics"] = _binary_metrics(y_val, p_val, threshold)
        pred_payload["p_positive"] = p_test
    else:
        summary["test_metrics"] = _multiclass_metrics(y_test, pred_test, labels_order)
        if len(pred_val):
            summary["val_metrics"] = _multiclass_metrics(y_val, pred_val, labels_order)

    per_chrom = []
    for c in sorted(set(chrom_test.tolist())):
        m = chrom_test == c
        per_chrom.append(
            {
                "chrom": c,
                "n_nodes": int(m.sum()),
                "macro_f1": float(f1_score(y_test[m], pred_test[m], average="macro", labels=labels_order, zero_division=0)),
            }
        )
    pd.DataFrame(per_chrom).to_csv(out_dir / "per_chrom_metrics.csv", index=False)
    pd.DataFrame(pred_payload).to_csv(out_dir / "test_predictions.csv.gz", index=False, compression="gzip")

    cm = confusion_matrix(y_test, pred_test, labels=labels_order)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / "confusion_test.csv")
    (out_dir / "classification_report_test.txt").write_text(
        classification_report(
            y_test,
            pred_test,
            labels=labels_order,
            target_names=class_names,
            zero_division=0,
        ),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["test_metrics"], indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Aligned cCRE baseline runner.")
    ap.add_argument("--full_segments", required=True)
    ap.add_argument("--full_links", required=True)
    ap.add_argument("--node_labels", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_chrs", nargs="+", required=True)
    ap.add_argument("--val_chrs", nargs="+", required=True)
    ap.add_argument("--method", choices=["logistic", "mlp", "random_forest"], default="logistic")
    ap.add_argument(
        "--feature_set",
        choices=["coordinate", "graph", "structural", "linearized_graph"],
        default="structural",
    )
    ap.add_argument(
        "--label_scheme",
        choices=["binary", "full9", "multiclass", "group3", "group4", "group5", "category_binary"],
        default="binary",
    )
    ap.add_argument("--positive_group", default=None)
    ap.add_argument("--all_ccre_as_negative", action="store_true")
    ap.add_argument(
        "--evaluation_universe",
        choices=["all", "benchmark_windows"],
        default="all",
    )
    ap.add_argument("--benchmark_manifest", default=None)
    ap.add_argument("--closures", nargs="+", choices=["strict", "1hop"], default=["strict", "1hop"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_aligned_baseline(
        full_segments=args.full_segments,
        full_links=args.full_links,
        node_labels=args.node_labels,
        out_dir=args.out_dir,
        test_chrs=args.test_chrs,
        val_chrs=args.val_chrs,
        method=args.method,
        feature_set=args.feature_set,
        label_scheme=args.label_scheme,
        positive_group=args.positive_group,
        background_only_negative=not args.all_ccre_as_negative,
        evaluation_universe=args.evaluation_universe,
        benchmark_manifest=args.benchmark_manifest,
        closures=args.closures,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
