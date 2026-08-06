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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from graph.io import read_links_csv
from tasks.ccre.baselines import _norm_chrom
from tasks.ccre.binary import _choose_threshold
from tasks.ccre.feature_policy import resolve_feature_policy
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


def _segment_number(name: object) -> int | None:
    value = str(name)
    if value.startswith("s") and value[1:].isdigit():
        return int(value[1:]) - 1
    return None


def _degree_by_requested_segid(
    links: pd.DataFrame,
    requested: set[int],
) -> dict[int, int]:
    degree: dict[int, int] = {}
    for left, right in links[["from_seg", "to_seg"]].itertuples(index=False):
        for name in (left, right):
            segid = _segment_number(name)
            if segid is not None and segid in requested:
                degree[segid] = degree.get(segid, 0) + 1
    return degree


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


def _sequence_kmer_features_from_sequences(
    seqs: list[str],
    *,
    k: int = 3,
    max_bases: int = 2048,
) -> tuple[np.ndarray, list[str]]:
    """Length-normalized mono/di/tri-nucleotide composition.

    Reverse complements are intentionally not collapsed: strand asymmetry can
    be informative, while both graph orientations are represented elsewhere.
    Ambiguous bases are reported as a separate fraction and break k-mers.
    Very long graph nodes use balanced prefix/suffix sampling to cap runtime;
    an inserted N prevents an artificial k-mer across the sampling boundary.
    """
    if k != 3:
        raise ValueError("The paper baseline currently supports k=3.")
    alphabet = "ACGT"
    names = (
        [f"frac_{base}" for base in alphabet]
        + ["frac_ambiguous", "cpg_fraction"]
        + [f"di_{a}{b}" for a in alphabet for b in alphabet]
        + [f"tri_{a}{b}{c}" for a in alphabet for b in alphabet for c in alphabet]
    )
    X = np.zeros((len(seqs), len(names)), dtype=np.float32)
    lookup = np.full(256, -1, dtype=np.int16)
    for idx, base in enumerate(alphabet):
        lookup[ord(base)] = idx
    for row_idx, seq in enumerate(seqs):
        seq = _sample_sequence(seq, max_bases)
        n = max(len(seq), 1)
        raw = np.frombuffer(seq.encode("ascii", errors="replace"), dtype=np.uint8)
        encoded = lookup[raw]
        valid = encoded >= 0
        mono = np.bincount(encoded[valid], minlength=4).astype(np.float32)
        valid_di_mask = valid[:-1] & valid[1:]
        di_codes = encoded[:-1][valid_di_mask] * 4 + encoded[1:][valid_di_mask]
        di = np.bincount(di_codes, minlength=16).astype(np.float32)
        valid_tri_mask = valid[:-2] & valid[1:-1] & valid[2:]
        tri_codes = (
            encoded[:-2][valid_tri_mask] * 16
            + encoded[1:-1][valid_tri_mask] * 4
            + encoded[2:][valid_tri_mask]
        )
        tri = np.bincount(tri_codes, minlength=64).astype(np.float32)
        valid_di = int(valid_di_mask.sum())
        valid_tri = int(valid_tri_mask.sum())
        cpg = int(di[1 * 4 + 2])
        valid_bases = float(mono.sum())
        X[row_idx, :4] = mono / max(valid_bases, 1.0)
        X[row_idx, 4] = float(n - valid_bases) / n
        X[row_idx, 5] = cpg / max(valid_di, 1)
        X[row_idx, 6:22] = di / max(valid_di, 1)
        X[row_idx, 22:] = tri / max(valid_tri, 1)
    return X, names


def _sample_sequence(seq: str, max_bases: int = 2048) -> str:
    seq = seq.upper()
    if len(seq) <= max_bases:
        return seq
    half = (max_bases - 1) // 2
    right = max_bases - 1 - half
    return seq[:half] + "N" + seq[-right:]


def _stream_sequence_kmer_features(
    full_segments: str | Path,
    segids: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Read only requested sequences in chunks, avoiding a multi-GB DataFrame."""
    positions: dict[int, list[int]] = {}
    for row_idx, segid in enumerate(segids.tolist()):
        positions.setdefault(int(segid), []).append(row_idx)
    empty, names = _sequence_kmer_features_from_sequences([], k=3)
    del empty
    X = np.zeros((len(segids), len(names)), dtype=np.float32)
    remaining = set(positions)
    fallback_segid = 0
    selected_sequences: list[str] = []
    selected_positions: list[list[int]] = []

    def flush() -> None:
        if not selected_sequences:
            return
        chunk_X, _ = _sequence_kmer_features_from_sequences(selected_sequences, k=3)
        for feature_row, row_indices in zip(chunk_X, selected_positions):
            X[row_indices] = feature_row
        selected_sequences.clear()
        selected_positions.clear()

    with _open_text(Path(full_segments)) as fh:
        for row in csv.DictReader(fh):
            name = row["name"]
            seq = row["seq"]
            raw_name = str(name)
            if raw_name.startswith("s") and raw_name[1:].isdigit():
                segid = int(raw_name[1:]) - 1
            else:
                segid = fallback_segid
            fallback_segid += 1
            if segid not in remaining:
                continue
            selected_sequences.append(_sample_sequence(str(seq)))
            selected_positions.append(positions[segid])
            remaining.remove(segid)
            if len(selected_sequences) >= 512:
                flush()
            if not remaining:
                break
    flush()
    if remaining:
        raise KeyError(
            f"{len(remaining)} requested segment IDs were absent from {full_segments}; "
            f"examples: {sorted(remaining)[:5]}"
        )
    return X, names


def _feature_matrix(
    *,
    feature_set: str,
    feature_policy: str,
    labels: pd.DataFrame,
    full_segments: str | Path,
    full_links: str | Path,
) -> tuple[np.ndarray, list[str]]:
    segids = labels["segid"].to_numpy(np.int64)
    if feature_set == "sequence_kmer":
        return _stream_sequence_kmer_features(full_segments, segids)

    links = read_links_csv(full_links)
    degree_by_segid = _degree_by_requested_segid(links, set(segids.tolist()))
    so = labels["SO"].to_numpy(float)
    ln = labels["LN"].to_numpy(float)
    degree = np.array([degree_by_segid.get(int(segid), 0) for segid in segids], dtype=float)
    structural_full = np.stack(
        [
            np.log1p(np.abs(so)),
            np.log1p(np.maximum(ln, 0)),
            np.zeros(len(labels), dtype=float),
            np.log1p(degree),
            np.zeros(len(labels), dtype=float),
            np.ones(len(labels), dtype=float),
        ],
        axis=1,
    ).astype(np.float32)
    baseline_full_names = [
        "log1p_SO",
        "log1p_LN",
        "SR",
        "log1p_degree",
        "orient",
        "is_grch38",
    ]
    if feature_policy == "leakage_safe":
        keep_names = ["log1p_SO", "log1p_LN", "log1p_degree", "orient"]
    elif feature_policy == "legacy_sr":
        keep_names = ["log1p_SO", "log1p_LN", "SR", "log1p_degree", "orient"]
    elif feature_policy == "legacy_reference":
        keep_names = baseline_full_names
    else:
        resolve_feature_policy(feature_policy)
        raise AssertionError("unreachable")
    baseline_name_to_full_idx = {name: i for i, name in enumerate(baseline_full_names)}
    structural = structural_full[:, [baseline_name_to_full_idx[n] for n in keep_names]]
    name_to_idx = {name: i for i, name in enumerate(keep_names)}
    if feature_set == "coordinate":
        names = [n for n in ["log1p_SO", "log1p_LN", "SR", "is_grch38"] if n in name_to_idx]
        return structural[:, [name_to_idx[n] for n in names]], names
    if feature_set == "graph":
        return structural[:, [name_to_idx["log1p_degree"]]], ["log1p_degree"]
    if feature_set == "structural":
        return structural, keep_names
    if feature_set == "linearized_graph":
        context = _linearized_context_features(labels, degree_by_segid)
        base_names = [
            n
            for n in ["log1p_SO", "log1p_LN", "SR", "is_grch38", "log1p_degree"]
            if n in name_to_idx
        ]
        X = np.concatenate([structural[:, [name_to_idx[n] for n in base_names]], context], axis=1)
        names = base_names + [
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
                n_jobs=1,
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
            n_jobs=1,
            random_state=seed,
        )
    if method == "sgd":
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                class_weight="balanced",
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                max_iter=2000,
                random_state=seed,
            ),
        )
    raise ValueError(f"Unknown method={method!r}")


def _binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(np.int64)
    out = {
        "threshold": float(threshold),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "positive_fraction": float(np.mean(y_true)),
        "brier": float(brier_score_loss(y_true, prob)),
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
    feature_policy: str,
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
        feature_policy=feature_policy,
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
        "feature_policy": feature_policy,
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
    binary_prob = pred_payload.get("p_positive")
    for c in sorted(set(chrom_test.tolist())):
        m = chrom_test == c
        row = {
            "chrom": c,
            "n_nodes": int(m.sum()),
            "macro_f1": float(
                f1_score(
                    y_test[m],
                    pred_test[m],
                    average="macro",
                    labels=labels_order,
                    zero_division=0,
                )
            ),
        }
        if binary_prob is not None:
            row.update(_binary_metrics(y_test[m], np.asarray(binary_prob)[m], threshold))
        per_chrom.append(row)
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
    ap.add_argument("--method", choices=["logistic", "mlp", "random_forest", "sgd"], default="logistic")
    ap.add_argument(
        "--feature_set",
        choices=["coordinate", "graph", "structural", "linearized_graph", "sequence_kmer"],
        default="structural",
    )
    ap.add_argument(
        "--feature_policy",
        choices=["leakage_safe", "legacy_sr", "legacy_reference"],
        default="leakage_safe",
        help=(
            "cCRE feature policy. leakage_safe removes both SR and is_grch38; "
            "legacy_sr matches older runs that kept SR but dropped is_grch38."
        ),
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
        feature_policy=args.feature_policy,
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
