"""cCRE classifiers over frozen GraphGenome-FM node embeddings.

This runner extracts node embeddings from a frozen link-pretraining checkpoint
on benchmark slices, averages repeated segment occurrences, and trains a
simple logistic or MLP classifier on the resulting fixed embeddings.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from graph.features import build_oid_metadata_from_segments
from graph.io import read_segments_csv
from graph.slicing import build_global_index
from tasks.ccre.aligned_baselines import _binary_metrics, _fit_model
from tasks.ccre.baselines import _norm_chrom
from tasks.ccre.binary import _choose_threshold
from tasks.ccre.label_groups import (
    canonical_scheme,
    category_binary_labels,
    class_names_for_scheme,
    count_labels,
    describe_scheme,
    map_label_indices,
)
from training.pretrain import load_slice, tensorize_slice
from evaluation.external import _build_model_from_checkpoint, _namespace_from_checkpoint
from utils.versioning import resolve_run_dir


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
        return y, ["background", positive_group], describe_scheme("category_binary", positive_group)
    y = map_label_indices(raw, scheme=scheme)
    return y, class_names_for_scheme(scheme), describe_scheme(scheme)


@torch.no_grad()
def _extract_embeddings(
    *,
    checkpoint: Path,
    manifest: Path,
    full_segments: Path,
    labeled_segids: set[int],
    closure: str,
    device_name: str,
    seed: int,
    max_slices: int | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for frozen embedding extraction.")
    device = torch.device(device_name)
    ckpt = torch.load(checkpoint, map_location=device)
    eval_args = _namespace_from_checkpoint(ckpt, seed=seed)

    segments = read_segments_csv(full_segments)
    seg_index, _ = build_global_index(segments)
    md = build_oid_metadata_from_segments(segments, seg_index)
    model, predictor = _build_model_from_checkpoint(ckpt, eval_args, device)
    del predictor

    manifest_df = pd.read_csv(manifest)
    if closure != "all":
        manifest_df = manifest_df[manifest_df["closure"].astype(str) == closure]
    if max_slices is not None:
        manifest_df = manifest_df.head(max_slices)

    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = defaultdict(int)
    for i, row in manifest_df.iterrows():
        sd_raw = load_slice(row, seg_index, md, segments, eval_args)
        if sd_raw is None:
            continue
        sd = tensorize_slice(sd_raw, device, eval_args)
        if eval_args.adaptive_window:
            from training.pretrain import _compute_adaptive_window_k

            eff_wk = _compute_adaptive_window_k(
                sd["branching_frac"],
                eval_args.adaptive_window_base,
                eval_args.adaptive_window_alpha,
            )
            for layer in model.linear_layers:
                layer.window_k = eff_wk

        h = model.encode_nodes(
            sd["X"],
            sd["so"],
            sd["src"],
            sd["dst"],
            sd["temps"],
            sd["edge_attr"],
            sd["orient"],
            sd["pop_ids"],
        ).cpu().numpy()
        for local_idx, oid in enumerate(sd_raw["nodes"].tolist()):
            segid = int(oid) // 2
            if segid not in labeled_segids:
                continue
            if segid not in sums:
                sums[segid] = h[local_idx].astype(np.float64)
            else:
                sums[segid] += h[local_idx]
            counts[segid] += 1
        if (i + 1) % 50 == 0:
            print(f"[ccre-emb] processed {i + 1} manifest rows; embedded {len(sums):,} labeled segids")

    means = {segid: (vec / max(counts[segid], 1)).astype(np.float32) for segid, vec in sums.items()}
    return means, dict(counts)


def run_embedding_baseline(
    *,
    checkpoint: str | Path,
    manifest: str | Path,
    full_segments: str | Path,
    node_labels: str | Path,
    out_dir: str | Path,
    test_chrs: list[str],
    val_chrs: list[str],
    method: str,
    label_scheme: str,
    positive_group: str | None,
    background_only_negative: bool,
    closure: str,
    device: str,
    seed: int,
    max_slices: int | None,
    save_embeddings: bool,
) -> dict[str, object]:
    out_dir = resolve_run_dir(Path(out_dir))
    labels = pd.read_csv(node_labels, compression="infer")
    labels["chrom"] = labels["chrom"].astype(str)
    y_all, class_names, label_desc = _build_labels(
        labels,
        label_scheme=label_scheme,
        positive_group=positive_group,
        background_only_negative=background_only_negative,
    )
    valid = y_all != -100
    labels = labels.loc[valid].reset_index(drop=True)
    y_all = y_all[valid]

    labeled_segids = set(labels["segid"].astype(int).tolist())
    emb_by_segid, counts = _extract_embeddings(
        checkpoint=Path(checkpoint),
        manifest=Path(manifest),
        full_segments=Path(full_segments),
        labeled_segids=labeled_segids,
        closure=closure,
        device_name=device,
        seed=seed,
        max_slices=max_slices,
    )
    keep = labels["segid"].astype(int).isin(emb_by_segid).to_numpy()
    labels = labels.loc[keep].reset_index(drop=True)
    y = y_all[keep]
    if len(labels) == 0:
        raise RuntimeError("No labeled nodes received embeddings. Check manifest/closure/checkpoint.")

    X = np.stack([emb_by_segid[int(segid)] for segid in labels["segid"].tolist()], axis=0)
    chrom = labels["chrom"].to_numpy()
    test_chr_norm = {_norm_chrom(c) for c in test_chrs}
    val_chr_norm = {_norm_chrom(c) for c in val_chrs}
    is_test = np.isin(chrom, list(test_chr_norm))
    is_val = np.isin(chrom, list(val_chr_norm))
    is_train = ~(is_test | is_val)

    X_train, y_train = X[is_train], y[is_train]
    X_val, y_val = X[is_val], y[is_val]
    X_test, y_test = X[is_test], y[is_test]
    if len(y_test) == 0:
        raise RuntimeError("No embedded test nodes after split.")
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training split has fewer than two classes after embedding coverage filter.")

    clf = _fit_model(method, seed)
    clf.fit(X_train, y_train)
    pred_test = clf.predict(X_test)
    labels_order = list(range(len(class_names)))

    summary: dict[str, object] = {
        "method": method,
        "feature_set": "frozen_graphgenomefm_embeddings",
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "closure": closure,
        "label": label_desc,
        "test_chrs": sorted(test_chr_norm),
        "val_chrs": sorted(val_chr_norm),
        "n_embedded_labeled_nodes": int(len(labels)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train_label_counts": count_labels(y_train, class_names),
        "test_label_counts": count_labels(y_test, class_names),
        "embedding_dim": int(X.shape[1]),
        "mean_occurrences_per_node": float(np.mean([counts[int(s)] for s in labels["segid"]])),
    }

    pred_payload = {
        "segid": labels["segid"].to_numpy(np.int64)[is_test],
        "chrom": chrom[is_test],
        "y_true": y_test,
        "y_pred": pred_test,
    }
    if len(class_names) == 2 and hasattr(clf, "predict_proba"):
        p_val = clf.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])
        threshold = _choose_threshold(y_val, p_val) if len(p_val) else 0.5
        p_test = clf.predict_proba(X_test)[:, 1]
        summary["test_metrics"] = _binary_metrics(y_test, p_test, threshold)
        pred_payload["p_positive"] = p_test
        if len(p_val):
            summary["val_metrics"] = _binary_metrics(y_val, p_val, threshold)
    else:
        summary["test_metrics"] = {
            "macro_f1": float(
                f1_score(y_test, pred_test, average="macro", labels=labels_order, zero_division=0)
            )
        }

    pd.DataFrame(pred_payload).to_csv(out_dir / "test_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(
        confusion_matrix(y_test, pred_test, labels=labels_order),
        index=class_names,
        columns=class_names,
    ).to_csv(out_dir / "confusion_test.csv")
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
    if save_embeddings:
        np.savez_compressed(
            out_dir / "embeddings_labeled_nodes.npz",
            segid=labels["segid"].to_numpy(np.int64),
            X=X,
            y=y,
            chrom=chrom,
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["test_metrics"], indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen GraphGenome-FM embedding baseline for cCRE.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--full_segments", required=True)
    ap.add_argument("--node_labels", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_chrs", nargs="+", required=True)
    ap.add_argument("--val_chrs", nargs="+", required=True)
    ap.add_argument("--method", choices=["logistic", "mlp", "random_forest"], default="logistic")
    ap.add_argument(
        "--label_scheme",
        choices=["binary", "full9", "multiclass", "group3", "group4", "group5", "category_binary"],
        default="binary",
    )
    ap.add_argument("--positive_group", default=None)
    ap.add_argument("--all_ccre_as_negative", action="store_true")
    ap.add_argument("--closure", choices=["strict", "1hop", "all"], default="strict")
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_slices", type=int, default=None)
    ap.add_argument("--save_embeddings", action="store_true")
    args = ap.parse_args()
    run_embedding_baseline(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        full_segments=args.full_segments,
        node_labels=args.node_labels,
        out_dir=args.out_dir,
        test_chrs=args.test_chrs,
        val_chrs=args.val_chrs,
        method=args.method,
        label_scheme=args.label_scheme,
        positive_group=args.positive_group,
        background_only_negative=not args.all_ccre_as_negative,
        closure=args.closure,
        device=args.device,
        seed=args.seed,
        max_slices=args.max_slices,
        save_embeddings=args.save_embeddings,
    )


if __name__ == "__main__":
    main()
