"""Serialized-graph cCRE baseline.

This is a DeepGene-style comparison baseline, not a full DeepGene reproduction.
It serializes labeled graph nodes by chromosome and genomic offset, builds
fixed-length node-context windows, and trains a compact Transformer classifier.
The model consumes no graph edges; it represents the "serialized graph" modeling
assumption Prof asked us to compare against graph-native GraphGenome-FM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from graph.io import read_segments_csv
from tasks.ccre.aligned_baselines import (
    _build_labels,
    _covered_segids_from_manifest,
    _feature_matrix,
)
from tasks.ccre.baselines import _norm_chrom
from tasks.ccre.binary import _choose_threshold
from tasks.ccre.label_groups import class_names_for_scheme, count_labels
from utils.versioning import resolve_run_dir


def _resolve_path(path: str | Path, root: Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    p2 = root / p
    if p2.exists():
        return p2
    return p


def _sequence_features(
    *,
    labels: pd.DataFrame,
    full_segments: str | Path,
) -> tuple[np.ndarray, list[str]]:
    """Return simple sequence-composition features per labeled segment."""
    segments = read_segments_csv(full_segments)
    seq_by_name = (
        segments.drop_duplicates("name").set_index("name")["seq"].astype(str).to_dict()
    )
    names = [f"frac_{b}" for b in ["A", "C", "G", "T", "N"]]
    rows = []
    for segid in labels["segid"].astype(int).tolist():
        seq = seq_by_name.get(f"s{segid + 1}", "")
        if not seq:
            rows.append([0.0] * len(names))
            continue
        seq = seq.upper()
        denom = max(len(seq), 1)
        rows.append([seq.count(b) / denom for b in ["A", "C", "G", "T", "N"]])
    return np.asarray(rows, dtype=np.float32), names


def _filter_labels(
    *,
    node_labels: str | Path,
    label_scheme: str,
    positive_group: str | None,
    background_only_negative: bool,
    evaluation_universe: str,
    benchmark_manifest: str | Path | None,
    closures: list[str],
    test_chrs: list[str],
    val_chrs: list[str],
) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, object]]:
    labels_df = pd.read_csv(node_labels, compression="infer")
    labels_df["chrom"] = labels_df["chrom"].astype(str)

    y_all, class_names, label_desc = _build_labels(
        labels_df,
        label_scheme=label_scheme,
        positive_group=positive_group,
        background_only_negative=background_only_negative,
    )
    valid = y_all != -100

    if evaluation_universe != "all":
        if benchmark_manifest is None:
            raise ValueError("--benchmark_manifest is required for benchmark_windows evaluation")
        test_sn = {f"GRCh38#0#{_norm_chrom(c)}" for c in test_chrs}
        val_sn = {f"GRCh38#0#{_norm_chrom(c)}" for c in val_chrs}
        covered = _covered_segids_from_manifest(
            Path(benchmark_manifest),
            root=Path.cwd(),
            closures=set(closures),
            split="all",
            test_chrs=test_sn,
            val_chrs=val_sn,
            labeled_segids=set(labels_df["segid"].astype(int).tolist()),
        )
        valid &= labels_df["segid"].astype(int).isin(covered).to_numpy()

    labels_df = labels_df.loc[valid].reset_index(drop=True)
    y = y_all[valid]
    return labels_df, y, class_names, label_desc


class SerializedNodeDataset(Dataset):
    def __init__(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        chrom: np.ndarray,
        order_by_chrom: dict[str, np.ndarray],
        row_to_rank: np.ndarray,
        row_indices: np.ndarray,
        radius: int,
    ) -> None:
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.chrom = chrom
        self.order_by_chrom = order_by_chrom
        self.row_to_rank = row_to_rank
        self.row_indices = row_indices.astype(np.int64)
        self.radius = int(radius)
        self.seq_len = 2 * self.radius + 1

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, item: int):
        row_idx = int(self.row_indices[item])
        chrom = str(self.chrom[row_idx])
        ordered = self.order_by_chrom[chrom]
        rank = int(self.row_to_rank[row_idx])
        left = rank - self.radius
        right = rank + self.radius + 1
        take_left = max(left, 0)
        take_right = min(right, len(ordered))
        window_rows = ordered[take_left:take_right]
        dest_start = take_left - left
        xw = np.zeros((self.seq_len, self.X.shape[1]), dtype=np.float32)
        pad_mask = np.ones(self.seq_len, dtype=bool)
        xw[dest_start : dest_start + len(window_rows)] = self.X[window_rows]
        pad_mask[dest_start : dest_start + len(window_rows)] = False
        return (
            torch.from_numpy(xw),
            torch.from_numpy(pad_mask),
            torch.tensor(self.y[row_idx], dtype=torch.long),
            torch.tensor(row_idx, dtype=torch.long),
        )


class SerializedGraphTransformer(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        n_classes: int,
        seq_len: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.center = seq_len // 2
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(seq_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: "torch.Tensor", pad_mask: "torch.Tensor") -> "torch.Tensor":
        h = self.input_proj(x) + self.pos_embed.unsqueeze(0)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.head(h[:, self.center, :])


def _metrics_binary(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
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


def _metrics_multiclass(y_true: np.ndarray, pred: np.ndarray, labels_order: list[int]) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=labels_order, zero_division=0
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


@torch.no_grad()
def _predict(model, loader, device, n_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, probs, row_ids = [], [], []
    for xb, mb, yb, rb in loader:
        logits = model(xb.to(device), mb.to(device))
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        ys.append(yb.numpy())
        probs.append(p)
        row_ids.append(rb.numpy())
    return np.concatenate(ys), np.concatenate(probs), np.concatenate(row_ids)


def run_serialized_baseline(
    *,
    full_segments: str | Path,
    full_links: str | Path,
    node_labels: str | Path,
    out_dir: str | Path,
    test_chrs: list[str],
    val_chrs: list[str],
    label_scheme: str,
    positive_group: str | None,
    background_only_negative: bool,
    evaluation_universe: str,
    benchmark_manifest: str | Path | None,
    closures: list[str],
    feature_policy: str,
    context_radius: int,
    hidden_dim: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    include_sequence_features: bool,
    max_train_nodes: int | None,
    max_val_nodes: int | None,
    max_test_nodes: int | None,
    seed: int,
    device_name: str,
) -> dict[str, object]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for serialized cCRE baseline.")
    rng = np.random.default_rng(seed)
    out_dir = resolve_run_dir(Path(out_dir))
    device = torch.device(device_name)

    labels_df, y, class_names, label_desc = _filter_labels(
        node_labels=node_labels,
        label_scheme=label_scheme,
        positive_group=positive_group,
        background_only_negative=background_only_negative,
        evaluation_universe=evaluation_universe,
        benchmark_manifest=benchmark_manifest,
        closures=closures,
        test_chrs=test_chrs,
        val_chrs=val_chrs,
    )
    X, feature_names = _feature_matrix(
        feature_set="linearized_graph",
        feature_policy=feature_policy,
        labels=labels_df,
        full_segments=full_segments,
        full_links=full_links,
    )
    if include_sequence_features:
        X_seq, seq_names = _sequence_features(labels=labels_df, full_segments=full_segments)
        X = np.concatenate([X, X_seq], axis=1)
        feature_names = feature_names + seq_names

    chrom = labels_df["chrom"].astype(str).to_numpy()
    test_chr_norm = {_norm_chrom(c) for c in test_chrs}
    val_chr_norm = {_norm_chrom(c) for c in val_chrs}
    is_test = np.isin(chrom, list(test_chr_norm))
    is_val = np.isin(chrom, list(val_chr_norm))
    is_train = ~(is_test | is_val)

    def sample_indices(mask: np.ndarray, max_n: int | None) -> np.ndarray:
        idx = np.where(mask)[0]
        if max_n is not None and len(idx) > max_n:
            idx = rng.choice(idx, size=max_n, replace=False)
        return np.sort(idx)

    train_idx = sample_indices(is_train, max_train_nodes)
    val_idx = sample_indices(is_val, max_val_nodes)
    test_idx = sample_indices(is_test, max_test_nodes)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("Train or test split is empty after filtering.")

    order_by_chrom: dict[str, np.ndarray] = {}
    row_to_rank = np.zeros(len(labels_df), dtype=np.int64)
    for c, grp in labels_df.assign(_row=np.arange(len(labels_df))).sort_values(
        ["chrom", "SO", "segid"]
    ).groupby("chrom", sort=False):
        order = grp["_row"].to_numpy(np.int64)
        order_by_chrom[str(c)] = order
        row_to_rank[order] = np.arange(len(order), dtype=np.int64)

    ds_train = SerializedNodeDataset(
        X=X, y=y, chrom=chrom, order_by_chrom=order_by_chrom, row_to_rank=row_to_rank,
        row_indices=train_idx, radius=context_radius
    )
    ds_val = SerializedNodeDataset(
        X=X, y=y, chrom=chrom, order_by_chrom=order_by_chrom, row_to_rank=row_to_rank,
        row_indices=val_idx, radius=context_radius
    )
    ds_test = SerializedNodeDataset(
        X=X, y=y, chrom=chrom, order_by_chrom=order_by_chrom, row_to_rank=row_to_rank,
        row_indices=test_idx, radius=context_radius
    )
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False) if len(ds_val) else None
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    n_classes = len(class_names)
    model = SerializedGraphTransformer(
        in_dim=X.shape[1],
        n_classes=n_classes,
        seq_len=2 * context_radius + 1,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)
    counts = np.bincount(y[train_idx], minlength=n_classes).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = -1.0
    bad_epochs = 0
    history = []
    labels_order = list(range(n_classes))
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for xb, mb, yb, _ in loader_train:
            optimizer.zero_grad()
            logits = model(xb.to(device), mb.to(device))
            loss = criterion(logits, yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            n_seen += len(yb)

        if loader_val is not None:
            y_val, p_val, _ = _predict(model, loader_val, device, n_classes)
            val_pred = p_val.argmax(axis=1)
            val_score = float(f1_score(y_val, val_pred, average="macro", labels=labels_order, zero_division=0))
        else:
            val_score = 0.0
        history.append({"epoch": epoch, "train_loss": total_loss / max(n_seen, 1), "val_macro_f1": val_score})
        if val_score > best_val + 1e-5:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    y_test, p_test, row_ids = _predict(model, loader_test, device, n_classes)
    pred_test = p_test.argmax(axis=1)

    if n_classes == 2:
        if loader_val is not None:
            y_val, p_val, _ = _predict(model, loader_val, device, n_classes)
            threshold = _choose_threshold(y_val, p_val[:, 1])
        else:
            threshold = 0.5
        test_metrics = _metrics_binary(y_test, p_test[:, 1], threshold)
        pred_payload_prob = p_test[:, 1]
    else:
        test_metrics = _metrics_multiclass(y_test, pred_test, labels_order)
        pred_payload_prob = None

    pred_payload = {
        "segid": labels_df["segid"].to_numpy(np.int64)[row_ids],
        "chrom": chrom[row_ids],
        "y_true": y_test,
        "y_pred": pred_test,
    }
    if pred_payload_prob is not None:
        pred_payload["p_positive"] = pred_payload_prob
    pd.DataFrame(pred_payload).to_csv(out_dir / "test_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    pd.DataFrame(
        confusion_matrix(y_test, pred_test, labels=labels_order),
        index=class_names,
        columns=class_names,
    ).to_csv(out_dir / "confusion_test.csv")
    (out_dir / "classification_report_test.txt").write_text(
        classification_report(y_test, pred_test, labels=labels_order, target_names=class_names, zero_division=0),
        encoding="utf-8",
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_names": feature_names,
            "args": {
                "context_radius": context_radius,
                "hidden_dim": hidden_dim,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "feature_policy": feature_policy,
            },
        },
        out_dir / "ckpt_best.pt",
    )

    summary = {
        "method": "serialized_graph_transformer",
        "deepgene_style": True,
        "full_deepgene_reproduction": False,
        "feature_policy": feature_policy,
        "feature_names": feature_names,
        "label": label_desc,
        "evaluation_universe": evaluation_universe,
        "closures": closures,
        "test_chrs": sorted(test_chr_norm),
        "val_chrs": sorted(val_chr_norm),
        "context_radius": int(context_radius),
        "sequence_length_nodes": int(2 * context_radius + 1),
        "include_sequence_features": bool(include_sequence_features),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "train_label_counts": count_labels(y[train_idx], class_names),
        "test_label_counts": count_labels(y[test_idx], class_names),
        "best_val_macro_f1": float(best_val),
        "test_metrics": test_metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(test_metrics, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepGene-style serialized graph Transformer cCRE baseline.")
    ap.add_argument("--full_segments", required=True)
    ap.add_argument("--full_links", required=True)
    ap.add_argument("--node_labels", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_chrs", nargs="+", required=True)
    ap.add_argument("--val_chrs", nargs="+", required=True)
    ap.add_argument(
        "--label_scheme",
        choices=["binary", "full9", "multiclass", "group3", "group4", "group5", "category_binary"],
        default="binary",
    )
    ap.add_argument("--positive_group", default=None)
    ap.add_argument("--all_ccre_as_negative", action="store_true")
    ap.add_argument("--evaluation_universe", choices=["all", "benchmark_windows"], default="benchmark_windows")
    ap.add_argument("--benchmark_manifest", default=None)
    ap.add_argument("--closures", nargs="+", choices=["strict", "1hop"], default=["strict", "1hop"])
    ap.add_argument(
        "--feature_policy",
        choices=["leakage_safe", "legacy_sr", "legacy_reference"],
        default="leakage_safe",
    )
    ap.add_argument("--context_radius", type=int, default=15)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--include_sequence_features", action="store_true")
    ap.add_argument("--max_train_nodes", type=int, default=None)
    ap.add_argument("--max_val_nodes", type=int, default=None)
    ap.add_argument("--max_test_nodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = ap.parse_args()
    run_serialized_baseline(
        full_segments=args.full_segments,
        full_links=args.full_links,
        node_labels=args.node_labels,
        out_dir=args.out_dir,
        test_chrs=args.test_chrs,
        val_chrs=args.val_chrs,
        label_scheme=args.label_scheme,
        positive_group=args.positive_group,
        background_only_negative=not args.all_ccre_as_negative,
        evaluation_universe=args.evaluation_universe,
        benchmark_manifest=args.benchmark_manifest,
        closures=args.closures,
        feature_policy=args.feature_policy,
        context_radius=args.context_radius,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        include_sequence_features=args.include_sequence_features,
        max_train_nodes=args.max_train_nodes,
        max_val_nodes=args.max_val_nodes,
        max_test_nodes=args.max_test_nodes,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()

