"""
Cross-slice shared training of DualStreamPangenomeGAT on cCRE node
classification. Mirrors training.pretrain but targets multi-class
node labels instead of binary edge labels.

Key design:
  * Reuses per-slice loading from 06b (node features, edge index, SO tensor,
    orient, pop_ids, branching_frac, adaptive window_k).
  * Swaps the final edge_predictor for NodeClsHead (softmax over 9 cCRE classes).
  * Drops is_grch38 from node features (feature index 3 in models.gat.build_node_features)
    to avoid a cCRE-label shortcut once we later extend to alt-haplotype nodes.
  * Trains with multi-class focal loss (models.heads.multiclass_focal_loss) and
    inverse-frequency class weights computed from the train split.
  * Held-out chromosome evaluation matches 06b's protocol so the number is
    directly comparable to the 0.980 strict-AUC anchor.

Expected runtime:
  ~CPU small benchmark (benchmark_v4, strict, 24 chr * 10 windows): 45-90 min
  for 60 epochs + patience 15.

Usage
-----
python -m tasks.ccre.train_gat \\
    --manifest        data/hprc/benchmark/manifest.csv \\
    --full_segments   data/hprc/full_segments.csv \\
    --node_labels     data/hprc/ccre/run_001/node_labels.csv.gz \\
    --out_dir         results/hprc/ccre_gat \\
    --hidden_dim 48 --n_heads 4 --n_layers 2 \\
    --epochs 60 --patience 15 \\
    --dual_stream \\
    --adaptive_window --adaptive_window_base 32 --adaptive_window_alpha 4.0 \\
    --multiscale_rope --n_rope_scales 3 \\
    --orientation_rope \\
    --focal_gamma 2.0 \\
    --warmup_epochs 5 \\
    --test_chrs GRCh38#0#chr8 GRCh38#0#chr19 GRCh38#0#chr22 \\
    --val_chrs GRCh38#0#chr16 \\
    --device cpu
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from graph.io import read_segments_csv
from graph.slicing import build_global_index
from graph.features import build_oid_metadata_from_segments
from graph.neg_sampling import (
    oriented_ids_from_links,
    slice_oriented_node_set,
    compute_oriented_degrees,
)
from analysis.network import analyze, build_adjacency
from models.attention_window import compute_branching_distances
from models.gat import build_node_features, build_edge_features, build_dense_edge_index
from models.dual_stream_gat import DualStreamPangenomeGAT, build_pop_ids_array
from models.heads import NodeClsHead, multiclass_focal_loss
from utils.versioning import resolve_run_dir
from tasks.ccre.encoding import CCRE_CLASSES, CCRE_CLASS_TO_IDX, N_CCRE_CLASSES
from tasks.ccre.label_groups import (
    class_names_for_scheme,
    map_label_indices,
    n_classes_for_scheme,
)

# -------- Drop `is_grch38` (feature index 3) from build_node_features output -----

IS_GRCH38_IDX: int = 3  # from models.gat feature layout


def _drop_is_grch38(X: np.ndarray) -> np.ndarray:
    """Remove the is_grch38 column from a (N, 7) feature matrix -> (N, 6)."""
    keep = [i for i in range(X.shape[1]) if i != IS_GRCH38_IDX]
    return X[:, keep]


# ---------------------------------------------------------------------------
# Slice loader (adapted from 06b_shared_train.load_slice)
# ---------------------------------------------------------------------------


def _compute_adaptive_window_k(
    branching_frac: float,
    base: int,
    alpha: float,
    wk_min: int = 32,
    wk_max: int = 256,
) -> int:
    k = int(base * (1.0 + alpha * branching_frac))
    return max(wk_min, min(wk_max, k))


def load_slice_ccre(
    row: pd.Series,
    seg_index,
    md: Dict,
    segid_to_label: Dict[int, int],
    args: argparse.Namespace,
) -> Optional[Dict]:
    """
    Load a slice for cCRE node classification.

    Differences from 06b.load_slice:
      * Builds per-node cCRE label vector from segid_to_label.
      * Drops the is_grch38 column from node features.
      * No edge_pred data needed (we don't do link prediction here).
    """
    seg_sub = pd.read_csv(row["segments_path"], compression="infer")
    links_sub = pd.read_csv(row["links_path"], compression="infer")

    if len(links_sub) == 0:
        return None

    u_struct, v_struct = oriented_ids_from_links(links_sub, seg_index)
    nodes = slice_oriented_node_set(u_struct, v_struct)
    deg_map = compute_oriented_degrees(u_struct, v_struct, nodes)

    adj = build_adjacency(u_struct, v_struct, nodes)
    stats = analyze(u_struct, v_struct, nodes, sample_paths=False)
    _ = compute_branching_distances(adj, stats.branching_nodes, 8)  # not used here

    X7 = build_node_features(
        nodes=nodes,
        oid_to_so=md["oid_to_so"],
        oid_to_ln=md["oid_to_ln"],
        oid_to_sr=md["oid_to_sr"],
        oid_to_is_grch38=md["oid_to_is_grch38"],
        oid_to_degree=deg_map,
        oid_to_component_id=stats.oid_to_component_id,
    )
    X = _drop_is_grch38(X7) if args.drop_is_grch38 else X7

    so_arr = np.array([md["oid_to_so"].get(int(n), 0) for n in nodes], dtype=np.int64)
    orient_arr = (nodes % 2).astype(np.int8)
    pop_ids_arr = (
        build_pop_ids_array(nodes, md["oid_to_sn"])
        if args.pop_cond and "oid_to_sn" in md
        else np.zeros(len(nodes), dtype=np.int64)
    )

    src, dst = build_dense_edge_index(u_struct, v_struct, nodes)

    edge_attr_arr = None
    if args.use_edge_features:
        edge_attr_arr = build_edge_features(u_struct, v_struct, nodes, md)

    # Build per-node labels: only labeled where the node's SEGID appears in the
    # cCRE label table. Unlabeled (alt-haplotype) nodes become ignore_index.
    # Both orientations inherit the same label.
    ignore_idx = -100
    n_nodes = len(nodes)
    y = np.full(n_nodes, fill_value=ignore_idx, dtype=np.int64)
    for i, oid in enumerate(nodes.tolist()):
        seg_id = int(oid) // 2
        if seg_id in segid_to_label:
            y[i] = int(segid_to_label[seg_id])
    y = map_label_indices(y, scheme=args.task, ignore_index=ignore_idx)
    n_labeled = int((y != ignore_idx).sum())
    if n_labeled < 4:
        return None

    return {
        "name": row["name"],
        "target_sn": row["target_sn"],
        "closure": row["closure"],
        "nodes": nodes,
        "node_feats": X,
        "so_arr": so_arr,
        "orient_arr": orient_arr,
        "pop_ids_arr": pop_ids_arr,
        "branching_frac": float(stats.branching_frac),
        "src": src,
        "dst": dst,
        "edge_attr": edge_attr_arr,
        "y": y,
        "n_nodes": n_nodes,
        "n_labeled": n_labeled,
    }


def tensorize_slice(sd: Dict, device, args) -> Dict:
    t = {
        "X": torch.tensor(sd["node_feats"], dtype=torch.float32, device=device),
        "so": torch.tensor(sd["so_arr"], dtype=torch.int64, device=device),
        "src": torch.tensor(sd["src"], dtype=torch.long, device=device),
        "dst": torch.tensor(sd["dst"], dtype=torch.long, device=device),
        "y": torch.tensor(sd["y"], dtype=torch.long, device=device),
        "temps": torch.ones(sd["n_nodes"], dtype=torch.float32, device=device),
        "edge_attr": (
            torch.tensor(sd["edge_attr"], dtype=torch.float32, device=device)
            if sd.get("edge_attr") is not None
            else None
        ),
        "orient": (
            torch.tensor(sd["orient_arr"], dtype=torch.int64, device=device)
            if args.orientation_rope
            else None
        ),
        "pop_ids": (
            torch.tensor(sd["pop_ids_arr"], dtype=torch.int64, device=device)
            if args.pop_cond
            else None
        ),
        "branching_frac": sd["branching_frac"],
        "name": sd["name"],
        "target_sn": sd["target_sn"],
        "closure": sd["closure"],
        "n_nodes": sd["n_nodes"],
        "n_labeled": sd["n_labeled"],
    }
    return t


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------


def compute_class_weights(
    slices: List[Dict],
    n_classes: int,
    device,
    mode: str = "inverse_sqrt",
) -> "torch.Tensor":
    """Compute per-class weights from class frequencies in the train slices.
    mode:
      * 'none'         : all ones
      * 'inverse'      : 1/freq   (too aggressive usually)
      * 'inverse_sqrt' : 1/sqrt(freq) (default; stabler)
    """
    counts = torch.zeros(n_classes, dtype=torch.float64)
    for sd in slices:
        y = sd["y"]
        # y is a CPU tensor or numpy; count per class
        if isinstance(y, torch.Tensor):
            yy = y[y != -100]
            vals, cnts = torch.unique(yy, return_counts=True)
            for v, c in zip(vals.tolist(), cnts.tolist()):
                counts[v] += c
        else:
            uniq, cnt = np.unique(y[y != -100], return_counts=True)
            for v, c in zip(uniq.tolist(), cnt.tolist()):
                counts[v] += c
    # Avoid div0
    freqs = counts / counts.clamp(min=1).sum()
    if mode == "none":
        w = torch.ones(n_classes, dtype=torch.float32)
    elif mode == "inverse":
        w = 1.0 / freqs.clamp(min=1e-6)
    elif mode == "inverse_sqrt":
        w = 1.0 / freqs.clamp(min=1e-6).sqrt()
    else:
        raise ValueError(f"unknown class_weight_mode: {mode}")
    # Normalize mean to 1 for stable LR
    w = (w / w.mean()).float().to(device)
    return w


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step_count = 0

    def step(self):
        self._step_count += 1
        if self._step_count <= self.warmup_epochs:
            factor = self._step_count / max(self.warmup_epochs, 1)
        else:
            progress = (self._step_count - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * factor


def train_one_epoch(
    model: "DualStreamPangenomeGAT",
    head: "NodeClsHead",
    optimizer,
    slices: List[Dict],
    class_weights: "torch.Tensor",
    args,
    epoch: int,
) -> float:
    model.train()
    head.train()
    rng = np.random.default_rng(args.seed + epoch)
    order = rng.permutation(len(slices))

    accum_steps = args.accum_steps
    total_loss = 0.0
    n_steps = 0
    optimizer.zero_grad()

    for i, si in enumerate(order):
        sd = slices[si]

        if args.adaptive_window:
            eff_wk = _compute_adaptive_window_k(
                sd["branching_frac"],
                args.adaptive_window_base,
                args.adaptive_window_alpha,
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
        )
        logits = head(h)  # (N, C)
        loss = multiclass_focal_loss(
            logits,
            sd["y"],
            gamma=args.focal_gamma,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
            ignore_index=-100,
        )
        loss = loss / accum_steps
        loss.backward()
        total_loss += loss.item() * accum_steps
        n_steps += 1

        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    if n_steps % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def evaluate(
    model: "DualStreamPangenomeGAT",
    head: "NodeClsHead",
    slices: List[Dict],
    args,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """Return (macro_f1, per_slice_df, pooled_preds_df)."""
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    model.eval()
    head.eval()

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_prob: List[np.ndarray] = []
    all_chrom: List[str] = []
    per_slice_rows: List[Dict] = []
    n_classes = n_classes_for_scheme(args.task)

    for sd in slices:
        if args.adaptive_window:
            eff_wk = _compute_adaptive_window_k(
                sd["branching_frac"],
                args.adaptive_window_base,
                args.adaptive_window_alpha,
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
        )
        logits = head(h)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        pred = logits.argmax(dim=-1).cpu().numpy()
        y = sd["y"].cpu().numpy()
        mask = y != -100
        if mask.sum() < 4:
            continue
        y_t = y[mask]
        y_p = pred[mask]
        p_pos = probs[mask, 1] if n_classes == 2 else None
        chrom_name = str(sd["target_sn"]).split("#")[-1]

        f1 = f1_score(
            y_t,
            y_p,
            average="macro",
            labels=list(range(n_classes)),
            zero_division=0,
        )
        row = {
            "name": sd["name"],
            "target_sn": sd["target_sn"],
            "chrom": chrom_name,
            "closure": sd["closure"],
            "n_nodes": sd["n_nodes"],
            "n_labeled": int(mask.sum()),
            "branching_frac": float(sd["branching_frac"]),
            "macro_f1": float(f1),
        }
        if n_classes == 2:
            row["positive_fraction"] = float(np.mean(y_t))
            row["balanced_accuracy"] = float(balanced_accuracy_score(y_t, y_p))
            if len(np.unique(y_t)) == 2:
                row["auroc"] = float(roc_auc_score(y_t, p_pos))
                row["auprc"] = float(average_precision_score(y_t, p_pos))
            else:
                row["auroc"] = float("nan")
                row["auprc"] = float("nan")
        per_slice_rows.append(row)
        all_true.append(y_t)
        all_pred.append(y_p)
        if p_pos is not None:
            all_prob.append(p_pos)
        all_chrom.extend([chrom_name] * len(y_t))

    if not per_slice_rows:
        return 0.0, pd.DataFrame(), pd.DataFrame()

    y_t = np.concatenate(all_true)
    y_p = np.concatenate(all_pred)
    pooled_f1 = float(
        f1_score(
            y_t,
            y_p,
            average="macro",
            labels=list(range(n_classes)),
            zero_division=0,
        )
    )

    per_slice = pd.DataFrame(per_slice_rows)
    pooled_data = {"chrom": all_chrom, "y_true": y_t, "y_pred": y_p}
    if n_classes == 2 and all_prob:
        pooled_data["p_ccre"] = np.concatenate(all_prob)
    pooled = pd.DataFrame(pooled_data)
    return pooled_f1, per_slice, pooled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required.")

    ap = argparse.ArgumentParser(
        description="Cross-slice shared training for cCRE node classification."
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--full_segments", required=True)
    ap.add_argument(
        "--node_labels", required=True, help="Output of scripts.10_encode_ccre_to_nodes"
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--task",
        default="multiclass",
        choices=["multiclass", "full9", "binary", "group3", "group4", "group5"],
        help="cCRE label scheme: full 9-way, binary, or reduced grouped labels.",
    )

    # Architecture (mirror 06b defaults)
    ap.add_argument("--hidden_dim", type=int, default=48)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Dual-stream
    ap.add_argument("--dual_stream", action="store_true", default=False)
    ap.add_argument("--no_rope", action="store_true", default=False)
    ap.add_argument("--no_fusion_gate", action="store_true", default=False)
    ap.add_argument(
        "--stream_mode",
        choices=["full", "coordinate", "graph"],
        default="full",
        help="Architecture ablation: full, coordinate-only, or graph-only backbone.",
    )
    ap.add_argument("--multiscale_rope", action="store_true", default=False)
    ap.add_argument("--n_rope_scales", type=int, default=3)
    ap.add_argument("--orientation_rope", action="store_true", default=False)
    ap.add_argument("--pop_cond", action="store_true", default=False)
    ap.add_argument("--pop_embed_dim", type=int, default=16)
    ap.add_argument("--pop_table", type=str, default=None)

    # Window
    ap.add_argument("--adaptive_window", action="store_true", default=False)
    ap.add_argument("--adaptive_window_base", type=int, default=32)
    ap.add_argument("--adaptive_window_alpha", type=float, default=4.0)
    ap.add_argument("--window_k", type=int, default=None)

    # Feature toggles
    ap.add_argument(
        "--drop_is_grch38",
        action="store_true",
        default=True,
        help="Drop is_grch38 from node features (default: on).",
    )
    ap.add_argument(
        "--keep_is_grch38",
        action="store_true",
        default=False,
        help="Override: keep is_grch38 (not recommended for cCRE).",
    )
    ap.add_argument("--use_edge_features", action="store_true", default=False)
    ap.add_argument(
        "--pretrained_checkpoint",
        default=None,
        help="Optional link-pretraining checkpoint whose backbone weights seed the cCRE model.",
    )
    ap.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=False,
        help="Freeze the pretrained backbone and train only the cCRE head.",
    )

    # Training
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--accum_steps", type=int, default=4)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument(
        "--class_weight_mode",
        type=str,
        default="inverse_sqrt",
        choices=["none", "inverse", "inverse_sqrt"],
    )

    # Other
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])

    # Cross-chromosome holdout (same format as 06b)
    ap.add_argument(
        "--test_chrs",
        nargs="+",
        default=None,
        help="Full SN strings (e.g. GRCh38#0#chr8) held out from training.",
    )
    ap.add_argument(
        "--val_chrs",
        nargs="+",
        default=None,
        help="Chromosomes for validation. For paper runs these should be "
        "disjoint from --test_chrs. If unset, uses the first test_chr for "
        "quick iteration.",
    )

    args = ap.parse_args()
    if args.keep_is_grch38:
        args.drop_is_grch38 = False

    device = torch.device(args.device)

    out_dir = resolve_run_dir(Path(args.out_dir))
    print(f"[12] output: {out_dir}")
    print(f"[12] device: {device}")
    print(f"[12] task: {args.task}")
    print(f"[12] drop_is_grch38: {args.drop_is_grch38}")
    if args.test_chrs:
        print(f"[12] test_chrs: {args.test_chrs}")

    # 1. Load seg metadata
    print(f"[12] Loading segments...")
    segments = read_segments_csv(args.full_segments)
    seg_index, _ = build_global_index(segments)
    md = build_oid_metadata_from_segments(segments, seg_index)

    if args.pop_cond and args.pop_table:
        from models.dual_stream_gat import load_pop_table

        n_loaded = load_pop_table(args.pop_table)
        print(f"[12] Loaded {n_loaded} population labels.")

    # 2. Load node labels -> segid_to_label
    print(f"[12] Loading node labels from {args.node_labels}")
    nl = pd.read_csv(args.node_labels, compression="infer")
    segid_to_label: Dict[int, int] = dict(
        zip(nl["segid"].astype(int).tolist(), nl["ccre_label"].astype(int).tolist())
    )
    print(f"[12] segid_to_label entries: {len(segid_to_label):,}")

    # 3. Load all slices
    manifest = pd.read_csv(args.manifest)
    print(f"[12] Loading {len(manifest)} slices ...")
    all_slices_raw = []
    for _, row in manifest.iterrows():
        sd = load_slice_ccre(row, seg_index, md, segid_to_label, args)
        if sd is not None:
            all_slices_raw.append(sd)
    print(f"[12] Loaded {len(all_slices_raw)} slices with >=4 labeled nodes")

    # 4. Cross-chr split
    test_chr_set = set(args.test_chrs) if args.test_chrs else set()
    val_chr_set = (
        set(args.val_chrs)
        if args.val_chrs
        else (set([list(test_chr_set)[0]]) if test_chr_set else set())
    )
    if val_chr_set & test_chr_set:
        print(
            "[12] WARNING: val_chrs overlap test_chrs. This is acceptable for "
            "quick iteration but not for final paper runs."
        )
    print(f"[12] val_chrs: {sorted(val_chr_set)}")

    train_slices_raw = [
        s
        for s in all_slices_raw
        if s["target_sn"] not in test_chr_set and s["target_sn"] not in val_chr_set
    ]
    val_slices_raw = [s for s in all_slices_raw if s["target_sn"] in val_chr_set]
    test_slices_raw = [s for s in all_slices_raw if s["target_sn"] in test_chr_set]
    print(f"[12] train slices: {len(train_slices_raw)}")
    print(f"[12] val   slices: {len(val_slices_raw)}")
    print(f"[12] test  slices: {len(test_slices_raw)}")

    available_targets = {str(s["target_sn"]) for s in all_slices_raw}
    if args.val_chrs and len(val_slices_raw) == 0:
        available_preview = ", ".join(sorted(available_targets)[:12])
        raise RuntimeError(
            "No validation slices matched --val_chrs. "
            f"Requested {sorted(val_chr_set)}; available targets include: {available_preview}"
        )
    if args.test_chrs and len(test_slices_raw) == 0:
        available_preview = ", ".join(sorted(available_targets)[:12])
        raise RuntimeError(
            "No test slices matched --test_chrs. "
            f"Requested {sorted(test_chr_set)}; available targets include: {available_preview}"
        )
    if len(train_slices_raw) == 0:
        raise RuntimeError("No training slices (check --test_chrs format).")

    # 5. Tensorize
    slices_train = [tensorize_slice(sd, device, args) for sd in train_slices_raw]
    slices_val = [tensorize_slice(sd, device, args) for sd in val_slices_raw]
    slices_test = [tensorize_slice(sd, device, args) for sd in test_slices_raw]
    n_classes = n_classes_for_scheme(args.task)
    class_names = class_names_for_scheme(args.task)

    # 6. Build model
    in_dim = slices_train[0]["X"].shape[1]
    effective_window_k = (
        args.window_k
        if args.window_k is not None
        else (64 if args.adaptive_window else None)
    )
    edge_feat_dim = (
        slices_train[0]["edge_attr"].shape[1]
        if slices_train[0].get("edge_attr") is not None
        else 0
    )

    model = DualStreamPangenomeGAT(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        edge_mlp_dim=args.hidden_dim * 2,  # unused here (we replace head)
        edge_feat_dim=edge_feat_dim,
        use_rope=not args.no_rope,
        use_fusion_gate=not args.no_fusion_gate,
        window_k=effective_window_k,
        use_multiscale_rope=args.multiscale_rope,
        n_rope_scales=args.n_rope_scales,
        use_orientation=args.orientation_rope,
        use_cross_attn=False,
        pop_embed_dim=args.pop_embed_dim if args.pop_cond else 0,
        stream_mode=args.stream_mode,
    ).to(device)

    if args.pretrained_checkpoint:
        ckpt = torch.load(args.pretrained_checkpoint, map_location=device)
        ckpt_in_dim = int(ckpt.get("in_dim", in_dim))
        if ckpt_in_dim != int(in_dim):
            raise RuntimeError(
                "Pretrained checkpoint input dimension does not match cCRE model. "
                f"checkpoint in_dim={ckpt_in_dim}, cCRE in_dim={int(in_dim)}. "
                "Use --keep_is_grch38 for checkpoints trained with the full 7 features, "
                "or train a compatible checkpoint."
            )
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[12] loaded pretrained backbone: {args.pretrained_checkpoint}")
        if missing:
            print(f"[12] missing checkpoint keys: {len(missing)}")
        if unexpected:
            print(f"[12] unexpected checkpoint keys: {len(unexpected)}")
        if args.freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
            print("[12] frozen pretrained backbone; training cCRE head only")

    head = NodeClsHead(
        hidden_dim=args.hidden_dim,
        n_classes=n_classes,
        mlp_dim=args.hidden_dim * 2,
        dropout=args.dropout,
    ).to(device)

    print(f"[12] backbone params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[12] head params:     {sum(p.numel() for p in head.parameters()):,}")

    # 7. Class weights on TRAIN only
    class_weights = compute_class_weights(
        slices_train,
        n_classes,
        device,
        mode=args.class_weight_mode,
    )
    print(f"[12] class weights ({args.class_weight_mode}):")
    for cls, w in zip(class_names, class_weights.cpu().tolist()):
        print(f"    {cls:12s} {w:.3f}")

    # 8. Optimizer + scheduler
    params = [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)

    # 9. Training loop
    best_val_f1 = -1.0
    best_model_state = None
    best_head_state = None
    patience_counter = 0
    history: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, head, optimizer, slices_train, class_weights, args, epoch
        )
        scheduler.step()

        val_f1 = 0.0
        if slices_val:
            val_f1, _, _ = evaluate(model, head, slices_val, args)
        lr_now = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_macro_f1": val_f1,
                "lr": lr_now,
            }
        )
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"  epoch {epoch:3d}  loss={train_loss:.4f}  "
                f"val_macro_f1={val_f1:.4f}  lr={lr_now:.2e}"
            )

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_head_state = copy.deepcopy(head.state_dict())
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print(f"  early stop @ epoch {epoch}  best_val_f1={best_val_f1:.4f}")
            break

    # 10. Restore best and evaluate on test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        head.load_state_dict(best_head_state)

    print(f"\n[12] === FINAL EVAL ===")
    print(f"  best_val_macro_f1: {best_val_f1:.4f}")

    if slices_test:
        test_f1, test_per_slice, test_pooled = evaluate(model, head, slices_test, args)
        test_metrics = {"macro_f1": float(test_f1)}
        if n_classes == 2 and not test_pooled.empty:
            from sklearn.metrics import (
                average_precision_score,
                balanced_accuracy_score,
                roc_auc_score,
            )

            y_true = test_pooled["y_true"].to_numpy()
            y_pred = test_pooled["y_pred"].to_numpy()
            p_ccre = test_pooled["p_ccre"].to_numpy()
            test_metrics["balanced_accuracy"] = float(
                balanced_accuracy_score(y_true, y_pred)
            )
            test_metrics["positive_fraction"] = float(np.mean(y_true))
            if len(np.unique(y_true)) == 2:
                test_metrics["auroc"] = float(roc_auc_score(y_true, p_ccre))
                test_metrics["auprc"] = float(
                    average_precision_score(y_true, p_ccre)
                )
        print(f"  HELD-OUT test_macro_f1 (pooled): {test_f1:.4f}")
        if n_classes == 2:
            print(
                "  HELD-OUT binary: "
                f"AUROC={test_metrics.get('auroc', float('nan')):.4f}  "
                f"AUPRC={test_metrics.get('auprc', float('nan')):.4f}  "
                f"balanced_acc={test_metrics.get('balanced_accuracy', float('nan')):.4f}"
            )
        print(f"  ^^^ THIS IS THE NUMBER THAT MATTERS FOR THE PAPER ^^^")
        test_per_slice.to_csv(out_dir / "test_per_slice.csv", index=False)
        test_pooled.to_csv(
            out_dir / "test_pooled_preds.csv.gz", index=False, compression="gzip"
        )

        # Per-chrom test macro-F1
        print("\n[12] Per-chromosome test macro-F1:")
        print(test_per_slice.groupby("chrom")["macro_f1"].mean().round(4).to_string())
        if n_classes == 2 and "auroc" in test_per_slice.columns:
            print("\n[12] Per-chromosome binary AUROC:")
            print(test_per_slice.groupby("chrom")["auroc"].mean().round(4).to_string())
    else:
        test_f1 = float("nan")
        test_metrics = {}

    # 11. Save artifacts
    torch.save(
        {
            "model_state": model.state_dict(),
            "head_state": head.state_dict(),
            "args": vars(args),
            "in_dim": in_dim,
            "n_classes": n_classes,
            "ccre_classes": class_names,
        },
        out_dir / "ckpt_best.pt",
    )
    print(f"[12] Saved checkpoint: {out_dir / 'ckpt_best.pt'}")

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    summary = {
        "task": args.task,
        "best_val_macro_f1": float(best_val_f1),
        "test_macro_f1": float(test_f1),
        "test_metrics": test_metrics,
        "n_train_slices": len(slices_train),
        "n_val_slices": len(slices_val),
        "n_test_slices": len(slices_test),
        "in_dim": int(in_dim),
        "args": vars(args),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[12] Saved summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
