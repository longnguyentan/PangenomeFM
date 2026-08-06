"""
Shared pretraining for the GraphGenome-FM graph encoder.

KEY DIFFERENCE from 06_gat_train.py:
  06  → trains a SEPARATE model per slice (240 independent models, ~190 edges each)
  06b → trains ONE model across ALL slices (shared weights, ~45K edges total)

Diagnostic finding driving this change:
  - Slices training 100+ epochs achieve 0.893 strict AUC
  - Slices training ≤35 epochs achieve 0.513 (chance)
  - epochs_run ↔ test_auc correlation: r = 0.849
  → The model CAN solve strict when it has enough signal.
  → 52% of slices are data-starved with only ~190 train edges each.

Approach:
  1. Load ALL slices and pre-split each into train/val/test edges
  2. Create ONE shared DualStreamPangenomeGAT
  3. Each epoch: iterate through all slices, accumulate gradients, step
  4. Validate per-slice, early-stop on mean val AUC
  5. Evaluate per-slice on test edges with the shared model

Also incorporates:
  - Focal loss (--focal_loss, --focal_gamma)
  - DropEdge augmentation (--drop_edge, --drop_edge_rate)
  - Edge features (--use_edge_features)
  - LR warmup + cosine decay (--warmup_epochs)
  - More expressive link predictor with element-wise features
  - JK-Net style layer aggregation (--jk_mode)

Usage:
    python -m training.pretrain \\
        --manifest data/hprc/benchmark/manifest.csv \\
        --full_segments data/hprc/full_segments.csv \\
        --out_dir results/v3/gat \\
        --hidden_dim 48 --n_heads 4 --n_layers 2 \\
        --epochs 100 --patience 20 \\
        --dual_stream \\
        --adaptive_window --adaptive_window_base 32 --adaptive_window_alpha 4.0 \\
        --multiscale_rope --n_rope_scales 3 \\
        --orientation_rope \\
        --focal_loss --focal_gamma 2.0 \\
        --drop_edge --drop_edge_rate 0.1 \\
        --warmup_epochs 5 \\
        --device cpu
"""

from __future__ import annotations

import argparse
import copy
import math
import json
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
from models.dual_stream_gat import (
    DualStreamPangenomeGAT,
    build_pop_ids_array,
)
from models.losses import focal_bce_loss
from evaluation.splits import normalize_chrom, validate_chromosome_split


# ---------------------------------------------------------------------------
# Drop-edge augmentation
# ---------------------------------------------------------------------------


def drop_edges(
    src: "torch.Tensor",
    dst: "torch.Tensor",
    drop_rate: float,
    edge_attr: Optional["torch.Tensor"] = None,
    training: bool = True,
) -> Tuple["torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
    """
    Randomly drop a fraction of structural edges during training.

    Regularisation for small-data regimes: forces the model to learn
    robust representations even when some edges are missing.

    Args:
        src, dst:   (E,) edge index tensors
        drop_rate:  fraction of edges to drop (e.g. 0.1 = 10%)
        training:   only drop during training; return unchanged at eval

    Returns:
        (src_aug, dst_aug) with a random subset of edges removed.
    """
    if not training or drop_rate <= 0.0:
        return src, dst, edge_attr
    E = src.size(0)
    mask = torch.rand(E, device=src.device) >= drop_rate
    edge_attr_aug = edge_attr[mask] if edge_attr is not None else None
    return src[mask], dst[mask], edge_attr_aug


def mask_positive_query_edges(
    src: "torch.Tensor",
    dst: "torch.Tensor",
    edge_attr: Optional["torch.Tensor"],
    q_u: "torch.Tensor",
    q_v: "torch.Tensor",
    labels: "torch.Tensor",
    idx: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
    """Remove positive query edges from the message-passing graph.

    This supports leakage-audited link prediction: the model may use the
    surrounding observed topology, but the candidate positive edge currently
    being scored is not available as a structural GAT edge.
    """
    if idx.numel() == 0:
        return src, dst, edge_attr

    pos_idx = idx[labels[idx] > 0.5]
    if pos_idx.numel() == 0:
        return src, dst, edge_attr

    # Hash directed pairs into one integer key.  The previous per-query loop
    # was quadratic in the number of structural/query edges and made
    # chromosome-scale windows effectively unusable.
    max_node = torch.max(
        torch.cat([src, dst, q_u[pos_idx], q_v[pos_idx]])
    ).to(torch.int64)
    key_base = max_node + 1
    structural_keys = src.to(torch.int64) * key_base + dst.to(torch.int64)
    positive_keys = (
        q_u[pos_idx].to(torch.int64) * key_base + q_v[pos_idx].to(torch.int64)
    )
    keep = ~torch.isin(structural_keys, torch.unique(positive_keys))

    edge_attr_masked = edge_attr[keep] if edge_attr is not None else None
    return src[keep], dst[keep], edge_attr_masked


# ---------------------------------------------------------------------------
# Enhanced link predictor (Priority 10)
# ---------------------------------------------------------------------------


class ExpressiveLinkPredictor(nn.Module):
    """
    More expressive link predictor that uses multiple interaction features:
        score = MLP([h_u, h_v, h_u * h_v, |h_u - h_v|])

    The element-wise product captures similarity between node embeddings.
    The absolute difference captures asymmetry.
    Together with concatenation, this gives the predictor 4x the input
    compared to a simple [h_u || h_v] MLP.
    """

    def __init__(self, hidden_dim: int, mlp_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        # Input: [h_u, h_v, h_u*h_v, |h_u-h_v|] = 4 * hidden_dim
        in_dim = 4 * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, 1),
        )

    def forward(self, h_u: "torch.Tensor", h_v: "torch.Tensor") -> "torch.Tensor":
        features = torch.cat(
            [h_u, h_v, h_u * h_v, (h_u - h_v).abs()],
            dim=-1,
        )
        return self.mlp(features).squeeze(-1)


if TORCH_AVAILABLE:

    class _GradientReversal(torch.autograd.Function):
        """Identity forward pass with a sign-reversed encoder gradient."""

        @staticmethod
        def forward(ctx, value: "torch.Tensor", scale: float) -> "torch.Tensor":
            ctx.scale = float(scale)
            return value.view_as(value)

        @staticmethod
        def backward(ctx, grad_output: "torch.Tensor"):
            return -ctx.scale * grad_output, None


    class DatasetDiscriminator(nn.Module):
        """Predict construction dataset from a pooled slice representation."""

        def __init__(self, hidden_dim: int, n_domains: int, dropout: float = 0.1):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_domains),
            )

        def forward(self, pooled: "torch.Tensor", grl_scale: float) -> "torch.Tensor":
            reversed_features = _GradientReversal.apply(pooled, grl_scale)
            return self.network(reversed_features)


# ---------------------------------------------------------------------------
# Slice loader (reuses logic from 06_gat_train)
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


def split_candidate_indices(
    n_candidates: int, split_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic 70/10/20 indices independent of model seed."""

    rng = np.random.default_rng(split_seed)
    indices = rng.permutation(n_candidates)
    n_test = int(n_candidates * 0.2)
    n_val = int(n_candidates * 0.1)
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    return train_idx, val_idx, test_idx


def load_slice(
    row: pd.Series,
    seg_index,
    md: Dict,
    full_segments: pd.DataFrame,
    args: argparse.Namespace,
) -> Optional[Dict]:
    """Load one manifest row into tensors."""
    seg_sub = pd.read_csv(row["segments_path"], compression="infer")
    links_sub = pd.read_csv(row["links_path"], compression="infer")
    edge_df = pd.read_csv(row["edge_pred_path"], compression="infer")

    if len(links_sub) == 0 or len(edge_df) == 0:
        return None

    u_struct, v_struct = oriented_ids_from_links(links_sub, seg_index)
    nodes = slice_oriented_node_set(u_struct, v_struct)
    deg_map = compute_oriented_degrees(u_struct, v_struct, nodes)

    adj = build_adjacency(u_struct, v_struct, nodes)
    stats = analyze(u_struct, v_struct, nodes, sample_paths=False)
    oid_to_radius = compute_branching_distances(adj, stats.branching_nodes, 8)

    X = build_node_features(
        nodes=nodes,
        oid_to_so=md["oid_to_so"],
        oid_to_ln=md["oid_to_ln"],
        oid_to_sr=md["oid_to_sr"],
        oid_to_is_grch38=md["oid_to_is_grch38"],
        oid_to_degree=deg_map,
        oid_to_component_id=stats.oid_to_component_id,
    )

    so_arr = np.array([md["oid_to_so"].get(int(n), 0) for n in nodes], dtype=np.int64)
    orient_arr = (nodes % 2).astype(np.int8)
    pop_ids_arr = (
        build_pop_ids_array(nodes, md["oid_to_sn"])
        if args.pop_cond and "oid_to_sn" in md
        else np.zeros(len(nodes), dtype=np.int64)
    )

    src, dst = build_dense_edge_index(u_struct, v_struct, nodes)
    oid_to_idx = {int(n): i for i, n in enumerate(nodes)}

    temp_arr = np.ones(len(nodes), dtype=np.float32)

    edge_attr_arr = None
    if args.use_edge_features:
        edge_attr_arr = build_edge_features(u_struct, v_struct, nodes, md)

    valid_mask = edge_df["u_oid"].isin(oid_to_idx) & edge_df["v_oid"].isin(oid_to_idx)
    edge_df_v = edge_df[valid_mask].reset_index(drop=True)
    if len(edge_df_v) == 0:
        return None

    query_u_arr = np.array(
        [oid_to_idx[int(x)] for x in edge_df_v["u_oid"]], dtype=np.int64
    )
    query_v_arr = np.array(
        [oid_to_idx[int(x)] for x in edge_df_v["v_oid"]], dtype=np.int64
    )
    labels_arr = edge_df_v["label"].to_numpy(dtype=np.float32)

    # Train/val/test split per slice
    n = len(labels_arr)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    train_idx, val_idx, test_idx = split_candidate_indices(n, split_seed)

    if len(train_idx) < 10:
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
        "temps": temp_arr,
        "edge_attr": edge_attr_arr,
        "query_u": query_u_arr,
        "query_v": query_v_arr,
        "labels": labels_arr,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "n_pos": int(labels_arr.sum()),
        "n_neg": int((1 - labels_arr).sum()),
    }


# ---------------------------------------------------------------------------
# Shared model training
# ---------------------------------------------------------------------------


def tensorize_slice(slice_data: Dict, device: "torch.device", args) -> Dict:
    """Convert numpy arrays to torch tensors."""
    d = {}
    d["X"] = torch.tensor(slice_data["node_feats"], dtype=torch.float32, device=device)
    d["so"] = torch.tensor(slice_data["so_arr"], dtype=torch.int64, device=device)
    d["src"] = torch.tensor(slice_data["src"], dtype=torch.long, device=device)
    d["dst"] = torch.tensor(slice_data["dst"], dtype=torch.long, device=device)
    d["temps"] = torch.tensor(slice_data["temps"], dtype=torch.float32, device=device)
    d["labels"] = torch.tensor(slice_data["labels"], dtype=torch.float32, device=device)
    d["q_u"] = torch.tensor(slice_data["query_u"], dtype=torch.long, device=device)
    d["q_v"] = torch.tensor(slice_data["query_v"], dtype=torch.long, device=device)
    d["train_idx"] = torch.tensor(
        slice_data["train_idx"], dtype=torch.long, device=device
    )
    d["val_idx"] = torch.tensor(slice_data["val_idx"], dtype=torch.long, device=device)
    d["test_idx"] = torch.tensor(
        slice_data["test_idx"], dtype=torch.long, device=device
    )
    d["edge_attr"] = (
        torch.tensor(slice_data["edge_attr"], dtype=torch.float32, device=device)
        if slice_data.get("edge_attr") is not None
        else None
    )
    d["orient"] = (
        torch.tensor(slice_data["orient_arr"], dtype=torch.int64, device=device)
        if args.orientation_rope
        else None
    )
    d["pop_ids"] = (
        torch.tensor(slice_data["pop_ids_arr"], dtype=torch.int64, device=device)
        if args.pop_cond
        else None
    )
    d["branching_frac"] = slice_data["branching_frac"]
    d["name"] = slice_data["name"]
    d["target_sn"] = slice_data["target_sn"]
    d["closure"] = slice_data["closure"]
    d["dataset"] = slice_data.get("dataset", "primary")
    d["n_nodes"] = len(slice_data["nodes"])
    d["n_pos"] = slice_data["n_pos"]
    d["n_neg"] = slice_data["n_neg"]
    return d


def maybe_tensorize_slice(
    slice_data: Dict, args: argparse.Namespace
) -> Tuple[Dict, bool]:
    """Return a tensor slice and whether it was created for this call.

    Lazy tensorization avoids keeping every genome-wide slice twice in RAM.
    The returned boolean lets callers drop their temporary reference promptly.
    """
    if "X" in slice_data:
        return slice_data, False
    return tensorize_slice(slice_data, torch.device(args.device), args), True


def leakage_safe_mask_indices(
    slice_data: Dict, query_idx: "torch.Tensor"
) -> "torch.Tensor":
    """Edges to hide for leakage-safe candidate-split reconstruction.

    Validation and test positives are never exposed while fitting a training
    chromosome.  The current training-query batch is additionally hidden,
    leaving the other training edges as graph context.
    """
    return torch.unique(
        torch.cat(
            [slice_data["val_idx"], slice_data["test_idx"], query_idx]
        )
    )


def train_one_epoch_shared(
    model: "DualStreamPangenomeGAT",
    predictor: Optional["ExpressiveLinkPredictor"],
    domain_classifier: Optional["DatasetDiscriminator"],
    dataset_to_idx: dict[str, int],
    optimizer: "torch.optim.Optimizer",
    slices: List[Dict],
    args: argparse.Namespace,
    epoch: int,
) -> float:
    """
    One epoch of shared training across all slices.

    For each slice:
      1. Optionally apply DropEdge to structural edges
      2. Encode nodes with shared model
      3. Score train edges with predictor
      4. Compute focal (or BCE) loss
      5. Accumulate gradients
    Step optimizer after every `accum_steps` slices.
    """
    model.train()
    if predictor is not None:
        predictor.train()
    if domain_classifier is not None:
        domain_classifier.train()

    # Shuffle slice order each epoch
    rng = np.random.default_rng(args.seed + epoch)
    order = rng.permutation(len(slices))

    total_loss = 0.0
    n_steps = 0
    accum_steps = getattr(args, "accum_steps", 4)  # accumulate over N slices

    optimizer.zero_grad()

    optimizer_steps_pending = 0
    for si in order:
        sd, _temporary = maybe_tensorize_slice(slices[si], args)
        train_idx = sd["train_idx"]
        if len(train_idx) < 10:
            continue

        # Per-slice adaptive window_k: dynamically set on the shared model
        # so each slice gets its own branching-aware window size.
        if args.adaptive_window:
            eff_wk = _compute_adaptive_window_k(
                sd["branching_frac"],
                args.adaptive_window_base,
                args.adaptive_window_alpha,
            )
            for layer in model.linear_layers:
                layer.window_k = eff_wk

        # Candidate mini-batches are essential here. Masking every training
        # positive simultaneously would erase nearly the whole structural
        # graph, while leaving validation/test positives visible would leak.
        permutation = torch.as_tensor(
            rng.permutation(len(train_idx)), dtype=torch.long, device=train_idx.device
        )
        shuffled_train_idx = train_idx[permutation]
        candidate_batch_size = max(1, int(args.batch_size))
        for batch_start in range(0, len(shuffled_train_idx), candidate_batch_size):
            batch_idx = shuffled_train_idx[
                batch_start : batch_start + candidate_batch_size
            ]

            src_for_mp = sd["src"]
            dst_for_mp = sd["dst"]
            edge_attr_for_mp = sd["edge_attr"]
            if args.mask_query_edges:
                mask_idx = leakage_safe_mask_indices(sd, batch_idx)
                src_for_mp, dst_for_mp, edge_attr_for_mp = mask_positive_query_edges(
                    src_for_mp,
                    dst_for_mp,
                    edge_attr_for_mp,
                    sd["q_u"],
                    sd["q_v"],
                    sd["labels"],
                    mask_idx,
                )

            src_aug, dst_aug, edge_attr_aug = drop_edges(
                src_for_mp,
                dst_for_mp,
                drop_rate=args.drop_edge_rate if args.drop_edge else 0.0,
                edge_attr=edge_attr_for_mp,
                training=True,
            )

            h = model.encode_nodes(
                sd["X"],
                sd["so"],
                src_aug,
                dst_aug,
                sd["temps"],
                edge_attr_aug,
                sd["orient"],
                sd["pop_ids"],
            )

            qu = sd["q_u"][batch_idx]
            qv = sd["q_v"][batch_idx]
            labels = sd["labels"][batch_idx]

            if predictor is not None:
                logits = predictor(h[qu], h[qv])
                probs = torch.sigmoid(logits)
            else:
                probs = model.edge_predictor(
                    torch.cat([h[qu], h[qv]], dim=-1)
                ).squeeze(-1)
                probs = torch.sigmoid(probs)

            n_pos = labels.sum().item()
            n_neg = len(labels) - n_pos
            pw = n_neg / max(n_pos, 1)

            if args.focal_loss:
                loss = focal_bce_loss(
                    probs,
                    labels,
                    gamma=args.focal_gamma,
                    alpha=args.focal_alpha,
                    pos_weight=pw,
                )
            else:
                weight = torch.where(
                    labels > 0.5,
                    torch.tensor(pw, device=labels.device),
                    torch.tensor(1.0, device=labels.device),
                )
                loss = F.binary_cross_entropy(probs, labels, weight=weight)

            if domain_classifier is not None:
                pooled = h.mean(dim=0, keepdim=True)
                domain_logits = domain_classifier(pooled, args.domain_grl_lambda)
                domain_target = torch.tensor(
                    [dataset_to_idx[str(sd["dataset"])]],
                    dtype=torch.long,
                    device=h.device,
                )
                domain_loss = F.cross_entropy(domain_logits, domain_target)
                loss = loss + args.domain_loss_weight * domain_loss

            loss = loss / accum_steps
            loss.backward()
            total_loss += loss.item() * accum_steps
            n_steps += 1
            optimizer_steps_pending += 1

            if optimizer_steps_pending == accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if predictor is not None:
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
                if domain_classifier is not None:
                    torch.nn.utils.clip_grad_norm_(
                        domain_classifier.parameters(), max_norm=1.0
                    )
                optimizer.step()
                optimizer.zero_grad()
                optimizer_steps_pending = 0

    # Final step for remaining slices
    if optimizer_steps_pending:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if predictor is not None:
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
        if domain_classifier is not None:
            torch.nn.utils.clip_grad_norm_(
                domain_classifier.parameters(), max_norm=1.0
            )
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def evaluate_shared(
    model: "DualStreamPangenomeGAT",
    predictor: Optional["ExpressiveLinkPredictor"],
    slices: List[Dict],
    split: str = "val",  # "val" or "test"
    args: Optional[argparse.Namespace] = None,
    prediction_rows: Optional[List[Dict]] = None,
    prediction_split_label: Optional[str] = None,
) -> Tuple[float, List[Dict]]:
    """
    Evaluate shared model per-slice.

    Returns:
        mean_auc: average AUC across all slices
        per_slice: list of {name, target_sn, closure, auc, ...}
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    if predictor is not None:
        predictor.eval()

    per_slice = []
    for slice_data in slices:
        sd, _temporary = maybe_tensorize_slice(slice_data, args)
        idx_key = f"{split}_idx"
        idx = sd[idx_key]
        if len(idx) < 4:
            continue

        # Per-slice adaptive window_k (must match training)
        if (
            args is not None
            and args.adaptive_window
            and hasattr(model, "linear_layers")
        ):
            bf = sd.get("branching_frac", 0.0)
            eff_wk = _compute_adaptive_window_k(
                bf, args.adaptive_window_base, args.adaptive_window_alpha
            )
            for layer in model.linear_layers:
                layer.window_k = eff_wk

        src_for_mp = sd["src"]
        dst_for_mp = sd["dst"]
        edge_attr_for_mp = sd["edge_attr"]
        if args is not None and getattr(args, "mask_query_edges", False):
            # Validation/test positives are absent from the structural graph
            # in every evaluation pass. For train metrics, also hide the
            # currently scored training positives.
            mask_idx = leakage_safe_mask_indices(sd, idx)
            src_for_mp, dst_for_mp, edge_attr_for_mp = mask_positive_query_edges(
                src_for_mp,
                dst_for_mp,
                edge_attr_for_mp,
                sd["q_u"],
                sd["q_v"],
                sd["labels"],
                mask_idx,
            )

        h = model.encode_nodes(
            sd["X"],
            sd["so"],
            src_for_mp,
            dst_for_mp,
            sd["temps"],
            edge_attr_for_mp,
            sd["orient"],
            sd["pop_ids"],
        )

        qu = sd["q_u"][idx]
        qv = sd["q_v"][idx]
        labels = sd["labels"][idx]

        if predictor is not None:
            logits = predictor(h[qu], h[qv])
            probs = torch.sigmoid(logits).cpu().numpy()
        else:
            probs = model.edge_predictor(torch.cat([h[qu], h[qv]], dim=-1)).squeeze(-1)
            probs = torch.sigmoid(probs).cpu().numpy()

        y = labels.cpu().numpy()
        if prediction_rows is not None:
            qu_np = qu.cpu().numpy()
            qv_np = qv.cpu().numpy()
            for u_local, v_local, yy, pp in zip(qu_np, qv_np, y, probs):
                prediction_rows.append(
                    {
                        "dataset": sd.get("dataset", "primary"),
                        "slice": sd["name"],
                        "target_sn": sd["target_sn"],
                        "closure": sd["closure"],
                        "split": prediction_split_label or split,
                        "u_local": int(u_local),
                        "v_local": int(v_local),
                        "y_true": float(yy),
                        "p_edge": float(pp),
                    }
                )
        if len(np.unique(y)) < 2:
            auc = 0.5
        else:
            auc = float(roc_auc_score(y, probs))

        per_slice.append(
            {
                "name": sd["name"],
                "dataset": sd.get("dataset", "primary"),
                "target_sn": sd["target_sn"],
                "closure": sd["closure"],
                "n_nodes": sd["n_nodes"],
                "n_edges": len(sd["src"]),
                "n_pos": sd["n_pos"],
                "n_neg": sd["n_neg"],
                "branching_frac": sd["branching_frac"],
                f"{split}_auc": auc,
            }
        )

    mean_auc = np.mean([r[f"{split}_auc"] for r in per_slice]) if per_slice else 0.5
    return mean_auc, per_slice


# ---------------------------------------------------------------------------
# LR schedule with warmup
# ---------------------------------------------------------------------------


class WarmupCosineScheduler:
    """Linear warmup + cosine decay."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step_count = 0

    def step(self):
        self._step_count += 1
        if self._step_count <= self.warmup_epochs:
            # Linear warmup
            factor = self._step_count / max(self.warmup_epochs, 1)
        else:
            # Cosine decay
            progress = (self._step_count - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * factor

    def state_dict(self) -> Dict[str, object]:
        """Return the minimal deterministic scheduler state for recovery."""

        return {
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.total_epochs,
            "base_lrs": list(self.base_lrs),
            "step_count": self._step_count,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore a state written by :meth:`state_dict`."""

        if int(state["warmup_epochs"]) != self.warmup_epochs:
            raise ValueError("Recovery warmup_epochs does not match this run.")
        if int(state["total_epochs"]) != self.total_epochs:
            raise ValueError("Recovery total_epochs does not match this run.")
        self.base_lrs = [float(value) for value in state["base_lrs"]]
        self._step_count = int(state["step_count"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required.")

    ap = argparse.ArgumentParser(
        description="Cross-slice SHARED training for pangenome link prediction."
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--full_segments", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--primary_dataset_name",
        default="primary",
        help="Stable dataset label stored in predictions for the primary manifest.",
    )
    ap.add_argument(
        "--extra_datasets",
        nargs="*",
        default=[],
        metavar="ITEM",
        help=(
            "Flat triples NAME MANIFEST FULL_SEGMENTS for additional graphs. "
            "All datasets share one encoder but retain dataset labels in outputs."
        ),
    )
    ap.add_argument(
        "--closures",
        nargs="+",
        choices=["strict", "1hop"],
        default=["strict", "1hop"],
        help="Train only selected closure types; useful for resumable long runs.",
    )

    # Architecture (same as 06)
    ap.add_argument("--hidden_dim", type=int, default=48)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Dual-stream flags
    ap.add_argument("--dual_stream", action="store_true", default=False)
    ap.add_argument("--no_rope", action="store_true", default=False)
    ap.add_argument("--no_fusion_gate", action="store_true", default=False)
    ap.add_argument(
        "--stream_mode",
        choices=["full", "coordinate", "graph"],
        default="full",
        help=(
            "Architecture ablation: full dual stream, coordinate stream only, "
            "or graph-topology stream only."
        ),
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
    ap.add_argument("--auto_window", action="store_true", default=False)
    ap.add_argument("--window_k", type=int, default=None)

    # ── NEW: Shared training settings ────────────────────────────────────────
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (lower than per-slice due to more data)",
    )
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience on mean val AUC",
    )
    ap.add_argument(
        "--recovery_every",
        type=int,
        default=1,
        help=(
            "Write an atomic training-recovery checkpoint every N epochs; "
            "set to 0 to disable (default: 1)."
        ),
    )
    ap.add_argument(
        "--resume_recovery",
        type=str,
        default=None,
        help=(
            "Resume a single --closures context from a recovery_*.pt file. "
            "Architecture, epoch budget, and context must match."
        ),
    )
    ap.add_argument(
        "--accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation: step every N slices",
    )
    ap.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="LR warmup epochs before cosine decay",
    )

    # ── Priority 2: Focal loss ───────────────────────────────────────────────
    ap.add_argument(
        "--focal_loss",
        action="store_true",
        default=False,
        help="Use focal loss instead of BCE (better for hard negatives)",
    )
    ap.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss focusing parameter (default 2.0)",
    )
    ap.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Focal loss balance factor (default 0.25)",
    )

    # ── Priority 3: DropEdge ─────────────────────────────────────────────────
    ap.add_argument(
        "--drop_edge",
        action="store_true",
        default=False,
        help="DropEdge augmentation during training",
    )
    ap.add_argument(
        "--drop_edge_rate",
        type=float,
        default=0.1,
        help="Fraction of structural edges to drop (default 0.1)",
    )

    # ── Priority 5: Edge features ────────────────────────────────────────────
    ap.add_argument("--use_edge_features", action="store_true", default=False)

    # ── Priority 10: Expressive predictor ────────────────────────────────────
    ap.add_argument(
        "--expressive_predictor",
        action="store_true",
        default=False,
        help="Use [h_u, h_v, h_u*h_v, |h_u-h_v|] MLP predictor",
    )

    # Other
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help=(
            "Candidate train/validation/test split seed. Defaults to --seed for "
            "backward compatibility; set explicitly for fair multi-seed runs."
        ),
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument(
        "--lazy_tensorize",
        action="store_true",
        help=(
            "Keep genome-wide slices as NumPy arrays and move one slice at a time "
            "to the selected device. This substantially reduces peak memory."
        ),
    )
    ap.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save pooled edge probabilities from final evaluations for reliability curves.",
    )
    ap.add_argument(
        "--mask_query_edges",
        action="store_true",
        help=(
            "Remove positive query edges from the message-passing graph before "
            "scoring each train/validation/test split. Use for leakage-audited "
            "paper reruns."
        ),
    )
    ap.add_argument(
        "--domain_adversarial",
        action="store_true",
        help=(
            "Add a gradient-reversal dataset classifier to multi-dataset "
            "training. Use only after auditing construction and ancestry confounding."
        ),
    )
    ap.add_argument(
        "--domain_loss_weight",
        type=float,
        default=0.1,
        help="Weight of dataset-classification loss in the joint objective.",
    )
    ap.add_argument(
        "--domain_grl_lambda",
        type=float,
        default=1.0,
        help="Gradient-reversal strength applied only to encoder gradients.",
    )

    # ── Cross-chromosome held-out validation ─────────────────────────────────
    ap.add_argument(
        "--test_chrs",
        nargs="+",
        default=None,
        help=(
            "Hold out these chromosomes entirely from training for final test.\n"
            "Use --val_chrs for chromosome-level validation/early stopping.\n"
            "This is the gold-standard test for cross-chr generalization.\n"
            "Example: --test_chrs GRCh38#0#chr1 GRCh38#0#chrY"
        ),
    )
    ap.add_argument(
        "--val_chrs",
        nargs="+",
        default=None,
        help=(
            "Chromosomes held out from training for early stopping. "
            "Example: --val_chrs GRCh38#0#chr16"
        ),
    )

    args = ap.parse_args()
    if len(args.extra_datasets) % 3:
        raise ValueError(
            "--extra_datasets must contain triples: NAME MANIFEST FULL_SEGMENTS"
        )
    if args.recovery_every < 0:
        raise ValueError("--recovery_every must be non-negative.")
    if args.resume_recovery and len(args.closures) != 1:
        raise ValueError(
            "--resume_recovery requires exactly one value in --closures."
        )
    device = torch.device(args.device)

    # ── Experiment label ─────────────────────────────────────────────────────
    _lp = ["shared"]
    if args.dual_stream:
        _lp.append("dual")
    if args.stream_mode != "full":
        _lp.append(args.stream_mode)
    if args.no_fusion_gate:
        _lp.append("nogate")
    if args.multiscale_rope:
        _lp.append(f"mscale{args.n_rope_scales}")
    if args.orientation_rope:
        _lp.append("orient")
    if args.adaptive_window:
        _lp.append(f"adpwk{args.adaptive_window_base}a{args.adaptive_window_alpha:.0f}")
    if args.focal_loss:
        _lp.append(f"focal{args.focal_gamma}")
    if args.drop_edge:
        _lp.append(f"dedge{args.drop_edge_rate}")
    if args.use_edge_features:
        _lp.append("efeat")
    if args.expressive_predictor:
        _lp.append("exppred")
    if args.mask_query_edges:
        _lp.append("maskedq")
    if args.split_seed is not None:
        _lp.append(f"splitseed{args.split_seed}")
    if args.extra_datasets:
        _lp.append(f"multidata{1 + len(args.extra_datasets) // 3}")
    if args.domain_adversarial:
        _lp.append(
            f"dannw{args.domain_loss_weight:g}l{args.domain_grl_lambda:g}"
        )
    if args.test_chrs:
        # Short label for held-out chromosomes
        chr_short = "_".join(c.split("#")[-1] for c in args.test_chrs)
        _lp.append(f"heldout_{chr_short}")
    if args.val_chrs:
        chr_short = "_".join(c.split("#")[-1] for c in args.val_chrs)
        _lp.append(f"val_{chr_short}")
    _lp.append(f"ep{args.epochs}")
    _lp.append(f"pat{args.patience}")
    exp_label = "__" + "_".join(_lp)
    # ─────────────────────────────────────────────────────────────────────────

    from utils.versioning import resolve_run_dir

    out_dir = resolve_run_dir(Path(args.out_dir))

    print(f"[06b] Cross-slice SHARED training")
    print(f"[06b] Device: {device}")
    print(f"[06b] Experiment: {exp_label}")
    print(f"[06b] focal_loss={args.focal_loss} (gamma={args.focal_gamma})")
    print(f"[06b] drop_edge={args.drop_edge} (rate={args.drop_edge_rate})")
    print(f"[06b] expressive_predictor={args.expressive_predictor}")
    print(f"[06b] mask_query_edges={args.mask_query_edges}")
    print(
        f"[06b] domain_adversarial={args.domain_adversarial} "
        f"(weight={args.domain_loss_weight}, grl_lambda={args.domain_grl_lambda})"
    )
    print(f"[06b] warmup={args.warmup_epochs}, accum_steps={args.accum_steps}")
    if args.test_chrs:
        print(f"[06b] CROSS-CHR HELD-OUT: {args.test_chrs}")
        print(f"[06b]   Model trains on all OTHER chromosomes, tests on held-out")
    if args.val_chrs:
        print(f"[06b] CROSS-CHR VALIDATION: {args.val_chrs}")

    if args.pop_cond and args.pop_table:
        from models.dual_stream_gat import load_pop_table

        n_loaded = load_pop_table(args.pop_table)
        print(f"[06b] Population table: {n_loaded} samples loaded")

    dataset_specs = [(args.primary_dataset_name, args.manifest, args.full_segments)]
    dataset_specs.extend(
        tuple(args.extra_datasets[i : i + 3])
        for i in range(0, len(args.extra_datasets), 3)
    )
    all_slices_raw = []
    for dataset_name, manifest_path, segments_path in dataset_specs:
        print(f"[06b] Loading metadata for dataset={dataset_name}: {segments_path}")
        full_segments = pd.read_csv(
            segments_path,
            compression="infer",
            usecols=["id", "name", "LN", "SN", "SO", "SR"],
        )
        seg_index, _ = build_global_index(full_segments)
        md = build_oid_metadata_from_segments(full_segments, seg_index)
        manifest = pd.read_csv(manifest_path)
        print(f"[06b] Loading {len(manifest)} slices from dataset={dataset_name}...")
        loaded_here = 0
        for _, row in manifest.iterrows():
            sd = load_slice(row, seg_index, md, full_segments, args)
            if sd is not None:
                sd["dataset"] = str(dataset_name)
                all_slices_raw.append(sd)
                loaded_here += 1
        print(f"[06b] Loaded {loaded_here} slices from dataset={dataset_name}")
        del full_segments, md

    print(f"[06b] Loaded {len(all_slices_raw)} slices successfully")
    dataset_names = sorted({str(s["dataset"]) for s in all_slices_raw})
    dataset_to_idx = {name: idx for idx, name in enumerate(dataset_names)}
    if args.domain_adversarial and len(dataset_names) < 2:
        raise ValueError(
            "--domain_adversarial requires at least two datasets via "
            "--extra_datasets."
        )

    available_targets = {str(s["target_sn"]) for s in all_slices_raw}
    available_chroms = {normalize_chrom(target) for target in available_targets}
    if args.test_chrs:
        missing = sorted(
            {normalize_chrom(chrom) for chrom in args.test_chrs} - available_chroms
        )
        if missing:
            available_preview = ", ".join(sorted(available_targets)[:12])
            raise ValueError(
                "Requested held-out chromosomes are absent from the benchmark: "
                f"{missing}. Available targets include: {available_preview}"
            )
    if args.val_chrs:
        missing = sorted(
            {normalize_chrom(chrom) for chrom in args.val_chrs} - available_chroms
        )
        if missing:
            available_preview = ", ".join(sorted(available_targets)[:12])
            raise ValueError(
                "Requested validation chromosomes are absent from the benchmark: "
                f"{missing}. Available targets include: {available_preview}"
            )

    # Split into strict and 1hop groups (train shared model per closure type)
    strict_slices_raw = [s for s in all_slices_raw if s["closure"] == "strict"]
    hop1_slices_raw = [s for s in all_slices_raw if s["closure"] == "1hop"]
    print(f"[06b] Strict: {len(strict_slices_raw)}, 1-hop: {len(hop1_slices_raw)}")

    # ── Cross-chromosome split ───────────────────────────────────────────────
    split_spec = validate_chromosome_split(
        val_chrs=args.val_chrs,
        test_chrs=args.test_chrs,
    )
    test_chr_set = set(split_spec["test_chrs"])
    val_chr_set = set(split_spec["val_chrs"])

    def _split_by_chr(slices_raw):
        """Split slices into train, validation chromosome, and heldout test."""
        train = [
            s
            for s in slices_raw
            if normalize_chrom(s["target_sn"]) not in test_chr_set
            and normalize_chrom(s["target_sn"]) not in val_chr_set
        ]
        val = [
            s for s in slices_raw if normalize_chrom(s["target_sn"]) in val_chr_set
        ]
        heldout = [
            s for s in slices_raw if normalize_chrom(s["target_sn"]) in test_chr_set
        ]
        return train, val, heldout

    if test_chr_set or val_chr_set:
        strict_train, strict_val, strict_heldout = _split_by_chr(strict_slices_raw)
        hop1_train, hop1_val, hop1_heldout = _split_by_chr(hop1_slices_raw)
        print(f"[06b] Cross-chr split:")
        print(
            f"  Strict: {len(strict_train)} train, "
            f"{len(strict_val)} val, {len(strict_heldout)} held-out"
        )
        print(
            f"  1-hop:  {len(hop1_train)} train, "
            f"{len(hop1_val)} val, {len(hop1_heldout)} held-out"
        )
    else:
        strict_train, strict_val, strict_heldout = strict_slices_raw, [], []
        hop1_train, hop1_val, hop1_heldout = hop1_slices_raw, [], []
    # ─────────────────────────────────────────────────────────────────────────

    # Process each closure type separately (different difficulty levels)
    all_closure_results = []
    for closure_name, train_slices_raw, val_slices_raw, heldout_slices_raw in [
        ("strict", strict_train, strict_val, strict_heldout),
        ("1hop", hop1_train, hop1_val, hop1_heldout),
    ]:
        if closure_name not in args.closures:
            continue
        if not train_slices_raw:
            continue
        print(f"\n{'='*60}")
        print(
            f"  TRAINING SHARED MODEL: {closure_name} ({len(train_slices_raw)} train slices)"
        )
        if heldout_slices_raw:
            print(
                f"  HELD-OUT for evaluation: {len(heldout_slices_raw)} slices ({args.test_chrs})"
            )
        if val_slices_raw:
            print(
                f"  CHR-HELDOUT validation: {len(val_slices_raw)} slices ({args.val_chrs})"
            )
        print(f"{'='*60}")

        # Window: for shared training, we dynamically set window_k per-slice
        # in train_one_epoch_shared and evaluate_shared. Init with a default.
        if args.window_k is not None:
            effective_window_k = args.window_k
        elif args.adaptive_window:
            # Default that gets overridden per-slice in forward pass
            effective_window_k = 64
            median_bf = np.median([s["branching_frac"] for s in train_slices_raw])
            print(f"  Adaptive window: per-slice (median_bf={median_bf:.3f})")
        elif args.auto_window:
            effective_window_k = 64
        else:
            effective_window_k = None

        edge_feat_dim = (
            train_slices_raw[0]["edge_attr"].shape[1]
            if train_slices_raw[0].get("edge_attr") is not None
            else 0
        )

        # Create shared model
        model = DualStreamPangenomeGAT(
            in_dim=train_slices_raw[0]["node_feats"].shape[1],
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            edge_mlp_dim=args.hidden_dim * 2,
            edge_feat_dim=edge_feat_dim,
            use_rope=not args.no_rope,
            use_fusion_gate=not args.no_fusion_gate,
            window_k=effective_window_k,
            use_multiscale_rope=args.multiscale_rope,
            n_rope_scales=args.n_rope_scales,
            use_orientation=args.orientation_rope,
            use_cross_attn=False,  # dropped based on experiments
            pop_embed_dim=args.pop_embed_dim if args.pop_cond else 0,
            stream_mode=args.stream_mode,
        ).to(device)

        # Optional: expressive predictor replaces default edge_predictor
        predictor = None
        if args.expressive_predictor:
            predictor = ExpressiveLinkPredictor(
                hidden_dim=args.hidden_dim,
                mlp_dim=args.hidden_dim * 2,
                dropout=args.dropout,
            ).to(device)

        domain_classifier = None
        if args.domain_adversarial:
            domain_classifier = DatasetDiscriminator(
                hidden_dim=args.hidden_dim,
                n_domains=len(dataset_to_idx),
                dropout=args.dropout,
            ).to(device)

        # Optimizer
        params = list(model.parameters())
        if predictor is not None:
            params += list(predictor.parameters())
        if domain_classifier is not None:
            params += list(domain_classifier.parameters())
        optimizer = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)

        # Tensorize eagerly for legacy/small runs, or one slice at a time for
        # full-coordinate coverage runs.
        slices_t = (
            train_slices_raw
            if args.lazy_tensorize
            else [tensorize_slice(sd, device, args) for sd in train_slices_raw]
        )

        # Tensorize HELD-OUT slices (only used for evaluation, never trained on)
        heldout_t = (
            (
                heldout_slices_raw
                if args.lazy_tensorize
                else [tensorize_slice(sd, device, args) for sd in heldout_slices_raw]
            )
            if heldout_slices_raw
            else []
        )
        val_chr_t = (
            (
                val_slices_raw
                if args.lazy_tensorize
                else [tensorize_slice(sd, device, args) for sd in val_slices_raw]
            )
            if val_slices_raw
            else []
        )

        total_train_edges = sum(len(s["train_idx"]) for s in slices_t)
        print(f"  Total train edges: {total_train_edges}")
        print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
        if predictor:
            print(
                f"  Predictor params: {sum(p.numel() for p in predictor.parameters()):,}"
            )

        # Training loop
        best_val_auc = -1.0
        best_state = None
        best_pred_state = None
        patience_counter = 0

        recovery_path = out_dir / f"recovery_{closure_name}{exp_label}.pt"
        start_epoch = 1
        if args.resume_recovery:
            resume_path = Path(args.resume_recovery)
            recovery = torch.load(resume_path, map_location=device, weights_only=False)
            if recovery.get("closure") != closure_name:
                raise ValueError(
                    f"Recovery context {recovery.get('closure')!r} does not match "
                    f"requested context {closure_name!r}."
                )
            if recovery.get("exp_label") != exp_label:
                raise ValueError(
                    "Recovery experiment label does not match the requested "
                    "architecture, split, or epoch settings."
                )
            model.load_state_dict(recovery["model_state"])
            if predictor is not None and recovery.get("predictor_state") is not None:
                predictor.load_state_dict(recovery["predictor_state"])
            if (
                domain_classifier is not None
                and recovery.get("domain_classifier_state") is not None
            ):
                domain_classifier.load_state_dict(recovery["domain_classifier_state"])
            optimizer.load_state_dict(recovery["optimizer_state"])
            scheduler.load_state_dict(recovery["scheduler_state"])
            best_val_auc = float(recovery["best_val_auc"])
            best_state = recovery.get("best_model_state")
            best_pred_state = recovery.get("best_predictor_state")
            patience_counter = int(recovery["patience_counter"])
            start_epoch = int(recovery["epoch"]) + 1
            torch.set_rng_state(recovery["torch_rng_state"].cpu())
            np.random.set_state(recovery["numpy_rng_state"])
            print(
                f"  Resumed {closure_name} from {resume_path} at epoch "
                f"{start_epoch} (best_val={best_val_auc:.4f})"
            )

        epoch = start_epoch - 1
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_one_epoch_shared(
                model,
                predictor,
                domain_classifier,
                dataset_to_idx,
                optimizer,
                slices_t,
                args,
                epoch,
            )
            scheduler.step()

            # Validate
            val_eval_slices = val_chr_t if val_chr_t else slices_t
            val_auc, val_details = evaluate_shared(
                model, predictor, val_eval_slices, "val", args
            )

            if epoch % 5 == 0 or epoch == 1:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch:3d}/{args.epochs}  "
                    f"loss={train_loss:.4f}  val_auc={val_auc:.4f}  "
                    f"lr={current_lr:.2e}"
                )

            # Early stopping
            if val_auc > best_val_auc + 1e-4:
                best_val_auc = val_auc
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
                if predictor is not None:
                    best_pred_state = copy.deepcopy(predictor.state_dict())
            else:
                patience_counter += 1

            should_stop = patience_counter >= args.patience
            if args.recovery_every and (
                epoch % args.recovery_every == 0 or should_stop
            ):
                recovery_tmp = recovery_path.with_name(recovery_path.name + ".tmp")
                torch.save(
                    {
                        "schema_version": 1,
                        "closure": closure_name,
                        "exp_label": exp_label,
                        "epoch": int(epoch),
                        "model_state": model.state_dict(),
                        "predictor_state": (
                            predictor.state_dict() if predictor is not None else None
                        ),
                        "domain_classifier_state": (
                            domain_classifier.state_dict()
                            if domain_classifier is not None
                            else None
                        ),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_val_auc": float(best_val_auc),
                        "best_model_state": best_state,
                        "best_predictor_state": best_pred_state,
                        "patience_counter": int(patience_counter),
                        "torch_rng_state": torch.get_rng_state(),
                        "numpy_rng_state": np.random.get_state(),
                        "args": vars(args),
                    },
                    recovery_tmp,
                )
                recovery_tmp.replace(recovery_path)

            if should_stop:
                print(f"  Early stop @ epoch {epoch}  best_val={best_val_auc:.4f}")
                break

        # Restore best
        if best_state is not None:
            model.load_state_dict(best_state)
        if predictor is not None and best_pred_state is not None:
            predictor.load_state_dict(best_pred_state)

        pred_rows: Optional[List[Dict]] = [] if args.save_predictions else None

        # Final evaluation on test AND val splits with best model (in-distribution)
        test_auc, test_details = evaluate_shared(
            model, predictor, slices_t, "test", args, pred_rows, "train_chr_test"
        )
        val_auc_final, val_details = evaluate_shared(
            model, predictor, slices_t, "val", args, pred_rows, "train_chr_val"
        )
        print(
            f"\n  FINAL {closure_name} (in-dist): test_auc={test_auc:.4f}  best_val={best_val_auc:.4f}"
        )

        # Merge per-slice val and test AUCs
        val_by_name = {vd["name"]: vd["val_auc"] for vd in val_details}

        # Build results DataFrame for IN-DISTRIBUTION slices
        results = []
        for td in test_details:
            td["best_val_auc"] = val_by_name.get(td["name"], best_val_auc)
            td["test_auc"] = td.pop("test_auc")
            td["epochs_run"] = epoch
            td["skipped"] = False
            td["split"] = "train_chr"  # these chromosomes were in the training set
            results.append(td)

        if val_chr_t:
            val_chr_test_auc, val_chr_details = evaluate_shared(
                model, predictor, val_chr_t, "test", args, pred_rows, "val_chr_test"
            )
            val_chr_val_auc, val_chr_val_details = evaluate_shared(
                model, predictor, val_chr_t, "val", args, pred_rows, "val_chr_val"
            )
            val_chr_by_name = {
                vd["name"]: vd["val_auc"] for vd in val_chr_val_details
            }
            print(
                f"\n  VAL-CHR {closure_name}: test_split_auc={val_chr_test_auc:.4f}  "
                f"(val_split={val_chr_val_auc:.4f})"
            )
            for vd in val_chr_details:
                vd["best_val_auc"] = val_chr_by_name.get(
                    vd["name"], val_chr_val_auc
                )
                vd["test_auc"] = vd.pop("test_auc")
                vd["epochs_run"] = epoch
                vd["skipped"] = False
                vd["split"] = "val_chr"
                results.append(vd)

        # ── HELD-OUT CHROMOSOME EVALUATION ───────────────────────────────────
        if heldout_t:
            # For held-out slices, we evaluate using ALL their edges as "test"
            # (not the per-slice train/val/test split, since the model never
            #  saw any of these slices during training)
            heldout_test_auc, heldout_details = evaluate_shared(
                model, predictor, heldout_t, "test", args, pred_rows, "heldout_chr_test"
            )
            # Also evaluate on the "train" portion to check consistency
            heldout_train_auc, _ = evaluate_shared(
                model, predictor, heldout_t, "train", args, pred_rows, "heldout_chr_train"
            )
            heldout_val_auc, heldout_val_details = evaluate_shared(
                model, predictor, heldout_t, "val", args, pred_rows, "heldout_chr_val"
            )

            heldout_val_by_name = {
                vd["name"]: vd["val_auc"] for vd in heldout_val_details
            }

            print(
                f"\n  HELD-OUT {closure_name}: test_auc={heldout_test_auc:.4f}  "
                f"(train_split={heldout_train_auc:.4f}, val_split={heldout_val_auc:.4f})"
            )
            print(f"  ^^^ THIS IS THE NUMBER THAT MATTERS FOR THE PAPER ^^^")

            for hd in heldout_details:
                hd["best_val_auc"] = heldout_val_by_name.get(
                    hd["name"], heldout_val_auc
                )
                hd["test_auc"] = hd.pop("test_auc")
                hd["epochs_run"] = epoch
                hd["skipped"] = False
                hd[
                    "split"
                ] = "heldout_chr"  # these chromosomes were NEVER seen during training
                results.append(hd)
        # ─────────────────────────────────────────────────────────────────────

        results_df = pd.DataFrame(results)
        results_df["exp_label"] = exp_label.lstrip("_")
        results_df["model"] = "SharedDualStreamGAT"
        results_df["use_rope"] = not args.no_rope
        results_df["multiscale_rope"] = args.multiscale_rope
        results_df["orientation_rope"] = args.orientation_rope
        results_df["use_fusion_gate"] = not args.no_fusion_gate
        results_df["stream_mode"] = args.stream_mode
        results_df["cross_attn_fusion"] = False
        results_df["pop_cond"] = args.pop_cond
        results_df["pop_embed_dim"] = args.pop_embed_dim if args.pop_cond else 0
        results_df["window_k"] = (
            "adaptive"
            if args.adaptive_window
            else ("auto" if args.auto_window else "full")
        )
        results_df["adaptive_window_base"] = (
            args.adaptive_window_base if args.adaptive_window else 0
        )
        results_df["adaptive_window_alpha"] = (
            args.adaptive_window_alpha if args.adaptive_window else 0.0
        )
        results_df["focal_loss"] = args.focal_loss
        results_df["focal_gamma"] = args.focal_gamma if args.focal_loss else 0.0
        results_df["drop_edge"] = args.drop_edge
        results_df["drop_edge_rate"] = args.drop_edge_rate if args.drop_edge else 0.0
        results_df["expressive_predictor"] = args.expressive_predictor
        results_df["min_branching_frac"] = 0.0
        results_df["hidden_dim"] = args.hidden_dim
        results_df["n_heads"] = args.n_heads
        results_df["n_layers"] = args.n_layers
        results_df["seed"] = args.seed
        results_df["split_seed"] = (
            args.seed if args.split_seed is None else args.split_seed
        )
        results_df["epochs_max"] = args.epochs
        results_df["patience"] = args.patience
        # Compatibility with 06 output format
        results_df["patience_1hop"] = args.patience
        results_df["patience_strict"] = args.patience
        results_df["k_hop"] = 1
        results_df["k_hop_chrs"] = "ALL"
        results_df["use_edge_features"] = args.use_edge_features
        results_df["topo_temp"] = "off"
        results_df["sinusoidal_node_feat"] = False
        results_df["sinusoidal_pe_dim"] = 0
        results_df["test_chrs"] = str(args.test_chrs or "none")

        # Save per-closure CSV
        closure_csv = out_dir / f"{closure_name}_results{exp_label}.csv"
        results_df.to_csv(closure_csv, index=False)
        print(f"  Saved: {closure_csv}")
        if pred_rows is not None:
            pred_csv = out_dir / f"{closure_name}_pooled_predictions{exp_label}.csv.gz"
            pd.DataFrame(pred_rows).to_csv(pred_csv, index=False, compression="gzip")
            print(f"  Saved pooled predictions: {pred_csv}")

        ckpt_path = out_dir / f"ckpt_{closure_name}{exp_label}.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "predictor_state": predictor.state_dict() if predictor is not None else None,
                "domain_classifier_state": (
                    domain_classifier.state_dict()
                    if domain_classifier is not None
                    else None
                ),
                "dataset_to_idx": dataset_to_idx,
                "args": vars(args),
                "closure": closure_name,
                "exp_label": exp_label,
                "in_dim": int(train_slices_raw[0]["node_feats"].shape[1]),
                "edge_feat_dim": int(edge_feat_dim),
                "stream_mode": args.stream_mode,
                "best_val_auc": float(best_val_auc),
                "epochs_run": int(epoch),
            },
            ckpt_path,
        )
        print(f"  Saved checkpoint: {ckpt_path}")

        # Accumulate for combined output
        all_closure_results.append(results_df)

        # Per-chromosome summary (separate train vs held-out)
        if len(results_df) > 0:
            for split_name in results_df["split"].unique():
                split_df = results_df[results_df["split"] == split_name]
                if "heldout" in split_name:
                    tag = "HELD-OUT"
                elif "val" in split_name:
                    tag = "VAL"
                else:
                    tag = "train"
                print(f"\n  Per-chromosome {closure_name} AUC [{tag}]:")
                for sn, grp in split_df.groupby("target_sn"):
                    chr_name = str(sn).split("#")[-1]
                    print(f"    {chr_name:6s}: {grp['test_auc'].mean():.4f}")

    # Save combined gat_results (both closure types, matches 06 format)
    if all_closure_results:
        combined = pd.concat(all_closure_results, ignore_index=True)
        combined_csv = out_dir / f"gat_results{exp_label}.csv"
        combined.to_csv(combined_csv, index=False)
        print(f"\n[06b] Combined results: {combined_csv}")

        # Summary
        strict_df = combined[combined["closure"] == "strict"]
        hop1_df = combined[combined["closure"] == "1hop"]

        for split_name in combined["split"].unique():
            split_sub = combined[combined["split"] == split_name]
            if "heldout" in split_name:
                tag = "HELD-OUT"
            elif "val" in split_name:
                tag = "VAL"
            else:
                tag = "train-chr"
            s_sub = split_sub[split_sub["closure"] == "strict"]
            h_sub = split_sub[split_sub["closure"] == "1hop"]
            if len(s_sub) > 0:
                print(
                    f"[06b] [{tag}] Strict AUC: {s_sub['test_auc'].mean():.4f} "
                    f"({(s_sub['test_auc'] > 0.5).sum()}/{len(s_sub)} above chance)"
                )
            if len(h_sub) > 0:
                print(f"[06b] [{tag}] 1-hop AUC:  {h_sub['test_auc'].mean():.4f}")

    print(f"\n[06b] Done. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
