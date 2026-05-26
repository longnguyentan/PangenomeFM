"""
src/models/gat.py

Graph Attention Network (GAT) for pangenome link prediction.

Prof's advice (from meeting):
  "So I think that would be a good variation... graph attention."
  "Pay attention to the heterogeneity of the data."
  "If it's a linear path, you just look at nearby nodes, but if it's more complex,
   your attention would get you a little bit more complicated because you can
   actually look at different parties in the neighborhood."
  "You want to pay attention from the last common node to the next common node."

Architecture:
  [Node features] → MLP encoder → K × GATConv layers → node embeddings
  [Edge (u,v)]    → MLP([h_u ∥ h_v]) → sigmoid → link probability

GAT variant used here:
  - Standard Veličković et al. (2018) multi-head attention
  - Temperature scaling per node (branching-point aware, from attention_window.py)
  - The temperature for node i modulates how "sharp" its attention is:
      sharp (low temp) at branching points → careful, discriminative aggregation
      soft (high temp) at linear chains → broader, averaging aggregation
  - Skip connection + layer norm after each GAT layer
  - No PyTorch Geometric dependency; uses sparse COO message passing

Node features (7 dims):
  - log1p(SO) normalized         (genomic offset)
  - log1p(LN) normalized         (segment length)
  - SR (float, normalized)       (sample rank / source tag)
  - is_grch38 (binary)           (reference sequence flag)
  - log1p(degree) normalized     (local degree)
  - orientation bit (0/1)        (strand)
  - component_id normalized      (which connected component)

Usage:
    model = PangenomeGAT(in_dim=7, hidden_dim=64, n_heads=4, n_layers=2)
    loss  = model.compute_loss(node_feats, edge_index, u_batch, v_batch, labels, temps)
    probs = model.predict_edges(node_feats, edge_index, u_batch, v_batch, temps)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Node feature builder (numpy → torch-ready arrays)
# ---------------------------------------------------------------------------


def build_node_features(
    nodes: np.ndarray,
    oid_to_so: dict,
    oid_to_ln: dict,
    oid_to_sr: dict,
    oid_to_is_grch38: dict,
    oid_to_degree: dict,
    oid_to_component_id: dict,
) -> np.ndarray:
    """
    Build (N, 7) float32 feature matrix for GAT node encoder.
    All features are normalized to [0, 1] or log-scaled.

    Returns numpy array of shape (N, 7) where N = len(nodes).
    """
    N = len(nodes)
    X = np.zeros((N, 7), dtype=np.float32)

    so_arr = np.array([oid_to_so.get(int(n), 0) for n in nodes], dtype=np.float64)
    ln_arr = np.array([oid_to_ln.get(int(n), 1) for n in nodes], dtype=np.float64)
    sr_arr = np.array([oid_to_sr.get(int(n), 0) for n in nodes], dtype=np.float32)
    grch_arr = np.array(
        [oid_to_is_grch38.get(int(n), 0) for n in nodes], dtype=np.float32
    )
    deg_arr = np.array([oid_to_degree.get(int(n), 0) for n in nodes], dtype=np.float64)
    comp_arr = np.array(
        [oid_to_component_id.get(int(n), 0) for n in nodes], dtype=np.float32
    )
    orient_arr = (np.array(nodes, dtype=np.int64) % 2).astype(np.float32)

    # log-normalize and scale to [0, 1]
    def _lognorm(a: np.ndarray) -> np.ndarray:
        a = np.log1p(np.abs(a))
        mx = a.max()
        return (a / mx).astype(np.float32) if mx > 0 else a.astype(np.float32)

    def _norm(a: np.ndarray) -> np.ndarray:
        mn, mx = a.min(), a.max()
        if mx > mn:
            return ((a - mn) / (mx - mn)).astype(np.float32)
        return np.zeros(len(a), dtype=np.float32)

    X[:, 0] = _lognorm(so_arr)
    X[:, 1] = _lognorm(ln_arr)
    X[:, 2] = _norm(sr_arr)
    X[:, 3] = grch_arr
    X[:, 4] = _lognorm(deg_arr)
    X[:, 5] = orient_arr
    X[:, 6] = _norm(comp_arr)

    return X


def build_dense_edge_index(
    u: np.ndarray,
    v: np.ndarray,
    nodes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Re-index u, v from oid space to dense [0, N-1] space.
    nodes must be sorted or have a consistent order.

    Returns (src, dst) arrays in dense index space.
    """
    oid_to_idx = {int(n): i for i, n in enumerate(nodes)}
    src = np.array([oid_to_idx[int(x)] for x in u], dtype=np.int64)
    dst = np.array([oid_to_idx[int(x)] for x in v], dtype=np.int64)
    return src, dst


def build_edge_features(
    u: np.ndarray,
    v: np.ndarray,
    nodes: np.ndarray,
    md: dict,
) -> np.ndarray:
    """
    Build (E, 3) float32 edge feature matrix for structural edges.

    Used when --use_edge_features is set in scripts/06_gat_train.py.
    Features (all normalized to [0, 1]):
        0: delta_SO  — absolute genomic coordinate difference
        1: same_orient — 1 if both endpoints have the same strand bit
        2: delta_SR  — absolute sample-rank difference

    These are injected into GATConvLayer as an additive attention bias,
    following the formula:
        e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) + W_edge(edge_attr_ij)
    """
    oid_to_idx = {int(n): i for i, n in enumerate(nodes)}
    src_idx = np.array([oid_to_idx[int(x)] for x in u], dtype=np.int64)
    dst_idx = np.array([oid_to_idx[int(x)] for x in v], dtype=np.int64)

    so_arr = np.array([md["oid_to_so"].get(int(n), 0) for n in nodes], dtype=np.float64)
    sr_arr = np.array([md["oid_to_sr"].get(int(n), 0) for n in nodes], dtype=np.float32)
    orient_arr = (np.array(nodes, dtype=np.int64) % 2).astype(np.float32)

    E = len(u)
    feats = np.zeros((E, 3), dtype=np.float32)

    delta_so = np.abs(so_arr[src_idx] - so_arr[dst_idx])
    so_max = delta_so.max()
    feats[:, 0] = (delta_so / so_max).astype(np.float32) if so_max > 0 else 0.0

    feats[:, 1] = (orient_arr[src_idx] == orient_arr[dst_idx]).astype(np.float32)

    delta_sr = np.abs(sr_arr[src_idx] - sr_arr[dst_idx])
    sr_max = delta_sr.max()
    feats[:, 2] = (delta_sr / sr_max).astype(np.float32) if sr_max > 0 else 0.0

    return feats


# ---------------------------------------------------------------------------
# GAT Layer (pure PyTorch, no PyG)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class GATConvLayer(nn.Module):
        """
        Single multi-head GAT layer with per-node temperature scaling.

        Implements Veličković et al. (2018) attention:
            e_ij = LeakyReLU(a^T [W h_i ∥ W h_j])
            α_ij = softmax_j(e_ij / temp_i)     ← temperature from attention_window.py
            h'_i = σ(Σ_j α_ij W h_j)

        For K heads, outputs are concatenated (last layer uses mean instead).
        """

        def __init__(
            self,
            in_dim: int,
            out_dim: int,
            n_heads: int = 4,
            dropout: float = 0.1,
            negative_slope: float = 0.2,
            last_layer: bool = False,
            edge_feat_dim: int = 0,
        ):
            super().__init__()
            self.n_heads = n_heads
            self.out_dim = out_dim
            self.last_layer = last_layer
            self.dropout = dropout
            self.edge_feat_dim = edge_feat_dim

            self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)
            # Attention vector: one per head
            self.a = nn.Parameter(torch.zeros(n_heads, 2 * out_dim))
            self.leaky_relu = nn.LeakyReLU(negative_slope)
            # Optional edge-feature projection: adds per-edge bias to attention logits
            # Active only when --use_edge_features is set (edge_feat_dim > 0)
            self.W_edge = (
                nn.Linear(edge_feat_dim, n_heads, bias=False)
                if edge_feat_dim > 0
                else None
            )

            nn.init.xavier_uniform_(self.W.weight, gain=1.414)
            nn.init.xavier_uniform_(self.a.data.unsqueeze(0), gain=1.414)
            if self.W_edge is not None:
                nn.init.xavier_uniform_(self.W_edge.weight, gain=1.414)

        def forward(
            self,
            x: "torch.Tensor",  # (N, in_dim)
            src: "torch.Tensor",  # (E,) dense indices
            dst: "torch.Tensor",  # (E,) dense indices
            temps: Optional["torch.Tensor"] = None,  # (N,) per-node temperature
            edge_attr: Optional["torch.Tensor"] = None,  # (E, edge_feat_dim)
        ) -> "torch.Tensor":
            N = x.size(0)
            H = self.n_heads
            D = self.out_dim

            # Linear projection: (N, H*D) → (N, H, D)
            Wh = self.W(x).view(N, H, D)  # (N, H, D)

            # Gather src/dst features for each edge
            Wh_src = Wh[src]  # (E, H, D)
            Wh_dst = Wh[dst]  # (E, H, D)

            # Attention coefficients: e_ij = LeakyReLU(a^T [Wh_i ∥ Wh_j])
            concat = torch.cat([Wh_src, Wh_dst], dim=-1)  # (E, H, 2D)
            e = (concat * self.a.unsqueeze(0)).sum(dim=-1)  # (E, H)
            e = self.leaky_relu(e)  # (E, H)

            # Edge feature bias: additive term from structural edge properties
            # Active only when --use_edge_features is set (edge_attr is not None)
            if edge_attr is not None and self.W_edge is not None:
                e = e + self.W_edge(edge_attr)  # (E, H)

            # Per-node temperature scaling (from branching-point distances)
            if temps is not None:
                # temps shape: (N,) → expand to (E, H) using src indices
                t_src = temps[src].unsqueeze(-1)  # (E, 1)
                e = e / t_src.clamp(min=0.1)  # sharpen/soften per source node

            # Sparse softmax over incoming edges per destination node
            # Using segment_softmax approach: exp(e - max_e) / sum(exp)
            alpha = self._sparse_softmax(e, dst, N)  # (E, H)

            # Dropout on attention weights
            if self.training and self.dropout > 0:
                alpha = F.dropout(alpha, p=self.dropout)

            # Aggregate: h'_i = Σ_j α_ij * Wh_j   (sum over incoming edges)
            msg = alpha.unsqueeze(-1) * Wh_src  # (E, H, D)  ← Wh_src = source features

            # Scatter sum to destination nodes
            out = torch.zeros(N, H, D, device=x.device)
            idx = dst.unsqueeze(-1).unsqueeze(-1).expand(-1, H, D)
            out.scatter_add_(0, idx, msg)  # (N, H, D)

            if self.last_layer:
                # Final layer: mean over heads
                return out.mean(dim=1)  # (N, D)
            else:
                # Intermediate layers: concat heads
                return F.elu(out.view(N, H * D))  # (N, H*D)

        @staticmethod
        def _sparse_softmax(
            e: "torch.Tensor",  # (E, H)
            dst: "torch.Tensor",  # (E,)
            N: int,
        ) -> "torch.Tensor":
            """Compute softmax of e grouped by destination node."""
            # Max per destination (for numerical stability)
            e_max = torch.zeros(N, e.size(1), device=e.device)
            e_max.scatter_reduce_(
                0, dst.unsqueeze(-1).expand_as(e), e, reduce="amax", include_self=True
            )
            e_shifted = e - e_max[dst]  # (E, H)
            exp_e = torch.exp(e_shifted)

            # Sum per destination
            exp_sum = torch.zeros(N, e.size(1), device=e.device)
            exp_sum.scatter_add_(0, dst.unsqueeze(-1).expand_as(exp_e), exp_e)

            # Normalize
            alpha = exp_e / (exp_sum[dst] + 1e-8)
            return alpha

    # ---------------------------------------------------------------------------
    # Full GAT model
    # ---------------------------------------------------------------------------

    class PangenomeGAT(nn.Module):
        """
        Multi-layer GAT for pangenome link prediction.

        Prof's design prescription:
          - Node encoder: maps raw segment features → latent space
          - GAT layers: propagate information with branching-point aware attention
          - Edge predictor: scores (u, v) pairs from concatenated node embeddings
          - Skip connections + LayerNorm for stability

        Args:
            in_dim:       Input node feature dimension (7 for our feature set)
            hidden_dim:   GAT hidden/output dimension per head
            n_heads:      Number of attention heads
            n_layers:     Number of GAT layers
            dropout:      Dropout rate
            edge_mlp_dim: Hidden dim for edge predictor MLP
        """

        def __init__(
            self,
            in_dim: int = 7,
            hidden_dim: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.1,
            edge_mlp_dim: int = 128,
            edge_feat_dim: int = 0,
        ):
            super().__init__()
            self.n_layers = n_layers
            self.edge_feat_dim = edge_feat_dim

            # Node encoder: raw features → first GAT input
            self.node_encoder = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),
            )

            # GAT layers
            gat_layers = []
            for i in range(n_layers):
                is_last = i == n_layers - 1
                layer_in = hidden_dim if i == 0 else hidden_dim * n_heads
                gat_layers.append(
                    GATConvLayer(
                        in_dim=layer_in,
                        out_dim=hidden_dim,
                        n_heads=n_heads,
                        dropout=dropout,
                        last_layer=is_last,
                        edge_feat_dim=edge_feat_dim,
                    )
                )
            self.gat_layers = nn.ModuleList(gat_layers)

            # Layer norms (one per GAT layer output)
            lnorm_dims = [hidden_dim * n_heads] * (n_layers - 1) + [hidden_dim]
            self.layer_norms = nn.ModuleList([nn.LayerNorm(d) for d in lnorm_dims])

            # Skip connection projections (when dims change)
            # First layer: hidden_dim → hidden_dim*n_heads
            self.skip_projs = nn.ModuleList()
            curr_dim = hidden_dim
            for i in range(n_layers):
                is_last = i == n_layers - 1
                out_dim = hidden_dim if is_last else hidden_dim * n_heads
                if curr_dim != out_dim:
                    self.skip_projs.append(nn.Linear(curr_dim, out_dim, bias=False))
                else:
                    self.skip_projs.append(nn.Identity())
                curr_dim = out_dim

            final_node_dim = hidden_dim  # after last GAT layer (mean over heads)

            # Edge predictor: MLP on concatenated node embeddings
            self.edge_predictor = nn.Sequential(
                nn.Linear(2 * final_node_dim, edge_mlp_dim),
                nn.LayerNorm(edge_mlp_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(edge_mlp_dim, edge_mlp_dim // 2),
                nn.ELU(),
                nn.Linear(edge_mlp_dim // 2, 1),
            )

        def encode_nodes(
            self,
            x: "torch.Tensor",  # (N, in_dim)
            src: "torch.Tensor",  # (E,) dense indices
            dst: "torch.Tensor",  # (E,) dense indices
            temps: Optional["torch.Tensor"] = None,  # (N,)
            edge_attr: Optional["torch.Tensor"] = None,  # (E, edge_feat_dim)
        ) -> "torch.Tensor":
            """Produce node embeddings via GAT layers."""
            h = self.node_encoder(x)  # (N, hidden_dim)

            for i, (gat, ln, skip) in enumerate(
                zip(self.gat_layers, self.layer_norms, self.skip_projs)
            ):
                h_new = gat(h, src, dst, temps, edge_attr)
                h_new = ln(h_new)
                # Residual connection
                h = h_new + skip(h)

            return h  # (N, hidden_dim)

        def predict_edges(
            self,
            x: "torch.Tensor",
            src: "torch.Tensor",
            dst: "torch.Tensor",
            query_u: "torch.Tensor",  # (B,) indices into node array for edge src
            query_v: "torch.Tensor",  # (B,) indices into node array for edge dst
            temps: Optional["torch.Tensor"] = None,
            edge_attr: Optional["torch.Tensor"] = None,  # (E, edge_feat_dim)
        ) -> "torch.Tensor":
            """
            Returns (B,) sigmoid probabilities for query edges.

            Args:
                x:         Node features (N, in_dim)
                src/dst:   Graph structure (E,) for GAT message passing
                query_u:   Dense indices of edge source nodes to score
                query_v:   Dense indices of edge destination nodes to score
                temps:     Per-node attention temperature (N,)
                edge_attr: Structural edge features (E, edge_feat_dim)
                           — only used when --use_edge_features is set
            """
            h = self.encode_nodes(x, src, dst, temps, edge_attr)
            h_u = h[query_u]  # (B, D)
            h_v = h[query_v]  # (B, D)
            edge_feat = torch.cat([h_u, h_v], dim=-1)  # (B, 2D)
            logits = self.edge_predictor(edge_feat).squeeze(-1)  # (B,)
            return torch.sigmoid(logits)

        def compute_loss(
            self,
            x: "torch.Tensor",
            src: "torch.Tensor",
            dst: "torch.Tensor",
            query_u: "torch.Tensor",
            query_v: "torch.Tensor",
            labels: "torch.Tensor",  # (B,) float32 binary labels
            temps: Optional["torch.Tensor"] = None,
            pos_weight: float = 1.0,
            edge_attr: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            """Binary cross-entropy loss with optional positive class weighting."""
            probs = self.predict_edges(x, src, dst, query_u, query_v, temps, edge_attr)
            weight = torch.where(
                labels > 0.5,
                torch.tensor(pos_weight, device=labels.device),
                torch.tensor(1.0, device=labels.device),
            )
            loss = F.binary_cross_entropy(probs, labels, weight=weight)
            return loss

    # ---------------------------------------------------------------------------
    # Training utilities
    # ---------------------------------------------------------------------------

    class EarlyStopping:
        def __init__(self, patience: int = 10, min_delta: float = 1e-4):
            self.patience = patience
            self.min_delta = min_delta
            self.best_val = -float("inf")
            self.counter = 0
            self.best_state = None

        def step(self, val_auc: float, model: nn.Module) -> bool:
            """Returns True if training should stop."""
            if val_auc > self.best_val + self.min_delta:
                self.best_val = val_auc
                self.counter = 0
                import copy

                self.best_state = copy.deepcopy(model.state_dict())
            else:
                self.counter += 1
            return self.counter >= self.patience

        def restore_best(self, model: nn.Module):
            if self.best_state is not None:
                model.load_state_dict(self.best_state)

    def train_one_epoch(
        model: PangenomeGAT,
        optimizer: "torch.optim.Optimizer",
        x: "torch.Tensor",
        src: "torch.Tensor",
        dst: "torch.Tensor",
        query_u: "torch.Tensor",
        query_v: "torch.Tensor",
        labels: "torch.Tensor",
        temps: Optional["torch.Tensor"],
        batch_size: int = 512,
        pos_weight: float = 1.0,
        edge_attr: Optional["torch.Tensor"] = None,
    ) -> float:
        """
        One epoch of training with mini-batching over query edges.
        Node encoding is done on the full graph; only edge queries are batched.
        Returns mean loss.
        """
        model.train()
        idx = torch.randperm(len(labels))
        total_loss = 0.0
        n_batches = 0

        # Encode nodes once (not inside the query-edge loop)
        # For small graphs, do this outside. For large ones, use gradient checkpointing.
        for start in range(0, len(labels), batch_size):
            end = min(start + batch_size, len(labels))
            b_idx = idx[start:end]
            b_u = query_u[b_idx]
            b_v = query_v[b_idx]
            b_labels = labels[b_idx]

            optimizer.zero_grad()
            loss = model.compute_loss(
                x, src, dst, b_u, b_v, b_labels, temps, pos_weight, edge_attr
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(
        model: PangenomeGAT,
        x: "torch.Tensor",
        src: "torch.Tensor",
        dst: "torch.Tensor",
        query_u: "torch.Tensor",
        query_v: "torch.Tensor",
        labels: "torch.Tensor",
        temps: Optional["torch.Tensor"],
        batch_size: int = 1024,
        edge_attr: Optional["torch.Tensor"] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Evaluate link prediction AUC.
        Returns (auc_score, predicted_probs).
        """
        from sklearn.metrics import roc_auc_score

        model.eval()
        all_probs = []

        for start in range(0, len(labels), batch_size):
            end = min(start + batch_size, len(labels))
            b_u = query_u[start:end]
            b_v = query_v[start:end]
            probs = model.predict_edges(x, src, dst, b_u, b_v, temps, edge_attr)
            all_probs.append(probs.cpu().numpy())

        probs_np = np.concatenate(all_probs)
        labels_np = labels.cpu().numpy()
        auc = float(roc_auc_score(labels_np, probs_np))
        return auc, probs_np

else:
    # Stub when PyTorch is not available
    class PangenomeGAT:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for PangenomeGAT. "
                "Install with: pip install torch"
            )
