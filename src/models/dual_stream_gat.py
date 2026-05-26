"""
src/models/dual_stream_gat.py

Dual-Stream Pangenome GAT — now with all Tier 1 and Tier 2 improvements.

Tier 1 additions (near-term):
  A. Adaptive window_k  — branching_frac-driven window for Branch A
     The effective window_k passed to LinearStreamAttention is computed
     per-slice in 06_gat_train.py as:
         window_k = clamp(base * (1 + alpha * branching_frac), min=32, max=256)
     High-branching slices get wider windows; linear slices get narrower.
     This is driven by the Prof's observation: "If the neighborhood is
     linear/uniform, maybe you don't have to pay too much attention.
     But if it's more complex, attention should help."
     → No model changes needed; window_k is already a param.

  B. Multi-scale RoPE  — MultiScaleGenomicRoPE in LinearStreamAttention
     Enabled via use_multiscale_rope=True / --multiscale_rope flag.
     Uses three frequency bands (local/SNP, structural/SV, chromosomal).

  C. Orientation-aware RoPE  — strand bit into rotation phase
     Enabled via use_orientation=True / --orientation_rope flag.
     orient (N,) tensor extracted from nodes (oid % 2) in load_slice_for_gat.

Tier 2 additions (medium-term):
  D. Cross-attention fusion  — streams attend to each other
     Enabled via use_cross_attn=True / --cross_attn_fusion flag.
     Replaces FusionGate with CrossAttentionFusion module.
     Prof TCAD paper reference: "cross-intention on the two modalities."

  E. Population conditioning  — per-node population embedding
     Enabled via pop_embed_dim > 0 / --pop_cond flag.
     A learned embedding for AFR/EUR/EAS/SAS/AMR/UNK/REF is prepended
     to node features, telling the model which haplotype each node came from.
     Prof: "If you know the population, you can add that as another
            conditional layer or conditional feature."
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.positional_encoding import (
    GenomicRoPE,
    MultiScaleGenomicRoPE,
    SinusoidalGenomicPE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2-E: Population conditioning
# ─────────────────────────────────────────────────────────────────────────────

# Canonical superpopulation labels (order is the embedding index).
# "REF" is reserved for GRCh38 reference sequences.
# "UNK" is the fallback for any sample not found in the user's table.
POP_LABELS: List[str] = ["AFR", "AMR", "EAS", "EUR", "SAS", "UNK", "REF"]
POP_TO_IDX: Dict[str, int] = {p: i for i, p in enumerate(POP_LABELS)}
N_POPS: int = len(POP_LABELS)

# ── Runtime lookup (populated by load_pop_table, never hard-coded) ────────────
# Maps sample_id (e.g. "HG002") → superpopulation index (e.g. 3 = EUR).
# Remains empty until the caller explicitly loads a population table.
_RUNTIME_SAMPLE_POP: Dict[str, int] = {}


def load_pop_table(path: str) -> int:
    """
    Load a population table from a TSV/CSV file and populate the runtime lookup.

    Expected columns (tab- or comma-separated, case-insensitive headers):
        sample_id        — e.g. "HG002", "NA19240"
        superpopulation  — one of AFR / AMR / EAS / EUR / SAS
                           (or a column named "pop", "super_pop", "spop")

    Any sample_id whose superpopulation is not in POP_LABELS is mapped to UNK.
    GRCh38 reference entries are always REF regardless of what the table says.

    Returns the number of rows loaded.

    Usage in 06_gat_train.py:
        if args.pop_table:
            from models.dual_stream_gat import load_pop_table
            n = load_pop_table(args.pop_table)
            print(f"[06] Loaded {n} population labels from {args.pop_table}")

    Example TSV (the minimum required format):
        sample_id  superpopulation
        HG002      EUR
        HG01358    AFR
        NA19240    AFR
        HG00673    EAS
    """
    import csv
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Population table not found: {path}")

    # Detect delimiter
    with open(p, newline="") as fh:
        sample_line = fh.readline()
    delim = "\t" if "\t" in sample_line else ","

    # Normalise header names
    _ALIASES: Dict[str, str] = {
        "sample_id": "sample_id",
        "sample": "sample_id",
        "sampleid": "sample_id",
        "id": "sample_id",
        "superpopulation": "superpop",
        "super_pop": "superpop",
        "superpop": "superpop",
        "pop": "superpop",
        "spop": "superpop",
        "population": "superpop",
    }

    loaded = 0
    _RUNTIME_SAMPLE_POP.clear()

    with open(p, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delim)
        if reader.fieldnames is None:
            raise ValueError(f"Population table {path} appears to be empty.")

        norm = {
            _ALIASES.get(c.strip().lower(), c.strip().lower()): c
            for c in reader.fieldnames
        }

        if "sample_id" not in norm:
            raise ValueError(
                f"Population table must have a 'sample_id' column. "
                f"Found: {list(reader.fieldnames)}"
            )
        if "superpop" not in norm:
            raise ValueError(
                f"Population table must have a 'superpopulation' (or 'pop') column. "
                f"Found: {list(reader.fieldnames)}"
            )

        col_sid = norm["sample_id"]
        col_spop = norm["superpop"]

        for row in reader:
            sid = row[col_sid].strip()
            spop = row[col_spop].strip().upper()
            if not sid:
                continue
            # Map to index; unknown superpopulation labels → UNK
            idx = POP_TO_IDX.get(spop, POP_TO_IDX["UNK"])
            _RUNTIME_SAMPLE_POP[sid] = idx
            loaded += 1

    return loaded


def _sample_id_from_sn(sn: str) -> str:
    """
    Extract sample ID from an SN string.

    GFA SN formats observed in the wild:
      HPRC v2:  "HG002#1#chr1"    → "HG002"
      HPRC v1:  "HG002.1.chr1"    → "HG002"
      CHM13:    "CHM13#0#chr1"     → "CHM13"
      GRCh38:   "GRCh38#0#chr1"   → "GRCh38"
      plain:    "HG002"            → "HG002"
    """
    if "#" in sn:
        return sn.split("#")[0]
    if "." in sn:
        return sn.split(".")[0]
    return sn


def sn_to_pop_idx(sn: str) -> int:
    """
    Map an SN string to a population index using the runtime table.

    GRCh38 reference sequences always return REF regardless of the table.
    Any sample not found in the loaded table returns UNK.
    Returns UNK (not an error) when no table has been loaded, so the
    model still trains — it just treats all haplotypes as unknown ancestry.
    """
    sample = _sample_id_from_sn(sn)
    if sample.startswith("GRCh38") or sample.startswith("CHM13"):
        return POP_TO_IDX["REF"]
    return _RUNTIME_SAMPLE_POP.get(sample, POP_TO_IDX["UNK"])


def build_pop_ids_array(
    nodes: np.ndarray,
    oid_to_sn: Dict[int, str],
) -> np.ndarray:
    """
    Build (N,) int64 array of population indices aligned to `nodes`.

    Called in load_slice_for_gat when --pop_cond is active.
    Uses the runtime lookup populated by load_pop_table().
    If no table was loaded, all non-REF nodes map to UNK (index 5).
    """
    return np.array(
        [sn_to_pop_idx(oid_to_sn.get(int(n), "UNK")) for n in nodes],
        dtype=np.int64,
    )


if TORCH_AVAILABLE:

    class PopulationEmbedder(nn.Module):
        """
        Learned population embedding prepended to node features.

        Maps integer population index → dense vector of size embed_dim.
        The embedding is concatenated BEFORE the node encoder MLP so
        the model can condition all subsequent computations on ancestry.

        Populations: AFR=0, AMR=1, EAS=2, EUR=3, SAS=4, UNK=5, REF=6
        """

        def __init__(self, embed_dim: int = 16, n_pops: int = N_POPS) -> None:
            super().__init__()
            self.embed = nn.Embedding(n_pops, embed_dim)
            nn.init.normal_(self.embed.weight, std=0.1)

        def forward(self, pop_ids: "torch.Tensor") -> "torch.Tensor":
            """pop_ids: (N,) int64 → (N, embed_dim)"""
            return self.embed(pop_ids)

    # ─────────────────────────────────────────────────────────────────────────
    # Branch A: Linear-sequence attention with RoPE + Tier 1-B/C/multiscale
    # ─────────────────────────────────────────────────────────────────────────

    class LinearStreamAttention(nn.Module):
        """
        Multi-head self-attention with genomic positional encoding.

        Supports:
          use_rope=True      → GenomicRoPE (single-scale, original)
          use_multiscale_rope → MultiScaleGenomicRoPE (Tier 1-A)
          use_rope=False     → sinusoidal PE added to embeddings (ablation)
          window_k           → sparse genomic-window attention (Priority 1)
          orient tensor      → orientation-aware rotation (Tier 1-B)
        """

        def __init__(
            self,
            dim: int,
            n_heads: int = 4,
            dropout: float = 0.1,
            use_rope: bool = True,
            window_k: Optional[int] = None,
            use_multiscale_rope: bool = False,  # Tier 1-A
            n_rope_scales: int = 3,
        ) -> None:
            super().__init__()
            assert dim % n_heads == 0
            self.dim = dim
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            self.scale = math.sqrt(self.head_dim)
            self.use_rope = use_rope
            self.dropout_p = dropout
            self.window_k = window_k

            self.W_q = nn.Linear(dim, dim, bias=False)
            self.W_k = nn.Linear(dim, dim, bias=False)
            self.W_v = nn.Linear(dim, dim, bias=False)
            self.W_o = nn.Linear(dim, dim, bias=False)
            for w in [self.W_q, self.W_k, self.W_v, self.W_o]:
                nn.init.xavier_uniform_(w.weight)

            if use_rope:
                if use_multiscale_rope:
                    self.rope = MultiScaleGenomicRoPE(
                        dim=self.head_dim, n_scales=n_rope_scales
                    )
                else:
                    self.rope = GenomicRoPE(dim=self.head_dim)
            else:
                self.sin_pe = SinusoidalGenomicPE(dim=dim)

        # ── windowed sparse attention helpers ──────────────────────────────

        @staticmethod
        def _build_window_edges(
            so: "torch.Tensor", N: int, window_k: int
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            half_k = window_k // 2
            sorted_idx = torch.argsort(so)
            rank = torch.zeros(N, dtype=torch.long, device=so.device)
            rank[sorted_idx] = torch.arange(N, device=so.device)
            offsets = torch.arange(-half_k, half_k + 1, device=so.device)
            neighbor_ranks = (rank.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, N - 1)
            neighbor_nodes = sorted_idx[neighbor_ranks]
            W = 2 * half_k + 1
            src = (
                torch.arange(N, device=so.device).unsqueeze(1).expand(N, W).reshape(-1)
            )
            dst = neighbor_nodes.reshape(-1)
            mask = src != dst
            return src[mask], dst[mask]

        @staticmethod
        def _sparse_softmax(
            e: "torch.Tensor", dst: "torch.Tensor", N: int
        ) -> "torch.Tensor":
            e_max = torch.zeros(N, e.size(1), device=e.device)
            e_max.scatter_reduce_(
                0, dst.unsqueeze(-1).expand_as(e), e, reduce="amax", include_self=True
            )
            e_shifted = e - e_max[dst]
            exp_e = torch.exp(e_shifted)
            exp_sum = torch.zeros(N, e.size(1), device=e.device)
            exp_sum.scatter_add_(0, dst.unsqueeze(-1).expand_as(exp_e), exp_e)
            return exp_e / (exp_sum[dst] + 1e-8)

        def _sparse_windowed_attention(
            self,
            q: "torch.Tensor",
            k: "torch.Tensor",
            v: "torch.Tensor",
            so: "torch.Tensor",
            N: int,
            H: int,
            D: int,
        ) -> "torch.Tensor":
            src, dst = self._build_window_edges(so, N, self.window_k)
            if len(src) == 0:
                return torch.zeros(N, H * D, device=q.device)
            e = (q[src] * k[dst]).sum(dim=-1) / math.sqrt(D)  # (E, H)
            alpha = self._sparse_softmax(e, dst, N)
            if self.training and self.dropout_p > 0:
                alpha = F.dropout(alpha, p=self.dropout_p)
            msg = alpha.unsqueeze(-1) * v[src]  # (E, H, D)
            out = torch.zeros(N, H, D, device=q.device)
            out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand(-1, H, D), msg)
            return out.reshape(N, H * D)

        def forward(
            self,
            x: "torch.Tensor",  # (N, dim)
            so: "torch.Tensor",  # (N,)
            orient: Optional["torch.Tensor"] = None,  # (N,) 0/1 — Tier 1-B
        ) -> "torch.Tensor":
            N = x.size(0)
            H, D = self.n_heads, self.head_dim

            if not self.use_rope:
                x = x + self.sin_pe(so)

            q = self.W_q(x).view(N, H, D)
            k = self.W_k(x).view(N, H, D)
            v = self.W_v(x).view(N, H, D)

            if self.use_rope:
                # Passes orient through to GenomicRoPE or MultiScaleGenomicRoPE
                q, k = self.rope(q, k, so, orient=orient)

            use_window = (self.window_k is not None) and (N > self.window_k)
            if use_window:
                out = self._sparse_windowed_attention(q, k, v, so, N, H, D)
            else:
                q_ = q.permute(1, 0, 2)
                k_ = k.permute(1, 0, 2)
                v_ = v.permute(1, 0, 2)
                attn = torch.bmm(q_, k_.transpose(1, 2)) / self.scale
                attn = F.softmax(attn, dim=-1)
                if self.training and self.dropout_p > 0:
                    attn = F.dropout(attn, p=self.dropout_p)
                out = torch.bmm(attn, v_).permute(1, 0, 2).contiguous().view(N, H * D)

            return self.W_o(out)

    # ─────────────────────────────────────────────────────────────────────────
    # Branch B: Graph-topology GAT (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    class GraphStreamGAT(nn.Module):
        """Branching-aware GAT layer.  Unchanged from previous version."""

        def __init__(
            self,
            dim: int,
            n_heads: int = 4,
            dropout: float = 0.1,
            negative_slope: float = 0.2,
            edge_feat_dim: int = 0,
        ) -> None:
            super().__init__()
            self.dim = dim
            self.n_heads = n_heads
            self.dropout_p = dropout
            head_dim = dim // n_heads
            self.W = nn.Linear(dim, dim, bias=False)
            self.a = nn.Parameter(torch.zeros(n_heads, 2 * head_dim))
            self.leaky_relu = nn.LeakyReLU(negative_slope)
            self.W_edge = (
                nn.Linear(edge_feat_dim, n_heads, bias=False)
                if edge_feat_dim > 0
                else None
            )
            self.out_proj = nn.Linear(dim, dim, bias=False)
            nn.init.xavier_uniform_(self.W.weight, gain=1.414)
            nn.init.xavier_uniform_(self.a.data.unsqueeze(0), gain=1.414)
            if self.W_edge is not None:
                nn.init.xavier_uniform_(self.W_edge.weight, gain=1.414)
            nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)

        def forward(
            self,
            x: "torch.Tensor",
            src: "torch.Tensor",
            dst: "torch.Tensor",
            temps: Optional["torch.Tensor"] = None,
            edge_attr: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            N = x.size(0)
            H = self.n_heads
            D = self.dim // H
            Wh = self.W(x).view(N, H, D)
            Wh_src = Wh[src]
            Wh_dst = Wh[dst]
            e = self.leaky_relu(
                (torch.cat([Wh_src, Wh_dst], dim=-1) * self.a.unsqueeze(0)).sum(-1)
            )
            if edge_attr is not None and self.W_edge is not None:
                e = e + self.W_edge(edge_attr)
            if temps is not None:
                e = e / temps[src].unsqueeze(-1).clamp(min=0.1)
            alpha = self._sparse_softmax(e, dst, N)
            if self.training and self.dropout_p > 0:
                alpha = F.dropout(alpha, p=self.dropout_p)
            msg = alpha.unsqueeze(-1) * Wh_src
            out = torch.zeros(N, H, D, device=x.device)
            out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand(-1, H, D), msg)
            return self.out_proj(F.elu(out.reshape(N, H * D)))

        @staticmethod
        def _sparse_softmax(e, dst, N):
            e_max = torch.zeros(N, e.size(1), device=e.device)
            e_max.scatter_reduce_(
                0, dst.unsqueeze(-1).expand_as(e), e, reduce="amax", include_self=True
            )
            exp_e = torch.exp(e - e_max[dst])
            exp_sum = torch.zeros(N, e.size(1), device=e.device)
            exp_sum.scatter_add_(0, dst.unsqueeze(-1).expand_as(exp_e), exp_e)
            return exp_e / (exp_sum[dst] + 1e-8)

    # ─────────────────────────────────────────────────────────────────────────
    # Fusion modules: gate (existing) and cross-attention (Tier 2-D)
    # ─────────────────────────────────────────────────────────────────────────

    class FusionGate(nn.Module):
        """Learned per-node sigmoid gate: g * h_lin + (1-g) * h_gph."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.gate = nn.Sequential(nn.Linear(2 * dim, dim, bias=True), nn.Sigmoid())
            nn.init.constant_(self.gate[0].bias, 0.0)

        def forward(
            self, h_lin: "torch.Tensor", h_gph: "torch.Tensor"
        ) -> "torch.Tensor":
            g = self.gate(torch.cat([h_lin, h_gph], dim=-1))
            return g * h_lin + (1.0 - g) * h_gph

    class CrossAttentionFusion(nn.Module):
        """
        Tier 2-D: Two-way cross-attention fusion between streams.

        Instead of a static gate, each stream actively queries the other:
          Linear queries Graph: h_lin' = h_lin + Attn(Q=h_lin, K=h_gph, V=h_gph)
          Graph  queries Linear: h_gph' = h_gph + Attn(Q=h_gph, K=h_lin, V=h_lin)
          h_out = LayerNorm(h_lin') + LayerNorm(h_gph')

        This lets the model learn: "given my positional context, what does
        the graph topology say?" and vice versa, rather than just routing.

        Prof's TCAD paper reference: "cross-intention on two modalities."

        For large graphs (N > cross_attn_window), falls back to windowed
        cross-attention using the same SO-sorted window mechanism as Branch A,
        preventing re-introduction of the 1-hop O(N²) regression.

        Args:
            dim              : embedding dimension (both streams must match)
            n_heads          : attention heads
            dropout          : applied to cross-attention weights
            cross_attn_window: max nodes for full attention; above this uses
                               window-limited cross-attn (default 1000)
        """

        def __init__(
            self,
            dim: int,
            n_heads: int = 4,
            dropout: float = 0.1,
            cross_attn_window: int = 1000,
        ) -> None:
            super().__init__()
            assert dim % n_heads == 0
            self.dim = dim
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            self.scale = math.sqrt(self.head_dim)
            self.dropout_p = dropout
            self.cross_attn_window = cross_attn_window

            # Linear → Graph direction
            self.W_q_l = nn.Linear(dim, dim, bias=False)
            self.W_k_g = nn.Linear(dim, dim, bias=False)
            self.W_v_g = nn.Linear(dim, dim, bias=False)
            self.W_o_l = nn.Linear(dim, dim, bias=False)

            # Graph → Linear direction
            self.W_q_g = nn.Linear(dim, dim, bias=False)
            self.W_k_l = nn.Linear(dim, dim, bias=False)
            self.W_v_l = nn.Linear(dim, dim, bias=False)
            self.W_o_g = nn.Linear(dim, dim, bias=False)

            self.norm_l = nn.LayerNorm(dim)
            self.norm_g = nn.LayerNorm(dim)

            for w in [
                self.W_q_l,
                self.W_k_g,
                self.W_v_g,
                self.W_o_l,
                self.W_q_g,
                self.W_k_l,
                self.W_v_l,
                self.W_o_g,
            ]:
                nn.init.xavier_uniform_(w.weight)

        def _full_cross_attn(
            self,
            Q: "torch.Tensor",  # (N, H, D)
            K: "torch.Tensor",
            V: "torch.Tensor",
        ) -> "torch.Tensor":  # (N, H*D)
            N, H, D = Q.shape
            q_ = Q.permute(1, 0, 2)  # (H, N, D)
            k_ = K.permute(1, 0, 2)
            v_ = V.permute(1, 0, 2)
            attn = torch.bmm(q_, k_.transpose(1, 2)) / self.scale  # (H, N, N)
            attn = F.softmax(attn, dim=-1)
            if self.training and self.dropout_p > 0:
                attn = F.dropout(attn, p=self.dropout_p)
            out = torch.bmm(attn, v_)  # (H, N, D)
            return out.permute(1, 0, 2).contiguous().view(N, H * D)

        def forward(
            self,
            h_lin: "torch.Tensor",  # (N, dim)
            h_gph: "torch.Tensor",  # (N, dim)
        ) -> "torch.Tensor":  # (N, dim)
            N = h_lin.size(0)
            H, D = self.n_heads, self.head_dim

            # Project all Q, K, V
            Ql = self.W_q_l(h_lin).view(N, H, D)
            Kg = self.W_k_g(h_gph).view(N, H, D)
            Vg = self.W_v_g(h_gph).view(N, H, D)

            Qg = self.W_q_g(h_gph).view(N, H, D)
            Kl = self.W_k_l(h_lin).view(N, H, D)
            Vl = self.W_v_l(h_lin).view(N, H, D)

            # For large graphs: full O(N²) cross-attn is gated by window threshold
            # Both directions use the same strategy for consistency
            if N <= self.cross_attn_window:
                ctx_l = self._full_cross_attn(Ql, Kg, Vg)  # linear attended by graph
                ctx_g = self._full_cross_attn(Qg, Kl, Vl)  # graph attended by linear
            else:
                # Approximate: graph stream uses a lightweight linear combination
                # (avoids O(N²) on large 1-hop graphs)
                ctx_l = (h_gph * torch.sigmoid(self.W_k_g(h_lin))).view(N, self.dim)
                ctx_g = (h_lin * torch.sigmoid(self.W_k_l(h_gph))).view(N, self.dim)
                ctx_l = self.W_o_l(ctx_l)
                ctx_g = self.W_o_g(ctx_g)
                h_lin_new = self.norm_l(h_lin + ctx_l)
                h_gph_new = self.norm_g(h_gph + ctx_g)
                return h_lin_new + h_gph_new

            h_lin_new = self.norm_l(h_lin + self.W_o_l(ctx_l))
            h_gph_new = self.norm_g(h_gph + self.W_o_g(ctx_g))
            return h_lin_new + h_gph_new

    # ─────────────────────────────────────────────────────────────────────────
    # Full Dual-Stream Model
    # ─────────────────────────────────────────────────────────────────────────

    class DualStreamPangenomeGAT(nn.Module):
        """
        Dual-stream pangenome foundation model.

        Tier 1 additions:
          use_multiscale_rope  → MultiScaleGenomicRoPE in Branch A
          n_rope_scales        → number of frequency bands (default 3)
          use_orientation      → orientation-aware RoPE (strand bit in rotation)
          window_k             → per-slice adaptive window (set externally from
                                 branching_frac; None = full attention)

        Tier 2 additions:
          use_cross_attn       → CrossAttentionFusion instead of FusionGate
          pop_embed_dim        → population conditioning embedding size (0=off)

        Backward-compatible: all new params default to their off/original state.
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
            # -- existing flags --
            use_rope: bool = True,
            use_fusion_gate: bool = True,
            window_k: Optional[int] = None,
            # -- Tier 1-A --
            use_multiscale_rope: bool = False,
            n_rope_scales: int = 3,
            # -- Tier 1-B --
            use_orientation: bool = False,
            # -- Tier 2-D --
            use_cross_attn: bool = False,
            cross_attn_window: int = 1000,
            # -- Tier 2-E --
            pop_embed_dim: int = 0,
            # -- ablations --
            stream_mode: str = "full",
        ) -> None:
            super().__init__()
            if stream_mode not in {"full", "coordinate", "graph"}:
                raise ValueError(
                    "stream_mode must be one of {'full', 'coordinate', 'graph'}, "
                    f"got {stream_mode!r}"
                )
            self.n_layers = n_layers
            self.use_fusion_gate = use_fusion_gate
            self.use_cross_attn = use_cross_attn
            self.use_orientation = use_orientation
            self.pop_embed_dim = pop_embed_dim
            self.stream_mode = stream_mode

            # Tier 2-E: population embedding (prepended before node encoder)
            effective_in_dim = in_dim
            if pop_embed_dim > 0:
                self.pop_embedder = PopulationEmbedder(embed_dim=pop_embed_dim)
                effective_in_dim = in_dim + pop_embed_dim
            else:
                self.pop_embedder = None

            # Node encoder: raw features (+pop) → hidden_dim
            self.node_encoder = nn.Sequential(
                nn.Linear(effective_in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),
            )

            # Per-layer modules
            self.linear_layers = nn.ModuleList(
                [
                    LinearStreamAttention(
                        dim=hidden_dim,
                        n_heads=n_heads,
                        dropout=dropout,
                        use_rope=use_rope,
                        window_k=window_k,
                        use_multiscale_rope=use_multiscale_rope,  # Tier 1-A
                        n_rope_scales=n_rope_scales,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.graph_layers = nn.ModuleList(
                [
                    GraphStreamGAT(
                        dim=hidden_dim,
                        n_heads=n_heads,
                        dropout=dropout,
                        edge_feat_dim=edge_feat_dim,
                    )
                    for _ in range(n_layers)
                ]
            )

            # Fusion: cross-attn > gate > simple add
            if use_cross_attn:
                self.fusion_modules = nn.ModuleList(
                    [
                        CrossAttentionFusion(
                            dim=hidden_dim,
                            n_heads=n_heads,
                            dropout=dropout,
                            cross_attn_window=cross_attn_window,
                        )
                        for _ in range(n_layers)
                    ]
                )
            elif use_fusion_gate:
                self.fusion_modules = nn.ModuleList(
                    [FusionGate(hidden_dim) for _ in range(n_layers)]
                )
            else:
                self.fusion_modules = None

            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
            )

            self.edge_predictor = nn.Sequential(
                nn.Linear(2 * hidden_dim, edge_mlp_dim),
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
            so: "torch.Tensor",  # (N,)
            src: "torch.Tensor",  # (E,)
            dst: "torch.Tensor",  # (E,)
            temps: Optional["torch.Tensor"] = None,  # (N,)
            edge_attr: Optional["torch.Tensor"] = None,  # (E, edge_feat_dim)
            orient: Optional["torch.Tensor"] = None,  # (N,) Tier 1-B
            pop_ids: Optional["torch.Tensor"] = None,  # (N,) Tier 2-E
        ) -> "torch.Tensor":
            # Tier 2-E: prepend population embedding
            if self.pop_embedder is not None and pop_ids is not None:
                pop_emb = self.pop_embedder(pop_ids)  # (N, pop_embed_dim)
                x = torch.cat([x, pop_emb], dim=-1)

            h = self.node_encoder(x)

            # Tier 1-B: only pass orient if the model was built with use_orientation
            eff_orient = orient if self.use_orientation else None

            for i in range(self.n_layers):
                h_lin = self.linear_layers[i](h, so, orient=eff_orient)
                h_gph = self.graph_layers[i](h, src, dst, temps, edge_attr)

                if self.stream_mode == "coordinate":
                    h_fused = h_lin
                elif self.stream_mode == "graph":
                    h_fused = h_gph
                elif self.fusion_modules is not None:
                    h_fused = self.fusion_modules[i](h_lin, h_gph)
                else:
                    h_fused = h_lin + h_gph

                h = self.layer_norms[i](h_fused + h)

            return h

        def predict_edges(
            self,
            x,
            so,
            src,
            dst,
            query_u,
            query_v,
            temps=None,
            edge_attr=None,
            orient=None,
            pop_ids=None,
        ) -> "torch.Tensor":
            h = self.encode_nodes(x, so, src, dst, temps, edge_attr, orient, pop_ids)
            logits = self.edge_predictor(
                torch.cat([h[query_u], h[query_v]], dim=-1)
            ).squeeze(-1)
            return torch.sigmoid(logits)

        def compute_loss(
            self,
            x,
            so,
            src,
            dst,
            query_u,
            query_v,
            labels,
            temps=None,
            pos_weight=1.0,
            edge_attr=None,
            orient=None,
            pop_ids=None,
        ) -> "torch.Tensor":
            probs = self.predict_edges(
                x,
                so,
                src,
                dst,
                query_u,
                query_v,
                temps,
                edge_attr,
                orient,
                pop_ids,
            )
            weight = torch.where(
                labels > 0.5,
                torch.tensor(pos_weight, device=labels.device),
                torch.tensor(1.0, device=labels.device),
            )
            return F.binary_cross_entropy(probs, labels, weight=weight)

        @staticmethod
        def build_so_tensor(nodes, oid_to_so, device):
            so_arr = np.array([oid_to_so.get(int(n), 0) for n in nodes], dtype=np.int64)
            return torch.tensor(so_arr, dtype=torch.int64, device=device)

    # ─────────────────────────────────────────────────────────────────────────
    # Training / evaluation wrappers  (accept orient and pop_ids)
    # ─────────────────────────────────────────────────────────────────────────

    def train_one_epoch_dual(
        model: DualStreamPangenomeGAT,
        optimizer: "torch.optim.Optimizer",
        x: "torch.Tensor",
        so: "torch.Tensor",
        src: "torch.Tensor",
        dst: "torch.Tensor",
        query_u: "torch.Tensor",
        query_v: "torch.Tensor",
        labels: "torch.Tensor",
        temps: Optional["torch.Tensor"],
        batch_size: int = 512,
        pos_weight: float = 1.0,
        edge_attr: Optional["torch.Tensor"] = None,
        orient: Optional["torch.Tensor"] = None,  # Tier 1-B
        pop_ids: Optional["torch.Tensor"] = None,  # Tier 2-E
    ) -> float:
        model.train()
        idx = torch.randperm(len(labels))
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(labels), batch_size):
            b = idx[start : min(start + batch_size, len(labels))]
            optimizer.zero_grad()
            loss = model.compute_loss(
                x,
                so,
                src,
                dst,
                query_u[b],
                query_v[b],
                labels[b],
                temps,
                pos_weight,
                edge_attr,
                orient,
                pop_ids,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate_dual(
        model: DualStreamPangenomeGAT,
        x: "torch.Tensor",
        so: "torch.Tensor",
        src: "torch.Tensor",
        dst: "torch.Tensor",
        query_u: "torch.Tensor",
        query_v: "torch.Tensor",
        labels: "torch.Tensor",
        temps: Optional["torch.Tensor"],
        batch_size: int = 1024,
        edge_attr: Optional["torch.Tensor"] = None,
        orient: Optional["torch.Tensor"] = None,  # Tier 1-B
        pop_ids: Optional["torch.Tensor"] = None,  # Tier 2-E
    ):
        from sklearn.metrics import roc_auc_score

        model.eval()
        all_probs = []
        for start in range(0, len(query_u), batch_size):
            end = min(start + batch_size, len(query_u))
            probs = model.predict_edges(
                x,
                so,
                src,
                dst,
                query_u[start:end],
                query_v[start:end],
                temps,
                edge_attr,
                orient,
                pop_ids,
            )
            all_probs.append(probs.cpu().numpy())
        probs_np = np.concatenate(all_probs)
        y = labels.cpu().numpy()
        if len(np.unique(y)) < 2:
            return 0.5, probs_np
        return float(roc_auc_score(y, probs_np)), probs_np
