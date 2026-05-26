"""
src/models/positional_encoding.py

Rotary Positional Embedding (RoPE) for genomic coordinates.

Original: GenomicRoPE following DeepGene (Zhang et al. 2024).

New in this version:
  Tier 1-A  MultiScaleGenomicRoPE
    Three parallel frequency bands matching biological scales:
      Scale 0: base=1_000   → 1 bp – 1 kbp   (SNP / local)
      Scale 1: base=10_000  → 1 kbp – 100 kbp (SV / structural, current default)
      Scale 2: base=100_000 → 100 kbp – 100 Mbp (chromosomal)
    dim//n_scales head-dimensions are assigned to each scale,
    then concatenated back.  No extra parameters vs single-scale.

  Tier 1-B  Orientation-aware RoPE  (orient param in GenomicRoPE.forward)
    Strand bit (oid % 2) carried into the rotation:
      Forward  (ori=0): R(+θ)  — no change
      Reverse  (ori=1): R(-θ)  — sin negated
    fwd-fwd pair: dot product ∝ cos(θ_i-θ_j)  — relative distance ✓
    rev-rev pair: dot product ∝ cos(θ_i-θ_j)  — same distance ✓
    fwd-rev pair: dot product ∝ cos(θ_i+θ_j)  — palindrome sensitivity ✓
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

GENOME_LENGTH_BP: int = 3_200_000_000


if TORCH_AVAILABLE:

    # ─────────────────────────────────────────────────────────────────────────
    # GenomicRoPE — original single-scale, now with orientation support
    # ─────────────────────────────────────────────────────────────────────────

    class GenomicRoPE(nn.Module):
        """
        Single-scale RoPE for genomic offsets (DeepGene eq.2-4).

        forward() now accepts optional `orient` (N,) tensor.
        When supplied, reverse-complement nodes (orient=1) get R(-θ).
        This is the Tier 1-B orientation-aware extension.
        """

        def __init__(
            self,
            dim: int,
            base: float = 10_000.0,
            scale_bp: int = GENOME_LENGTH_BP,
        ) -> None:
            super().__init__()
            assert dim % 2 == 0, "RoPE requires even head dimension"
            self.dim = dim
            self.scale_bp = float(scale_bp)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            )
            self.register_buffer("inv_freq", inv_freq)  # (dim/2,)

        def _cos_sin(self, so: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            pos = so.float() / self.scale_bp  # (N,)
            angles = torch.outer(pos, self.inv_freq)  # (N, dim/2)
            return angles.cos(), angles.sin()

        @staticmethod
        def _rotate_half(x: "torch.Tensor") -> "torch.Tensor":
            half = x.shape[-1] // 2
            return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

        def forward(
            self,
            q: "torch.Tensor",  # (N, H, D)
            k: "torch.Tensor",  # (N, H, D)
            so: "torch.Tensor",  # (N,)
            orient: Optional["torch.Tensor"] = None,  # (N,) 0/1 — Tier 1-B
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            cos_t, sin_t = self._cos_sin(so)  # (N, dim/2)

            # Tier 1-B: reverse-complement nodes → negate sin (R(-θ))
            if orient is not None:
                sign = 1.0 - 2.0 * orient.float()  # (N,) in {+1, -1}
                sin_t = sin_t * sign.unsqueeze(-1)

            cos_t = torch.cat([cos_t, cos_t], dim=-1).unsqueeze(1)  # (N,1,D)
            sin_t = torch.cat([sin_t, sin_t], dim=-1).unsqueeze(1)

            q_rot = q * cos_t + self._rotate_half(q) * sin_t
            k_rot = k * cos_t + self._rotate_half(k) * sin_t
            return q_rot, k_rot

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 1-A — MultiScaleGenomicRoPE
    # ─────────────────────────────────────────────────────────────────────────

    class MultiScaleGenomicRoPE(nn.Module):
        """
        Multi-scale RoPE: three frequency bands for three biological scales.

        dim is split evenly across n_scales sub-dimensions.
        Each sub-dim gets its own frequency ladder (base parameter).
        Results are concatenated back to (N, H, D).

        Also supports orientation-aware encoding (orient param).

        Args:
            dim         : total head dimension (must divide by 2*n_scales)
            n_scales    : number of frequency bands (default 3)
            scale_bases : base per band; default [1e3, 1e4, 1e5]
            scale_bp    : SO normaliser (default 3.2 Gbp)
        """

        _DEFAULT_BASES: List[float] = [1_000.0, 10_000.0, 100_000.0]

        def __init__(
            self,
            dim: int,
            n_scales: int = 3,
            scale_bases: Optional[List[float]] = None,
            scale_bp: int = GENOME_LENGTH_BP,
        ) -> None:
            super().__init__()
            assert (
                dim % (n_scales * 2) == 0
            ), f"dim ({dim}) must be divisible by 2*n_scales ({2*n_scales})"
            self.dim = dim
            self.n_scales = n_scales
            self.scale_bp = float(scale_bp)
            self.sub_dim = dim // n_scales

            if scale_bases is None:
                scale_bases = self._DEFAULT_BASES[:n_scales]
            assert len(scale_bases) == n_scales, "len(scale_bases) must equal n_scales"

            for s, base in enumerate(scale_bases):
                inv_freq = 1.0 / (
                    base
                    ** (
                        torch.arange(0, self.sub_dim, 2, dtype=torch.float32)
                        / self.sub_dim
                    )
                )
                self.register_buffer(f"inv_freq_{s}", inv_freq)

        def _cos_sin_s(
            self, so: "torch.Tensor", s: int
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            inv_f = getattr(self, f"inv_freq_{s}")
            pos = so.float() / self.scale_bp
            angles = torch.outer(pos, inv_f)
            return angles.cos(), angles.sin()

        @staticmethod
        def _rotate_half(x: "torch.Tensor") -> "torch.Tensor":
            half = x.shape[-1] // 2
            return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

        def _apply_rope(
            self,
            seg: "torch.Tensor",  # (N, H, sub_dim)
            cos_t: "torch.Tensor",  # (N, sub_dim/2)
            sin_t: "torch.Tensor",  # (N, sub_dim/2)
            sign: Optional["torch.Tensor"],  # (N,) or None
        ) -> "torch.Tensor":
            if sign is not None:
                sin_t = sin_t * sign.unsqueeze(-1)
            c = torch.cat([cos_t, cos_t], dim=-1).unsqueeze(1)  # (N,1,sub_dim)
            s_ = torch.cat([sin_t, sin_t], dim=-1).unsqueeze(1)
            return seg * c + self._rotate_half(seg) * s_

        def forward(
            self,
            q: "torch.Tensor",  # (N, H, D)
            k: "torch.Tensor",  # (N, H, D)
            so: "torch.Tensor",  # (N,)
            orient: Optional["torch.Tensor"] = None,  # (N,) 0/1
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            sign = None
            if orient is not None:
                sign = 1.0 - 2.0 * orient.float()  # (N,) in {+1,-1}

            S = self.sub_dim
            q_parts, k_parts = [], []
            for s in range(self.n_scales):
                cos_s, sin_s = self._cos_sin_s(so, s)
                q_parts.append(
                    self._apply_rope(q[..., s * S : (s + 1) * S], cos_s, sin_s, sign)
                )
                k_parts.append(
                    self._apply_rope(k[..., s * S : (s + 1) * S], cos_s, sin_s, sign)
                )

            return torch.cat(q_parts, dim=-1), torch.cat(k_parts, dim=-1)

    # ─────────────────────────────────────────────────────────────────────────
    # Sinusoidal fallback (ablation --no_rope)
    # ─────────────────────────────────────────────────────────────────────────

    class SinusoidalGenomicPE(nn.Module):
        """Fixed sinusoidal PE added to node embeddings (no Q/K rotation)."""

        def __init__(self, dim: int, base: float = 10_000.0) -> None:
            super().__init__()
            assert dim % 2 == 0
            self.dim = dim
            self.base = base

        def forward(self, so: "torch.Tensor") -> "torch.Tensor":
            pos = so.float() / GENOME_LENGTH_BP
            inv_freq = 1.0 / (
                self.base
                ** (
                    torch.arange(0, self.dim, 2, dtype=torch.float32, device=so.device)
                    / self.dim
                )
            )
            angles = torch.outer(pos, inv_freq)
            pe = torch.zeros(len(so), self.dim, device=so.device)
            pe[:, 0::2] = angles.sin()
            pe[:, 1::2] = angles.cos()
            return pe


# ─────────────────────────────────────────────────────────────────────────────
# Numpy helpers (no-torch pipeline code)
# ─────────────────────────────────────────────────────────────────────────────


def sinusoidal_pe_numpy(
    so_values: np.ndarray,
    dim: int = 64,
    base: float = 10_000.0,
) -> np.ndarray:
    """Sinusoidal PE without PyTorch. Returns (N, dim) float32."""
    pos = so_values.astype(np.float64) / GENOME_LENGTH_BP
    i = np.arange(0, dim // 2, dtype=np.float64)
    inv_freq = 1.0 / (base ** (2 * i / dim))
    angles = np.outer(pos, inv_freq)
    pe = np.zeros((len(pos), dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles).astype(np.float32)
    pe[:, 1::2] = np.cos(angles).astype(np.float32)
    return pe
