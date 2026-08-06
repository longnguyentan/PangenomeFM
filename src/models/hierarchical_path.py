"""Hierarchical path pooling and paired haplotype objectives."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def mean_pool(values: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Pool rows by a contiguous zero-based group index."""

    if values.ndim != 2 or groups.ndim != 1 or len(values) != len(groups):
        raise ValueError("Expected values=[n,d] and groups=[n].")
    if len(values) == 0:
        return values.new_zeros((0, values.shape[1]))
    if groups.min().item() < 0:
        raise ValueError("Group indices must be nonnegative.")
    n_groups = int(groups.max().item()) + 1
    pooled = values.new_zeros((n_groups, values.shape[1]))
    counts = values.new_zeros((n_groups, 1))
    pooled.index_add_(0, groups, values)
    counts.index_add_(
        0, groups, torch.ones((len(groups), 1), device=values.device, dtype=values.dtype)
    )
    return pooled / counts.clamp_min(1.0)


class HierarchicalPathEncoder(nn.Module):
    """Pool node embeddings into windows, events, and haplotype paths."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.node_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.window_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.event_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_to_window: torch.Tensor,
        window_to_event: torch.Tensor,
        event_to_path: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        nodes = self.node_projection(node_embeddings)
        windows = self.window_projection(mean_pool(nodes, node_to_window))
        if len(windows) != len(window_to_event):
            raise ValueError("window_to_event must have one entry per pooled window.")
        events = self.event_projection(mean_pool(windows, window_to_event))
        if len(events) != len(event_to_path):
            raise ValueError("event_to_path must have one entry per pooled event.")
        paths = mean_pool(events, event_to_path)
        return {
            "node_embeddings": nodes,
            "window_embeddings": windows,
            "event_embeddings": events,
            "path_embeddings": paths,
        }


class AntisymmetricHaplotypeHead(nn.Module):
    """Predict H1-minus-H2 values with exact swap antisymmetry."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(
        self, haplotype_1: torch.Tensor, haplotype_2: torch.Tensor
    ) -> torch.Tensor:
        if haplotype_1.shape != haplotype_2.shape:
            raise ValueError("Paired haplotype embeddings must have identical shapes.")
        return self.network(haplotype_1 - haplotype_2).squeeze(-1)


class CrossGraphContrastiveLoss(nn.Module):
    """Symmetric InfoNCE for same-locus embeddings from two graph domains."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = float(temperature)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape or left.ndim != 2:
            raise ValueError("Cross-graph batches must have matching [batch, dim] shapes.")
        if len(left) < 2:
            raise ValueError("At least two matched anchors are required.")
        left = F.normalize(left, dim=-1)
        right = F.normalize(right, dim=-1)
        logits = left @ right.T / self.temperature
        targets = torch.arange(len(left), device=left.device)
        return 0.5 * (
            F.cross_entropy(logits, targets)
            + F.cross_entropy(logits.T, targets)
        )
