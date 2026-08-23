"""Multimodal, path-aware components for PangenomeFM v2.

The published PangenomeFM encoder is intentionally topology native.  This
module implements the next-stage contract without changing that v1 code path:

* a trainable nucleotide encoder or a projection of frozen sequence profiles;
* fusion of sequence, topology, coordinates, population and haplotype context;
* ordered path pooling; and
* optional edge, masked-token, path-order, branch-choice and cross-graph
  objectives.

Every objective is opt-in.  Missing labels are therefore represented by an
absent batch key rather than by invented targets.  The module accepts
precomputed topology profiles so it can be piloted with the verified v1
checkpoints before end-to-end graph training is attempted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F


PAD_TOKEN = 0
MASK_TOKEN = 6
DEFAULT_VOCAB_SIZE = 7  # PAD, A, C, G, T, N, MASK


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights for the independently auditable v2 pretraining objectives."""

    edge: float = 1.0
    masked_token: float = 1.0
    path_order: float = 0.25
    branch_choice: float = 0.5
    cross_graph: float = 0.1

    def as_dict(self) -> dict[str, float]:
        return {
            "edge": self.edge,
            "masked_token": self.masked_token,
            "path_order": self.path_order,
            "branch_choice": self.branch_choice,
            "cross_graph": self.cross_graph,
        }


class SegmentSequenceEncoder(nn.Module):
    """Encode nucleotide tokens and retain token states for masked LM."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_length: int = 4096,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)
        self.token_embedding = nn.Embedding(
            vocab_size, hidden_dim, padding_idx=PAD_TOKEN
        )
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2:
            raise ValueError("sequence_tokens must have shape [nodes, length]")
        if tokens.shape[1] > self.max_length:
            raise ValueError(
                f"sequence length {tokens.shape[1]} exceeds max_length={self.max_length}"
            )
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.vocab_size):
            raise ValueError("sequence_tokens contain an out-of-vocabulary token")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None]
        padding_mask = tokens.eq(PAD_TOKEN)
        hidden = self.norm(self.encoder(hidden, src_key_padding_mask=padding_mask))
        valid = (~padding_mask).to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return pooled, hidden, self.lm_head(hidden)


class MultimodalFusion(nn.Module):
    """Fuse modalities while exposing their weights for interpretation."""

    def __init__(self, hidden_dim: int, n_modalities: int, dropout: float) -> None:
        super().__init__()
        self.n_modalities = int(n_modalities)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * n_modalities, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_modalities),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        modalities: list[torch.Tensor],
        availability: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(modalities) != self.n_modalities:
            raise ValueError("Unexpected number of modalities")
        if not modalities or any(value.shape != modalities[0].shape for value in modalities):
            raise ValueError("All modality tensors must have the same [nodes, hidden] shape")
        scores = self.score(torch.cat(modalities, dim=-1))
        if availability is not None:
            if availability.shape != scores.shape:
                raise ValueError("availability must have shape [nodes, modalities]")
            if (~availability.bool()).all(dim=1).any():
                raise ValueError("Every node must retain at least one modality")
            scores = scores.masked_fill(~availability.bool(), float("-inf"))
        weights = scores.softmax(dim=-1)
        stacked = torch.stack(modalities, dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return self.output(fused), weights


class OrderedPathEncoder(nn.Module):
    """Pool ordered, padded node-index paths into path representations."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_path_length: int = 1024,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.max_path_length = int(max_path_length)
        self.position_embedding = nn.Embedding(max_path_length, hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.cls, std=0.02)

    def forward(
        self, node_embeddings: torch.Tensor, path_nodes: torch.Tensor
    ) -> torch.Tensor:
        if path_nodes.ndim != 2:
            raise ValueError("path_nodes must have shape [paths, max_path_length]")
        if path_nodes.shape[1] > self.max_path_length:
            raise ValueError("path_nodes exceeds configured max_path_length")
        padding = path_nodes.lt(0)
        if padding.all(dim=1).any():
            raise ValueError("A path cannot consist entirely of padding")
        safe_nodes = path_nodes.clamp_min(0)
        if safe_nodes.numel() and safe_nodes.max() >= len(node_embeddings):
            raise ValueError("path_nodes references a node outside node_embeddings")
        positions = torch.arange(path_nodes.shape[1], device=path_nodes.device)
        values = node_embeddings[safe_nodes] + self.position_embedding(positions)[None]
        values = values.masked_fill(padding.unsqueeze(-1), 0.0)
        cls = self.cls.expand(len(path_nodes), -1, -1)
        encoded = self.encoder(
            torch.cat([cls, values], dim=1),
            src_key_padding_mask=torch.cat(
                [torch.zeros((len(path_nodes), 1), dtype=torch.bool, device=padding.device), padding],
                dim=1,
            ),
        )
        return self.norm(encoded[:, 0])


class InvariantPairHead(nn.Module):
    """Score unordered pairs from symmetric comparison features."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, 1),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape:
            raise ValueError("Pair tensors must have identical shapes")
        features = torch.cat([left * right, (left - right).abs()], dim=-1)
        return self.network(features).squeeze(-1)


class DirectionalPairHead(nn.Module):
    """Score ordered pairs; swapping inputs can change the prediction."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, 1),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape:
            raise ValueError("Pair tensors must have identical shapes")
        features = torch.cat([left, right, left - right, left * right], dim=-1)
        return self.network(features).squeeze(-1)


class PangenomeFoundationModelV2(nn.Module):
    """Joint sequence/topology/path model with independently weighted losses."""

    def __init__(
        self,
        topology_dim: int,
        hidden_dim: int = 128,
        *,
        coordinate_dim: int = 4,
        sequence_feature_dim: int = 0,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        n_heads: int = 4,
        sequence_layers: int = 2,
        path_layers: int = 2,
        dropout: float = 0.1,
        max_sequence_length: int = 4096,
        max_path_length: int = 1024,
        n_populations: int = 0,
        n_haplotype_states: int = 0,
        modality_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if topology_dim <= 0 or hidden_dim <= 0:
            raise ValueError("topology_dim and hidden_dim must be positive")
        if sequence_feature_dim < 0 or coordinate_dim < 0:
            raise ValueError("feature dimensions cannot be negative")
        if not 0 <= modality_dropout < 1:
            raise ValueError("modality_dropout must be in [0, 1)")
        self.hidden_dim = int(hidden_dim)
        self.coordinate_dim = int(coordinate_dim)
        self.sequence_feature_dim = int(sequence_feature_dim)
        self.modality_dropout = float(modality_dropout)

        self.topology_projection = nn.Sequential(
            nn.Linear(topology_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.sequence_encoder: Optional[SegmentSequenceEncoder]
        self.sequence_projection: Optional[nn.Module]
        if sequence_feature_dim:
            self.sequence_encoder = None
            self.sequence_projection = nn.Sequential(
                nn.Linear(sequence_feature_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.sequence_encoder = SegmentSequenceEncoder(
                hidden_dim,
                vocab_size=vocab_size,
                n_heads=n_heads,
                n_layers=sequence_layers,
                dropout=dropout,
                max_length=max_sequence_length,
            )
            self.sequence_projection = None

        self.coordinate_projection = (
            nn.Sequential(
                nn.Linear(coordinate_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            if coordinate_dim
            else None
        )
        self.population_embedding = (
            nn.Embedding(n_populations, hidden_dim) if n_populations else None
        )
        self.haplotype_embedding = (
            nn.Embedding(n_haplotype_states, hidden_dim)
            if n_haplotype_states
            else None
        )

        n_modalities = 2 + int(self.coordinate_projection is not None)
        n_modalities += int(self.population_embedding is not None)
        n_modalities += int(self.haplotype_embedding is not None)
        self.fusion = MultimodalFusion(hidden_dim, n_modalities, dropout)
        self.path_encoder = OrderedPathEncoder(
            hidden_dim,
            n_heads=n_heads,
            n_layers=path_layers,
            dropout=dropout,
            max_path_length=max_path_length,
        )
        self.edge_head = InvariantPairHead(hidden_dim, dropout)
        self.path_order_head = DirectionalPairHead(hidden_dim, dropout)
        self.branch_bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1, bias=False)

    def _availability(self, n_nodes: int, n_modalities: int, device: torch.device) -> torch.Tensor:
        available = torch.ones((n_nodes, n_modalities), dtype=torch.bool, device=device)
        if self.training and self.modality_dropout:
            dropped = torch.rand((n_nodes, n_modalities), device=device) < self.modality_dropout
            # Topology is kept as the anchor modality; this preserves the v1
            # semantics and avoids nodes with no usable input.
            dropped[:, 0] = False
            available &= ~dropped
        return available

    def encode_nodes(
        self,
        *,
        topology_embeddings: torch.Tensor,
        sequence_tokens: Optional[torch.Tensor] = None,
        sequence_features: Optional[torch.Tensor] = None,
        coordinates: Optional[torch.Tensor] = None,
        population_ids: Optional[torch.Tensor] = None,
        haplotype_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if topology_embeddings.ndim != 2:
            raise ValueError("topology_embeddings must have shape [nodes, features]")
        n_nodes = len(topology_embeddings)
        modalities = [self.topology_projection(topology_embeddings)]
        token_logits: Optional[torch.Tensor] = None

        if self.sequence_projection is not None:
            if sequence_features is None:
                raise ValueError("sequence_features are required by this model")
            sequence_summary = self.sequence_projection(sequence_features)
        else:
            if sequence_tokens is None or self.sequence_encoder is None:
                raise ValueError("sequence_tokens are required by this model")
            sequence_summary, _, token_logits = self.sequence_encoder(sequence_tokens)
        modalities.append(sequence_summary)

        if self.coordinate_projection is not None:
            if coordinates is None:
                raise ValueError("coordinates are required by this model")
            modalities.append(self.coordinate_projection(coordinates))
        if self.population_embedding is not None:
            if population_ids is None:
                raise ValueError("population_ids are required by this model")
            modalities.append(self.population_embedding(population_ids))
        if self.haplotype_embedding is not None:
            if haplotype_ids is None:
                raise ValueError("haplotype_ids are required by this model")
            modalities.append(self.haplotype_embedding(haplotype_ids))

        if any(len(value) != n_nodes for value in modalities):
            raise ValueError("Every modality must contain the same number of nodes")
        availability = self._availability(n_nodes, len(modalities), topology_embeddings.device)
        fused, weights = self.fusion(modalities, availability)
        result = {"node_embeddings": fused, "modality_weights": weights}
        if token_logits is not None:
            result["token_logits"] = token_logits
        return result

    def edge_logits(
        self, node_embeddings: torch.Tensor, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        return self.edge_head(node_embeddings[left], node_embeddings[right])

    def branch_logits(
        self,
        node_embeddings: torch.Tensor,
        sources: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if candidates.ndim != 2 or len(sources) != len(candidates):
            raise ValueError("branch candidates must have shape [groups, choices]")
        source = node_embeddings[sources].unsqueeze(1).expand(-1, candidates.shape[1], -1)
        target = node_embeddings[candidates]
        return self.branch_bilinear(source, target).squeeze(-1)

    @staticmethod
    def cross_graph_loss(left: torch.Tensor, right: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        if left.ndim != 2 or left.shape != right.shape or len(left) < 2:
            raise ValueError("cross-graph anchors require matching [batch, dim] tensors")
        left = F.normalize(left, dim=-1)
        right = F.normalize(right, dim=-1)
        logits = left @ right.T / temperature
        targets = torch.arange(len(left), device=left.device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))

    def compute_objectives(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        weights: ObjectiveWeights = ObjectiveWeights(),
        cross_graph_temperature: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode_nodes(
            topology_embeddings=batch["topology_embeddings"],
            sequence_tokens=batch.get("sequence_tokens"),
            sequence_features=batch.get("sequence_features"),
            coordinates=batch.get("coordinates"),
            population_ids=batch.get("population_ids"),
            haplotype_ids=batch.get("haplotype_ids"),
        )
        nodes = encoded["node_embeddings"]
        losses: dict[str, torch.Tensor] = {}

        if {"edge_u", "edge_v", "edge_labels"}.issubset(batch):
            logits = self.edge_logits(nodes, batch["edge_u"], batch["edge_v"])
            losses["edge"] = F.binary_cross_entropy_with_logits(
                logits, batch["edge_labels"].float()
            )
        if "masked_token_labels" in batch:
            if "token_logits" not in encoded:
                raise ValueError("masked-token labels require trainable sequence tokens")
            losses["masked_token"] = F.cross_entropy(
                encoded["token_logits"].reshape(-1, encoded["token_logits"].shape[-1]),
                batch["masked_token_labels"].reshape(-1),
                ignore_index=-100,
            )
        if {"path_nodes", "path_pair_left", "path_pair_right", "path_order_labels"}.issubset(batch):
            paths = self.path_encoder(nodes, batch["path_nodes"])
            logits = self.path_order_head(
                paths[batch["path_pair_left"]], paths[batch["path_pair_right"]]
            )
            losses["path_order"] = F.binary_cross_entropy_with_logits(
                logits, batch["path_order_labels"].float()
            )
        if {"branch_sources", "branch_candidates", "branch_labels"}.issubset(batch):
            logits = self.branch_logits(
                nodes, batch["branch_sources"], batch["branch_candidates"]
            )
            losses["branch_choice"] = F.cross_entropy(logits, batch["branch_labels"])
        if {"cross_graph_left_nodes", "cross_graph_right_nodes"}.issubset(batch):
            losses["cross_graph"] = self.cross_graph_loss(
                nodes[batch["cross_graph_left_nodes"]],
                nodes[batch["cross_graph_right_nodes"]],
                cross_graph_temperature,
            )
        if not losses:
            raise ValueError("Batch does not contain any supported objective labels")

        total = sum(weights.as_dict()[name] * value for name, value in losses.items())
        return {
            **losses,
            "total": total,
            "node_embeddings": nodes,
            "modality_weights": encoded["modality_weights"],
        }
