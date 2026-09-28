"""GraphMAE-style feature objective on the native encoder, not official GraphMAE.

The established mask/re-mask and scaled-cosine principles are from Hou et al.
(KDD 2022, https://arxiv.org/abs/2205.10803). This independently implemented
adaptation groups oriented handles by segment and predicts frozen NT features.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.dual_stream_gat import GraphStreamGAT, bidirectional_messages


def segment_mask(node_oids: torch.Tensor, rate: float, seed: int) -> torch.Tensor:
    """Hide both orientations together; never leave a target's reverse copy."""
    if not 0 < rate < 1 or node_oids.ndim != 1:
        raise ValueError('Require a one-dimensional handle vector and 0 < mask rate < 1')
    ids = node_oids.detach().cpu().numpy() // 2
    groups = np.unique(ids)
    if len(groups) < 2:
        raise ValueError('Need at least two biological segments')
    n = min(len(groups)-1, max(1, int(len(groups)*rate)))
    chosen = np.random.default_rng(seed).choice(groups, n, replace=False)
    return torch.as_tensor(np.isin(ids, chosen), device=node_oids.device)


def segment_target_statistics(slices: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training-only moments; overlapping windows/orientations counted once."""
    seen = {}
    for sd in slices:
        for oid, value in zip(sd['nodes'], sd['node_feats'][:, 7:]):
            key = int(oid)//2
            if key in seen and not np.array_equal(seen[key], value):
                raise ValueError('Same segment has inconsistent frozen targets')
            seen[key] = value
    ids = np.array(sorted(seen), dtype=np.int64)
    if len(ids) < 2:
        raise ValueError('Insufficient unique training targets')
    targets = np.stack([seen[i] for i in ids]).astype(np.float64)
    if targets.shape[1] != 512 or not np.isfinite(targets).all():
        raise ValueError('Expected finite, frozen 512D NT targets')
    return ids, targets.mean(0).astype(np.float32), np.maximum(targets.std(0), 1e-6).astype(np.float32)


def segment_cosine_loss(prediction, target, segment_ids, alpha: float = 2.):
    """Mean scaled cosine error with equal weight per masked segment."""
    if prediction.shape != target.shape or len(target) != len(segment_ids) or not len(target):
        raise ValueError('Aligned, nonempty masked targets required')
    error = (1-(F.normalize(prediction, dim=-1)*F.normalize(target, dim=-1)).sum(-1)).clamp(0, 2).pow(alpha)
    _, inverse = torch.unique(segment_ids, return_inverse=True)
    counts = torch.bincount(inverse).to(error.dtype)
    sums = torch.zeros_like(counts).scatter_add_(0, inverse, error)
    return (sums/counts).mean()


class MaskedFeatureObjective(nn.Module):
    """All input columns masked, latent re-masked, graph-only shallow decoder.

    Genomic position/orientation remains an explicit encoder covariate. The
    decoder sees unmasked structural edges; adjacency is not a prediction target.
    Input NT values stay identical to Q. Target moments come from training only.
    """
    def __init__(self, encoder, in_dim: int, hidden_dim: int, n_heads: int,
                 mean: np.ndarray, scale: np.ndarray, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.mask_token = nn.Parameter(torch.zeros(in_dim))
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decoder = GraphStreamGAT(hidden_dim, n_heads, dropout=0., bidirectional=True)
        self.output = nn.Linear(hidden_dim, 512)
        self.register_buffer('target_mean', torch.as_tensor(mean))
        self.register_buffer('target_scale', torch.as_tensor(scale))
        # The unused link head is never optimized by feature reconstruction.
        for name, parameter in encoder.named_parameters():
            parameter.requires_grad_(not freeze_encoder and not name.startswith('edge_predictor.'))

    def forward(self, sd: dict, mask: torch.Tensor):
        x = sd['X']
        if mask.dtype != torch.bool or mask.shape != (len(x),) or not mask.any() or mask.all():
            raise ValueError('Need both masked and visible nodes')
        ids = sd['node_oids']//2
        if torch.isin(ids[mask], ids[~mask]).any():
            raise ValueError('Reverse handle of a masked segment remains visible')
        use_x = torch.where(mask[:, None], self.mask_token[None, :], x)
        h = self.encoder.encode_nodes(use_x, sd['so'], sd['src'], sd['dst'], sd['temps'],
            sd['edge_attr'], sd['orient'], sd['pop_ids'])
        # No direct encoder-output copy from a masked node to its decoder.
        latent = self.projection(h).masked_fill(mask[:, None], 0.)
        src, dst, _, direction = bidirectional_messages(sd['src'], sd['dst'])
        predicted = self.output(self.decoder(latent, src, dst, edge_direction=direction))
        target = (x[:, 7:]-self.target_mean)/self.target_scale
        return segment_cosine_loss(predicted[mask], target[mask], ids[mask]), int(torch.unique(ids[mask]).numel())
