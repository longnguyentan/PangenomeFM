"""Opt-in query-chunked attention, keeping historical checkpoint behavior intact.

Exact mode retains global softmax attention. Window mode uses a unique local
key set (including self), normalized for each query. Recompute chunks during
backpropagation so retained attention activations do not grow quadratically.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def coordinate_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
                         chunk_size: int, dropout_p: float = 0.,
                         positions: torch.Tensor | None = None,
                         window_k: int | None = None,
                         recompute: bool = True) -> torch.Tensor:
    """Inputs/outputs are [nodes, heads, head_dim]; no biological labels.

    window_k denotes twice the rank radius (radius=floor(window_k/2)), matching
    the old window-width convention. Out-of-range keys are masked, not repeated.
    Chunking changes floating-point reduction/dropout order, not exact-mode
    mathematical attention. A new protocol must disclose this numerical change.
    """
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape or not len(q):
        raise ValueError('Nonempty aligned [nodes, heads, head_dim] arrays required')
    if chunk_size < 1 or not 0 <= dropout_p < 1:
        raise ValueError('Require positive chunk size and 0 <= dropout < 1')
    if window_k is not None:
        if window_k < 1 or positions is None or positions.shape != (len(q),):
            raise ValueError('Window attention requires positions and positive window_k')
        # Stable tie handling retains canonical node order for coincident positions.
        order = torch.argsort(positions, stable=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(len(q), device=q.device)
        offsets = torch.arange(-(window_k//2), window_k//2+1, device=q.device)

    def global_chunk(query, key, value):
        return F.scaled_dot_product_attention(query.transpose(0,1), key.transpose(0,1),
            value.transpose(0,1), dropout_p=dropout_p).transpose(0,1)

    def local_chunk(query, key, value, neighbors, valid):
        logits = (query[:,None,:,:]*key[neighbors]).sum(-1)/math.sqrt(query.shape[-1])
        logits = logits.masked_fill(~valid[:,:,None], float('-inf'))
        weights = torch.softmax(logits, dim=1)
        weights = F.dropout(weights, p=dropout_p, training=dropout_p > 0)
        return (weights[:,:,:,None]*value[neighbors]).sum(1)

    outputs = []
    for start in range(0, len(q), chunk_size):
        query = q[start:start+chunk_size]
        inputs = [query, k, v]
        function = global_chunk
        if window_k is not None:
            neighbor_ranks = rank[start:start+chunk_size,None] + offsets[None,:]
            valid = (neighbor_ranks >= 0) & (neighbor_ranks < len(q))
            neighbors = order[neighbor_ranks.clamp(0,len(q)-1)]
            inputs.extend([neighbors, valid])
            function = local_chunk
        if recompute and torch.is_grad_enabled() and any(x.requires_grad for x in [q,k,v]):
            outputs.append(checkpoint(function, *inputs, use_reentrant=False, preserve_rng_state=True))
        else:
            outputs.append(function(*inputs))
    return torch.cat(outputs, dim=0)
