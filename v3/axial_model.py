"""Sequence-aware multiallelic axial imputer. Independent PyTorch implementation.

Chunked linkage attention follows the STICI design principle; population-axis
attention is an extension, not an upstream STICI model/weight reproduction.
No donor, variant-ID, target-genotype, frequency or carrier embeddings exist.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

MISSING = -1
MASK = -2

@dataclass(frozen=True)
class ModelConfig:
    width: int = 176
    heads: int = 8
    layers: int = 6
    row_window: int = 256
    row_halo: int = 64
    dropout: float = 0.0
    checkpoint_layers: bool = True
    candidate_block: int = 256

class SequenceAlleleEncoder(nn.Module):
    """CNN over actual DNA; pool complete windows with base masks and counts.

    Windows must carry a convolution halo of four bases on each side. The core
    mask counts each original base once. Padding and unknown DNA are distinct.
    This is an executable CNN option, not an NT-v2/Caduceus substitute claim.
    """
    def __init__(self, width=192):
        super().__init__()
        self.bases = nn.Embedding(6, width, padding_idx=5) # A C G T N PAD
        self.conv1 = nn.Conv1d(width, width, 5, padding=2)
        self.conv2 = nn.Conv1d(width, width, 5, padding=2)
        self.project = nn.Linear(width + 2, width)

    def accumulate(self, codes, core_mask):
        if codes.ndim != 2 or codes.shape != core_mask.shape:
            raise ValueError('exact DNA/core mask dimensions required')
        if ((codes < 0) | (codes > 5)).any() or ((codes == 5) & core_mask).any():
            raise ValueError('padding cannot be an observed DNA base')
        valid = (codes != 5)[:, None]
        x = self.bases(codes).transpose(1, 2)
        x = torch.nn.functional.gelu(self.conv1(x)) * valid
        x = (torch.nn.functional.gelu(self.conv2(x)) * valid).transpose(1, 2)
        return (x * core_mask[..., None]).sum((0, 1)), core_mask.sum()

    def finish(self, sums, counts, lengths, n_bases):
        if sums.shape[:-1] != counts.shape or counts.shape != lengths.shape or lengths.shape != n_bases.shape:
            raise ValueError('one complete coverage count per allele required')
        if (counts != lengths).any() or (lengths < 1).any() or (n_bases > lengths).any():
            raise ValueError('complete literal coverage required; no prefix embedding')
        features = torch.stack((torch.log1p(lengths), n_bases / lengths), -1).to(sums)
        return self.project(torch.cat((sums / counts[..., None], features), -1))

class AxialBlock(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c = c
        self.row_norm = nn.LayerNorm(c.width); self.column_norm = nn.LayerNorm(c.width)
        self.row = nn.MultiheadAttention(c.width, c.heads, dropout=c.dropout, batch_first=True)
        self.column = nn.MultiheadAttention(c.width, c.heads, dropout=c.dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(c.width)
        self.ff = nn.Sequential(nn.Linear(c.width, c.width * 4), nn.GELU(),
                                nn.Dropout(c.dropout), nn.Linear(c.width * 4, c.width))
        self.local_conv = nn.Conv1d(c.width, c.width, 5, padding=2, groups=c.width)

    def forward(self, x, panel_rows):
        # A query never enters any panel state: later layers cannot relay hidden
        # query information through a reference and back to another query.
        h, s, d = x.shape; z = self.row_norm(x); chunks = []
        for start in range(0, s, self.c.row_window):
            stop = min(s, start + self.c.row_window)
            lo, hi = max(0, start - self.c.row_halo), min(s, stop + self.c.row_halo)
            chunks.append(self.row(z[:, start:stop], z[:, lo:hi], z[:, lo:hi], need_weights=False)[0])
        x = x + torch.cat(chunks, 1)
        x = x + self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        z = self.column_norm(x).transpose(0, 1) # sites x haplotypes x width
        p = z[:, :panel_rows]
        panel = self.column(p, p, p, need_weights=False)[0]
        queries = self.column(z[:, panel_rows:], p, p, need_weights=False)[0] if h > panel_rows else z[:, panel_rows:]
        x = x + torch.cat((panel, queries), 1).transpose(0, 1)
        return x + self.ff(self.ff_norm(x))

class PangenomeAxialImputer(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__(); self.config = config
        if config.width % config.heads or min(config.layers, config.row_window, config.candidate_block) < 1:
            raise ValueError('valid measured model dimensions required')
        self.missing = nn.Parameter(torch.randn(config.width) * .02)
        self.masked = nn.Parameter(torch.randn(config.width) * .02)
        self.site = nn.Sequential(nn.Linear(4, config.width), nn.GELU(), nn.Linear(config.width, config.width))
        self.blocks = nn.ModuleList([AxialBlock(config) for _ in range(config.layers)])
        self.output = nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, config.width, bias=False))

    def hidden(self, observations, alleles, candidate_valid, site_features, panel_rows):
        h, s = observations.shape
        if alleles.ndim != 3 or alleles.shape[0] != s or alleles.shape[2] != self.config.width:
            raise ValueError('site x full candidate x sequence vector required')
        if candidate_valid.shape != alleles.shape[:2] or site_features.shape != (s, 4) or not 0 < panel_rows <= h:
            raise ValueError('exact site/candidate/panel bindings required')
        if not candidate_valid.any(-1).all() or (observations < MASK).any():
            raise ValueError('explicit missing and masked codes required')
        known = observations >= 0; safe = observations.clamp_min(0)
        if (safe >= alleles.shape[1]).any() or not candidate_valid.gather(1, safe.T).T[known].all():
            raise ValueError('observed allele outside exact per-site universe')
        sites = torch.arange(s, device=alleles.device)[None].expand(h, s)
        x = alleles[sites, safe]
        x = torch.where((observations == MISSING)[..., None], self.missing, x)
        x = torch.where((observations == MASK)[..., None], self.masked, x)
        x = x + self.site(site_features)[None]
        for block in self.blocks:
            x = checkpoint(block, x, panel_rows, use_reentrant=False) if self.training and self.config.checkpoint_layers else block(x, panel_rows)
        return self.output(x)

    def forward(self, observations, alleles, candidate_valid, site_features, panel_rows):
        z = self.hidden(observations, alleles, candidate_valid, site_features, panel_rows)
        logits = torch.einsum('hsd,scd->hsc', z, alleles) / math.sqrt(self.config.width)
        return logits.masked_fill(~candidate_valid[None], -torch.inf)

    def full_candidate_nll(self, hidden, alleles, valid, truth, scored):
        """Exact full softmax via blocked logsumexp; no sampled alternatives."""
        if hidden.shape[:2] != truth.shape or truth.shape != scored.shape:
            raise ValueError('exact target and score mask dimensions required')
        if not scored.any(): return hidden.sum() * 0, 0
        if (truth[scored] < 0).any() or (truth[scored] >= alleles.shape[1]).any():
            raise ValueError('missing cannot become a negative/reference label')
        s = torch.arange(truth.shape[1], device=truth.device)[None].expand_as(truth)
        if not valid[s[scored], truth[scored]].all(): raise ValueError('truth outside source candidate universe')
        normalizer = torch.full_like(truth, -torch.inf, dtype=hidden.dtype)
        for a in range(0, alleles.shape[1], self.config.candidate_block):
            b = min(a + self.config.candidate_block, alleles.shape[1])
            logits = torch.einsum('hsd,scd->hsc', hidden, alleles[:, a:b]) / math.sqrt(self.config.width)
            value = torch.logsumexp(logits.masked_fill(~valid[None, :, a:b], -torch.inf), -1)
            normalizer = torch.logaddexp(normalizer, value)
        target = (hidden[scored] * alleles[s[scored], truth[scored]]).sum(-1) / math.sqrt(self.config.width)
        return (normalizer[scored] - target).sum(), int(scored.sum())
