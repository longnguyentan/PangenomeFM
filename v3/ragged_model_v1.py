"""Full-candidate axial head without chromosome-wide candidate padding.

Independent extension of the preserved v3 axial core. Flat sequence vectors and
exact per-site offsets retain every candidate. Loss blocks are checkpointed;
no full haplotype x site x maxC tensor or sampled-softmax approximation is used.
"""
import math
import torch
from torch.utils.checkpoint import checkpoint
from axial_model import PangenomeAxialImputer, MISSING, MASK

class RaggedPangenomeImputer(PangenomeAxialImputer):
    def validate(self, alleles, offsets, sites):
        if alleles.ndim != 2 or alleles.shape[1] != self.config.width or offsets.dtype != torch.long or offsets.shape != (sites + 1,):
            raise ValueError('flat complete allele vectors and exact long offsets required')
        if alleles.device != offsets.device or offsets[0] != 0 or offsets[-1] != len(alleles) or (offsets[1:] <= offsets[:-1]).any():
            raise ValueError('complete nonempty per-site candidate partition required')
        if not torch.isfinite(alleles).all(): raise ValueError('finite complete sequence vectors required')
        return offsets[1:] - offsets[:-1]

    def hidden_ragged(self, observations, alleles, offsets, site_features, panel_rows):
        if observations.ndim != 2 or observations.dtype != torch.long:
            raise ValueError('exact long haplotype/site observations required')
        h, s = observations.shape; counts = self.validate(alleles, offsets, s)
        if site_features.shape != (s, 4) or not torch.isfinite(site_features).all() or not 0 < panel_rows <= h:
            raise ValueError('exact finite geometry and TRAIN panel required')
        if (observations < MASK).any() or (observations >= counts[None]).any():
            raise ValueError('observed allele outside per-site full candidate universe')
        safe = observations.clamp_min(0)
        x = alleles[offsets[:-1][None] + safe]
        x = torch.where((observations == MISSING)[..., None], self.missing, x)
        x = torch.where((observations == MASK)[..., None], self.masked, x)
        x = x + self.site(site_features)[None]
        for block in self.blocks:
            x = checkpoint(block, x, panel_rows, use_reentrant=False) if self.training and self.config.checkpoint_layers else block(x, panel_rows)
        return self.output(x)

    def ragged_nll(self, hidden, alleles, offsets, truth, scored):
        if hidden.ndim != 3 or hidden.shape[:2] != truth.shape or truth.shape != scored.shape or truth.dtype != torch.long or scored.dtype != torch.bool:
            raise ValueError('exact hidden/truth/boolean score universe required')
        counts = self.validate(alleles, offsets, truth.shape[1])
        if ((truth < 0) & scored).any() or ((truth >= counts[None]) & scored).any():
            raise ValueError('missing/out-of-universe truth cannot be scored')
        total = hidden.sum() * 0; n = int(scored.sum())
        for site in range(truth.shape[1]):
            rows = scored[:, site]
            if not rows.any(): continue
            z = hidden[rows, site]; lo, hi = int(offsets[site]), int(offsets[site+1])
            normalizer = torch.full((len(z),), -torch.inf, dtype=z.dtype, device=z.device)
            def block_lse(q, a): return torch.logsumexp(q @ a.T / math.sqrt(self.config.width), -1)
            for begin in range(lo, hi, self.config.candidate_block):
                a = alleles[begin:min(hi, begin+self.config.candidate_block)]
                value = checkpoint(block_lse, z, a, use_reentrant=False) if torch.is_grad_enabled() and (z.requires_grad or a.requires_grad) else block_lse(z, a)
                normalizer = torch.logaddexp(normalizer, value)
            target = (z * alleles[lo + truth[rows, site]]).sum(-1) / math.sqrt(self.config.width)
            total = total + (normalizer-target).sum()
        return total, n

    def site_logits(self, hidden, alleles, offsets, site, rows):
        """Iterate full ordered candidate blocks for a caller's score accumulator."""
        self.validate(alleles, offsets, hidden.shape[1])
        if not 0 <= site < hidden.shape[1]: raise ValueError('exact site index required')
        lo, hi = int(offsets[site]), int(offsets[site+1])
        for begin in range(lo, hi, self.config.candidate_block):
            yield begin-lo, hidden[rows, site] @ alleles[begin:min(hi, begin+self.config.candidate_block)].T / math.sqrt(self.config.width)
