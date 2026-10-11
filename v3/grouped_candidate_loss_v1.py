"""Exact full-C likelihood grouped by candidate count, without max-C padding.

Grouping changes the evaluation order only. Every original candidate contributes
once to logsumexp; sites/cells retain source order and all missing denominators.
"""
import math
import torch
from torch.utils.checkpoint import checkpoint

def cell_nll(hidden,alleles,offsets,truth,scored,candidate_block=256,site_block=64):
    h,s,d=hidden.shape
    if truth.shape!=(h,s) or scored.shape!=truth.shape or truth.dtype!=torch.long or scored.dtype!=torch.bool or offsets.shape!=(s+1,) or offsets.dtype!=torch.long or alleles.ndim!=2 or alleles.shape[1]!=d or offsets[0]!=0 or offsets[-1]!=len(alleles) or not torch.isfinite(hidden).all() or not torch.isfinite(alleles).all():raise ValueError('exact finite ragged full candidate universe required')
    counts=offsets[1:]-offsets[:-1]
    if (counts<2).any() or (truth < -1).any() or ((truth>=counts[None])&(truth>=0)).any() or ((truth<0)&scored).any() or min(candidate_block,site_block)<1:raise ValueError('full source candidates and available scored truth required')
    groups={}
    for i,c in enumerate(counts.detach().cpu().tolist()):groups.setdefault(c,[]).append(i)
    pieces=[];positions=[]
    for count,indices in sorted(groups.items()):
        for start in range(0,len(indices),site_block):
            ids=torch.tensor(indices[start:start+site_block],dtype=torch.long,device=hidden.device);z=hidden[:,ids];lo=offsets[ids];normalizer=hidden.new_full((h,len(ids)),-torch.inf)
            for begin in range(0,count,candidate_block):
                cols=torch.arange(begin,min(count,begin+candidate_block),device=hidden.device);a=alleles[lo[:,None]+cols[None]]
                def block(q,x):return torch.logsumexp(torch.einsum('hkd,kcd->hkc',q,x)/math.sqrt(d),-1)
                value=checkpoint(block,z,a,use_reentrant=False) if torch.is_grad_enabled() and (z.requires_grad or a.requires_grad) else block(z,a)
                normalizer=torch.logaddexp(normalizer,value)
            target=(z*alleles[lo[None]+truth[:,ids].clamp_min(0)]).sum(-1)/math.sqrt(d)
            pieces.append(normalizer-target);positions.extend(indices[start:start+site_block])
    values=torch.cat(pieces,1);order=torch.argsort(torch.tensor(positions,device=hidden.device));values=values[:,order]
    return torch.where(scored,values,torch.full_like(values,float('nan')))

def weighted_loss(values,scored,structural,weight=10.):
    if scored.shape!=values.shape or scored.dtype!=torch.bool or structural.shape!=values.shape[1:] or structural.dtype!=torch.bool or not math.isfinite(weight) or weight<=0:raise ValueError('exact complete score/type weights required')
    n=int(scored.sum())
    if n==0:raise ValueError('empty training denominator; never a zero loss')
    if not torch.isfinite(values[scored]).all():raise ValueError('nonfinite complete candidate training loss')
    w=torch.where(structural,weight,1.).to(values)[None].expand_as(values);den=w[scored].sum();return (values[scored]*w[scored]).sum()/den,dict(scored_haplotypes=n,weighted_denominator=float(den.detach()),unweighted_NLL_sum=float(values[scored].detach().sum()))
