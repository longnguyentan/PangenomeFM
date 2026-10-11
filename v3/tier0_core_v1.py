"""Matched masked-pretext baselines and paired extended-real family estimands.

Lookup identities never enter model inputs. Caller supplies already closed full
candidate/missing namespaces. Nearest selection sees observations, never hidden
labels; exact-zero baseline probabilities remain infinite NLL.
"""
import math
import numpy as np

def nearest_observed(reference, observations):
 reference=np.asarray(reference);observations=np.asarray(observations)
 if reference.ndim!=2 or observations.ndim!=2 or reference.shape[1]!=observations.shape[1] or not len(reference) or (reference < -1).any() or (observations < -2).any():raise ValueError('complete reference and masked query namespaces required')
 chosen=np.full(len(observations),-1,dtype=np.int32);errors=np.full_like(chosen,-1);common=np.zeros_like(chosen)
 for q,observed in enumerate(observations):
  valid=(reference>=0)&(observed[None]>=0);n=valid.sum(1,dtype=np.int64);e=((reference!=observed[None])&valid).sum(1,dtype=np.int64);best=-1
  for r in np.flatnonzero(n):
   if best<0 or e[r]*n[best]<e[best]*n[r]:best=int(r)
  if best>=0:chosen[q]=best;errors[q]=int(e[best]);common[q]=int(n[best])
 return chosen,errors,common

def haplotype_losses(logits,truth,scored,counts,copies):
 logits=np.asarray(logits,dtype=np.float64);truth=np.asarray(truth);scored=np.asarray(scored);counts=np.asarray(counts);copies=np.asarray(copies)
 if logits.ndim!=3 or logits.shape[0]!=2 or truth.ndim!=1 or logits.shape[1]!=len(truth) or scored.shape!=truth.shape or scored.dtype!=np.bool_ or copies.shape!=truth.shape or counts.shape!=(logits.shape[2],) or len(counts)<2 or not np.isfinite(logits).all() or (counts<0).any() or not np.isfinite(counts).all() or ((truth < -1)|(truth>=len(counts))).any() or ((copies < -1)|(copies>=len(counts))).any() or (scored&(truth<0)).any():raise ValueError('fullC finite logits and exact scored/missing candidate namespace')
 H=len(truth);states=np.full((4,H),4,dtype=np.uint8);states[:,truth<0]=1;states[:,scored]=0;loss=np.full((4,H),np.nan);safe=np.maximum(truth,0)
 for m in range(2):
  z=logits[m]-logits[m].max(1,keepdims=True);v=np.log(np.exp(z).sum(1))-z[np.arange(H),safe];loss[m,scored]=v[scored]
 if int(counts.sum())==0:states[2,scored]=2
 else:
  zero=scored&(counts[safe]==0);finite=scored&~zero;states[2,zero]=3;loss[2,zero]=np.inf;loss[2,finite]=np.log(float(counts.sum()))-np.log(counts[safe[finite]].astype(np.float64))
 missing=scored&(copies<0);zero=scored&~missing&(copies!=truth);match=scored&~missing&~zero;states[3,missing]=2;states[3,zero]=3;loss[3,zero]=np.inf;loss[3,match]=0.
 return states,loss

def accumulate(sums,den,neginf,states,loss,families,block):
 available=np.isin(states,[0,3])
 for other_index,other in enumerate([1,2,3]):
  for cohort in range(2):
   use=available[0]&available[other]
   if cohort:use&=states[other]==0
   for h in np.flatnonzero(use):
    ix=(cohort,other_index,int(families[h]),block);den[ix]+=1
    if states[other,h]==3:neginf[ix]+=1
    else:sums[ix]+=float(loss[0,h]-loss[other,h])

def statistic(sums,den,neginf,wf,wb):
 n=den@wb;v=sums@wb;z=neginf@wb;used=(n>0)&(wf>0);F=int(wf[used].sum())
 if F==0:return dict(state='UNAVAILABLE_NO_DEFINED_FAMILY',value=None,defined_family_draws=0)
 if np.any((z>0)&used):return dict(state='NEGATIVE_INFINITY_EXACT_BASELINE_ZERO',value=None,defined_family_draws=F)
 return dict(state='FINITE_COMMON_AVAILABLE',value=float(np.sum(wf[used]*v[used]/n[used])/F),defined_family_draws=F)

def summarize(values):
 defined=sorted([-math.inf if x['state'].startswith('NEGATIVE_INFINITY') else x['value'] for x in values if not x['state'].startswith('UNAVAILABLE')]);quant=[]
 for frac in [.025,.5,.975]:
  v=defined[max(0,math.ceil(frac*len(defined))-1)] if defined else None;quant.append('NEGATIVE_INFINITY' if v==-math.inf else v)
 return dict(replicates=len(values),finite=sum(x['state'].startswith('FINITE') for x in values),negative_infinite=sum(x['state'].startswith('NEGATIVE_INFINITY') for x in values),undefined=sum(x['state'].startswith('UNAVAILABLE') for x in values),empirical_nearest_rank_percentiles=quant,interpretation='Paired dependence sensitivity, conditional on defined replicates; all frozen families retained in draws; not unconditional or joint simultaneous CI; reference1Mb is not validated LD')
