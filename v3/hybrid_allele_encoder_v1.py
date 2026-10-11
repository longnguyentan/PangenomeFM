"""Complete small-literal CNN plus frozen complete NT-bank projection.

Lookup keys never enter tensors. Explicit missing/common-flank clipping states
are separate from observed empty flanks. CNN pools every original base exactly
once, with four-base convolution halos and shared forward/RC orientation means.
"""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

DNA_CODES={c:i for i,c in enumerate(b'ACGTN')}
RC_TABLE=bytes.maketrans(b'ACGTN',b'TGCAN')

def pack_small(records,core_bases=256):
    if type(core_bases)is not int or core_bases<1:raise ValueError('positive complete CNN window required')
    codes=[];roles=[];groups=[];expected=[];features=[]
    for row,r in enumerate(records):
        own=r['DNA']
        if not isinstance(own,bytes) or not own or own.upper().translate(None,b'ACGTN'):
            raise ValueError('whole actual allele DNA required before CNN')
        own=own.upper();left=r['left'];right=r['right']
        for x in (left,right):
            if x is not None and (not isinstance(x,bytes) or x.upper().translate(None,b'ACGTN')):raise ValueError('exact flank or explicit unavailable required')
        for x,clip in [(left,r['left_clipped']),(right,r['right_clipped'])]:
            if (x is None)!=(clip is None) or (x is not None and (type(clip)is not int or clip<0 or len(x)+clip!=64)):raise ValueError('exact common64bp flank/clipping state required')
        for orient in (0,1):
            l,o,rr=(left,own,right) if not orient else (None if right is None else right.upper().translate(RC_TABLE)[::-1],own.translate(RC_TABLE)[::-1],None if left is None else left.upper().translate(RC_TABLE)[::-1])
            l=None if l is None else l.upper();rr=None if rr is None else rr.upper();sequence=(l or b'')+o+(rr or b'');lens=[len(l or b''),len(o),len(rr or b'')];role=[0]*lens[0]+[1]*lens[1]+[2]*lens[2];expected.append(lens)
            lc,rc=(r['left_clipped'],r['right_clipped']) if not orient else (r['right_clipped'],r['left_clipped'])
            features.append([float(l is not None),float(rr is not None),0. if lc is None else lc/64,0. if rc is None else rc/64,math.log1p(len(o)),o.count(b'N')/len(o)])
            for begin in range(0,len(sequence),core_bases):
                stop=min(len(sequence),begin+core_bases);lo=max(0,begin-4);hi=min(len(sequence),stop+4);offset=4-(begin-lo);width=core_bases+8;c=[5]*width;z=[-1]*width
                c[offset:offset+hi-lo]=[DNA_CODES[b] for b in sequence[lo:hi]]
                z[4:4+stop-begin]=role[begin:stop];codes.append(c);roles.append(z);groups.append(2*row+orient)
    if not records:return dict(codes=torch.empty((0,core_bases+8),dtype=torch.long),roles=torch.empty((0,core_bases+8),dtype=torch.long),groups=torch.empty(0,dtype=torch.long),counts=torch.empty((0,3),dtype=torch.long),features=torch.empty((0,6)),alleles=0)
    return dict(codes=torch.tensor(codes,dtype=torch.long),roles=torch.tensor(roles,dtype=torch.long),groups=torch.tensor(groups,dtype=torch.long),counts=torch.tensor(expected,dtype=torch.long),features=torch.tensor(features,dtype=torch.float32),alleles=len(records))

class CompleteSmallCNN(nn.Module):
    def __init__(self,width,window_batch=64):
        super().__init__();self.width=width;self.window_batch=window_batch
        self.bases=nn.Embedding(6,width,padding_idx=5);self.conv1=nn.Conv1d(width,width,5,padding=2);self.conv2=nn.Conv1d(width,width,5,padding=2)
        self.merge=nn.Sequential(nn.Linear(width*3+6,width),nn.GELU(),nn.Linear(width,width))
    def window_sums(self,codes,roles):
        valid=(codes!=5)[:,None];x=self.bases(codes).transpose(1,2);x=torch.nn.functional.gelu(self.conv1(x))*valid;x=torch.nn.functional.gelu(self.conv2(x))*valid;x=x.transpose(1,2)
        return torch.stack([(x*(roles==role)[...,None]).sum(1) for role in range(3)],1)
    def forward(self,packed):
        device=self.bases.weight.device;n=packed['alleles'];counts=packed['counts'].to(device);features=packed['features'].to(device,dtype=self.bases.weight.dtype)
        if not n:return self.bases.weight.new_empty((0,self.width))
        if counts.shape!=(2*n,3) or features.shape!=(2*n,6) or (counts[:,1]<1).any():raise ValueError('whole literal plus explicit flank states required')
        sums=self.bases.weight.new_zeros((2*n,3,self.width));seen=torch.zeros_like(counts)
        for begin in range(0,len(packed['codes']),self.window_batch):
            c=packed['codes'][begin:begin+self.window_batch].to(device);r=packed['roles'][begin:begin+self.window_batch].to(device);g=packed['groups'][begin:begin+self.window_batch].to(device)
            if c.shape!=r.shape or ((c<0)|(c>5)).any() or ((r < -1)|(r>2)).any() or ((c==5)&(r>=0)).any() or (g<0).any() or (g>=2*n).any():raise ValueError('exact full window/code/core population required')
            value=checkpoint(self.window_sums,c,r,use_reentrant=False) if self.training and torch.is_grad_enabled() else self.window_sums(c,r)
            sums=sums.index_add(0,g,value);seen.index_add_(0,g,torch.stack([(r==role).sum(1) for role in range(3)],1))
        if not torch.equal(seen,counts):raise ValueError('CNN complete base/role EOF differs')
        mean=sums/counts.clamp_min(1)[...,None];x=self.merge(torch.cat((mean.flatten(1),features),1))
        return x.reshape(n,2,self.width).mean(1)

class HybridAlleleEncoder(nn.Module):
    def __init__(self,width,window_batch=64):
        super().__init__();self.width=width;self.small=CompleteSmallCNN(width,window_batch);self.nt=nn.Sequential(nn.Linear(4096+8+2,width),nn.GELU(),nn.Linear(width,width))
    def structural(self,banks,masks,lengths,n_bases):
        if banks.shape[1:]!=(8,512) or masks.shape!=banks.shape[:2] or masks.dtype!=torch.bool or lengths.shape!=banks.shape[:1] or n_bases.shape!=lengths.shape or (lengths<1).any() or ((n_bases<0)|(n_bases>lengths)).any() or not masks.any(1).all() or not torch.isfinite(banks).all():raise ValueError('complete8-bank masked structural literal required')
        banks=banks*masks[...,None];meta=torch.stack((torch.log1p(lengths),n_bases/lengths),-1).to(banks);rev=torch.arange(7,-1,-1,device=banks.device)
        f=torch.cat((banks.flatten(1),masks.to(banks),meta),-1);r=torch.cat((banks[:,rev].flatten(1),masks[:,rev].to(banks),meta),-1)
        return (self.nt(f)+self.nt(r))/2
    def forward(self,total,small_indices,small_packed,structural_indices,banks,masks,lengths,n_bases):
        device=self.nt[0].weight.device;ids=torch.cat((small_indices,structural_indices))
        if ids.dtype!=torch.long or ids.device!=device or len(ids)!=total or not torch.equal(torch.sort(ids).values,torch.arange(total,device=device)):raise ValueError('every full original candidate encoded once; no hidden subset')
        a=self.small(small_packed);b=self.structural(banks,masks,lengths,n_bases) if len(structural_indices) else self.nt[0].weight.new_empty((0,self.width))
        if len(a)!=len(small_indices) or len(b)!=len(structural_indices):raise ValueError('exact complete candidate/CNN/NT joins required')
        result=self.nt[0].weight.new_zeros((total,self.width));return result.index_copy(0,small_indices,a).index_copy(0,structural_indices,b)
