"""Long's explicit pre-fit revision: ordered linkage and isolated query chunks.

Site indices and reference base-pair positions are geometry only. Panel rows
never read query states in any layer. Query chunks share the panel computation;
they are not independent panel forwards. All candidate vectors remain ragged.
This new version does not alter any previously registered/executed model.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from axial_model import ModelConfig, MISSING, MASK
from ragged_model_v1 import RaggedPangenomeImputer

@dataclass(frozen=True)
class PositionalConfig(ModelConfig):
    query_chunk: int = 64
    rope_theta: float = 10000.
    bp_scale: float = 1000.

def validate_geometry(indices, positions, sites, device):
    for value in (indices, positions):
        if value.shape != (sites,) or value.dtype != torch.long or value.device != device:
            raise ValueError('exact source-ordered int64 site indices and bp positions required')
        if (value < 0).any() or (value[1:] < value[:-1]).any():
            raise ValueError('negative or decreasing source geometry')
    if (indices[1:] == indices[:-1]).any():
        raise ValueError('site indices must uniquely retain source order')

def rotary(x, indices, theta):
    # Float64 phase construction avoids loss of integer site positions before
    # trigonometry. Rotate adjacent channels; no genotype/ID embedding table.
    d=x.shape[-1]
    inv=theta ** (-torch.arange(0,d,2,device=x.device,dtype=torch.float64)/d)
    phase=indices.to(torch.float64)[:,None]*inv[None]
    cos,sin=phase.cos().to(x),phase.sin().to(x)
    pairs=x.reshape(*x.shape[:-1],d//2,2)
    a,b=pairs[...,0],pairs[...,1]
    return torch.stack((a*cos-b*sin,a*sin+b*cos),-1).flatten(-2)

class PositionalAxialBlock(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c
        if c.width%c.heads or (c.width//c.heads)%2 or not 1<=c.query_chunk<=64 or c.bp_scale<=0 or c.rope_theta<=1:
            raise ValueError('even rotary head dimension and query chunk in 1..64 required')
        self.row_norm=nn.LayerNorm(c.width);self.column_norm=nn.LayerNorm(c.width)
        self.qkv=nn.Linear(c.width,3*c.width);self.row_output=nn.Linear(c.width,c.width)
        self.column=nn.MultiheadAttention(c.width,c.heads,dropout=c.dropout,batch_first=True)
        self.ff_norm=nn.LayerNorm(c.width)
        self.ff=nn.Sequential(nn.Linear(c.width,4*c.width),nn.GELU(),nn.Dropout(c.dropout),nn.Linear(4*c.width,c.width))
        self.local_conv=nn.Conv1d(c.width,c.width,5,padding=2,groups=c.width)
        self.register_buffer('distance_slopes',2.**(-8.*(torch.arange(c.heads,dtype=torch.float32)+1)/c.heads))

    def distance_bias(self,query_bp,key_bp,dtype):
        distance=(query_bp[:,None]-key_bp[None,:]).abs().to(torch.float64)
        return (-self.distance_slopes[:,None,None]*torch.log1p(distance/self.c.bp_scale)[None]).to(dtype)

    def forward(self,x,panel_rows,site_indices,positions):
        h,s,d=x.shape;validate_geometry(site_indices,positions,s,x.device)
        if not 0<panel_rows<=h:raise ValueError('nonempty isolated TRAIN panel required')
        z=self.row_norm(x)
        q,k,v=self.qkv(z).reshape(h,s,3,self.c.heads,d//self.c.heads).permute(2,0,3,1,4).unbind(0)
        q,k=rotary(q,site_indices,self.c.rope_theta),rotary(k,site_indices,self.c.rope_theta)
        chunks=[]
        for start in range(0,s,self.c.row_window):
            stop=min(s,start+self.c.row_window)
            lo,hi=max(0,start-self.c.row_halo),min(s,stop+self.c.row_halo)
            bias=self.distance_bias(positions[start:stop],positions[lo:hi],q.dtype)[None]
            y=F.scaled_dot_product_attention(q[:,:,start:stop],k[:,:,lo:hi],v[:,:,lo:hi],attn_mask=bias,dropout_p=self.c.dropout if self.training else 0.)
            chunks.append(y.transpose(1,2).reshape(h,stop-start,d))
        x=x+self.row_output(torch.cat(chunks,1))
        x=x+self.local_conv(x.transpose(1,2)).transpose(1,2)
        z=self.column_norm(x).transpose(0,1);p=z[:,:panel_rows]
        panel=self.column(p,p,p,need_weights=False)[0]
        queries=[]
        for start in range(panel_rows,h,self.c.query_chunk):
            queries.append(self.column(z[:,start:min(h,start+self.c.query_chunk)],p,p,need_weights=False)[0])
        x=x+torch.cat([panel]+queries,1).transpose(0,1)
        return x+self.ff(self.ff_norm(x))

class PositionalRaggedImputer(RaggedPangenomeImputer):
    def __init__(self,config=PositionalConfig()):
        nn.Module.__init__(self);self.config=config
        if min(config.layers,config.row_window,config.candidate_block)<1:raise ValueError('positive model dimensions required')
        self.missing=nn.Parameter(torch.randn(config.width)*.02)
        self.masked=nn.Parameter(torch.randn(config.width)*.02)
        self.site=nn.Sequential(nn.Linear(4,config.width),nn.GELU(),nn.Linear(config.width,config.width))
        self.blocks=nn.ModuleList([PositionalAxialBlock(config) for _ in range(config.layers)])
        self.output=nn.Sequential(nn.LayerNorm(config.width),nn.Linear(config.width,config.width,bias=False))

    def hidden_ragged(self,observations,alleles,offsets,site_features,panel_rows,site_indices,positions):
        if observations.ndim!=2 or observations.dtype!=torch.long:raise ValueError('exact long haplotype/site observations required')
        h,s=observations.shape;counts=self.validate(alleles,offsets,s)
        validate_geometry(site_indices,positions,s,alleles.device)
        if site_features.shape!=(s,4) or not torch.isfinite(site_features).all() or not 0<panel_rows<=h:
            raise ValueError('exact finite site features and panel required')
        if (observations<MASK).any() or (observations>=counts[None]).any():raise ValueError('observed allele outside full candidate universe')
        x=alleles[offsets[:-1][None]+observations.clamp_min(0)]
        x=torch.where((observations==MISSING)[...,None],self.missing,x)
        x=torch.where((observations==MASK)[...,None],self.masked,x)+self.site(site_features)[None]
        for block in self.blocks:
            args=(x,panel_rows,site_indices,positions)
            x=checkpoint(block,*args,use_reentrant=False) if self.training and self.config.checkpoint_layers else block(*args)
        return self.output(x)
