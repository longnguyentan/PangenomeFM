"""Identically initialized learned/frozen-backbone pair with atomic checks.

Both complete candidate losses and gradients are checked before either optimizer
update. Profile weights cannot seed a scientific fit. RNG/cursor resume is not
implemented; output reuse or implicit retry must be refused by the lifecycle.
"""
import copy
import torch
from torch import nn
from axial_model import ModelConfig
from ragged_model_v1 import RaggedPangenomeImputer
from hybrid_allele_encoder_v1 import HybridAlleleEncoder
from grouped_candidate_loss_v1 import cell_nll,weighted_loss

class SequenceAxialModel(nn.Module):
    def __init__(self,config=ModelConfig()):
        super().__init__();self.encoder=HybridAlleleEncoder(config.width);self.imputer=RaggedPangenomeImputer(config)
    def forward(self,inputs,observations,offsets,site_features,panel_rows):
        vectors=self.encoder(**inputs);hidden=self.imputer.hidden_ragged(observations,vectors,offsets,site_features,panel_rows);return hidden,vectors

def twins(config,device='cpu',seed=20261011):
    torch.manual_seed(seed);trained=SequenceAxialModel(config).to(device);random=copy.deepcopy(trained)
    for parameter in random.parameters():parameter.requires_grad_(False)
    for parameter in random.imputer.output.parameters():parameter.requires_grad_(True)
    if any(not torch.equal(a,b) for a,b in zip(trained.state_dict().values(),random.state_dict().values())):raise ValueError('paired initial state differs')
    return trained,random

def optimizer(model,learning_rate=.0003,weight_decay=.01):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=learning_rate,weight_decay=weight_decay)

def validate_gradients(models,clip=1.,inactive=None):
    norms=[]
    for index,model in enumerate(models):
        allowed=set() if inactive is None else inactive[index]
        params=[p for name,p in model.named_parameters() if p.requires_grad and name not in allowed]
        if any(p.grad is not None for name,p in model.named_parameters() if name in allowed):raise ValueError('declared inactive input branch unexpectedly active')
        if not params or any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise ValueError('every trained parameter needs finite paired gradients before either update')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('frozen backbone unexpectedly has a gradient')
        norm=torch.nn.utils.clip_grad_norm_(params,clip,error_if_nonfinite=True)
        if not torch.isfinite(norm):raise ValueError('nonfinite paired gradient norm')
        norms.append(float(norm))
    return norms

def paired_step(models,optimizers,batch,SV_weight=10.,clip=1.,before_update=lambda:None):
    for model,opt in zip(models,optimizers):model.train();opt.zero_grad(set_to_none=True)
    statistics=[]
    for model in models:
        z,a=model(batch['allele_inputs'],batch['observations'],batch['offsets'],batch['site_features'],batch['panel_rows'])
        values=cell_nll(z,a,batch['offsets'],batch['truth'],batch['scored'],model.imputer.config.candidate_block)
        loss,stat=weighted_loss(values,batch['scored'],batch['structural'],SV_weight);loss.backward();stat['weighted_loss']=float(loss.detach());statistics.append(stat)
    if statistics[0]['scored_haplotypes']!=statistics[1]['scored_haplotypes'] or statistics[0]['weighted_denominator']!=statistics[1]['weighted_denominator']:raise ValueError('paired source/loss exposure differs')
    inactive=[]
    for model in models:
        names=set()
        if not batch['allele_inputs']['small_packed']['alleles']:names.update(n for n,p in model.named_parameters() if n.startswith('encoder.small.') and p.requires_grad)
        if not len(batch['allele_inputs']['structural_indices']):names.update(n for n,p in model.named_parameters() if n.startswith('encoder.nt.') and p.requires_grad)
        inactive.append(names)
    norms=validate_gradients(models,clip,inactive);before_update()
    for opt in optimizers:opt.step()
    for model in models:
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite post-update state; preserve failure, no retry')
    return dict(trained=statistics[0],random=statistics[1],gradient_norms=norms)
