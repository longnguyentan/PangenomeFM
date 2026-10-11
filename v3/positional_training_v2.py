"""Fresh identically initialized positional pair; complete atomic updates."""
import copy
import torch
from torch import nn
from positional_axial_v2 import PositionalConfig,PositionalRaggedImputer
from hybrid_allele_encoder_v1 import HybridAlleleEncoder
from grouped_candidate_loss_v1 import cell_nll,weighted_loss
from paired_axial_training_v1 import validate_gradients,optimizer

class PositionalSequenceModel(nn.Module):
    def __init__(self,config=PositionalConfig()):
        super().__init__();self.encoder=HybridAlleleEncoder(config.width);self.imputer=PositionalRaggedImputer(config)
    def forward(self,inputs,observations,offsets,site_features,panel_rows,site_indices,positions):
        vectors=self.encoder(**inputs)
        hidden=self.imputer.hidden_ragged(observations,vectors,offsets,site_features,panel_rows,site_indices,positions)
        return hidden,vectors

def twins(config,device='cpu',seed=20261011):
    torch.manual_seed(seed);trained=PositionalSequenceModel(config).to(device);random=copy.deepcopy(trained)
    for p in random.parameters():p.requires_grad_(False)
    for p in random.imputer.output.parameters():p.requires_grad_(True)
    if any(not torch.equal(a,b) for a,b in zip(trained.state_dict().values(),random.state_dict().values())):raise ValueError('paired initial state differs')
    return trained,random

def paired_step(models,optimizers,batch,SV_weight=10.,clip=1.,before_update=lambda:None):
    for model,opt in zip(models,optimizers):model.train();opt.zero_grad(set_to_none=True)
    statistics=[]
    for model in models:
        z,a=model(batch['allele_inputs'],batch['observations'],batch['offsets'],batch['site_features'],batch['panel_rows'],batch['site_indices'],batch['positions'])
        values=cell_nll(z,a,batch['offsets'],batch['truth'],batch['scored'],model.imputer.config.candidate_block)
        loss,stat=weighted_loss(values,batch['scored'],batch['structural'],SV_weight);loss.backward();stat['weighted_loss']=float(loss.detach());statistics.append(stat)
    if statistics[0]['scored_haplotypes']!=statistics[1]['scored_haplotypes'] or statistics[0]['weighted_denominator']!=statistics[1]['weighted_denominator']:raise ValueError('paired exposure differs')
    inactive=[]
    for model in models:
        names=set()
        if not batch['allele_inputs']['small_packed']['alleles']:names.update(n for n,p in model.named_parameters() if n.startswith('encoder.small.') and p.requires_grad)
        if not len(batch['allele_inputs']['structural_indices']):names.update(n for n,p in model.named_parameters() if n.startswith('encoder.nt.') and p.requires_grad)
        inactive.append(names)
    norms=validate_gradients(models,clip,inactive);before_update()
    for opt in optimizers:opt.step()
    if any(not torch.isfinite(p).all() for model in models for p in model.parameters()):raise ValueError('nonfinite post-update state')
    return dict(trained=statistics[0],random=statistics[1],gradient_norms=norms)
