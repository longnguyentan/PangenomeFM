from argparse import Namespace

import numpy as np
import pytest
import torch

from evaluation.external import _build_model_from_checkpoint, _namespace_from_checkpoint
from training.masked_features import MaskedFeatureObjective, segment_cosine_loss, segment_mask, segment_target_statistics
from training.pretrain import encoder_parameter_digest
from training.pretrain_masked_features import make_encoder, mask_seed, validation


def fixture():
    rng = np.random.default_rng(9)
    # Twelve segments, both oriented handles. Same sequence vector per segment.
    features = np.repeat(rng.normal(size=(12, 519)).astype(np.float32), 2, axis=0)
    raw = dict(name='window', target_sn='GRCh38#0#chr3', closure='1hop', nodes=np.arange(24),
        node_feats=features, so_arr=np.repeat(np.arange(12)*100,2), orient_arr=np.arange(24)%2,
        pop_ids_arr=np.zeros(24), src=np.arange(23), dst=np.arange(1,24), temps=np.ones(24),
        labels=np.ones(1), query_u=np.array([0]), query_v=np.array([1]), train_idx=np.array([0]),
        val_idx=np.array([],dtype=int), test_idx=np.array([],dtype=int), edge_attr=None,
        branching_frac=0., n_pos=1, n_neg=0)
    return raw


def test_segment_mask_grouping_determinism_and_rejection():
    ids = torch.tensor([0,1,2,3,4,5,6,7,8,9])
    mask = segment_mask(ids, .3, 42)
    assert torch.equal(mask[::2], mask[1::2])
    assert torch.equal(mask, segment_mask(ids,.3,42))
    assert int(mask.sum()) == 2
    with pytest.raises(ValueError):
        segment_mask(torch.tensor([0,1]), .3, 42)
    assert mask_seed(42,'chr1',1) != mask_seed(42,'chr1',2)


def test_target_moments_deduplicate_orientations_and_windows():
    raw = fixture()
    ids, mean, scale = segment_target_statistics([raw, raw])
    assert len(ids) == 12
    expected = raw['node_feats'][::2,7:].astype(np.float64)
    np.testing.assert_allclose(mean,expected.mean(0),rtol=1e-6)
    np.testing.assert_allclose(scale,expected.std(0),rtol=1e-6)
    bad = dict(raw,node_feats=raw['node_feats'].copy())
    bad['node_feats'][0,7] += 1
    with pytest.raises(ValueError,match='inconsistent'):
        segment_target_statistics([raw,bad])


def test_scaled_cosine_segment_weighting():
    target=torch.tensor([[1.,0.],[1.,0.],[1.,0.]])
    pred=torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
    assert segment_cosine_loss(pred,target,torch.tensor([1,1,2])).item() == pytest.approx(.5)


@pytest.mark.parametrize('frozen',[True,False])
@pytest.mark.parametrize('stream',['full','coordinate'])
def test_learning_freezing_remasking_and_native_checkpoint_reload(frozen,stream):
    from training.pretrain import tensorize_slice
    torch.manual_seed(42)
    raw=fixture()
    _,mean,scale=segment_target_statistics([raw])
    args=Namespace(hidden_dim=12,n_heads=2,n_layers=1,dropout=0.,stream_mode=stream,
        orientation_rope=True,pop_cond=False,device='cpu',seed=42)
    encoder=make_encoder(args)
    initial=encoder_parameter_digest(encoder)
    objective=MaskedFeatureObjective(encoder,519,12,2,mean,scale,frozen)
    sd=tensorize_slice(raw,torch.device('cpu'),args)
    mask=segment_mask(sd['node_oids'],.3,42)
    # Decoder pre-hook proves masked latent rows were actually zeroed.
    seen=[]
    hook=objective.decoder.register_forward_pre_hook(lambda _module,inputs: seen.append(inputs[0].detach().clone()))
    opt=torch.optim.AdamW([p for p in objective.parameters() if p.requires_grad],lr=.001)
    loss,n=objective(sd,mask)
    assert n == 3 and torch.isfinite(loss)
    loss.backward()
    opt.step()
    assert (encoder_parameter_digest(encoder)==initial) == frozen
    assert torch.count_nonzero(seen[0][mask]).item() == 0
    hook.remove()
    bad=mask.clone()
    bad[1]=~bad[0]
    with pytest.raises(ValueError,match='Reverse'):
        objective(sd,bad)
    score,frame=validation(objective,[raw],args,{'mask_rate':.3})
    assert len(frame)==3 and np.isfinite(score)
    payload=dict(model_state=encoder.state_dict(),predictor_state=None,in_dim=519,edge_feat_dim=0,
        args=dict(vars(args),multiscale_rope=True,graph_message_direction='bidirectional',objective='masked_nt_features'))
    rebuilt,_=_build_model_from_checkpoint(payload,_namespace_from_checkpoint(payload,42),'cpu')
    encoder.eval()
    values=(sd['X'],sd['so'],sd['src'],sd['dst'],sd['temps'],sd['edge_attr'],sd['orient'],sd['pop_ids'])
    torch.testing.assert_close(encoder.encode_nodes(*values),rebuilt.encode_nodes(*values),rtol=0,atol=0)
