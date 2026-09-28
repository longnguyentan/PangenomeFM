"""Compare optional attention to independent dense equations and checkpoint replay."""
import math

import pytest
import torch

from evaluation.external import _build_model_from_checkpoint, _namespace_from_checkpoint
from models.coordinate_attention import coordinate_attention
from models.dual_stream_gat import DualStreamPangenomeGAT, LinearStreamAttention


def dense_oracle(q, k, v, positions=None, window_k=None):
    logits = torch.einsum('ihd,jhd->hij', q, k)/math.sqrt(q.shape[-1])
    if window_k is not None:
        order = torch.argsort(positions, stable=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(len(order))
        logits = logits.masked_fill((rank[:,None]-rank[None,:]).abs()[None,:,:] > window_k//2, float('-inf'))
    return torch.einsum('hij,jhd->ihd', torch.softmax(logits,-1), v)


@pytest.mark.parametrize('window_k', [None,1,2,5,30])
def test_matches_dense_equation_and_gradients_with_ties_and_boundary_keys(window_k):
    torch.manual_seed(63)
    values = [torch.randn(9,2,4,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    positions = torch.tensor([4,2,0,2,9,1,5,8,3])
    actual = coordinate_attention(*values,chunk_size=3,positions=positions,window_k=window_k)
    expected = dense_oracle(*values, positions, window_k)
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)
    actual_grad = torch.autograd.grad(actual.square().sum(),values)
    expected_grad = torch.autograd.grad(expected.square().sum(),values)
    for a,b in zip(actual_grad,expected_grad):
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('window_k', [None,4])
def test_recomputed_backward_preserves_same_chunked_dropout(window_k):
    torch.manual_seed(80)
    values = [torch.randn(9,2,4,requires_grad=True) for _ in range(3)]
    outputs, grads = [], []
    for recompute in [False,True]:
        torch.manual_seed(12)
        out=coordinate_attention(*values,chunk_size=3,positions=torch.arange(9),window_k=window_k,
            dropout_p=.2,recompute=recompute)
        outputs.append(out)
        grads.append(torch.autograd.grad(out.square().sum(),values))
    torch.testing.assert_close(*outputs,rtol=0,atol=0)
    for a,b in zip(*grads):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_full_layer_exact_mode_matches_legacy_dense_with_rotary_positions():
    torch.manual_seed(81)
    legacy=LinearStreamAttention(24,4,dropout=.1,use_multiscale_rope=True).double().eval()
    exact=LinearStreamAttention(24,4,dropout=.1,use_multiscale_rope=True,
        coordinate_attention_mode='chunked_exact',attention_chunk_size=3).double().eval()
    exact.load_state_dict(legacy.state_dict())
    x=torch.randn(9,24,dtype=torch.float64)
    so=torch.arange(9)*100
    orientation=torch.arange(9)%2
    torch.testing.assert_close(legacy(x,so,orientation),exact(x,so,orientation),rtol=1e-10,atol=1e-10)
    with pytest.raises(ValueError, match='global'):
        LinearStreamAttention(24,4,window_k=4,coordinate_attention_mode='chunked_exact')
    with pytest.raises(ValueError, match='positive window'):
        LinearStreamAttention(24,4,coordinate_attention_mode='chunked_window')


@pytest.mark.parametrize('mode', ['legacy','chunked_exact','chunked_window'])
def test_saved_architecture_metadata_restores_attention_mode_and_predictions(mode):
    torch.manual_seed(71)
    options=dict(in_dim=7,hidden_dim=24,n_heads=4,n_layers=2,dropout=.1,edge_mlp_dim=48,
        coordinate_attention_mode=mode,attention_chunk_size=3,window_k=4 if mode=='chunked_window' else None)
    model=DualStreamPangenomeGAT(**options).eval()
    args={k:v for k,v in options.items() if k not in {'in_dim','edge_mlp_dim'}}
    if mode=='legacy':
        args.pop('coordinate_attention_mode')
        args.pop('attention_chunk_size')  # actual old checkpoints have neither
    ckpt=dict(in_dim=7,model_state=model.state_dict(),args=args)
    restored,_=_build_model_from_checkpoint(ckpt,_namespace_from_checkpoint(ckpt,42),torch.device('cpu'))
    assert all(layer.coordinate_attention_mode==mode for layer in restored.linear_layers)
    x=torch.randn(9,7)
    so=torch.arange(9)*100
    src,dst=torch.arange(8),torch.arange(1,9)
    with torch.no_grad():
        torch.testing.assert_close(model.encode_nodes(x,so,src,dst),restored.encode_nodes(x,so,src,dst),rtol=0,atol=0)


def test_legacy_sparse_issue_is_reproducible_without_changing_historical_behavior():
    torch.manual_seed(50)
    legacy=LinearStreamAttention(8,2,dropout=0,window_k=4)
    q,k,v=[torch.randn(5,2,4) for _ in range(3)]
    so=torch.arange(5)
    src,dst=legacy._build_window_edges(so,5,4)
    # Boundary clamping repeats the key 0 for query 1; current code does not deduplicate.
    assert len(set(zip(src.tolist(),dst.tolist()))) < len(src)
    current=legacy._sparse_windowed_attention(q,k,v,so,5,2,4).reshape(5,2,4)
    # Correct query-normalized attention on the SAME multiset, excluding self,
    # separates the query/key reversal bug from the new unique-key/self policy.
    outgoing=torch.stack([torch.softmax((q[i]*k[dst[src==i]]).sum(-1)/2,dim=0)[:,:,None]
        .mul(v[dst[src==i]]).sum(0) for i in range(5)])
    assert not torch.allclose(current,outgoing)
