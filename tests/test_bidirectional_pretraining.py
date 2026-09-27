from argparse import Namespace
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from evaluation.external import _build_model_from_checkpoint, _namespace_from_checkpoint
from models.dual_stream_gat import DualStreamPangenomeGAT, bidirectional_messages
from training.pretrain import (
    LinearLinkPredictor, drop_edges, mask_positive_query_edges, validation_only_results,
)


def make_model(direction="incoming"):
    return DualStreamPangenomeGAT(
        in_dim=7, hidden_dim=8, n_heads=2, n_layers=1, dropout=0,
        edge_mlp_dim=16, stream_mode="graph", graph_message_direction=direction,
    ).eval()


def test_message_types_attributes_and_self_edges():
    src, dst = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 2])
    attributes = torch.arange(9).reshape(3, 3).float()
    u, v, a, kind = bidirectional_messages(src, dst, attributes)
    assert list(zip(u.tolist(), v.tolist(), kind.tolist())) == [
        (0, 1, 0), (1, 2, 0), (2, 2, 0), (1, 0, 1), (2, 1, 1),
    ]
    torch.testing.assert_close(a, attributes[torch.tensor([0, 1, 2, 0, 1])])
    # Augmentation cannot restore an edge removed by DropEdge.
    u, v, a = drop_edges(src, dst, 1, attributes, training=True)
    u, v, a, kind = bidirectional_messages(u, v, a)
    assert len(u) == len(v) == len(kind) == 0 and a.shape == (0, 3)


def test_query_mask_removes_all_four_traversals_before_reverse_messages():
    # Query 0->2, ordinary reverse, reverse-complement, reverse of RC,
    # plus an unrelated link that should survive in both message directions.
    src, dst = torch.tensor([0, 2, 3, 1, 4]), torch.tensor([2, 0, 1, 3, 6])
    query = dict(q_u=torch.tensor([0]), q_v=torch.tensor([2]), labels=torch.ones(1),
                 idx=torch.tensor([0]), node_oids=torch.arange(8))
    old_u, old_v, _ = mask_positive_query_edges(src, dst, None, **query)
    assert set(zip(old_u.tolist(), old_v.tolist())) == {(2, 0), (1, 3), (4, 6)}
    u, v, a = mask_positive_query_edges(src, dst, None, **query, mask_reverse_messages=True)
    u, v, _, _ = bidirectional_messages(u, v, a)
    assert set(zip(u.tolist(), v.tolist())) == {(4, 6), (6, 4)}


def test_bidirectional_graph_branch_sees_successor_and_learns_direction_bias():
    torch.manual_seed(42)
    incoming = make_model()
    bidir = make_model("bidirectional")
    bidir.load_state_dict(incoming.state_dict(), strict=False)
    x = torch.randn(3, 7, requires_grad=True)
    src, dst = torch.tensor([0, 1]), torch.tensor([1, 2])
    args = (torch.arange(3), src, dst, torch.ones(3))
    y_in = incoming.encode_nodes(x, *args)
    g_in = torch.autograd.grad(y_in[0, 0], x)[0]
    y_bi = bidir.encode_nodes(x, *args)
    g_bi = torch.autograd.grad(y_bi[0, 0], x, retain_graph=True)[0]
    assert g_in[1].abs().sum() == 0
    assert g_bi[1].abs().sum() > 1e-6
    y_bi[1, 0].backward()
    assert bidir.graph_layers[0].direction_bias.grad.abs().sum() > 1e-6


def test_default_constructor_keeps_legacy_state_and_rng():
    torch.manual_seed(7)
    default = DualStreamPangenomeGAT(hidden_dim=8, n_heads=2, n_layers=1)
    torch.manual_seed(7)
    explicit = DualStreamPangenomeGAT(
        hidden_dim=8, n_heads=2, n_layers=1, graph_message_direction="incoming",
    )
    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert not any("direction_bias" in key for key in default.state_dict())
    for key in default.state_dict():
        torch.testing.assert_close(default.state_dict()[key], explicit.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("direction", ["incoming", "bidirectional"])
@pytest.mark.parametrize("geometry", [False, True])
def test_checkpoint_reload_and_random_twin(direction, geometry):
    torch.manual_seed(8)
    model = make_model(direction)
    predictor = LinearLinkPredictor(8, geom_dim=4 if geometry else 0)
    saved = dict(model_state=model.state_dict(), predictor_state=predictor.state_dict(), in_dim=7,
                 edge_feat_dim=0, args=dict(hidden_dim=8, n_heads=2, n_layers=1, dropout=0,
                 stream_mode="graph", graph_message_direction=direction, linear_predictor=True,
                 pair_geometry=geometry))
    ns = _namespace_from_checkpoint(saved, seed=42)
    restored, head = _build_model_from_checkpoint(saved, ns, torch.device("cpu"))
    inputs = (torch.randn(3, 7), torch.arange(3), torch.tensor([0, 1]), torch.tensor([1, 2]))
    z, z_restored = model.encode_nodes(*inputs), restored.encode_nodes(*inputs)
    torch.testing.assert_close(z, z_restored, rtol=0, atol=0)
    geom = torch.zeros(1, 4) if geometry else None
    torch.testing.assert_close(predictor(z[:1], z[1:2], geom), head(z[:1], z[1:2], geom))
    spec = importlib.util.spec_from_file_location("random_checkpoint", Path("scripts/make_random_init_checkpoint.py"))
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    twin = type(model)(**script._constructor_kwargs(model, saved, ns))
    assert twin.state_dict().keys() == model.state_dict().keys()
    assert twin.graph_message_direction == direction


def test_linear_interactions_distinguish_balanced_repair_pairs():
    head = LinearLinkPredictor(2)
    with torch.no_grad():
        head.linear.weight.copy_(torch.tensor([[1., 1., 0., 0.]]))
        head.linear.bias.zero_()
    u = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    v = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [1., 0.]])
    torch.testing.assert_close(head(u, v), torch.tensor([1., 1., 0., 0.]))


def test_validation_only_final_scoring_never_requests_test(monkeypatch):
    calls = []

    def fake_evaluate(model, predictor, slices, split, args, rows, label):
        calls.append((slices, split, label))
        assert split == "val"
        return .6, [dict(name=label, val_auc=.6)]

    monkeypatch.setattr("training.pretrain.evaluate_shared", fake_evaluate)
    rows = validation_only_results(None, None, ["train"], ["val"], Namespace(), [], 2)
    assert calls == [(["train"], "val", "train_chr_val"), (["val"], "val", "val_chr_val")]
    assert len(rows) == 2 and all(np.isnan(r["test_auc"]) for r in rows)
    assert {r["evaluation_scope"] for r in rows} == {"validation_only"}


@pytest.mark.parametrize("direction,linear,frozen,sequence", [("incoming", False, False, False), ("bidirectional", True, False, False),
                                                    ("bidirectional", True, True, False), ("incoming", False, True, False),
                                                    ("bidirectional", False, False, True)])
def test_native_training_cli_validation_only_checkpoint(tmp_path, direction, linear, frozen, sequence):
    """Exercise loading, masking, training, final scoring and checkpoint reload."""
    segments, manifest = [], []
    for chromosome in [1, 2, 3]:
        n = 152
        ids = np.arange(n) + (chromosome - 1) * n
        seg = pd.DataFrame(dict(id=ids, name=ids.astype(str), LN=100,
                               SN=f"GRCh38#0#chr{chromosome}", SO=np.arange(n) * 100, SR=0))
        left = np.r_[ids[:-1], ids[:-2:3]]
        right = np.r_[ids[1:], ids[2::3]]
        links = pd.DataFrame(dict(from_seg=left.astype(str), to_seg=right.astype(str),
                                 from_orient="+", to_orient="+"))
        sp, lp = tmp_path / f"chr{chromosome}_segments.csv", tmp_path / f"chr{chromosome}_links.csv"
        seg.to_csv(sp, index=False)
        links.to_csv(lp, index=False)
        segments.append(seg)
        manifest.append(dict(name=f"window_chr{chromosome}", target_sn=f"GRCh38#0#chr{chromosome}",
                             closure="strict", segments_path=str(sp), links_path=str(lp)))
    full, mf, output = tmp_path / "segments.csv", tmp_path / "manifest.csv", tmp_path / "output"
    pd.concat(segments).to_csv(full, index=False)
    pd.DataFrame(manifest).to_csv(mf, index=False)
    cmd = [sys.executable, "-m", "training.pretrain", "--manifest", str(mf),
           "--full_segments", str(full), "--out_dir", str(output), "--closures", "strict",
           "--hidden_dim", "8", "--n_heads", "2", "--n_layers", "1", "--epochs", "1",
           "--patience", "1", "--warmup_epochs", "0", "--dropout", "0", "--batch_size", "512",
           "--objective", "junction_repair", "--node_structure_source", "visible",
           "--mask_query_edges", "--val_chrs", "chr2", "--test_chrs", "chr3",
           "--validation_only", "--save_predictions", "--graph_message_direction", direction]
    if linear:
        cmd.append("--linear_predictor")
    if frozen:
        cmd.append("--freeze_encoder")
    if sequence:
        import json
        cache = tmp_path / 'nt.npz'
        np.savez(cache, segid=np.arange(456), embeddings=np.random.default_rng(42).normal(size=(456, 512)).astype(np.float32))
        Path(f'{cache}.audit.json').write_text(json.dumps(dict(status='complete', downstream_label_access='none')))
        cmd += ['--node_extra_features', 'cache', '--node_feature_cache', str(cache), '--node_feature_min_coverage', '1.0']
    env = dict(os.environ, PYTHONPATH="src:.", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    predictions = pd.read_csv(next(output.rglob("*pooled_predictions*.csv.gz")))
    assert not predictions.astype(str).apply(lambda c: c.str.contains("heldout|chr3|_test")).any().any()
    saved = torch.load(next(output.rglob("ckpt_*.pt")), map_location="cpu", weights_only=False)
    assert saved["args"]["validation_only"] and saved["args"]["graph_message_direction"] == direction
    assert saved["args"]["linear_predictor"] == linear
    assert saved["args"]["freeze_encoder"] == frozen
    assert (saved["initial_encoder_sha256"] == saved["final_encoder_sha256"]) == frozen
    model, head = _build_model_from_checkpoint(saved, _namespace_from_checkpoint(saved, 42), torch.device("cpu"))
    assert model.graph_message_direction == direction
    assert isinstance(head, LinearLinkPredictor) == linear
    assert saved['in_dim'] == (519 if sequence else 7)


def test_pilot_gate_is_context_specific_without_relaxing_coverage(tmp_path):
    import json
    from scripts.server.run_junction_geometry_pilot import check_context

    # The aggregate audit can fail strict while the one-hop protocol passes.
    native = dict(junction_geometry_match='signed_gap_bins', junction_geometry_bin_ratio=1.25,
                  node_structure_source='visible', objective='junction_repair', drop_edge_rate=.1)
    (tmp_path / 'audit.json').write_text(json.dumps(dict(status='failed', native_args=native)))
    pd.DataFrame([dict(context='1hop', chrom='chr1', exclusion='retained',
                       n_val_candidates=10, n_test_candidates=10)]).to_csv(tmp_path / 'per_window.csv', index=False)
    scores = pd.DataFrame([dict(context='1hop', baseline=b, n_val=300, auprc=.51, auroc=.52)
                           for b in ['geometry', 'geometry_and_visible_degree']])
    scores.to_csv(tmp_path / 'validation_baselines.csv', index=False)
    check_context(tmp_path, '1hop', 'incoming', {'chr1'})
    with pytest.raises(ValueError, match='coverage'):
        check_context(tmp_path, 'strict', 'incoming', {'chr1'})
    with pytest.raises(ValueError, match='coverage'):
        check_context(tmp_path, '1hop', 'incoming', {'chr1', 'chr2'})
    scores.loc[0, 'auprc'] = .70
    scores.to_csv(tmp_path / 'validation_baselines.csv', index=False)
    with pytest.raises(ValueError, match='gate'):
        check_context(tmp_path, '1hop', 'incoming', {'chr1'})


def test_pilot_commands_keep_validation_scope_and_fixed_budget():
    from scripts.server.run_junction_geometry_pilot import commands
    receipt = dict(manifest=dict(path='manifest.csv'), full_segments=dict(path='segments.csv'),
                   native_args=dict(seed=42, split_seed=20260806, test_chrs=['chr1'], val_chrs=['chr2']))
    jobs = commands(Path('results/pilot'), receipt, '1hop')
    assert len(jobs) == 4
    for name, cmd in jobs.items():
        assert '--validation_only' in cmd and '--save_predictions' in cmd
        assert '--pair_geometry' not in cmd
        assert cmd[cmd.index('--epochs') + 1] == '10'
        assert ('--linear_predictor' in cmd) == name.endswith('_linear')
        assert cmd[cmd.index('--graph_message_direction') + 1] == name.split('_')[0]


def test_sequence_pilot_keeps_fixed_budget_and_requires_complete_inputs():
    from scripts.server.run_junction_geometry_pilot import commands
    receipt = dict(manifest=dict(path='manifest.csv'), full_segments=dict(path='segments.csv'),
                   native_args=dict(seed=42, split_seed=20260806, test_chrs=['chr1'], val_chrs=['chr2']))
    jobs = commands(Path('results/nt_pilot'), receipt, '1hop',
                    arms=['bidirectional_default', 'bidirectional_linear'], node_feature_cache=Path('nt.npz'))
    assert list(jobs) == ['bidirectional_default', 'bidirectional_linear']
    for cmd in jobs.values():
        assert cmd[cmd.index('--node_feature_min_coverage') + 1] == '1.0'
        assert cmd[cmd.index('--node_feature_cache') + 1] == 'nt.npz'
        assert '--validation_only' in cmd and cmd[cmd.index('--epochs') + 1] == '10'
    with pytest.raises(ValueError, match='unique'):
        commands(Path('out'), receipt, '1hop', arms=['bidirectional_default'] * 2)


def test_sequence_pilot_rejects_missing_benchmark_inputs(tmp_path, monkeypatch):
    from scripts.server.run_junction_geometry_pilot import check_sequence_cache
    contract = dict(model_name='InstaDeepAI/nucleotide-transformer-v2-50m-multi-species',
                    resolved_revision='81b29e5786726d891dbf929404ef20adca5b36f1',
                    maximum_token_length=1000, maximum_raw_bases=6000, full_segments_sha256='graph',
                    pooling='mean final hidden state over non-special, non-padding tokens',
                    raw_sequence_sampling='long nodes retain balanced prefix and suffix separated by N before tokenizer truncation',
                    truncation_policy='tokenizer truncation at maximum_token_length; complete raw sequences remain in the source table')
    monkeypatch.setattr('scripts.server.merge_node_sequence_fm_caches.sequence_contract', lambda _: contract)
    monkeypatch.setattr('scripts.server.complete_node_sequence_fm_cache.benchmark_targets',
                        lambda *args: (np.array([1, 2]), {}))
    cache = tmp_path / 'nt.npz'
    receipt = dict(full_segments=dict(path='segments', sha256='graph'), manifest=dict(path='manifest'))
    np.savez(cache, segid=[1], embeddings=np.ones((1, 512)))
    with pytest.raises(ValueError, match='miss 1'):
        check_sequence_cache(cache, receipt, '1hop')
    np.savez(cache, segid=[1, 2], embeddings=np.ones((2, 512)))
    assert check_sequence_cache(cache, receipt, '1hop')['coverage'] == 1.0
    contract['full_segments_sha256'] = 'other graph'
    with pytest.raises(ValueError, match='different graph'):
        check_sequence_cache(cache, receipt, '1hop')
