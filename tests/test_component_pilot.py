from argparse import Namespace
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.server.run_component_context_pilot import probe_command
from scripts.server.preflight_component_pilot import verify_matched_banks
from scripts.server.run_ccre_frozen_probe_fold import evaluate_feature_sets, validate_checkpoint_holdout
from tasks.transfer.component_pilot_report import compare
from training.component_pilot import (
    ARMS, CONTEXTS, SEEDS, load_matched_slices, pair_slice, readiness_receipts,
    reconstruction_mask, target_statistics, validate_plan, guard_extraction_scope,
)
from training.pretrain import encoder_parameter_digest, tensorize_slice
from training.pretrain_masked_features import make_encoder
from training.masked_features import MaskedFeatureObjective, segment_mask, segment_target_statistics


def plan():
    return json.loads(Path('configs/masked_nt_component_pilot_20260928.json').read_text())


def raw(nodes, name='w', chrom='chr3'):
    nodes = np.asarray(nodes, np.int64)
    values = np.random.default_rng(7).normal(size=(20, 519)).astype(np.float32)
    return dict(name=name, target_sn='GRCh38#0#' + chrom, closure='1hop', nodes=nodes,
        node_feats=values[nodes // 2], so_arr=nodes // 2 * 100, orient_arr=nodes % 2,
        pop_ids_arr=np.zeros(len(nodes)), src=np.array([0, 1, 2]), dst=np.array([1, 2, 3]),
        temps=np.ones(len(nodes)), labels=np.ones(1), query_u=np.array([0]), query_v=np.array([1]),
        train_idx=np.array([0]), val_idx=np.array([], int), test_idx=np.array([], int),
        edge_attr=None, branching_frac=0., n_pos=1, n_neg=0)


def paired():
    old = raw([0, 2, 4, 6])
    expanded = raw(np.arange(20))
    expanded.update(src=np.array([0, 2, 4, 6, 8]), dst=np.array([2, 4, 6, 8, 10]))
    expanded = pair_slice(old, expanded)
    old['mask_segment_ids'] = np.unique(old['nodes'] // 2)
    return old, expanded


def test_legacy_no_override_masks_and_target_statistics_are_bitwise_identical():
    examples = [raw(np.arange(20)), raw(np.arange(8, 28))]
    for example in examples:
        for seed in [42, 314159, 20260806]:
            for rate in [.1, .3, .7]:
                oids = torch.tensor(example['nodes'])
                assert torch.equal(reconstruction_mask(example, oids, rate, seed), segment_mask(oids, rate, seed))
    for actual, expected in zip(target_statistics(examples), segment_target_statistics(examples)):
        assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()


def test_matched_masks_hide_new_reverse_handle_without_changing_target_count():
    old, expanded = paired()
    for seed in [42, 51, 314159]:
        m1 = reconstruction_mask(old, torch.tensor(old['nodes']), .3, seed)
        m2 = reconstruction_mask(expanded, torch.tensor(expanded['nodes']), .3, seed)
        assert set(old['nodes'][m1] // 2) == set(expanded['nodes'][m2] // 2)
        assert len(set(expanded['nodes'][m2] // 2)) == 1
        assert torch.equal(m2[::2], m2[1::2])
        assert not m2[8:].any()
    for a, b in zip(target_statistics([old, old]), target_statistics([expanded])):
        np.testing.assert_array_equal(a, b)
    expanded['node_feats'][8:, 7:] = 100000
    for a, b in zip(target_statistics([old]), target_statistics([expanded])):
        np.testing.assert_array_equal(a, b)


def test_pair_rejects_removed_oriented_handle_edge_changed_nt_and_identity():
    old, expanded = paired()
    with pytest.raises(ValueError, match='handle'):
        pair_slice(old, dict(expanded, nodes=expanded['nodes'][1:]))
    with pytest.raises(ValueError, match='edge'):
        pair_slice(old, dict(expanded, src=np.array([0]), dst=np.array([2])))
    bad = dict(expanded, node_feats=expanded['node_feats'].copy())
    bad['node_feats'][0, 7] += 1
    with pytest.raises(ValueError, match='Frozen NT'):
        pair_slice(old, bad)
    with pytest.raises(ValueError, match='Unmatched'):
        pair_slice(old, dict(expanded, name='other'))


@pytest.mark.parametrize('context', CONTEXTS)
def test_loader_never_loads_test_or_changes_original_eligibility(monkeypatch, context):
    import training.component_pilot as module
    rows = pd.DataFrame([dict(name=n, target_sn='GRCh38#0#' + c, closure='1hop')
                         for n, c in [('train', 'chr3'), ('val', 'chr2'), ('test', 'chr1'), ('excluded', 'chr4')]])
    components = rows.assign(closure='component', segments_path='/unused', links_path='/unused')
    components.attrs['verified_records'] = {n: dict(segments={'sha256': 'x'}, links={'sha256': 'x'}) for n in rows.name}
    loaded = []
    def loader(row, _index, _metadata, _segments, args, audit=None):
        loaded.append((row['name'], args.extraction_mode))
        assert row['name'] != 'test'
        if audit is not None:
            audit['exclusion'] = 'retained' if row['name'] != 'excluded' else 'fewer_than_10_training_candidates'
        if row['name'] == 'excluded':
            return None
        old, expanded = paired()
        result = dict(expanded if args.extraction_mode else old, name=row['name'], target_sn=row.target_sn)
        result['nodes'] = result['nodes'] + (100 if row['name'] == 'val' else 0)
        return result
    monkeypatch.setattr(module, 'load_slice', loader)
    monkeypatch.setattr(module, 'verified_fingerprint', lambda *args: {})
    args = Namespace(test_chrs=['chr1'], val_chrs=['chr2'], extraction_mode=False)
    train, val, audits = load_matched_slices(rows, components, None, None, None, args, context)
    assert [r['name'] for r in train] == ['train'] and [r['name'] for r in val] == ['val']
    assert len(audits) == 3 and all(n != 'test' for n, _ in loaded)
    assert ('excluded', True) not in loaded


@pytest.mark.parametrize('frozen', [False, True])
def test_chunked_matched_objective_backprop_and_frozen_control(frozen):
    old, expanded = paired()
    _, mean, scale = target_statistics([old])
    args = Namespace(hidden_dim=12, n_heads=2, n_layers=1, dropout=0., stream_mode='full',
        coordinate_attention_mode='chunked_exact', attention_chunk_size=3, orientation_rope=True, pop_cond=False)
    torch.manual_seed(42)
    encoder = make_encoder(args)
    initial = encoder_parameter_digest(encoder)
    objective = MaskedFeatureObjective(encoder, 519, 12, 2, mean, scale, frozen)
    sd = tensorize_slice(expanded, torch.device('cpu'), args)
    mask = reconstruction_mask(expanded, sd['node_oids'], .3, 42)
    opt = torch.optim.AdamW([p for p in objective.parameters() if p.requires_grad], lr=.001)
    loss, count = objective(sd, mask)
    assert count == 1 and torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in objective.parameters() if p.grad is not None)
    opt.step()
    assert (encoder_parameter_digest(encoder) == initial) == frozen


def test_readiness_requires_all_jobs_and_real_cuda_profiles(tmp_path):
    p = plan()
    validate_plan(p)
    jobs = [dict(fold=f, seed=s) for f, s in product(['fold_a', 'fold_b', 'fold_c', 'fold_d', 'fold_e'], SEEDS)]
    paths = {}
    for role in ['replication', 'v1_reference', 'exact_gpu', 'window_gpu', 'preflight']:
        data = dict(status='complete', completed_jobs=jobs)
        if role == 'preflight':
            data.update(config={'sha256': 'config'}, biological_labels_used=False, cuda_used=False,
                optimizer_steps=0, test_windows_loaded=0, exact_target_moments=True,
                exact_target_populations=True, n_train_windows=1, n_validation_windows=1,
                n_verified_mask_samples=69)
        if role.endswith('gpu'):
            data.update(mode='chunked_exact' if role == 'exact_gpu' else 'chunked_window', device='cuda:0',
                chunk_size=512, all_gradients_finite=True, encoder_weights_unchanged=True,
                biological_labels_used=False, weight_updates=0, cuda_peak_allocated_bytes=100,
                n_handles=28287, context_audit={'sha256': p['component_pilot']['context_audit_sha256']})
        paths[role] = tmp_path/(role + '.json')
        paths[role].write_text(json.dumps(data))
    assert set(readiness_receipts(paths, p, config_sha256='config')) == set(paths)
    with pytest.raises(ValueError, match='preflight'):
        readiness_receipts(paths, p, config_sha256='changed-config')
    data = json.loads(paths['replication'].read_text())
    data['completed_jobs'] = jobs[:-1]
    paths['replication'].write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Incomplete'):
        readiness_receipts(paths, p, config_sha256='config')
    data['completed_jobs'] = jobs
    paths['replication'].write_text(json.dumps(data))
    data = json.loads(paths['exact_gpu'].read_text())
    data['device'] = 'cpu'
    paths['exact_gpu'].write_text(json.dumps(data))
    with pytest.raises(ValueError, match='memory/correctness'):
        readiness_receipts(paths, p, config_sha256='config')
    p['component_pilot']['coordinate_attention_mode'] = 'legacy'
    with pytest.raises(ValueError, match='Unsupported'):
        validate_plan(p)


def test_validation_probe_accepts_only_train_and_validation_and_records_populations():
    kwargs = dict(segids=np.arange(12), chromosomes=np.array(['chr3']*8 + ['chr2']*4),
        labels=np.array([0, 1]*6), features={'coordinate': np.arange(24).reshape(12, 2)},
        test_chrs={'chr1'}, val_chrs={'chr2'}, seed=42, validation_only=True, probe_max_iter=4000)
    metrics, _, predictions = evaluate_feature_sets(**kwargs)
    assert metrics.n_test.eq(0).all() and set(predictions.chromosome) == {'chr2'}
    assert metrics.train_targets_sha256.str.len().eq(64).all()
    with pytest.raises(ValueError, match='Empty downstream split'):
        evaluate_feature_sets(**dict(kwargs, validation_only=False))


def test_probe_command_is_explicitly_matched_and_validation_only():
    for context, task in product(CONTEXTS, ['ccre', 'sv']):
        command = probe_command(checkpoint=Path('/ckpt'), manifest=Path('/original'), graph=Path('/graph'),
            context_manifest=Path('/component/manifest.csv'), sw=Path('/sw'), nt=Path('/nt'),
            topology=Path('/h'), out=Path('/out'), seed=42, context=context, task=task,
            fold=dict(test=['chr1'], validation=['chr2']))
        assert '--exclude-test-extraction' in command and '--validation-only' in command
        assert command[command.index('--closure') + 1] == context
        assert ('--component-context-manifest' in command) == (context == 'component')
        assert '--save-probes' in command and command[command.index('--probe-max-iter')+1] == '4000'


def test_native_component_extraction_uses_original_occurrences_and_no_test_windows(tmp_path, monkeypatch):
    import tasks.ccre.embedding_baseline as extraction
    import training.component_pilot as policy
    import tasks.transfer.traitgym as provenance

    old, expanded = paired()
    originals = pd.DataFrame([dict(name='w', closure='1hop', target_sn='GRCh38#0#chr3'),
                              dict(name='test', closure='1hop', target_sn='GRCh38#0#chr1')])
    manifest = tmp_path/'original.csv'
    originals.to_csv(manifest, index=False)
    context_frame = originals.assign(closure='component', segments_path='/not-read', links_path='/not-read')
    context_frame.attrs['verified_records'] = {'w': dict(segments={'sha256': 's'}, links={'sha256': 'l'})}
    monkeypatch.setattr(policy, 'verified_context_manifest', lambda *a: context_frame)
    monkeypatch.setattr(provenance, 'verified_fingerprint', lambda *a: {})
    monkeypatch.setattr(extraction, 'read_segments_csv', lambda *a: pd.DataFrame())
    monkeypatch.setattr(extraction, 'build_global_index', lambda *a: (None, None))
    monkeypatch.setattr(extraction, 'build_oid_metadata_from_segments', lambda *a: {})
    args = Namespace(objective='masked_nt_features', node_extra_features='cache', stream_mode='full',
        adaptive_window=False, orientation_rope=True, pop_cond=False)
    monkeypatch.setattr(extraction, '_namespace_from_checkpoint', lambda *a, **k: Namespace(**vars(args)))
    class Encoder:
        def encode_nodes(self, x, *args):
            return x[:, :2]
    monkeypatch.setattr(extraction, '_build_model_from_checkpoint', lambda *a: (Encoder(), None))
    loaded = []
    def loader(row, _index, _md, _segments, args):
        assert row['name'] == 'w'
        loaded.append(args.objective)
        return expanded if row.closure == 'component' else old
    monkeypatch.setattr(extraction, 'load_slice', loader)
    checkpoint = tmp_path/'checkpoint.pt'
    def save_checkpoint(closure):
        torch.save({'plan': plan(), 'args': {'test_chrs': ['chr1']}, 'closure': closure}, checkpoint)
    save_checkpoint('1hop')
    kwargs = dict(checkpoint=checkpoint, manifest=manifest, full_segments=tmp_path/'unused',
        labeled_segids=set(range(10)), device_name='cpu', seed=42, target_chrs={'chr3'},
        extraction_candidate_policy='manuscript', return_canonical_audit=True)
    one, counts_one, _ = extraction._extract_embeddings(closure='1hop', **kwargs)
    save_checkpoint('component')
    wide, counts_wide, audit = extraction._extract_embeddings(closure='component',
        component_context_manifest=tmp_path/'manifest.csv', **kwargs)
    assert set(one) == set(wide) == {0, 1, 2, 3}
    assert counts_one == counts_wide == {0: 1, 1: 1, 2: 1, 3: 1}
    assert loaded == ['edge_masking', 'edge_masking', 'masked_nt_features']
    assert audit['extracted_chromosomes'] == ['chr3']
    assert audit['pooling_occurrences'] == 'original_one_hop_oriented_handles'
    for key in one:
        np.testing.assert_array_equal(one[key], wide[key])


@pytest.mark.parametrize('context', CONTEXTS)
def test_pilot_checkpoint_direct_entrypoints_cannot_score_or_extract_test(tmp_path, context):
    payload = dict(plan=plan(), closure=context, args=dict(test_chrs=['chr1'], val_chrs=['chr2'], seed=42))
    path = tmp_path/'pilot.pt'
    torch.save(payload, path)
    kwargs = dict(checkpoint=path, test_chrs={'chr1'}, closure=context, seed=42)
    for validation, exclude in [(False, False), (True, False), (False, True)]:
        with pytest.raises(ValueError, match='validation-only'):
            validate_checkpoint_holdout(**kwargs, validation_only=validation, exclude_test_extraction=exclude)
    validate_checkpoint_holdout(**kwargs, validation_only=True, exclude_test_extraction=True)
    with pytest.raises(ValueError, match='validation chromosomes'):
        validate_checkpoint_holdout(**kwargs, validation_only=True, exclude_test_extraction=True, val_chrs={'chr4'})
    for chromosomes in [None, {'chr1'}, {'chr1', 'chr3'}]:
        with pytest.raises(ValueError, match='non-test'):
            guard_extraction_scope(payload, chromosomes, context, 'manuscript', None)
    guard_extraction_scope(payload, {'chr2', 'chr3'}, context, 'manuscript', None)


def test_cpu_preflight_checks_every_declared_seed_epoch_and_validation_mask():
    old, expanded = paired()
    p = plan()
    result = verify_matched_banks([[old], [old]], [[expanded], [expanded]], p)
    assert result['n_verified_mask_samples'] == 3*(20+3)
    assert len(result['mask_bank_sha256']) == 64
    with pytest.raises(ValueError, match='populations'):
        verify_matched_banks([[old], [old]], [[], [expanded]], p)


def report_frame():
    rows = []
    for seed, context, arm, task, feature in product(SEEDS, CONTEXTS, ARMS, ['sv', 'ccre'], ['cs', 'csh', 'cst', 'csht']):
        value = .5 + .1*(context == 'component') + .05*arm.endswith('trained')
        rows.append(dict(seed=seed, context=context, model=arm, task=task, feature=feature, fold='fold_a',
            n_test=0, n_train=10, n_val=6, n_evaluated=6, positive_prevalence=.5,
            evaluation_partition='development_validation', targets_sha256='v', train_targets_sha256='t',
            validation_targets_sha256='v', scores_sha256='s',
            **{m: value for m in ['auprc', 'auroc', 'normalized_ap', 'balanced_accuracy', 'f1']}))
    return pd.DataFrame(rows)


def test_report_requires_exact_population_controls_and_complete_matrix():
    frame = report_frame()
    diff = compare(frame, plan())
    assert len(diff) == 3*2*2*5*11
    np.testing.assert_allclose(diff.loc[diff.comparison.eq('component minus 1hop / full_trained'), 'difference'], .1)
    with pytest.raises(ValueError, match='Incomplete'):
        compare(frame.iloc[:-1], plan())
    frame.loc[0, 'train_targets_sha256'] = 'different'
    with pytest.raises(ValueError, match='population'):
        compare(frame, plan())
    frame = report_frame()
    frame.loc[0, 'scores_sha256'] = 'different'
    with pytest.raises(ValueError, match='baseline'):
        compare(frame, plan())
