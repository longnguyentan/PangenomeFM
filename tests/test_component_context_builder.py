import numpy as np
import pandas as pd
import pytest

from graph.component_context import (
    attention_storage, build_component_index, complete_context, fold_membership_audit,
)
from graph.slicing import induced_subgraph


def fixture():
    # r0--a--b--r1 and r1--c--r2. Adding r1 must not recursively add c/r2.
    nodes = pd.DataFrame(dict(name=['r0','r1','r2','a','b','c','orphan'],
        SN=['GRCh38#0#chr1']*3+['sample#1#chr1']*4,
        SO=[0,100,200,0,0,0,0], LN=[10]*7))
    links = pd.DataFrame([('r0','+','a','-'),('a','-','b','+'),('b','+','r1','-'),
        ('r1','+','c','+'),('c','-','r2','+')],
        columns=['from_seg','from_orient','to_seg','to_orient'])
    return nodes, links


def test_complete_component_retains_far_nodes_all_anchors_and_orientations():
    nodes, links = fixture()
    index = build_component_index(nodes, links)
    selected, labels = complete_context(index, np.array([0]))
    assert selected.tolist() == [0,1,3,4] and len(labels) == 1
    assert len(index.members) == 3  # retain the unanchored alternative in the denominator
    sub, edges, actual = induced_subgraph(index.nodes, links, index.names, selected, False,
        from_id=index.source, to_id=index.target, segments_aligned_to_index=True)
    pd.testing.assert_frame_equal(edges, links.iloc[:3].reset_index(drop=True))
    assert sub.name.tolist() == ['r0','r1','a','b']
    np.testing.assert_array_equal(actual, selected)
    selected2, labels2 = complete_context(index, np.array([0,1]))
    assert selected2.tolist() == [0,1,2,3,4,5] and len(labels2) == 2


def test_cross_chromosome_and_malformed_inputs_fail():
    nodes, links = fixture()
    nodes.loc[1, 'SN'] = 'GRCh38#0#chr2'
    with pytest.raises(ValueError, match='crosses'):
        complete_context(build_component_index(nodes, links), np.array([0]))
    nodes, links = fixture()
    for core in [np.array([], dtype=int), np.array([0,0]), np.array([-1]), np.array([1.2])]:
        with pytest.raises(ValueError, match='Core must'):
            complete_context(build_component_index(nodes, links), core)
    with pytest.raises(ValueError, match='reference segments'):
        complete_context(build_component_index(nodes, links), np.array([3]))
    links.loc[0, 'from_orient'] = '?'
    with pytest.raises(ValueError, match='orientation'):
        build_component_index(nodes, links)
    nodes.loc[0, 'SO'] = -1
    with pytest.raises(ValueError, match='coordinates'):
        build_component_index(nodes, links)


def test_reference_isolate_is_kept_and_fold_ids_are_actually_disjoint():
    nodes, links = fixture()
    nodes.loc[2, 'SN'] = 'GRCh38#0#chr2'
    index = build_component_index(nodes, links.iloc[:3])
    selected, components = complete_context(index, np.array([2]))
    assert selected.tolist() == [2] and not len(components)
    folds = [dict(name='a', validation=['chr2'], test=['chr3'])]
    frame = fold_membership_audit([np.array([0,3,4]), np.array([2]), np.array([6])],
        ['chr1','chr2','chr3'], folds, 7)
    assert frame.iloc[0].train_segments == 3 and frame.iloc[0].train_test_overlap == 0
    with pytest.raises(ValueError, match='leakage'):
        fold_membership_audit([np.array([0,3]),np.array([3,2])], ['chr1','chr2'], folds, 7)


def test_attention_estimates_are_bounds_not_total_memory_or_exclusion_rules():
    r = attention_storage(50000)
    assert r['dense_score_bytes'] == 40_000_000_000
    assert r['sparse_score_bytes_upper_bound'] == 4*50000*129*4
    assert attention_storage(10)['dense_score_bytes'] == attention_storage(10)['sparse_score_bytes_upper_bound']
    with pytest.raises(ValueError):
        attention_storage(0)


def test_preparation_materializes_oriented_native_tables_and_keeps_failed_qc(tmp_path):
    import hashlib
    from pathlib import Path
    import json
    from scripts.server.prepare_component_contexts import run

    nodes, links = fixture()
    sources = tmp_path/'inputs'
    sources.mkdir()
    nodes.to_csv(sources/'segments.csv', index=False)
    links.to_csv(sources/'links.csv', index=False)
    pd.DataFrame([dict(name='w1',closure='1hop',target_sn='GRCh38#0#chr1',start=0,end=10)]).to_csv(sources/'manifest.csv', index=False)
    np.savez(sources/'nt.npz', segid=np.arange(len(nodes)), embeddings=np.ones((len(nodes),512),np.float32))
    (sources/'folds.json').write_text(json.dumps(dict(rotating_chromosome_folds=[dict(name='a',validation=['chr2'],test=['chr3'])])))
    plan = dict(expected_segments=7,heads=4,scalar_bytes=4,sparse_window_k=128,gpu_bytes=24*2**30)
    for field, file in [('full_segments','segments.csv'),('full_links','links.csv'),('manifest','manifest.csv'),('sequence_cache','nt.npz'),('fold_config','folds.json')]:
        plan[field] = str(sources/file)
        plan[field+'_sha256'] = hashlib.sha256((sources/file).read_bytes()).hexdigest()
    (sources/'nt.npz.audit.json').write_text(json.dumps(dict(status='complete',downstream_label_access='none',
        full_segments_sha256=plan['full_segments_sha256'],output_sha256=plan['sequence_cache_sha256'],model_parameters_frozen=True)))
    result = run(plan, tmp_path/'out', materialize=True)
    assert result['status']=='complete' and result['training_ready'] is False
    from scripts.server.verify_component_contexts import verify
    verification = verify(tmp_path/'out')
    assert verification['n_verified_windows']==1
    frame = pd.read_csv(tmp_path/'out'/'manifest.csv')
    assert frame.n_segments.tolist()==[4] and frame.original_one_hop_partial_components.tolist()==[1]
    assert frame.removed_reference_flanks_from_one_hop.tolist()==[0]
    pd.testing.assert_frame_equal(pd.read_csv(frame.links_path.iloc[0]), links.iloc[:3].reset_index(drop=True))
    with np.load(tmp_path/'out'/'context_segment_ids.npz') as ids:
        assert ids['segment_ids'].tolist()==[0,1,3,4]
    edge_path=Path(frame.links_path.iloc[0])
    broken=pd.read_csv(edge_path)
    broken.loc[0,'from_orient']='-'
    broken.to_csv(edge_path,index=False)
    with pytest.raises(ValueError,match='differs from original'):
        verify(tmp_path/'out')
    # An invalid core produces a retained failure row and no usable training manifest.
    pd.DataFrame([dict(name='empty',closure='1hop',target_sn='GRCh38#0#chr2',start=0,end=10)]).to_csv(sources/'manifest.csv',index=False)
    plan['manifest_sha256']=hashlib.sha256((sources/'manifest.csv').read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='Unsafe/empty'):
        run(plan, tmp_path/'failed', materialize=True)
    assert pd.read_csv(tmp_path/'failed'/'windows.csv').status.tolist()==['failed']
    assert not (tmp_path/'failed'/'manifest.csv').exists()
