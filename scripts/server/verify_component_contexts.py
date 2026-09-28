#!/usr/bin/env python3
"""Replay materialized graph tables against original canonical membership/edges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from graph.component_context import (
    REFERENCE_PREFIX, attention_storage, build_component_index, complete_context,
    fold_membership_audit,
)
from graph.neg_sampling import oriented_ids_from_links, slice_oriented_node_set
from graph.slicing import (
    build_incident_edge_index, induced_subgraph, segids_in_window_by_sn,
)
from tasks.entex.prepare import fingerprint
from tasks.transfer.traitgym import verified_fingerprint, write_json


def verify(root: Path) -> dict:
    audit=json.loads((root/'audit.json').read_text())
    if audit['status']!='complete' or not audit['materialized']:
        raise ValueError('Require a complete materialized context set')
    plan=audit['plan']
    for key in ['full_segments','full_links','manifest','fold_config']:
        verified_fingerprint(Path(plan[key]),plan[key+'_sha256'])
    original=pd.read_csv(plan['full_segments'],usecols=lambda c:c!='seq')
    links=pd.read_csv(plan['full_links'])
    index=build_component_index(original,links)
    names,nodes,source,target=index.names,index.nodes,index.source,index.target
    if len(nodes)!=plan['expected_segments']:
        raise ValueError('Canonical graph size differs from plan')
    sn=nodes.SN.astype(str).to_numpy()
    rr=index.reference[source]&index.reference[target]
    if (rr & (sn[source]!=sn[target])).any():
        raise ValueError('Cross-reference-contig links need an explicit split policy')
    original_manifest=pd.read_csv(plan['manifest'])
    expected_windows=original_manifest.loc[original_manifest.closure.eq('1hop')].reset_index(drop=True)
    if (expected_windows.empty or expected_windows.name.duplicated().any()
            or expected_windows.duplicated(['target_sn','start','end']).any()):
        raise ValueError('Original one-hop interval universe must be nonempty and unique')
    frame=pd.read_csv(root/'manifest.csv')
    identity=['name','target_sn','start','end']
    try:
        pd.testing.assert_frame_equal(frame[identity],expected_windows[identity],check_dtype=False)
    except (AssertionError,KeyError) as exc:
        raise ValueError('Materialized interval universe differs from original manifest') from exc
    if not frame.status.eq('complete').all() or not frame.closure.eq('component').all():
        raise ValueError('Every original interval must have a complete component context')
    with np.load(root/'context_segment_ids.npz',allow_pickle=False) as z:
        pointer,ids,windows=z['indptr'],z['segment_ids'],z['names']
    if (pointer.ndim!=1 or ids.ndim!=1 or pointer.dtype.kind not in 'iu' or ids.dtype.kind not in 'iu'
            or len(pointer)!=len(frame)+1 or pointer[0]!=0 or pointer[-1]!=len(ids)
            or not (np.diff(pointer)>0).all() or not np.array_equal(windows,frame.name.to_numpy(str))):
        raise ValueError('Context membership is misaligned with manifest')
    incident_ptr,incident_edges=build_incident_edge_index(source,target,len(nodes))
    records=[]
    expected_memberships=[]
    chromosomes=[]
    for i,row in enumerate(frame.itertuples(index=False)):
        chosen=ids[pointer[i]:pointer[i+1]]
        if not np.array_equal(chosen,np.unique(chosen)) or (chosen<0).any() or (chosen>=len(nodes)).any():
            raise ValueError('Invalid serialized canonical membership')
        core=segids_in_window_by_sn(sn,nodes.SO.to_numpy(),nodes.LN.to_numpy(),
            row.target_sn,int(row.start),int(row.end))
        complete,components=complete_context(index,core)
        _,_,old_ids=induced_subgraph(nodes,links,names,core,True,
            from_id=source,to_id=target,incident_indptr=incident_ptr,
            incident_edge_ids=incident_edges,segments_aligned_to_index=True)
        expected_membership=np.union1d(complete,old_ids)
        if not np.array_equal(chosen,expected_membership):
            raise ValueError(f'Context membership is not complete for original interval {row.name}')
        expected_memberships.append(expected_membership)
        chromosomes.append(str(row.target_sn).removeprefix(REFERENCE_PREFIX))
        keep=np.zeros(len(nodes),bool)
        keep[chosen]=True
        expected_nodes=nodes.iloc[chosen].reset_index(drop=True)
        actual_nodes=pd.read_csv(row.segments_path)
        actual_nodes['name']=actual_nodes.name.astype('string')
        expected_edges=links.loc[keep[source]&keep[target]].reset_index(drop=True)
        actual_edges=pd.read_csv(row.links_path)
        try:
            pd.testing.assert_frame_equal(expected_nodes,actual_nodes,check_dtype=False)
            pd.testing.assert_frame_equal(expected_edges,actual_edges,check_dtype=False)
        except AssertionError as exc:
            raise ValueError(f'Materialized graph differs from original at {row.name}') from exc
        u,v=oriented_ids_from_links(expected_edges,names)
        handles=slice_oriented_node_set(u,v)
        missing=np.setdiff1d(chosen,np.unique(handles//2))
        old_components=set(index.labels[old_ids[~index.reference[old_ids]]].tolist())
        memory=attention_storage(len(handles),heads=plan['heads'],scalar_bytes=plan['scalar_bytes'],
            window_k=plan['sparse_window_k']) if len(handles) else dict(dense_score_bytes=0,sparse_score_bytes_upper_bound=0)
        expected_counts=dict(n_core_segments=len(core),n_components=len(components),n_segments=len(chosen),
            n_alternative_segments=int((~index.reference[chosen]).sum()),n_links=len(expected_edges),
            n_oriented_handles=len(handles),n_isolated_segments=len(missing),
            n_isolated_core_segments=len(np.intersect1d(core,missing)),original_one_hop_segments=len(old_ids),
            original_one_hop_partial_components=sum(not np.isin(index.members[c],old_ids).all() for c in old_components),
            added_segments_over_one_hop=len(np.setdiff1d(chosen,old_ids)),
            removed_reference_flanks_from_one_hop=len(np.setdiff1d(old_ids,chosen)),
            dense_score_alone_exceeds_gpu=memory['dense_score_bytes']>plan['gpu_bytes'],**memory)
        if row.chrom!=chromosomes[-1] or any(getattr(row,key,None)!=value for key,value in expected_counts.items()):
            raise ValueError(f'Context summary counts differ from reconstructed graph at {row.name}')
        records.append(dict(name=row.name,segments=fingerprint(Path(row.segments_path)),
            links=fingerprint(Path(row.links_path)),n_segments=len(chosen),n_links=len(actual_edges)))
    folds=json.loads(Path(plan['fold_config']).read_text())['rotating_chromosome_folds']
    expected_overlap=fold_membership_audit(expected_memberships,chromosomes,folds,len(nodes))
    try:
        pd.testing.assert_frame_equal(expected_overlap,pd.read_csv(root/'fold_overlap.csv'),check_dtype=False)
        pd.testing.assert_frame_equal(frame,pd.read_csv(root/'windows.csv'),check_dtype=False)
    except AssertionError as exc:
        raise ValueError('Context split or window evidence differs from independent replay') from exc
    if audit['n_windows']!=len(expected_windows) or audit['failed_windows']!=0:
        raise ValueError('Context audit window counts differ from original interval universe')
    return dict(status='complete',context_audit=fingerprint(root/'audit.json'),
        manifest=fingerprint(root/'manifest.csv'),membership=fingerprint(root/'context_segment_ids.npz'),
        fold_overlap=fingerprint(root/'fold_overlap.csv'),n_verified_windows=len(records),
        exact_segment_metadata=True,exact_oriented_induced_edges=True,
        complete_membership_recomputed=True,original_interval_universe_verified=True,
        summary_counts_recomputed=True,fold_membership_recomputed=True,
        biological_labels_used=False,records=records,implementation=fingerprint(Path(__file__)))


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--contexts',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Refusing to overwrite verification receipt')
    result=verify(args.contexts)
    write_json(args.out,result)


if __name__=='__main__':
    main()
