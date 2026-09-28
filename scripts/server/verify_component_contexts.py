#!/usr/bin/env python3
"""Replay materialized graph tables against original canonical membership/edges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from graph.slicing import build_global_index, map_links_to_segids
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
    names,nodes=build_global_index(original)
    links=pd.read_csv(plan['full_links'])
    source,target=map_links_to_segids(links,names)
    frame=pd.read_csv(root/'manifest.csv')
    with np.load(root/'context_segment_ids.npz',allow_pickle=False) as z:
        pointer,ids,windows=z['indptr'],z['segment_ids'],z['names']
    if (len(pointer)!=len(frame)+1 or pointer[0]!=0 or pointer[-1]!=len(ids)
            or not (np.diff(pointer)>0).all() or not np.array_equal(windows,frame.name.to_numpy(str))):
        raise ValueError('Context membership is misaligned with manifest')
    records=[]
    for i,row in enumerate(frame.itertuples(index=False)):
        chosen=ids[pointer[i]:pointer[i+1]]
        if not np.array_equal(chosen,np.unique(chosen)) or (chosen<0).any() or (chosen>=len(nodes)).any():
            raise ValueError('Invalid serialized canonical membership')
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
        records.append(dict(name=row.name,segments=fingerprint(Path(row.segments_path)),
            links=fingerprint(Path(row.links_path)),n_segments=len(chosen),n_links=len(actual_edges)))
    return dict(status='complete',context_audit=fingerprint(root/'audit.json'),
        manifest=fingerprint(root/'manifest.csv'),membership=fingerprint(root/'context_segment_ids.npz'),
        fold_overlap=fingerprint(root/'fold_overlap.csv'),n_verified_windows=len(records),
        exact_segment_metadata=True,exact_oriented_induced_edges=True,
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
