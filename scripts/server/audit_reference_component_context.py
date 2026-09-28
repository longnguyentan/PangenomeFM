#!/usr/bin/env python3
"""Audit alternative-component coverage without treating components as bubbles."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from graph.slicing import build_global_index, map_links_to_segids
from tasks.entex.prepare import fingerprint
from tasks.transfer.traitgym import write_json

PRIMARY = {f'chr{i}' for i in range(1, 23)} | {'chrX', 'chrY'}


def component_context(segments: pd.DataFrame, links: pd.DataFrame, cached_ids: np.ndarray) -> pd.DataFrame:
    index, nodes = build_global_index(segments)
    if len(nodes) != len(segments):
        raise ValueError('Canonical segment names must be unique')
    if len(np.unique(cached_ids)) != len(cached_ids) or (cached_ids < 0).any() or (cached_ids >= len(nodes)).any():
        raise ValueError('Cache IDs must be unique canonical rows')
    ref = nodes.SN.astype(str).str.startswith('GRCh38#0#').to_numpy()
    if not ref.any() or ref.all():
        raise ValueError('Require both GRCh38 reference and alternative segments')
    source, target = map_links_to_segids(links, index)
    alt_edges = ~ref[source] & ~ref[target]
    adjacency = coo_matrix((np.ones(alt_edges.sum(), dtype=np.uint8),
        (source[alt_edges], target[alt_edges])), shape=(len(nodes), len(nodes))).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    present = np.zeros(len(nodes), dtype=bool)
    present[cached_ids] = True
    alts = np.flatnonzero(~ref)
    frame = pd.DataFrame(dict(component=labels[alts], segment_id=alts,
        bases=nodes.LN.to_numpy()[alts], in_cache=present[alts]))
    result = frame.groupby('component').agg(n_alternative_segments=('segment_id', 'size'),
        alternative_bases=('bases', 'sum'), n_cached=('in_cache', 'sum'))
    crossing = ref[source] != ref[target]
    a = np.where(ref[source[crossing]], target[crossing], source[crossing])
    r = np.where(ref[source[crossing]], source[crossing], target[crossing])
    boundary = pd.DataFrame(dict(component=labels[a], reference_id=r)).drop_duplicates()
    if len(boundary):
        boundary['reference_sn'] = nodes.SN.to_numpy()[boundary.reference_id]
        boundary['chromosome'] = boundary.reference_sn.str.removeprefix('GRCh38#0#')
        boundary['start'] = nodes.SO.to_numpy()[boundary.reference_id]
        boundary['end'] = boundary.start + nodes.LN.to_numpy()[boundary.reference_id]
        metadata = boundary.groupby('component').agg(n_reference_anchors=('reference_id', 'size'),
            n_reference_contigs=('reference_sn', 'nunique'),
            reference_contigs=('chromosome', lambda s: '|'.join(sorted(set(s)))),
            minimum_anchor_start=('start', 'min'), maximum_anchor_end=('end', 'max'))
        result = result.join(metadata)
    else:
        for col in ['n_reference_anchors', 'n_reference_contigs', 'minimum_anchor_start', 'maximum_anchor_end']:
            result[col] = np.nan
        result['reference_contigs'] = ''
    for col in ['n_reference_anchors', 'n_reference_contigs']:
        result[col] = result[col].fillna(0).astype(int)
    result['reference_contigs'] = result.reference_contigs.fillna('')
    result['status'] = np.select([result.n_reference_anchors.eq(0), result.n_reference_contigs.gt(1),
        ~result.reference_contigs.isin(PRIMARY), result.n_reference_anchors.eq(1)],
        ['unanchored', 'multiple_reference_contigs', 'nonprimary_reference_contig', 'single_reference_anchor'],
        default='multiple_anchors_one_primary_chromosome')
    result['anchor_span_bp'] = (result.maximum_anchor_end-result.minimum_anchor_start).where(result.n_reference_contigs.eq(1))
    result['coverage'] = np.select([result.n_cached.eq(0), result.n_cached.eq(result.n_alternative_segments)],
                                   ['absent', 'complete'], default='partial')
    return result.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--full-segments', type=Path, required=True)
    ap.add_argument('--full-links', type=Path, required=True)
    ap.add_argument('--sequence-cache', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    segments = pd.read_csv(args.full_segments, usecols=['name', 'SN', 'SO', 'LN'])
    links = pd.read_csv(args.full_links, usecols=['from_seg', 'to_seg'])
    with np.load(args.sequence_cache, allow_pickle=False) as cache:
        cached = cache['segid']
    frame = component_context(segments, links, cached)
    frame.to_parquet(args.out_dir/'alternative_components.parquet', index=False)
    summary = frame.groupby(['status', 'coverage']).agg(components=('component', 'size'),
        alternative_segments=('n_alternative_segments', 'sum'), cached_alternative_segments=('n_cached', 'sum'),
        alternative_bases=('alternative_bases', 'sum')).reset_index()
    summary.to_csv(args.out_dir/'coverage.csv', index=False)
    write_json(args.out_dir/'audit.json', dict(status='complete',
        graph=fingerprint(args.full_segments), links=fingerprint(args.full_links), cache=fingerprint(args.sequence_cache),
        n_segments=len(segments), n_reference_segments=int(segments.SN.astype(str).str.startswith('GRCh38#0#').sum()),
        n_alternative_components=len(frame), n_alternative_segments=int(frame.n_alternative_segments.sum()),
        component_size_quantiles=frame.n_alternative_segments.quantile([0, .5, .9, .95, .99, 1]).to_dict(),
        n_partial_components=int(frame.coverage.eq('partial').sum()),
        scope='Undirected connected components after removing reference nodes; no biological labels; no graph/checkpoint changes.',
        limitations=['Connectivity is not directed reconvergence, a bubble call or a phased allele.',
          'Cache membership audits the benchmark-union representation scope, not each individual window.',
          'Single-anchor, unanchored, multichromosome and nonprimary cases require separate treatment before fold-safe context construction.',
          'Even two reference anchors do not prove a single simple or oriented bubble.']))


if __name__ == '__main__':
    main()
