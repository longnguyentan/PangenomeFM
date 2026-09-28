#!/usr/bin/env python3
"""Plan complete alternative-component contexts on the exact manuscript graph.

Retain every original one-hop interval in QC, including unsafe or empty cases.
Optional materialization writes graph tables only, not link-prediction labels.
No biological models are fitted and no historical manifest is overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from evaluation.modality_factorial import load_frozen_node_embedding_cache
from graph.component_context import (
    REFERENCE_PREFIX, attention_storage, build_component_index,
    complete_context, fold_membership_audit,
)
from graph.neg_sampling import oriented_ids_from_links, slice_oriented_node_set
from graph.slicing import (
    build_incident_edge_index, induced_subgraph, segids_in_window_by_sn,
)
from tasks.entex.prepare import fingerprint
from tasks.transfer.traitgym import verified_fingerprint, write_json


def run(plan: dict, out: Path, materialize: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    record = dict(status='running', plan=plan, materialized=materialize,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        implementation=fingerprint(Path(__file__)), biological_labels_used=False)
    write_json(out/'status.json', record)
    try:
        record['sources'] = {name: verified_fingerprint(Path(plan[name]), plan[name+'_sha256'])
            for name in ['full_segments', 'full_links', 'manifest', 'sequence_cache', 'fold_config']}
        # Sequence strings are unnecessary for frozen NT inputs or context membership.
        segments = pd.read_csv(plan['full_segments'], usecols=lambda c: c != 'seq')
        links = pd.read_csv(plan['full_links'])
        idx = build_component_index(segments, links)
        if len(idx.nodes) != plan['expected_segments']:
            raise ValueError('Unexpected canonical graph size')
        rr = idx.reference[idx.source] & idx.reference[idx.target]
        sn = idx.nodes.SN.astype(str).to_numpy()
        cross = rr & (sn[idx.source] != sn[idx.target])
        record['cross_reference_contig_links'] = int(cross.sum())
        if cross.any():
            raise ValueError('Cross-reference-contig links need an explicit split policy')
        ids, values, cache_audit = load_frozen_node_embedding_cache(Path(plan['sequence_cache']))
        if not np.array_equal(np.sort(ids), np.arange(len(idx.nodes))):
            raise ValueError('Full canonical NT coverage required')
        if values.shape != (len(ids), 512) or not np.isfinite(values).all():
            raise ValueError('Malformed frozen NT cache')
        if (cache_audit.get('full_segments_sha256') != plan['full_segments_sha256']
                or cache_audit.get('output_sha256') != plan['sequence_cache_sha256']
                or cache_audit.get('model_parameters_frozen') is not True):
            raise ValueError('Frozen cache provenance differs from graph or artifact')
        del values
        manifest = pd.read_csv(plan['manifest'])
        rows = manifest.loc[manifest.closure.eq('1hop')].copy()
        if rows.empty or rows.duplicated(['target_sn','start','end']).any() or rows.name.duplicated().any():
            raise ValueError('Original one-hop interval universe must be nonempty and unique')
        incident_ptr, incident_edges = build_incident_edge_index(idx.source, idx.target, len(idx.nodes))
        all_ids, chroms, summaries = [], [], []
        pointer = [0]
        included_components: set[int] = set()
        for row in rows.itertuples(index=False):
            name = str(row.name)
            if Path(name).name != name or name in {'.','..'} or row.start >= row.end:
                raise ValueError('Unsafe window name or malformed interval')
            core = segids_in_window_by_sn(sn, idx.nodes.SO.to_numpy(), idx.nodes.LN.to_numpy(),
                row.target_sn, int(row.start), int(row.end))
            info = dict(name=name, target_sn=row.target_sn, chrom=str(row.target_sn).removeprefix(REFERENCE_PREFIX),
                start=int(row.start), end=int(row.end), n_core_segments=len(core), status='pending')
            try:
                chosen, components = complete_context(idx, core)
                # Preserve every old one-hop reference flank: the new context is
                # an exact node/edge superset, with complete attached alternatives.
                _, _, old_ids = induced_subgraph(idx.nodes, links, idx.names, core, True,
                    from_id=idx.source, to_id=idx.target, incident_indptr=incident_ptr,
                    incident_edge_ids=incident_edges, segments_aligned_to_index=True)
                chosen = np.union1d(chosen, old_ids)
                sub, edge, selected = induced_subgraph(idx.nodes, links, idx.names, chosen, False,
                    from_id=idx.source, to_id=idx.target, incident_indptr=incident_ptr,
                    incident_edge_ids=incident_edges, segments_aligned_to_index=True)
                if not np.array_equal(selected, chosen):
                    raise ValueError('Induced graph lost selected segments')
                u, v = oriented_ids_from_links(edge, idx.names)
                handles = slice_oriented_node_set(u, v)
                represented = np.unique(handles//2)
                missing = np.setdiff1d(chosen, represented)
                core_missing = np.intersect1d(core, missing)
                memory = attention_storage(len(handles), heads=plan['heads'],
                    scalar_bytes=plan['scalar_bytes'], window_k=plan['sparse_window_k']) if len(handles) else dict(dense_score_bytes=0, sparse_score_bytes_upper_bound=0)
                old_components = set(idx.labels[old_ids[~idx.reference[old_ids]]].tolist())
                partial = sum(not np.isin(idx.members[c], old_ids).all() for c in old_components)
                info.update(status='complete', n_components=len(components), n_segments=len(chosen),
                    n_alternative_segments=int((~idx.reference[chosen]).sum()), n_links=len(edge),
                    n_oriented_handles=len(handles), n_isolated_segments=len(missing),
                    n_isolated_core_segments=len(core_missing), original_one_hop_segments=len(old_ids),
                    original_one_hop_partial_components=int(partial),
                    added_segments_over_one_hop=len(np.setdiff1d(chosen, old_ids)),
                    removed_reference_flanks_from_one_hop=len(np.setdiff1d(old_ids, chosen)),
                    nt_coverage=1., dense_score_alone_exceeds_gpu=memory.get('dense_score_bytes', 0) > plan['gpu_bytes'],
                    **memory)
                all_ids.append(chosen)
                chroms.append(info['chrom'])
                pointer.append(pointer[-1]+len(chosen))
                included_components.update(map(int, components))
                if materialize:
                    stem = out/'slices'/name
                    stem.parent.mkdir(exist_ok=True)
                    sub.to_csv(str(stem)+'_segments.csv.gz', index=False)
                    edge.to_csv(str(stem)+'_links.csv.gz', index=False)
                    info.update(segments_path=str(stem.resolve())+'_segments.csv.gz',
                        links_path=str(stem.resolve())+'_links.csv.gz', closure='component')
            except ValueError as exc:
                info.update(status='failed', error=str(exc))
            summaries.append(info)
        frame = pd.DataFrame(summaries)
        frame.to_csv(out/'windows.csv', index=False)
        if not frame.status.eq('complete').all():
            raise ValueError('Unsafe/empty contexts retained in windows.csv; no training manifest issued')
        folds = json.loads(Path(plan['fold_config']).read_text())['rotating_chromosome_folds']
        fold_membership_audit(all_ids, chroms, folds, len(idx.nodes)).to_csv(out/'fold_overlap.csv', index=False)
        np.savez_compressed(out/'context_segment_ids.npz', indptr=np.asarray(pointer, np.int64),
            segment_ids=np.concatenate(all_ids), names=frame.name.to_numpy(str))
        if materialize:
            frame.to_csv(out/'manifest.csv', index=False)
        coverage = np.unique(np.concatenate(all_ids))
        record.update(status='complete', n_windows=len(frame), failed_windows=0,
            n_unique_context_segments=len(coverage), n_unique_alternative_segments=int((~idx.reference[coverage]).sum()),
            n_components_covered=len(included_components), n_components_total=len(idx.members),
            n_dense_score_over_gpu=int(frame.dense_score_alone_exceeds_gpu.sum()),
            n_windows_with_incomplete_original_components=int(frame.original_one_hop_partial_components.gt(0).sum()),
            n_windows_with_isolated_core=int(frame.n_isolated_core_segments.gt(0).sum()),
            context_size_quantiles=frame.n_segments.quantile([0,.5,.9,.95,.99,1]).to_dict(),
            dense_score_bytes_quantiles=frame.dense_score_bytes.quantile([0,.5,.9,.95,.99,1]).to_dict(),
            training_ready=False,
            limitations=['Undirected alternative components are not directed bubbles or phased haplotypes.',
              'Expansion begins only at original core nodes; added reference anchors do not recursively expand.',
              'All windows retained; memory flags do not authorize excluding complex windows.',
              'Score-matrix bytes are a lower bound on dense model memory, not peak GPU profiling.',
              'Sparse attention semantics require repair/verification before wider-context training.',
              'Materialized tables are compatible graph inputs, not a trained model or performance result.',
              'All original one-hop reference flanks retained; newly added anchors do not recursively trigger more components.'])
        write_json(out/'audit.json', record)
    except Exception as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(out/'status.json', record)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--materialize', action='store_true')
    args = ap.parse_args()
    run(json.loads(args.config.read_text()), args.out_dir, args.materialize)


if __name__ == '__main__':
    main()
