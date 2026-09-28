#!/usr/bin/env python3
"""Append missing reference vectors from original small structural windows.

Historical caches excluded windows with too few reconstruction candidates.
Inference does not need candidate labels. Existing vectors are never recomputed.
This explicit coverage extension is saved separately from its immutable parent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.external import _build_model_from_checkpoint, _namespace_from_checkpoint
from evaluation.modality_factorial import load_frozen_node_embedding_cache
from graph.features import build_oid_metadata_from_segments
from graph.io import read_segments_csv
from graph.slicing import build_global_index
from scripts.server.run_ccre_frozen_probe_matrix import build_jobs, checkpoint_for
from tasks.entex.prepare import fingerprint
from tasks.transfer.traitgym import verified_fingerprint, write_json
from training.pretrain import _compute_adaptive_window_k, load_slice, tensorize_slice


def append_only(ids, values, additions: dict[int, np.ndarray]):
    if len(np.unique(ids)) != len(ids) or set(ids) & additions.keys():
        raise ValueError('Append-only extension cannot replace any original segment')
    extra = np.array(sorted(additions), dtype=ids.dtype)
    if not len(extra):
        raise ValueError('No missing segments to append')
    extra_values = np.stack([additions[i] for i in extra]).astype(values.dtype)
    if extra_values.shape[1:] != values.shape[1:] or not np.isfinite(extra_values).all():
        raise ValueError('Invalid appended vectors')
    return np.concatenate([ids, extra]), np.concatenate([values, extra_values])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['full-segments', 'manifest', 'cache-root', 'results-root', 'target-cache', 'out-root']:
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(Path('configs/entex_v1.json').read_text())
    graph = verified_fingerprint(args.full_segments, config['full_segments_sha256'])
    manifest_source = fingerprint(args.manifest)
    manifest = pd.read_csv(args.manifest)
    segments = read_segments_csv(args.full_segments)
    index, _ = build_global_index(segments)
    md = build_oid_metadata_from_segments(segments, index)
    with np.load(args.target_cache) as z:
        required = z['segid'].astype(int)
    jobs = build_jobs(json.loads(Path(config['manuscript_config']).read_text()))
    args.out_root.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='running', completed_runs=0, planned_runs=len(jobs),
        graph=graph, manifest=manifest_source, target_cache=fingerprint(args.target_cache),
        implementation=fingerprint(Path(__file__)), no_biological_labels=True,
        existing_vector_policy='Bitwise unchanged; only missing reference segments appended', jobs=[])
    write_json(args.out_root/'status.json', receipt)
    try:
        for job in jobs:
            relative = Path(job.fold)/f'seed_{job.seed}'/(job.closure+'.npz')
            source = args.cache_root/relative
            ids, values, audit = load_frozen_node_embedding_cache(source)
            parent = verified_fingerprint(source, audit['output_sha256'])
            checkpoint = checkpoint_for(args.results_root, job)
            expected = dict(checkpoint_sha256=fingerprint(checkpoint)['sha256'], graph_sha256=graph['sha256'],
                manifest_sha256=manifest_source['sha256'], seed=job.seed, closure=job.closure, canonical_conflict_policy='exclude')
            if audit['identity'] != expected:
                raise ValueError('Original cache resource identity differs')
            missing = sorted(map(int, set(required)-set(ids)))
            if not missing:
                raise ValueError('Expected a coverage gap; do not silently rewrite complete caches')
            # Small-window recovery is deliberately restricted to reference targets.
            target = segments.iloc[missing]
            if not target.SN.str.startswith('GRCh38#0#').all():
                raise ValueError('Only reference-anchored cache completion is supported')
            candidate = manifest.loc[manifest.closure.eq(job.closure)]
            select = np.zeros(len(candidate), dtype=bool)
            for row in target.itertuples():
                select |= (candidate.target_sn.eq(row.SN) & candidate.start.lt(row.SO+row.LN) & candidate.end.gt(row.SO)).to_numpy()
            selected = candidate.loc[select]
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            options = _namespace_from_checkpoint(payload, job.seed)
            model, _ = _build_model_from_checkpoint(payload, options, torch.device('cpu'))
            options.extraction_mode = True
            # Reuse the existing label-free structural loader path, retaining the
            # checkpoint's feature policy/architecture. No junction training occurs.
            options.objective = 'junction_repair'
            options.canonical_conflict_policy = 'exclude'
            sums, counts, used = {}, {}, []
            for _, row in selected.iterrows():
                raw = load_slice(row, index, md, segments, options)
                if raw is None:
                    continue
                # This recovery must not change the context of previously eligible windows.
                sd = tensorize_slice(raw, torch.device('cpu'), options)
                if options.adaptive_window:
                    window = _compute_adaptive_window_k(raw['branching_frac'],
                        options.adaptive_window_base, options.adaptive_window_alpha)
                    for layer in model.linear_layers:
                        layer.window_k = window
                with torch.no_grad():
                    h = model.encode_nodes(sd['X'], sd['so'], sd['src'], sd['dst'], sd['temps'],
                        sd['edge_attr'], sd['orient'], sd['pop_ids']).numpy()
                covered = []
                for oid, vector in zip(raw['nodes'], h):
                    segid = int(oid)//2
                    if segid in missing:
                        sums[segid] = sums.get(segid, np.zeros_like(vector))+vector
                        counts[segid] = counts.get(segid, 0)+1
                        covered.append(segid)
                used.append(dict(name=str(row['name']), covered=sorted(set(covered)),
                    segments=fingerprint(Path(row['segments_path'])), links=fingerprint(Path(row['links_path']))))
            if set(sums) != set(missing):
                raise ValueError('Original windows cannot cover every missing reference target')
            new_ids, new_values = append_only(ids, values, {s: sums[s]/counts[s] for s in sums})
            out = args.out_root/relative
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, segid=new_ids, embeddings=new_values)
            with np.load(out) as z:
                if not np.array_equal(z['segid'][:len(ids)], ids) or not np.array_equal(z['embeddings'][:len(ids)], values):
                    raise ValueError('Original vectors changed during serialization')
            updated = dict(audit, output_sha256=fingerprint(out)['sha256'], coverage_extension=dict(
                parent_cache=parent, appended_segments=missing, original_vectors_bitwise_identical=True,
                loader_policy='Original reference-overlapping structural windows; candidate-count eligibility bypassed only for missing targets',
                windows=used, encoder_training=False))
            write_json(Path(str(out)+'.audit.json'), updated)
            receipt['jobs'].append(dict(fold=job.fold, seed=job.seed, context=job.closure, appended_segments=missing, output=fingerprint(out)))
            receipt['completed_runs'] += 1
            write_json(args.out_root/'status.json', receipt)
            print(f'Completed {job.name}: appended {missing}', flush=True)
        receipt['status'] = 'complete'
    except Exception as exc:
        receipt.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(args.out_root/'status.json', receipt)


if __name__ == '__main__':
    main()
