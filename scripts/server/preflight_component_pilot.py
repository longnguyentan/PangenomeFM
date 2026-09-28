#!/usr/bin/env python3
"""CPU-only all-window matched-target replay; no fitting or biological labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from evaluation.external import _namespace_from_checkpoint
from evaluation.splits import normalize_chrom
from graph.features import build_oid_metadata_from_segments
from graph.slicing import build_global_index
from tasks.entex.prepare import fingerprint
from tasks.transfer.traitgym import verified_fingerprint, write_json
from training.component_pilot import load_matched_slices, reconstruction_mask, target_statistics, validate_plan, verified_context_manifest
from training.pretrain_masked_features import mask_seed


def verify_matched_banks(one, complete, plan):
    """Return a digest of every fixed training/validation target sample."""
    digest, count = hashlib.sha256(), 0
    for partition, old_rows, new_rows in zip(['train', 'validation'], one, complete):
        if [r['name'] for r in old_rows] != [r['name'] for r in new_rows]:
            raise ValueError('Context window populations differ')
        for old, new in zip(old_rows, new_rows):
            if not np.array_equal(old['mask_segment_ids'], new['mask_segment_ids']):
                raise ValueError('Context reconstruction target populations differ')
            oids = [torch.as_tensor(r['nodes']) for r in [old, new]]
            masks = ([(epoch, 0) for epoch in range(1, plan['optimization']['epochs'] + 1)]
                     if partition == 'train' else [(0, view) for view in range(3)])
            for seed in plan['seeds']:
                for epoch, view in masks:
                    bank_seed = mask_seed(seed, old['name'], epoch, view)
                    chosen = [np.unique(r['nodes'][reconstruction_mask(r, oid, plan['mask_rate'], bank_seed).numpy()] // 2)
                              for r, oid in zip([old, new], oids)]
                    if not np.array_equal(*chosen):
                        raise ValueError('Context reconstruction mask banks differ')
                    digest.update(json.dumps([partition, old['name'], seed, epoch, view]).encode())
                    digest.update(chosen[0].astype('<i8').tobytes())
                    count += 1
    return dict(n_verified_mask_samples=count, mask_bank_sha256=digest.hexdigest())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'template-checkpoint', 'manifest', 'full-segments', 'nt-cache', 'component-contexts', 'out-dir']:
        ap.add_argument('--' + name, type=Path, required=True)
    cli = ap.parse_args()
    plan = json.loads(cli.config.read_text())
    validate_plan(plan)
    cli.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    record = dict(status='running', scope='CPU loading/mask/moment audit only; no model fitting or probe labels',
        biological_labels_used=False, test_windows_loaded=0, optimizer_steps=0, cuda_used=False,
        config=fingerprint(cli.config), implementation=fingerprint(Path(__file__)))
    write_json(cli.out_dir/'status.json', record)
    try:
        record['sources'] = [verified_fingerprint(path, plan[key + '_sha256']) for key, path in
            [('manifest', cli.manifest), ('full_segments', cli.full_segments), ('nt_cache', cli.nt_cache)]]
        record['template'] = fingerprint(cli.template_checkpoint)
        template = torch.load(cli.template_checkpoint, map_location='cpu', weights_only=False)
        args = _namespace_from_checkpoint(template, plan['seeds'][0])
        expected = dict(plan['encoder'], node_extra_features='cache', node_structure_source='visible',
            junction_geometry_match='signed_gap_bins', junction_geometry_bin_ratio=1.25,
            split_seed=20260806, pop_cond=False, use_edge_features=False)
        if template['in_dim'] != 519 or any(getattr(args, k, None) != v for k, v in expected.items()):
            raise ValueError('Template input/loader policy differs from fixed pilot')
        folds = json.loads(Path('configs/server_full_multicohort_20260806.json').read_text())['rotating_chromosome_folds']
        fold = next(f for f in folds if f['name'] == 'fold_a')
        if set(args.test_chrs) != set(fold['test']) or set(args.val_chrs) != set(fold['validation']):
            raise ValueError('Template is not fold A')
        args.device, args.node_feature_cache, args.node_feature_min_coverage = 'cpu', str(cli.nt_cache), 1.
        args.canonical_conflict_policy, args.objective, args.extraction_mode = 'exclude', 'junction_repair', False
        args.drop_edge, args.drop_edge_rate, args.mask_query_edges = False, 0., False
        args.linear_predictor, args.pair_geometry = False, False
        manifest = pd.read_csv(cli.manifest)
        components = verified_context_manifest(cli.component_contexts, plan, manifest)
        rows = manifest.loc[manifest.closure.eq('1hop') & ~manifest.target_sn.map(normalize_chrom).isin(args.test_chrs)]
        segments = pd.read_csv(cli.full_segments, usecols=lambda c: c != 'seq')
        index, _ = build_global_index(segments)
        metadata = build_oid_metadata_from_segments(segments, index)
        one_train, one_val, one_audit = load_matched_slices(rows, components, index, metadata, segments, args, '1hop')
        wide_train, wide_val, wide_audit = load_matched_slices(rows, components, index, metadata, segments, args, 'component')
        for actual, expected in zip(target_statistics(one_train), target_statistics(wide_train)):
            if actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
                raise ValueError('Original/component target identities or moments differ')
        bank = verify_matched_banks([one_train, one_val], [wide_train, wide_val], plan)
        for name, audits in [('1hop', one_audit), ('component', wide_audit)]:
            pd.DataFrame(audits).to_csv(cli.out_dir/(name + '_windows.csv'), index=False)
        record.update(status='complete', n_train_windows=len(one_train), n_validation_windows=len(one_val),
            exact_target_moments=True, exact_target_populations=True, **bank,
            tables={name: fingerprint(cli.out_dir/name) for name in ['1hop_windows.csv', 'component_windows.csv']},
            elapsed_seconds=time.time() - started)
        write_json(cli.out_dir/'audit.json', record)
    except Exception as exc:
        record.update(status='failed', error=repr(exc), elapsed_seconds=time.time() - started)
        raise
    finally:
        write_json(cli.out_dir/'status.json', record)


if __name__ == '__main__':
    main()
