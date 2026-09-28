"""Matched, label-free target policy for the complete-component development pilot.

Only original one-hop segments are reconstruction targets. Added segments are
visible context, so context expansion cannot silently change the objective's
population, target moments, mask count, or window weights.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.splits import normalize_chrom
from tasks.entex.prepare import fingerprint
from tasks.transfer.traitgym import verified_fingerprint
from training.masked_features import segment_mask, segment_target_statistics
from training.pretrain import load_slice


ARMS = ['full_trained', 'full_random', 'coordinate_trained', 'coordinate_random']
CONTEXTS = ['1hop', 'component']
SEEDS = [42, 314159, 20260806]


def validate_plan(plan: dict) -> None:
    policy = plan.get('component_pilot', {})
    if (plan.get('fold') != 'fold_a' or plan.get('contexts') != CONTEXTS
            or plan.get('arms') != ARMS or plan.get('seeds') != SEEDS
            or policy.get('target_population') != 'original_one_hop_segments'
            or policy.get('training_window_policy') != 'original_junction_eligible'
            or policy.get('coordinate_attention_mode') != 'chunked_exact'
            or policy.get('attention_chunk_size') != 512
            or policy.get('downstream_partition') != 'development_validation'
            or policy.get('validation_masks') != 3
            or policy.get('added_nodes') != 'visible_context_only'):
        raise ValueError('Unsupported complete-component pilot policy')


def readiness_receipts(paths: dict[str, Path], plan: dict, *, config_sha256: str) -> dict:
    """Read execution health only; never read biological metrics or predictions."""
    required = {'replication', 'v1_reference', 'exact_gpu', 'window_gpu', 'preflight'}
    if set(paths) != required:
        raise ValueError('Require complete replication, v1, both GPU profiles and CPU preflight receipts')
    reports = {name: json.loads(Path(path).read_text()) for name, path in paths.items()}
    if any(r.get('status') != 'complete' for r in reports.values()):
        raise ValueError('Preceding biological matrix/reference/profiles are not complete')
    preflight = reports['preflight']
    if (preflight.get('config', {}).get('sha256') != config_sha256
            or preflight.get('biological_labels_used') is not False
            or preflight.get('cuda_used') is not False or preflight.get('optimizer_steps') != 0
            or preflight.get('test_windows_loaded') != 0
            or preflight.get('exact_target_moments') is not True
            or preflight.get('exact_target_populations') is not True
            or preflight.get('n_train_windows', 0) < 1 or preflight.get('n_validation_windows', 0) < 1
            or preflight.get('n_verified_mask_samples', 0) != len(plan['seeds']) * (
                preflight.get('n_train_windows', 0) * plan['optimization']['epochs']
                + preflight.get('n_validation_windows', 0) * 3)):
        raise ValueError('Missing or mismatched CPU all-window preflight')
    expected_jobs = {(f, s) for f in ['fold_a', 'fold_b', 'fold_c', 'fold_d', 'fold_e'] for s in SEEDS}
    for name in ['replication', 'v1_reference']:
        jobs = [(j['fold'], j['seed']) for j in reports[name].get('completed_jobs', [])]
        if len(jobs) != 15 or set(jobs) != expected_jobs:
            raise ValueError(f'Incomplete {name} execution matrix')
    for name, mode in [('exact_gpu', 'chunked_exact'), ('window_gpu', 'chunked_window')]:
        r = reports[name]
        if (r.get('mode') != mode or not str(r.get('device', '')).startswith('cuda')
                or r.get('chunk_size') != plan['component_pilot']['attention_chunk_size']
                or r.get('all_gradients_finite') is not True
                or r.get('encoder_weights_unchanged') is not True
                or r.get('biological_labels_used') is not False
                or r.get('weight_updates') != 0 or not r.get('cuda_peak_allocated_bytes')
                or r.get('n_handles') != 28287
                or r.get('context_audit', {}).get('sha256') != plan['component_pilot']['context_audit_sha256']):
            raise ValueError(f'Invalid {name} memory/correctness receipt')
    return {name: fingerprint(Path(path)) for name, path in paths.items()}


def verified_context_manifest(root: Path, plan: dict, original: pd.DataFrame) -> pd.DataFrame:
    """Bind every materialized input to the independently replayed graph receipt."""
    verified_fingerprint(root / 'materialization_replay_v2.json', plan['component_pilot']['materialization_replay_sha256'])
    replay = json.loads((root / 'materialization_replay_v2.json').read_text())
    verified_fingerprint(root / 'audit.json', plan['component_pilot']['context_audit_sha256'])
    if (replay.get('status') != 'complete' or not replay.get('exact_segment_metadata')
            or not replay.get('exact_oriented_induced_edges') or replay.get('biological_labels_used')
            or not all(replay.get(k) is True for k in ['complete_membership_recomputed',
                'original_interval_universe_verified', 'summary_counts_recomputed', 'fold_membership_recomputed'])):
        raise ValueError('Independent exact materialization replay required')
    for key, filename in [('context_audit', 'audit.json'), ('manifest', 'manifest.csv'),
                          ('membership', 'context_segment_ids.npz'), ('fold_overlap', 'fold_overlap.csv')]:
        verified_fingerprint(root / filename, replay[key]['sha256'])
    audit = json.loads((root / 'audit.json').read_text())
    if (audit.get('status') != 'complete' or not audit.get('materialized')
            or audit.get('failed_windows') or audit.get('n_windows') != 608):
        raise ValueError('Require all 608 completed contexts without omissions')
    for left, right in [('full_segments', 'full_segments'), ('manifest', 'manifest'),
                        ('sequence_cache', 'nt_cache')]:
        if audit['plan'][left + '_sha256'] != plan[right + '_sha256']:
            raise ValueError('Component graph/manifest/NT provenance differs from pilot')
    overlap = pd.read_csv(root / 'fold_overlap.csv')
    if (set(overlap.fold) != {'fold_a', 'fold_b', 'fold_c', 'fold_d', 'fold_e'}
            or overlap.fold.duplicated().any()
            or overlap.filter(regex='_overlap$').shape[1] != 3
            or (overlap.filter(regex='_overlap$').to_numpy() != 0).any()):
        raise ValueError('Actual context-segment chromosome separation failed')
    frame = pd.read_csv(root / 'manifest.csv')
    old = original.loc[original.closure.eq('1hop')]
    columns = ['name', 'target_sn', 'start', 'end']
    if (len(frame) != 608 or not frame.closure.eq('component').all()
            or frame.name.duplicated().any() or old.name.duplicated().any()
            or replay['n_verified_windows'] != len(frame)):
        raise ValueError('Context interval universe differs')
    pd.testing.assert_frame_equal(old[columns].sort_values('name').reset_index(drop=True),
        frame[columns].sort_values('name').reset_index(drop=True), check_dtype=False)
    records = {r['name']: r for r in replay['records']}
    if len(records) != len(frame) or set(records) != set(frame.name):
        raise ValueError('Materialization replay does not cover exact interval universe')
    # File digests are checked by the loader only for train/validation rows.
    # Test-window tables are neither opened nor encoded by this pilot.
    frame.attrs['verified_records'] = records
    return frame


def guard_probe_scope(payload: dict, *, validation_only: bool, exclude_test_extraction: bool):
    if 'component_pilot' in payload.get('plan', {}):
        validate_plan(payload['plan'])
        if not validation_only or not exclude_test_extraction:
            raise ValueError('Component pilot checkpoints require validation-only probes and excluded test extraction')


def guard_extraction_scope(payload: dict, target_chrs, closure: str, candidate_policy: str, max_slices):
    if 'component_pilot' not in payload.get('plan', {}):
        return
    validate_plan(payload['plan'])
    heldout = {normalize_chrom(c) for c in payload['args']['test_chrs']}
    targets = {normalize_chrom(c) for c in target_chrs or []}
    if (not targets or targets & heldout or closure != payload['closure']
            or candidate_policy != 'manuscript' or max_slices is not None):
        raise ValueError('Pilot extraction requires its explicit source context, original eligibility and non-test chromosomes')


def pair_slice(original: dict, expanded: dict) -> dict:
    """Attach a fixed target set, checking canonical identities and frozen values."""
    if original['name'] != expanded['name'] or original['target_sn'] != expanded['target_sn']:
        raise ValueError('Unmatched context window')
    old_nodes, new_nodes = np.asarray(original['nodes']), np.asarray(expanded['nodes'])
    if not np.isin(old_nodes, new_nodes).all():
        raise ValueError('Expanded context removed an original oriented handle')
    ids = np.unique(old_nodes // 2)
    positions = {int(n): i for i, n in enumerate(new_nodes)}
    indices = [positions[int(n)] for n in old_nodes]
    if not np.array_equal(original['node_feats'][:, 7:], expanded['node_feats'][indices, 7:]):
        raise ValueError('Frozen NT target changed across contexts')
    old_edges = {(int(old_nodes[s]), int(old_nodes[t])) for s, t in zip(original['src'], original['dst'])}
    new_edges = {(int(new_nodes[s]), int(new_nodes[t])) for s, t in zip(expanded['src'], expanded['dst'])}
    if not old_edges.issubset(new_edges):
        raise ValueError('Expanded context removed an original oriented edge')
    return dict(expanded, mask_segment_ids=ids)


def load_matched_slices(original_rows, component_rows, index, metadata, segments, args, context):
    """Apply eligibility to the old graph once; never filter on expanded graph size."""
    if context not in CONTEXTS:
        raise ValueError('Unknown pilot context')
    components = component_rows.set_index('name', drop=False)
    records = component_rows.attrs['verified_records']
    expanded_args = copy.copy(args)
    expanded_args.objective, expanded_args.extraction_mode = 'masked_nt_features', True
    train, validation, audits = [], [], []
    for _, row in original_rows.iterrows():
        chrom = normalize_chrom(row.target_sn)
        if chrom in args.test_chrs:
            continue
        audit = dict(window=row['name'], chrom=chrom, context=context)
        old = load_slice(row, index, metadata, segments, args, audit)
        if old is not None:
            target_ids = np.unique(old['nodes'] // 2)
            if context == 'component':
                component_row = components.loc[row['name']]
                for key in ['segments', 'links']:
                    verified_fingerprint(Path(component_row[key + '_path']), records[row['name']][key]['sha256'])
                expanded = load_slice(component_row, index, metadata, segments, expanded_args)
                if expanded is None:
                    raise ValueError('Eligible original window has unusable complete context')
                raw = pair_slice(old, expanded)
            else:
                raw = dict(old, mask_segment_ids=target_ids)
            audit.update(n_target_segments=len(target_ids), n_context_segments=len(np.unique(raw['nodes'] // 2)),
                target_ids_sha256=hashlib.sha256(target_ids.astype('<i8').tobytes()).hexdigest())
            (validation if chrom in args.val_chrs else train).append(raw)
        audits.append(audit)
    if not train or not validation:
        raise ValueError('Empty matched pilot training/validation partition')
    train_context = np.unique(np.concatenate([r['nodes'] // 2 for r in train]))
    val_context = np.unique(np.concatenate([r['nodes'] // 2 for r in validation]))
    if np.intersect1d(train_context, val_context).size:
        raise ValueError('Context-only nodes leak across training/validation')
    return train, validation, audits


def reconstruction_mask(raw: dict, node_oids: torch.Tensor, rate: float, seed: int) -> torch.Tensor:
    """Use identical segment samples across contexts, hiding all reverse handles."""
    if 'mask_segment_ids' not in raw:
        return segment_mask(node_oids, rate, seed)
    targets = np.asarray(raw['mask_segment_ids'], dtype=np.int64)
    if not np.array_equal(targets, np.unique(targets)) or len(targets) < 2:
        raise ValueError('Require sorted unique reconstruction targets')
    ids = node_oids.detach().cpu().numpy() // 2
    if not np.isin(targets, ids).all():
        raise ValueError('Reconstruction target is absent from context')
    target_oids = torch.as_tensor(targets * 2)
    chosen = targets[segment_mask(target_oids, rate, seed).numpy()]
    return torch.as_tensor(np.isin(ids, chosen), device=node_oids.device)


def target_statistics(slices):
    """Count only original reconstruction targets, once per biological segment."""
    if isinstance(slices, list) and all('mask_segment_ids' not in raw for raw in slices):
        return segment_target_statistics(slices)
    def target_views():
        for raw in slices:
            keep = np.isin(raw['nodes'] // 2, raw.get('mask_segment_ids', raw['nodes'] // 2))
            yield dict(nodes=raw['nodes'][keep], node_feats=raw['node_feats'][keep])
    return segment_target_statistics(target_views())
