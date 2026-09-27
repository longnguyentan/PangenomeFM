#!/usr/bin/env python3
"""Gate and run the fixed one-context, four-arm validation-only junction pilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import numpy as np

from tasks.entex.prepare import fingerprint

ARMS = ['incoming_default', 'incoming_linear', 'bidirectional_default', 'bidirectional_linear']


def check_sequence_cache(path: Path, receipt: dict, context: str) -> dict:
    """Require the same frozen NT policy and complete native benchmark coverage."""
    from scripts.server.complete_node_sequence_fm_cache import benchmark_targets
    from scripts.server.merge_node_sequence_fm_caches import sequence_contract

    contract = sequence_contract(path)
    if contract['full_segments_sha256'] != receipt['full_segments']['sha256']:
        raise ValueError('Sequence inputs use a different graph')
    expected = dict(model_name='InstaDeepAI/nucleotide-transformer-v2-50m-multi-species',
                    resolved_revision='81b29e5786726d891dbf929404ef20adca5b36f1',
                    maximum_token_length=1000, maximum_raw_bases=6000,
                    pooling='mean final hidden state over non-special, non-padding tokens',
                    raw_sequence_sampling='long nodes retain balanced prefix and suffix separated by N before tokenizer truncation',
                    truncation_policy='tokenizer truncation at maximum_token_length; complete raw sequences remain in the source table')
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError('Sequence inputs differ from the manuscript NT model/preprocessing')
    required, scope = benchmark_targets(Path(receipt['full_segments']['path']),
                                        Path(receipt['manifest']['path']), context)
    with np.load(path, allow_pickle=False) as cache:
        ids, values = cache['segid'], cache['embeddings']
        if (len(np.unique(ids)) != len(ids) or values.shape != (len(ids), 512)
                or not np.isfinite(values).all()):
            raise ValueError('Invalid sequence input cache')
        missing = np.setdiff1d(required, ids)
    if len(missing):
        raise ValueError(f'Sequence inputs miss {len(missing)} native benchmark nodes')
    return dict(cache=fingerprint(path), contract=contract, benchmark_scope=scope,
                coverage=1.0, biological_labels_used=False,
                orientation_policy='same frozen segment sequence vector for both oriented handles')


def check_context(audit_dir: Path, context: str, direction: str, required: set[str]) -> dict:
    receipt = json.loads((audit_dir / 'audit.json').read_text())
    args = receipt['native_args']
    expected = dict(junction_geometry_match='signed_gap_bins', junction_geometry_bin_ratio=1.25,
                    node_structure_source='visible', objective='junction_repair', drop_edge_rate=0.1)
    if any(args.get(k) != v for k, v in expected.items()):
        raise ValueError('Audit configuration differs from the fixed pilot protocol')
    if args.get('graph_message_direction', 'incoming') != direction:
        raise ValueError('Audit message direction mismatch')
    counts = pd.read_csv(audit_dir / 'per_window.csv')
    counts = counts.loc[counts.context.eq(context) & counts.exclusion.eq('retained')]
    for split in ['val', 'test']:
        covered = set(counts.loc[counts[f'n_{split}_candidates'].ge(4), 'chrom'])
        if required - covered:
            raise ValueError(f'{context}/{split} coverage fails: {sorted(required - covered)}')
    scores = pd.read_csv(audit_dir / 'validation_baselines.csv')
    scores = scores.loc[scores.context.eq(context) & scores.baseline.isin(['geometry', 'geometry_and_visible_degree'])]
    if len(scores) != 2 or scores.n_val.min() < 200:
        raise ValueError('Insufficient context-specific validation control evidence')
    if not (scores[['auprc', 'auroc']].ge(0).all().all()
            and scores[['auprc', 'auroc']].le(0.60).all().all()):
        raise ValueError('Nuisance baseline exceeds the predeclared pilot gate')
    return receipt


def commands(root: Path, receipt: dict, context: str, *,
             arms: list[str] | None = None, node_feature_cache: Path | None = None,
             stream_mode: str = 'full', seed: int = 42) -> dict[str, list[str]]:
    native = receipt['native_args']
    if native['seed'] != 42 or native['split_seed'] != 20260806:
        raise ValueError('Fixed fold-A pilot requires the predefined seeds')
    if seed not in {42, 314159, 20260806}:
        raise ValueError('Use a predefined manuscript model seed')
    common = [sys.executable, '-u', '-m', 'training.pretrain',
        '--manifest', receipt['manifest']['path'], '--full_segments', receipt['full_segments']['path'],
        '--closures', context, '--objective', 'junction_repair', '--junction_scope', 'branching',
        '--junction_span_size', '16', '--junction_geometry_match', 'signed_gap_bins',
        '--junction_geometry_bin_ratio', '1.25', '--node_structure_source', 'visible',
        '--mask_query_edges', '--validation_only', '--save_predictions', '--lazy_tensorize',
        '--hidden_dim', '48', '--n_layers', '2', '--n_heads', '4', '--multiscale_rope',
        '--orientation_rope', '--seed', str(seed), '--split_seed', '20260806',
        '--drop_edge', '--drop_edge_rate', '0.1', '--batch_size', '512', '--accum_steps', '4',
        '--epochs', '10', '--patience', '3', '--warmup_epochs', '1', '--recovery_every', '1',
        '--lr', '0.0005', '--weight_decay', '0.0001', '--device', 'cuda',
        '--test_chrs', *native['test_chrs'], '--val_chrs', *native['val_chrs']]
    result = {}
    if stream_mode not in {'full', 'coordinate'}:
        raise ValueError('Use the full model or its native coordinate-stream ablation')
    if stream_mode != 'full':
        common += ['--stream_mode', stream_mode]
    selected = ARMS if arms is None else arms
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(ARMS):
        raise ValueError('Select unique existing pilot arms')
    if node_feature_cache is not None:
        common += ['--node_extra_features', 'cache', '--node_feature_cache', str(node_feature_cache),
                   '--node_feature_min_coverage', '1.0']
    for direction in ['incoming', 'bidirectional']:
        for head in ['default', 'linear']:
            name = f'{direction}_{head}'
            if name not in selected:
                continue
            result[name] = common + ['--out_dir', str(root / name), '--graph_message_direction', direction]
            if head == 'linear':
                result[name].append('--linear_predictor')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incoming-audit', type=Path, required=True)
    parser.add_argument('--bidirectional-audit', type=Path, required=True)
    parser.add_argument('--context', choices=['strict', '1hop'], default='1hop')
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    parser.add_argument('--stream-mode', choices=['full', 'coordinate'], default='full')
    parser.add_argument('--seed', type=int, choices=[42, 314159, 20260806], default=42)
    parser.add_argument('--node-feature-cache', type=Path,
                        help='Audited benchmark-complete frozen NT inputs; defines a separate multimodal model')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--random-encoder-control', action='store_true',
                        help='Repeat the fixed four arms with frozen random backbones and fitted heads')
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.arms) or len(args.gpus) != len(args.arms):
        raise ValueError('Use a different GPU for each fixed arm')
    required = set(json.loads(Path('configs/server_full_multicohort_20260806.json').read_text())['primary_chromosomes'])
    incoming = check_context(args.incoming_audit, args.context, 'incoming', required)
    bidir = check_context(args.bidirectional_audit, args.context, 'bidirectional', required)
    for key in ['manifest', 'full_segments']:
        if incoming[key]['sha256'] != bidir[key]['sha256']:
            raise ValueError('Control audits use different resources')
        if fingerprint(Path(incoming[key]['path']))['sha256'] != incoming[key]['sha256']:
            raise ValueError('Audited resource changed')
    if incoming['split'] != bidir['split']:
        raise ValueError('Control audits use different chromosome partitions')
    sequence_inputs = check_sequence_cache(args.node_feature_cache, incoming, args.context) if args.node_feature_cache else None
    args.out_root.mkdir(parents=True, exist_ok=False)
    jobs = commands(args.out_root, incoming, args.context, arms=args.arms,
                    node_feature_cache=args.node_feature_cache, stream_mode=args.stream_mode, seed=args.seed)
    if args.random_encoder_control:
        for command in jobs.values():
            command.append('--freeze_encoder')
    record = dict(status='planned', commands=jobs, gpus=args.gpus, context=args.context,
                  incoming_audit=fingerprint(args.incoming_audit / 'audit.json'),
                  bidirectional_audit=fingerprint(args.bidirectional_audit / 'audit.json'),
                  code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  biological_labels_used=False, heldout_predictions_requested=False)
    record['sequence_inputs'] = sequence_inputs
    record['stream_mode'] = args.stream_mode
    record['seed'] = args.seed
    record['representation'] = ('topology_native' if not sequence_inputs else
                                'sequence_conditioned_coordinate' if args.stream_mode == 'coordinate'
                                else 'sequence_conditioned_graph')
    record['random_encoder_control'] = args.random_encoder_control
    path = args.out_root / 'status.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    if not args.execute:
        return
    processes, logs = {}, []
    record['status'] = 'running'
    path.write_text(json.dumps(record, indent=2) + '\n')
    try:
        for (name, command), gpu in zip(jobs.items(), args.gpus):
            log = (args.out_root / f'{name}.log').open('w')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONPATH='src:.')
            processes[name] = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        codes = {name: process.wait() for name, process in processes.items()}
    finally:
        for log in logs:
            log.close()
    record.update(status='complete' if not any(codes.values()) else 'failed', returncodes=codes)
    path.write_text(json.dumps(record, indent=2) + '\n')
    if any(codes.values()):
        raise RuntimeError('One or more fixed model arms failed; inspect logs')


if __name__ == '__main__':
    main()
