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

from tasks.entex.prepare import fingerprint


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


def commands(root: Path, receipt: dict, context: str) -> dict[str, list[str]]:
    native = receipt['native_args']
    if native['seed'] != 42 or native['split_seed'] != 20260806:
        raise ValueError('Fixed fold-A pilot requires the predefined seeds')
    common = [sys.executable, '-u', '-m', 'training.pretrain',
        '--manifest', receipt['manifest']['path'], '--full_segments', receipt['full_segments']['path'],
        '--closures', context, '--objective', 'junction_repair', '--junction_scope', 'branching',
        '--junction_span_size', '16', '--junction_geometry_match', 'signed_gap_bins',
        '--junction_geometry_bin_ratio', '1.25', '--node_structure_source', 'visible',
        '--mask_query_edges', '--validation_only', '--save_predictions', '--lazy_tensorize',
        '--hidden_dim', '48', '--n_layers', '2', '--n_heads', '4', '--multiscale_rope',
        '--orientation_rope', '--seed', '42', '--split_seed', '20260806',
        '--drop_edge', '--drop_edge_rate', '0.1', '--batch_size', '512', '--accum_steps', '4',
        '--epochs', '10', '--patience', '3', '--warmup_epochs', '1', '--recovery_every', '1',
        '--lr', '0.0005', '--weight_decay', '0.0001', '--device', 'cuda',
        '--test_chrs', *native['test_chrs'], '--val_chrs', *native['val_chrs']]
    result = {}
    for direction in ['incoming', 'bidirectional']:
        for head in ['default', 'linear']:
            name = f'{direction}_{head}'
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
    parser.add_argument('--gpus', type=int, nargs=4, default=[0, 1, 2, 3])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if len(set(args.gpus)) != 4:
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
    args.out_root.mkdir(parents=True, exist_ok=False)
    jobs = commands(args.out_root, incoming, args.context)
    record = dict(status='planned', commands=jobs, gpus=args.gpus, context=args.context,
                  incoming_audit=fingerprint(args.incoming_audit / 'audit.json'),
                  bidirectional_audit=fingerprint(args.bidirectional_audit / 'audit.json'),
                  code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  biological_labels_used=False, heldout_predictions_requested=False)
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
