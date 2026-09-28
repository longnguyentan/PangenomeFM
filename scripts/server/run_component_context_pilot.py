#!/usr/bin/env python3
"""Queue the fixed, matched fold-A context pilot after prior jobs and GPU QC."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tasks.entex.prepare import fingerprint
from tasks.transfer.traitgym import verified_fingerprint, write_json
from training.component_pilot import readiness_receipts, validate_plan


def probe_command(*, checkpoint, manifest, graph, context_manifest, sw, nt, topology,
                  out, seed, context, task, fold):
    command = [sys.executable, f'scripts/server/run_{task}_frozen_probe_fold.py',
        '--checkpoint', str(checkpoint), '--manifest', str(manifest),
        '--full-segments', str(graph), '--fold', 'fold_a', '--seed', str(seed),
        '--closure', context, '--device', 'cuda', '--out-dir', str(out),
        '--external-sequence-cache', str(nt), '--minimum-external-coverage', '1.0',
        '--topology-control-cache', str(topology), '--validation-only',
        '--exclude-test-extraction', '--save-probes', '--probe-max-iter', '4000',
        '--extraction-candidate-policy', 'manuscript',
        '--test-chrs', *fold['test'], '--val-chrs', *fold['validation']]
    if context == 'component':
        command += ['--component-context-manifest', str(context_manifest)]
    if task == 'sv':
        command += ['--examples', str(sw/'data/processed/hgsvc3_sv_breakpoint_examples_20260809/sv_breakpoint_examples.csv.gz'),
            '--feature-cache', str(sw/'data/processed/hprc_r2_hgsvc3_sv_features_20260809.npz')]
    else:
        command += ['--node-labels', str(sw/'data/downstream/ccre/hprc_r2_screen_v4/node_labels.csv.gz'),
            '--feature-cache', str(sw/'data/processed/hprc_r2_ccre_screen_v4_features.npz')]
    suffix = '_pair' if task == 'sv' else ''
    command += ['--feature-sets', *[name + suffix for name in [
        'coordinate_plus_frozen_sequence_fm',
        'coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm',
        'coordinate_plus_frozen_sequence_fm_plus_topology_control',
        'coordinate_plus_frozen_sequence_fm_plus_topology_control_plus_frozen_pangenomefm']]]
    return command


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'template-checkpoint', 'main-checkout', 'component-contexts',
                 'nt-cache', 'topology-control-cache', 'out-root', 'after-replication',
                 'after-v1-reference', 'after-exact-gpu', 'after-window-gpu', 'after-preflight']:
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--gpus', nargs=4, type=int, default=[0, 1, 2, 3])
    ap.add_argument('--wait-hours', type=float, default=12)
    cli = ap.parse_args()
    plan = json.loads(cli.config.read_text())
    validate_plan(plan)
    if len(set(cli.gpus)) != 4 or not 0 < cli.wait_hours <= 12:
        raise ValueError('Require four distinct GPUs and a bounded readiness deadline')
    dependencies = dict(replication=cli.after_replication, v1_reference=cli.after_v1_reference,
                        exact_gpu=cli.after_exact_gpu, window_gpu=cli.after_window_gpu, preflight=cli.after_preflight)
    root, sw = cli.out_root.resolve(), cli.main_checkout.resolve()/'server_workspace'
    root.mkdir(parents=True, exist_ok=False)
    record = dict(status='waiting', stage='dependencies', plan=plan,
        config=fingerprint(cli.config), implementation=fingerprint(Path(__file__)),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        biological_test_predictions=False, dependencies={k: str(v) for k, v in dependencies.items()},
        planned_pretraining_runs=24, planned_probe_runs=48, completed_jobs=[], commands=[])
    write_json(root/'status.json', record)
    env = dict(os.environ, PYTHONPATH='src:.', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')

    def run(command, log, gpu):
        with log.open('w') as handle:
            subprocess.run(command, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                           stdout=handle, stderr=subprocess.STDOUT, check=True)

    def batch(jobs):
        record['commands'].extend(dict(command=c, gpu=g) for c, _, g in jobs)
        write_json(root/'status.json', record)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run, *job) for job in jobs]
            for future in futures:
                future.result()

    try:
        deadline = time.monotonic() + cli.wait_hours * 3600
        while True:
            reports = [json.loads(p.read_text()) if p.exists() else {} for p in dependencies.values()]
            if any(r.get('status') in ['failed', 'cancelled'] for r in reports):
                raise ValueError('A prerequisite failed; preserve the failure before new GPU work')
            if all(r.get('status') == 'complete' for r in reports):
                verified = readiness_receipts(dependencies, plan, config_sha256=fingerprint(cli.config)['sha256'])
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                               '--format=csv,noheader,nounits'], text=True)
                memory = {int(i): int(m) for i, m in (s.split(',') for s in raw.splitlines())}
                if all(memory[g] < 256 for g in cli.gpus):
                    break
            if time.monotonic() > deadline:
                raise TimeoutError('Pilot dependency/resource readiness deadline exceeded')
            time.sleep(30)
        readiness = root/'readiness.json'
        write_json(readiness, dict(status='complete', config=fingerprint(cli.config), dependencies=verified))
        verified_fingerprint(cli.nt_cache, plan['nt_cache_sha256'])
        manifest = sw/'data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv'
        graph = sw/'data/processed/hprc_r2_sv/full_segments.csv.gz'
        verified_fingerprint(manifest, plan['manifest_sha256'])
        verified_fingerprint(graph, plan['full_segments_sha256'])
        topology_audit = json.loads(Path(str(cli.topology_control_cache) + '.audit.json').read_text())
        if (topology_audit.get('status') != 'complete' or topology_audit.get('processing_version') != 2
                or topology_audit.get('full_segments_sha256') != plan['full_segments_sha256']
                or topology_audit.get('downstream_label_access') != 'none'):
            raise ValueError('Shared handcrafted control cache provenance differs')
        record['topology_control'] = fingerprint(cli.topology_control_cache)
        folds = json.loads(Path('configs/server_full_multicohort_20260806.json').read_text())['rotating_chromosome_folds']
        fold = next(f for f in folds if f['name'] == plan['fold'])
        for seed, context in product(plan['seeds'], plan['contexts']):
            folder = root/f'seed_{seed}'/context
            folder.mkdir(parents=True)
            record.update(status='running', stage=f'{seed}/{context}/pretraining')
            jobs = []
            for arm, gpu in zip(plan['arms'], cli.gpus):
                command = [sys.executable, '-u', '-m', 'training.pretrain_masked_features',
                    '--config', str(cli.config.resolve()), '--template-checkpoint', str(cli.template_checkpoint.resolve()),
                    '--manifest', str(manifest), '--full-segments', str(graph), '--nt-cache', str(cli.nt_cache.resolve()),
                    '--out-dir', str(folder/arm), '--seed', str(seed), '--arm', arm,
                    '--context', context, '--component-contexts', str(cli.component_contexts.resolve()),
                    '--pilot-readiness', str(readiness)]
                jobs.append((command, folder/(arm + '.log'), gpu))
            batch(jobs)
            record.update(stage=f'{seed}/{context}/frozen_validation')
            for task in ['sv', 'ccre']:
                jobs = []
                for arm, gpu in zip(plan['arms'], cli.gpus):
                    command = probe_command(checkpoint=folder/arm/'checkpoint.pt', manifest=manifest,
                        graph=graph, context_manifest=cli.component_contexts.resolve()/'manifest.csv', sw=sw,
                        nt=cli.nt_cache.resolve(), topology=cli.topology_control_cache.resolve(),
                        out=folder/'probes'/arm/task, seed=seed, context=context, task=task, fold=fold)
                    jobs.append((command, folder/f'{arm}_{task}.log', gpu))
                batch(jobs)
            record['completed_jobs'].append(dict(seed=seed, context=context))
            write_json(root/'status.json', record)
        record.update(stage='report')
        write_json(root/'status.json', record)
        with (root/'report.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'tasks.transfer.component_pilot_report',
                '--root', str(root), '--out-dir', str(root/'analysis')], env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        record.update(status='complete', stage='complete')
    except Exception as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(root/'status.json', record)


if __name__ == '__main__':
    main()
