#!/usr/bin/env python3
"""Run all declared masked-feature arms/seeds, then frozen validation probes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from tasks.entex.prepare import fingerprint
from tasks.transfer.traitgym import write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'template-checkpoint', 'main-checkout', 'nt-cache', 'topology-control-cache', 'out-root']:
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--gpus', type=int, nargs=4, default=[0,1,2,3])
    args = ap.parse_args()
    if len(set(args.gpus)) != 4:
        raise ValueError('Use four distinct GPUs')
    plan = json.loads(args.config.read_text())
    args.out_root.mkdir(parents=True, exist_ok=False)
    root = args.out_root.resolve()
    sw = args.main_checkout.resolve()/'server_workspace'
    record = dict(status='running', plan=plan, native_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config=fingerprint(args.config), planned_pretraining_runs=12, completed_seeds=[], commands=[],
        planned_probe_runs=24, biological_test_predictions=False)
    write_json(root/'status.json', record)
    env = dict(os.environ, PYTHONPATH='src:.', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLCONFIGDIR='/tmp/pfm-mpl-masked')

    def run(command, log, gpu=None):
        execution = dict(env)
        if gpu is not None:
            execution['CUDA_VISIBLE_DEVICES'] = str(gpu)
        with log.open('w') as handle:
            subprocess.run(command, env=execution, stdout=handle, stderr=subprocess.STDOUT, check=True)

    try:
        for seed in plan['seeds']:
            directory = root/f'seed_{seed}'
            directory.mkdir()
            jobs = []
            for arm, gpu in zip(plan['arms'], args.gpus):
                command = [sys.executable,'-u','-m','training.pretrain_masked_features',
                    '--config',str(args.config.resolve()),'--template-checkpoint',str(args.template_checkpoint.resolve()),
                    '--manifest',str(sw/'data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv'),
                    '--full-segments',str(sw/'data/processed/hprc_r2_sv/full_segments.csv.gz'),
                    '--nt-cache',str(args.nt_cache.resolve()),'--out-dir',str(directory/arm),
                    '--seed',str(seed),'--arm',arm]
                jobs.append((command, directory/(arm+'.log'), gpu))
                record['commands'].append(dict(command=command,gpu=gpu))
            record['stage']=f'seed_{seed}/pretraining'
            write_json(root/'status.json',record)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(run,*job) for job in jobs]
                for future in futures:
                    future.result()
            # Weight identity, not a performance gate, decides technical readiness.
            for kind in ['full','coordinate']:
                trained=json.loads((directory/(kind+'_trained')/'status.json').read_text())
                random=json.loads((directory/(kind+'_random')/'status.json').read_text())
                if (trained['status']!='complete' or random['status']!='complete'
                        or trained['initial_encoder_sha256']!=random['initial_encoder_sha256']
                        or random['final_encoder_sha256']!=random['initial_encoder_sha256']):
                    raise ValueError('Invalid trained/random backbone identities')
            command=[sys.executable,'scripts/server/run_v2_review_controls.py',
                '--main-checkout',str(args.main_checkout.resolve()),'--out-root',str(directory/'biological'),
                '--fold',plan['fold'],'--seed',str(seed),'--contexts','1hop','--validation-only',
                '--primary-features-only','--probe-max-iter','4000','--extraction-candidate-policy','manuscript',
                '--topology-control-cache',str(args.topology_control_cache.resolve()),
                '--probe-gpus',*map(str,args.gpus),'--models',*plan['arms']]
            for arm in plan['arms']:
                command += ['--candidate-checkpoint',f'{arm}={directory/arm/"checkpoint.pt"}']
            record['stage']=f'seed_{seed}/frozen_validation'
            record['commands'].append(dict(command=command))
            write_json(root/'status.json',record)
            run(command,directory/'biological.log')
            record['completed_seeds'].append(seed)
            write_json(root/'status.json',record)
        record['stage']='report'
        write_json(root/'status.json',record)
        run([sys.executable,'-m','tasks.transfer.masked_feature_report','--root',str(root),
            '--out-dir',str(root.parent/(root.name+'_analysis'))],root/'report.log')
        record['status']='complete'
    except Exception as exc:
        record.update(status='failed',error=repr(exc))
        raise
    finally:
        write_json(root/'status.json',record)
if __name__ == '__main__':
    main()
