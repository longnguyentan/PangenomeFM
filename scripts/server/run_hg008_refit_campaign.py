#!/usr/bin/env python3
"""Create/run the complete new HGSVC-only refit matrix; never mix with old replay."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys


def refit_plan(base: dict, root: Path) -> dict:
    plan = copy.deepcopy(base)
    plan['run_name'] = 'hg008_prospective_refit_20260927'
    for task in plan['tasks']:
        command = task['command']
        command[0] = sys.executable
        command[command.index('--out-root') + 1] = str(root / 'probes')
        command.extend(['--protocol', 'prospective_refit'])
        task['description'] = 'New saved HGSVC-only probe; deterministic refit, not historical replay'
        task['id'] = task['id'].replace('hg008_', 'hg008_refit_', 1)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=False)
    base = json.loads(Path('configs/hg008_transfer_jobs_v1.json').read_text())
    plan = refit_plan(base, args.out_root)
    config = args.out_root / 'jobs.json'
    config.write_text(json.dumps(plan, indent=2) + '\n')
    if not args.execute:
        print(f'Prepared {len(plan["tasks"])} jobs: {config}')
        return
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONPATH='src:.')
    processes, logs = [], []
    try:
        for stage in sorted({task['stage'] for task in plan['tasks']}):
            log = (args.out_root / f'{stage}.log').open('w')
            logs.append(log)
            processes.append(subprocess.Popen([
                sys.executable, 'scripts/server/run_foundation_model_roadmap.py',
                '--config', str(config), '--execute', '--stage', stage,
                '--result-root', str(args.out_root / 'execution' / stage),
                '--fail-on-blocked',
            ], env=env, stdout=log, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
    finally:
        for log in logs:
            log.close()
    if any(codes):
        raise RuntimeError(f'Incomplete refit matrix; preserved logs/receipts: {codes}')
    subprocess.run([sys.executable, '-m', 'tasks.transfer.summarize', '--probe-root',
                    str(args.out_root / 'probes'), '--out-dir', str(args.out_root / 'analysis')],
                   env=env, check=True)


if __name__ == '__main__':
    main()
