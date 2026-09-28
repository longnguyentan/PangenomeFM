"""Same prespecified experiment, independent fold processes; no scientific changes."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/tuv43532/PangenomeFM_evidence_report_20260927')
BASE = ROOT/'results/foundation_evidence_20260927'
HERE = BASE/'traitgym_probe_float64_driver'
ENV = dict(os.environ, PYTHONPATH='src:.', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', MPLCONFIGDIR='/tmp/pfm-mpl')
PLAN = json.loads((HERE/'launch.json').read_text())
STATE = dict(status='running', native_commit=PLAN['native_commit'], stage='smoke_fit', maximum_parallel_folds=5, completed_folds=[])


def save():
    (HERE/'parallel_status.json').write_text(json.dumps(STATE, indent=2)+'\n')


def execute(command, name):
    with (HERE/f'{name}.log').open('x') as log:
        subprocess.run(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)


def fingerprint(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def merge_shards(roots, destination):
    receipts = [json.loads((p/'status.json').read_text()) for p in roots]
    qc = [json.loads((p/'qc.json').read_text()) for p in roots]
    if any(q != qc[0] for q in qc):
        raise ValueError('Different QC across execution shards')
    for receipt in receipts:
        if (receipt['status'] != 'complete' or receipt['completed_runs'] != receipt['planned_runs']
                or receipt['plan'] != receipts[0]['plan'] or receipt['sources'] != receipts[0]['sources']
                or receipt['implementation'] != receipts[0]['implementation']
                or receipt['probe_implementation'] != receipts[0]['probe_implementation']
                or receipt['runtime'] != receipts[0]['runtime']):
            raise ValueError('Incomplete shards or inconsistent frozen inputs/runtime')
    jobs = [j for receipt in receipts for j in receipt['jobs']]
    sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
    from scripts.server.run_ccre_frozen_probe_matrix import build_jobs
    from dataclasses import asdict
    cfg = json.loads((ROOT/receipts[0]['plan']['reference_config']).read_text())
    manuscript = json.loads((ROOT/cfg['manuscript_config']).read_text())
    expected = [asdict(j) for j in build_jobs(manuscript)]
    def canonical(job):
        return job['fold'], job['seed'], job['closure']
    if len(jobs) != len(expected) or sorted(jobs, key=canonical) != sorted(expected, key=canonical):
        raise ValueError('Shards do not form exactly the original 30 fold/seed/context jobs')
    destination.mkdir(exist_ok=False)
    for root, receipt in zip(roots, receipts):
        folds = sorted({j['fold'] for j in receipt['jobs']})
        for dataset in receipt['datasets']:
            (destination/dataset).mkdir(exist_ok=True)
            for fold in folds:
                shutil.copytree(root/dataset/fold, destination/dataset/fold)
    (destination/'qc.json').write_text(json.dumps(qc[0], indent=2)+'\n')
    receipt = dict(receipts[0], status='complete', scope='full_matrix', jobs=expected,
                   completed_runs=sum(r['completed_runs'] for r in receipts),
                   planned_runs=sum(r['planned_runs'] for r in receipts),
                   command=sys.argv, execution_shards=[fingerprint(p/'status.json') for p in roots],
                   merger=fingerprint(Path(__file__)), no_scientific_protocol_change=True)
    assert receipt['completed_runs'] == 60
    (destination/'status.json').write_text(json.dumps(receipt, indent=2)+'\n')


save()
try:
    assert subprocess.check_output(['git', 'rev-parse', '--short=7', 'HEAD'], cwd=ROOT, text=True).strip() == PLAN['native_commit']
    execute(PLAN['commands'][0], 'smoke_fit')
    STATE['stage'] = 'smoke_report_and_fixed_C_gate'
    save()
    execute(PLAN['commands'][1], 'parallel_smoke_report')
    execute(PLAN['commands'][2], 'parallel_smoke_gate')
    assert json.loads((HERE/'smoke_gate.json').read_text())['status'] == 'pass'
    STATE['stage'] = 'five_parallel_folds'
    save()
    commands, roots = {}, []
    for fold in ['fold_a', 'fold_b', 'fold_c', 'fold_d', 'fold_e']:
        command = list(PLAN['commands'][3])
        path = BASE/'traitgym_probe_float64_shards'/fold
        command[command.index('--out-root')+1] = str(path)
        command += ['--folds', fold]
        commands[fold] = command
        roots.append(path)
    (HERE/'parallel_commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        pending = {pool.submit(execute, command, fold): fold for fold, command in commands.items()}
        for future in concurrent.futures.as_completed(pending):
            fold = pending[future]
            try:
                future.result()
                STATE['completed_folds'].append(fold)
            except Exception as error:
                failures.append(dict(fold=fold, error=repr(error)))
                STATE['failures'] = failures
            save()
    if failures:
        raise RuntimeError(f'Failed execution shards: {failures}')
    STATE['stage'] = 'merge_and_replay'
    save()
    merge_shards(roots, BASE/'traitgym_probe_float64_full')
    execute(PLAN['commands'][4], 'parallel_full_report')
    STATE.update(status='complete', stage='done')
except Exception as error:
    STATE.update(status='failed', error=repr(error))
    raise
finally:
    STATE['updated'] = time.time()
    save()
