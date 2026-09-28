"""Merge completed TraitGym execution shards without altering scientific runs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

from scripts.server.run_ccre_frozen_probe_matrix import build_jobs
from tasks.entex.prepare import fingerprint
from tasks.transfer.traitgym import write_json


def validate_job_matrix(jobs: list[dict], expected: list[dict]) -> list[dict]:
    """Normalize JSON array containers, preserving every chromosome and job field."""
    expected = json.loads(json.dumps(expected))
    def key(job):
        return job['fold'], job['seed'], job['closure']
    if (len(jobs) != len(expected) or len({key(j) for j in jobs}) != len(jobs)
            or sorted(jobs, key=key) != sorted(expected, key=key)):
        raise ValueError('Shards do not form exactly the original fold/seed/context matrix')
    return expected


def merge(roots: list[Path], destination: Path, expected: list[dict]) -> dict:
    if not roots or len(set(p.resolve() for p in roots)) != len(roots):
        raise ValueError('Require distinct execution shards')
    receipts = [json.loads((p/'status.json').read_text()) for p in roots]
    qc = [json.loads((p/'qc.json').read_text()) for p in roots]
    if any(q != qc[0] for q in qc):
        raise ValueError('Different QC across execution shards')
    for receipt in receipts:
        if (receipt['status'] != 'complete' or receipt['completed_runs'] != receipt['planned_runs']
                or receipt['planned_runs'] != len(receipt['jobs'])*len(receipt['datasets'])
                or any(receipt[k] != receipts[0][k] for k in [
                    'plan', 'sources', 'implementation', 'probe_implementation', 'runtime', 'datasets'])):
            raise ValueError('Incomplete shards or inconsistent frozen inputs/runtime')
    jobs = validate_job_matrix([j for r in receipts for j in r['jobs']], expected)
    destination.mkdir(exist_ok=False)
    for root, receipt in zip(roots, receipts):
        folds = sorted({j['fold'] for j in receipt['jobs']})
        for dataset in receipt['datasets']:
            (destination/dataset).mkdir(exist_ok=True)
            for fold in folds:
                shutil.copytree(root/dataset/fold, destination/dataset/fold)
    write_json(destination/'qc.json', qc[0])
    receipt = dict(receipts[0], status='complete', scope='full_matrix', jobs=jobs,
        completed_runs=sum(r['completed_runs'] for r in receipts),
        planned_runs=sum(r['planned_runs'] for r in receipts), command=sys.argv,
        execution_shards=[fingerprint(p/'status.json') for p in roots],
        merger=fingerprint(Path(__file__)), no_scientific_protocol_change=True,
        recovery='JSON lists and native dataclass tuples normalized before exact comparison; original failed driver retained.')
    write_json(destination/'status.json', receipt)
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shards', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    roots = sorted(p.parent for p in args.shards.glob('*/status.json'))
    if not roots:
        raise ValueError('No execution shards')
    receipt = json.loads((roots[0]/'status.json').read_text())
    cfg = json.loads(Path(receipt['plan']['reference_config']).read_text())
    manuscript = json.loads(Path(cfg['manuscript_config']).read_text())
    merge(roots, args.out_dir, [asdict(j) for j in build_jobs(manuscript)])


if __name__ == '__main__':
    main()
