"""Consolidate SV-type execution shards with exact input and job identities."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

from scripts.server.run_ccre_frozen_probe_matrix import build_jobs
from tasks.entex.prepare import fingerprint
from tasks.transfer.traitgym import write_json
from tasks.transfer.traitgym_shards import validate_job_matrix


def merge(roots: list[Path], destination: Path, expected: list[dict]) -> dict:
    if not roots or len({p.resolve() for p in roots}) != len(roots):
        raise ValueError('Require distinct execution shards')
    receipts = [json.loads((p/'status.json').read_text()) for p in roots]
    for r in receipts:
        if (r['status'] != 'complete' or r['completed_runs'] != r['planned_runs']
                or r['planned_runs'] != len(r['jobs']) or r['n_excluded'] != 0
                or r['encoder_training'] is not False
                or any(r[k] != receipts[0][k] for k in [
                    'plan', 'sources', 'implementation', 'n_events', 'class_counts'])):
            raise ValueError('Incomplete shards or inconsistent frozen inputs')
    jobs = validate_job_matrix([j for r in receipts for j in r['jobs']], expected)
    destination.mkdir(parents=True, exist_ok=False)
    for root, r in zip(roots, receipts):
        for j in r['jobs']:
            relative = Path(j['fold'])/f"seed_{j['seed']}"/j['closure']
            shutil.copytree(root/relative, destination/relative)
    result = dict(receipts[0], jobs=jobs, completed_runs=len(jobs), planned_runs=len(jobs),
        execution_shards=[fingerprint(p/'status.json') for p in roots],
        merger=fingerprint(Path(__file__)), no_scientific_protocol_change=True)
    write_json(destination/'status.json', result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shards', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    roots = sorted(p.parent for p in args.shards.glob('*/status.json'))
    if not roots:
        raise ValueError('No execution shards')
    receipt = json.loads((roots[0]/'status.json').read_text())
    reference = json.loads(Path(receipt['plan']['reference_config']).read_text())
    manuscript = json.loads(Path(reference['manuscript_config']).read_text())
    merge(roots, args.out_dir, [asdict(j) for j in build_jobs(manuscript)])


if __name__ == '__main__':
    main()
