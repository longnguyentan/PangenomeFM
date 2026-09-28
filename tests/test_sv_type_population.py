import json

import pandas as pd
import pytest

from tasks.transfer.sv_type_probe import validate_population
from tasks.transfer.sv_type_shards import merge


def test_population_retains_different_events_at_same_anchor():
    x = pd.DataFrame(dict(variant_id=['a', 'b', 'c'], locus_id=['x', 'x', 'y'],
        chrom=['chr1']*3, start=[1, 1, 5], end=[2, 2, 6], svtype=['INS', 'DEL', 'INV']))
    loci = x[['locus_id', 'chrom', 'start', 'end']].drop_duplicates()
    plan = dict(n_events=3, class_counts=dict(INS=1, DEL=1, INV=1))
    validate_population(x, loci, plan)
    with pytest.raises(ValueError, match='population'):
        validate_population(x.iloc[:2], loci, plan)
    with pytest.raises(ValueError, match='coordinates'):
        validate_population(x, loci.assign(end=20), plan)
    with pytest.raises(ValueError, match='population'):
        validate_population(x, pd.concat([loci, loci]), plan)


def test_sv_shards_reject_incomplete_and_inconsistent_matrix(tmp_path):
    roots, jobs = [], []
    for fold in ['a', 'b']:
        root = tmp_path/fold
        out = root/fold/'seed_42'/'strict'
        out.mkdir(parents=True)
        (out/'metrics.csv').write_text('test')
        job = dict(fold=fold, seed=42, closure='strict', test=('chr1',))
        receipt = dict(status='complete', completed_runs=1, planned_runs=1,
            jobs=[job], n_excluded=0, encoder_training=False, plan={}, sources=[],
            implementation='same', n_events=3, class_counts={'INS': 1, 'DEL': 1, 'INV': 1})
        (root/'status.json').write_text(json.dumps(receipt))
        roots.append(root)
        jobs.append(job)
    result = merge(roots, tmp_path/'merged', jobs)
    assert result['completed_runs'] == 2
    assert (tmp_path/'merged'/'a'/'seed_42'/'strict'/'metrics.csv').read_text() == 'test'
    with pytest.raises(ValueError, match='matrix'):
        merge(roots[:1], tmp_path/'incomplete', jobs)
    r = json.loads((roots[0]/'status.json').read_text())
    r['implementation'] = 'different'
    (roots[0]/'status.json').write_text(json.dumps(r))
    with pytest.raises(ValueError, match='inconsistent'):
        merge(roots, tmp_path/'different', jobs)
