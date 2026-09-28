import json

import pytest

from tasks.transfer.traitgym_shards import merge, validate_job_matrix


def test_json_array_normalization_preserves_exact_holdouts():
    expected = [dict(fold='fold_a', seed=42, closure='strict', test=('chr1',), validation=('chr2',))]
    jobs = json.loads(json.dumps(expected))
    assert validate_job_matrix(jobs, expected) == jobs
    with pytest.raises(ValueError, match='exactly'):
        validate_job_matrix(jobs*2, expected)
    jobs[0]['test'] = ['chr3']
    with pytest.raises(ValueError, match='exactly'):
        validate_job_matrix(jobs, expected)


def test_merger_checks_completion_sources_and_refuses_overwrite(tmp_path):
    roots, jobs = [], []
    for fold in ['a', 'b']:
        root = tmp_path/fold
        (root/'dataset'/fold).mkdir(parents=True)
        (root/'dataset'/fold/'result.txt').write_text('unchanged prediction fixture')
        job = dict(fold=fold, seed=42, closure='strict', test=[fold], validation=['c'])
        state = dict(status='complete', completed_runs=1, planned_runs=1, jobs=[job], datasets=['dataset'],
                     plan={}, sources=[], implementation={}, probe_implementation={}, runtime={})
        (root/'status.json').write_text(json.dumps(state))
        (root/'qc.json').write_text('{}')
        roots.append(root)
        jobs.append(job)
    result = merge(roots, tmp_path/'out', jobs)
    assert result['completed_runs'] == 2 and result['scope'] == 'full_matrix'
    assert (tmp_path/'out/dataset/a/result.txt').read_text() == 'unchanged prediction fixture'
    with pytest.raises(FileExistsError):
        merge(roots, tmp_path/'out', jobs)
    state['sources'] = ['different cache']
    (roots[-1]/'status.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='inconsistent'):
        merge(roots, tmp_path/'bad', jobs)
    assert not (tmp_path/'bad').exists()
