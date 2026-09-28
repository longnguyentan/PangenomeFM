import json
import numpy as np
import pandas as pd
import pytest

from tasks.entex.prepare import fingerprint
from tasks.transfer.genotyping_report import replay
from tasks.transfer.regression_probe import regression_metrics


def fixture(tmp_path):
    loci = pd.DataFrame(dict(locus_id=list('abcdef'), chrom=['chr1']*2+['chr2']*2+['chr3']*2,
                             quality=[1., 2., 3., 4., 5., 6.]))
    plan = dict(task='test', targets=['quality'], feature_sets=['ridge'], ridge_alphas=[1, 10])
    job = dict(fold='fold_a', seed=42, closure='strict', test=['chr3'], validation=['chr2'])
    common = dict(task='test', target='quality', feature_set='ridge', fold='fold_a', seed=42, context='strict')
    p = pd.DataFrame(dict(locus_id=['e', 'f'], chrom=['chr3']*2, y_true=[5., 6.], prediction=[4.5, 6.2], **common))
    p.to_parquet(tmp_path/'predictions.parquet', index=False)
    pd.DataFrame([dict(**common, n_train=2, n_val=2, n_test=2, alpha=10, validation_mae=.1,
                       **regression_metrics(p.y_true, p.prediction))]).to_csv(tmp_path/'metrics.csv', index=False)
    pd.DataFrame([dict(target='quality', feature_set='ridge', alpha=a, validation_mae=m)
                  for a, m in [(1, .2), (10, .1)]]).to_csv(tmp_path/'validation_selection.csv', index=False)
    (tmp_path/'audit.json').write_text(json.dumps(dict(status='complete', n_excluded=0, feature_coverage=1,
        predictions=fingerprint(tmp_path/'predictions.parquet'))))
    return loci, job, plan


def test_replay_and_reject_wrong_outcome_or_selection(tmp_path):
    loci, job, plan = fixture(tmp_path)
    assert len(replay(tmp_path, loci, job, plan)) == 1
    bad = loci.copy()
    bad.loc[5, 'quality'] += 1
    with pytest.raises(ValueError, match='universe'):
        replay(tmp_path, bad, job, plan)
    m = pd.read_csv(tmp_path/'metrics.csv')
    m['alpha'] = 1
    m.to_csv(tmp_path/'metrics.csv', index=False)
    with pytest.raises(ValueError, match='validation rule'):
        replay(tmp_path, loci, job, plan)


def test_replay_rejects_tampered_predictions(tmp_path):
    loci, job, plan = fixture(tmp_path)
    p = pd.read_parquet(tmp_path/'predictions.parquet')
    p['prediction'] = np.nan
    p.to_parquet(tmp_path/'predictions.parquet')
    with pytest.raises(ValueError, match='Stale'):
        replay(tmp_path, loci, job, plan)
