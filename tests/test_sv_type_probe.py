import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tasks.entex.prepare import fingerprint
from tasks.transfer.sv_type_probe import evaluate_types
from tasks.transfer.sv_type_report import replay, summarize
from tasks.transfer.sv_types import CLASSES, normalize


def test_real_multiclass_targets_native_probe_replay_and_macro(tmp_path):
    source = pd.DataFrame([{'ID': f'chr{c}-{101+i}-{kind}-100', '#CHROM': f'chr{c}', 'POS': 100+i,
        'END': 101+i if kind == 'INS' else 200+i, 'SVTYPE': kind, 'SVLEN': 100}
        for c in range(1, 4) for kind in CLASSES for i in range(30)])
    examples = normalize(source)
    cs = 'coordinate_plus_frozen_sequence_fm'
    full = cs+'_plus_frozen_pangenomefm'
    rng = np.random.default_rng(51)
    matrices = {key: rng.normal(size=(len(examples), 4)).astype(np.float32) for key in ['coordinate', cs, full]}
    plan = dict(task='synthetic', custom_feature_sets={}, feature_sets=list(matrices), probe_max_iter=4000,
        metrics=['auprc','auroc','normalized_ap','balanced_accuracy','f1','precision','recall'],
        comparisons=[['T_given_CS', full, cs]], n_bootstrap=40, statistics_seed=1)
    job = SimpleNamespace(test=['chr1'], validation=['chr2'], seed=42, fold='fold_a', closure='strict')
    metrics, predictions = evaluate_types(examples, matrices, plan, job)
    assert len(metrics) == 9 and set(metrics.target_class) == set(CLASSES)
    for kind, label in CLASSES.items():
        part = predictions.loc[predictions.target_class.eq(kind)]
        assert part.y_true.eq(part.label.eq(label)).all()
    metrics.to_csv(tmp_path/'metrics.csv',index=False)
    predictions.to_parquet(tmp_path/'predictions.parquet',index=False)
    (tmp_path/'audit.json').write_text(json.dumps(dict(status='complete', n_excluded=0, feature_coverage=1.,
        all_probes_converged=True, predictions=fingerprint(tmp_path/'predictions.parquet'))))
    checked = replay(tmp_path, examples, plan, vars(job))
    absolute, _, _ = summarize(checked, plan)
    for feature in plan['feature_sets']:
        expected = metrics.loc[metrics.feature_set.eq(feature)].auprc.mean()
        actual = absolute.loc[absolute.target_class.eq('macro') & absolute.feature_set.eq(feature) & absolute.metric.eq('auprc'),'mean'].item()
        assert actual == pytest.approx(expected)
    metrics.loc[0,'auprc'] += .1
    metrics.to_csv(tmp_path/'metrics.csv',index=False)
    with pytest.raises(ValueError, match='differs from prediction replay'):
        replay(tmp_path, examples, plan, vars(job))
    corrupted = examples.copy()
    corrupted.loc[corrupted.svtype.eq('INV'), 'label'] = 0
    with pytest.raises(ValueError, match='multiclass'):
        evaluate_types(corrupted, matrices, plan, job)
    with pytest.raises(ValueError, match='Insufficient'):
        evaluate_types(examples, matrices, plan, SimpleNamespace(**dict(vars(job),test=['chr4'])))
