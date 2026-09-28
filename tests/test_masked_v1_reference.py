from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.server.run_masked_v1_reference import command
from tasks.transfer.masked_v1_reference import compare_reference


def test_reference_command_preserves_frozen_old_model_and_uniform_probe_budget():
    c = command(Path('/main'), Path('/H.npz'), Path('/new'), 'fold_c', 42)
    for flag, value in [('--models', 'v1'), ('--contexts', '1hop'), ('--probe-max-iter', '4000'),
                        ('--fold', 'fold_c'), ('--seed', '42')]:
        assert c[c.index(flag)+1] == value
    assert '--save-probes' in c and '--validation-only' not in c
    assert '--candidate-checkpoint' not in c
    assert c[-1] == '/new/fold_c/seed_42'


def test_v1_reference_comparison_retains_all_folds_and_rejects_mismatched_evidence():
    plan = json.loads(Path('configs/masked_nt_v1_reference_20260928.json').read_text())
    rep = json.loads(Path(plan['replication_protocol']).read_text())
    rep['n_bootstrap'] = 20
    rows = []
    for fold, seed, model, task, feature in product(rep['folds'], rep['seeds'], rep['arms'], rep['tasks'], rep['features']):
        value = 0.8 if fold != 'fold_b' else 0.4
        if feature in ['cs', 'csh']:
            value = 0.5
        rows.append(dict(fold=fold, seed=seed, model=model, task=task, feature=feature,
            context='1hop', evaluation_partition='test', n_train=100, n_val=50, n_test=40, n_evaluated=40,
            targets_sha256=f'{fold}/{task}', scores_sha256=feature if feature in ['cs','csh'] else model,
            positive_prevalence=0.5, **{m:value for m in rep['metrics']}))
    candidate = pd.DataFrame(rows)
    reference = candidate.loc[candidate.model.eq('full_trained')].copy().assign(model='v1')
    reference.loc[reference.feature.isin(['cst', 'csht']), rep['metrics']] = 0.6
    paired, summary = compare_reference(candidate, reference, plan, rep)
    assert len(paired) == 5*3*4*2*2*5
    assert np.allclose(summary.loc[summary.scope.eq('development_excluded_four_folds'), 'mean'], 0.2)
    assert np.allclose(summary.loc[summary.scope.eq('development_exposed_fold_only'), 'mean'], -0.2)
    assert summary.loc[summary.scope.eq('development_exposed_fold_only'), ['ci95_low', 'ci95_high']].isna().all().all()
    assert summary.loc[summary.scope.eq('development_excluded_four_folds'), ['ci95_low', 'ci95_high']].notna().all().all()
    with pytest.raises(ValueError, match='Missing'):
        compare_reference(candidate, reference.iloc[:-1], plan, rep)
    changed = reference.copy()
    changed.loc[changed.index[0], 'targets_sha256'] = 'different population'
    with pytest.raises(ValueError, match='populations'):
        compare_reference(candidate, changed, plan, rep)
    changed = reference.copy()
    changed.loc[changed.feature.eq('cs'), 'scores_sha256'] = 'different baseline'
    with pytest.raises(ValueError, match='baseline'):
        compare_reference(candidate, changed, plan, rep)
    changed = reference.copy()
    changed.loc[changed.index[0], 'auprc'] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        compare_reference(candidate, changed, plan, rep)
