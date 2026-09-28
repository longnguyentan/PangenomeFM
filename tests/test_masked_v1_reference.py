from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.server.run_masked_v1_reference import command
from tasks.entex.prepare import fingerprint
from tasks.transfer.masked_v1_reference import compare_reference, load_candidate_replication


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


@pytest.fixture
def bound_candidate_report(tmp_path):
    frame = pd.DataFrame({'model': ['full_trained', 'full_random'],
                          'auprc': [0.25, np.nextafter(0.6, 1.)]})
    table = tmp_path / 'audited_per_run.csv'
    frame.to_csv(table, index=False)
    audit = dict(status='complete', all_saved_predictions_replayed=True,
                 all_probes_converged=True, audited_per_run=fingerprint(table))
    # The bytes, rather than the original server's absolute path, identify a
    # report copied intact to another machine.
    audit['audited_per_run']['path'] = '/original/server/report/audited_per_run.csv'
    (tmp_path / 'audit.json').write_text(json.dumps(audit))
    return tmp_path, frame, audit


def test_candidate_report_binds_exact_bytes_and_retains_negative_comparisons(bound_candidate_report):
    root, expected, audit = bound_candidate_report
    actual, sources = load_candidate_replication(root)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.auprc.iloc[0] < actual.auprc.iloc[1]
    assert sources['candidate_table']['sha256'] == audit['audited_per_run']['sha256']
    assert sources['candidate_audit']['sha256'] == fingerprint(root / 'audit.json')['sha256']


@pytest.mark.parametrize('change', ['missing_digest', 'modified_table', 'wrong_size', 'incomplete'])
def test_candidate_report_rejects_unbound_or_changed_evidence(bound_candidate_report, change):
    root, _, audit = bound_candidate_report
    match = 'differs'
    if change == 'missing_digest':
        del audit['audited_per_run']
        match = 'fingerprint is missing; regenerate'
    elif change == 'modified_table':
        table = root / 'audited_per_run.csv'
        original = table.read_bytes()
        changed = original.replace(b'0.25', b'0.75')
        assert changed != original and len(changed) == len(original)
        table.write_bytes(changed)
    elif change == 'wrong_size':
        audit['audited_per_run']['bytes'] += 1
    else:
        audit['all_saved_predictions_replayed'] = False
        match = 'incomplete'
    (root / 'audit.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError, match=match):
        load_candidate_replication(root)
