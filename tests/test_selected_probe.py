import copy

import numpy as np
import pandas as pd
import pytest

from scripts.server.run_ccre_frozen_probe_fold import evaluate_feature_sets
from tasks.transfer.selected_probe import (choose_mixture, completion_summary, evaluate_selected,
                                          select_logistic, validate_selection)


def fixture():
    rng = np.random.default_rng(170)
    x = rng.normal(size=(150, 4)).astype(np.float32)
    y = np.tile([0, 0, 0, 0, 1], 30)
    outputs = {f'{name}_{kind}': dict(estimator=kind, input=name)
               for kind in ['linear', 'fixed', 'histgb'] for name in ['V', 'CS']}
    outputs['blend'] = dict(estimator='fusion', variant_input='V', locus_input='CS')
    plan = dict(probe_max_iter=4000, selected_probe=dict(C_grid=[.01, 1., 10.], mixture_weights=[0., .5, 1.],
        linear_inputs=['V', 'CS'], histgb_inputs=['V', 'CS'], outputs=outputs,
        histgb=dict(max_iter=5, max_leaf_nodes=3, min_samples_leaf=5, l2_regularization=1.)))
    args = dict(segids=np.arange(150), chromosomes=np.repeat(['chr1', 'chr2', 'chr3'], 50),
                labels=y, features={'V': x[:, :1], 'CS': x}, test_chrs={'chr1'}, val_chrs={'chr2'}, seed=42, plan=plan)
    return args


def test_fixed_c_replays_native_and_test_labels_do_not_select():
    args = fixture()
    metrics, _, pred, sweep, val = evaluate_selected(**args)
    native_args = {k: v for k, v in args.items() if k != 'plan'}
    native_args.update(probe_max_iter=4000, feature_access={'V': 'V', 'CS': 'C+S'})
    _, _, original = evaluate_feature_sets(**native_args)
    for name in ['V', 'CS']:
        for col in ['p_raw', 'p_calibrated', 'threshold', 'y_pred']:
            np.testing.assert_array_equal(pred.loc[pred.feature_set.eq(name+'_fixed'), col],
                                          original.loc[original.feature_set.eq(name), col])
    changed = dict(args, labels=args['labels'].copy())
    changed['labels'][:50] = 1 - changed['labels'][:50]
    updated, _, p2, s2, _ = evaluate_selected(**changed)
    pd.testing.assert_frame_equal(sweep, s2)
    np.testing.assert_array_equal(metrics.selected_parameter, updated.selected_parameter)
    np.testing.assert_array_equal(pred.p_raw, p2.p_raw)
    validate_selection(sweep, val.rename(columns={'segid': 'example_id'}), metrics, args['plan'])
    summary = completion_summary(metrics)
    assert summary['all_probes_completed'] and summary['all_linear_probes_converged']
    assert summary['all_probes_converged'] is None
    assert metrics.loc[metrics.probe_solver.eq('hist_gradient_boosting'), 'probe_converged'].isna().all()


def test_selection_and_mixture_tampering_is_rejected():
    args = fixture()
    metrics, _, _, sweep, val = evaluate_selected(**args)
    val = val.rename(columns={'segid': 'example_id'})
    bad = sweep.copy()
    bad.loc[0, 'selected'] = not bad.loc[0, 'selected']
    with pytest.raises(ValueError, match='validation rule'):
        validate_selection(bad, val, metrics, args['plan'])
    with pytest.raises(ValueError, match='grid'):
        validate_selection(sweep.iloc[1:], val, metrics, args['plan'])
    bad = val.copy()
    bad.loc[bad.feature_set.eq('blend'), 'p_raw'] = .5
    with pytest.raises(ValueError, match='AP differs|do not replay'):
        validate_selection(sweep, bad, metrics, args['plan'])


def test_endpoint_ties_and_invalid_grids():
    y = np.array([0, 1, 0, 1])
    selected, _ = choose_mixture(y, np.full(4, .3), np.full(4, .3), [0., .5, 1.])
    assert selected['value'] == 1.
    x = np.zeros((4, 1))
    _, chosen, _ = select_logistic(x, y, x, y, 42, [.01, 1., 10.], 4000)
    assert chosen['value'] == .01
    for grid in [[.5], [0, 0, 1], [0, 1, 2]]:
        with pytest.raises(ValueError, match='grid'):
            choose_mixture(y, y, y, grid)
    for grid in [[.1], [1., 1.], [0, 1], [1, np.nan]]:
        with pytest.raises(ValueError):
            select_logistic(x, y, x, y, 42, grid, 4000)


def test_nonfinite_and_partition_guards():
    args = fixture()
    bad = copy.deepcopy(args)
    bad['features']['CS'][0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        evaluate_selected(**bad)
    with pytest.raises(ValueError, match='disjoint'):
        evaluate_selected(**dict(args, val_chrs={'chr1'}))


def test_selected_report_replays_validation_and_test(tmp_path):
    import json
    from sklearn.metrics import balanced_accuracy_score
    from tasks.entex.prepare import fingerprint
    from tasks.transfer.traitgym import IDENTITY, weighted_chromosome_ap
    from tasks.transfer.traitgym_report import replay_run
    args = fixture()
    metrics, per_chr, predictions, selection, validation = evaluate_selected(**args)
    plan = dict(args['plan'], feature_sets=list(args['plan']['selected_probe']['outputs']),
                feature_max_iter={k: (5 if s['estimator'] == 'histgb' else 4000)
                    for k, s in args['plan']['selected_probe']['outputs'].items()})
    metrics['balanced_accuracy'] = [balanced_accuracy_score(p.y_true, p.y_pred) for f in metrics.feature_set
        for p in [predictions.loc[predictions.feature_set.eq(f)]]]
    metrics['chromosome_weighted_auprc'] = [weighted_chromosome_ap(per_chr.loc[per_chr.feature_set.eq(f)]) for f in metrics.feature_set]
    metrics['normalized_ap'] = (metrics.auprc-metrics.positive_fraction)/(1-metrics.positive_fraction)
    for frame in [predictions, validation]:
        frame.rename(columns={'segid': 'example_id'}, inplace=True)
        frame['variant_id'] = frame.example_id.astype(str)
        frame['locus_id'] = frame.variant_id
        frame['chrom'] = frame.chromosome
        frame['start'], frame['end'] = frame.example_id, frame.example_id+1
        frame['ref'], frame['alt'], frame['label'], frame['match_group'] = 'A', 'C', frame.y_true, 'synthetic'
        assert set(IDENTITY) <= set(frame)
    for frame in [metrics, predictions, validation]:
        for k, v in dict(task='test', dataset='complex_traits', fold='fold_a', seed=42, context='strict').items():
            frame[k] = v
    metrics.to_csv(tmp_path/'metrics.csv', index=False)
    predictions.to_parquet(tmp_path/'predictions.parquet', index=False)
    validation.to_parquet(tmp_path/'validation_predictions.parquet', index=False)
    selection.to_csv(tmp_path/'validation_selection.csv', index=False)
    audit = dict(status='complete', n_excluded=0)
    for key, name in [('predictions','predictions.parquet'),('validation_predictions','validation_predictions.parquet'),('validation_selection','validation_selection.csv')]:
        audit[key] = fingerprint(tmp_path/name)
    (tmp_path/'audit.json').write_text(json.dumps(audit))
    replayed, _ = replay_run(tmp_path, plan, ['chr1'], ['chr2'])
    assert len(replayed) == 7
    metrics.loc[metrics.feature_set.eq('CS_linear'), 'selected_parameter'] = 999
    metrics.to_csv(tmp_path/'metrics.csv', index=False)
    with pytest.raises(ValueError, match='Selected parameter'):
        replay_run(tmp_path, plan, ['chr1'], ['chr2'])


def test_mixture_precision_is_explicit_even_with_float32_inputs():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(412)
    y = np.tile([0, 1], 50)
    v = rng.uniform(size=100).astype(np.float32)
    b = rng.uniform(size=100).astype(np.float32)
    _, sweep = choose_mixture(y, v, b, [0., .25, .5, .75, 1.])
    for row in sweep:
        w = row['value']
        assert row['validation_auprc'] == average_precision_score(y, w*v.astype(np.float64)+(1-w)*b.astype(np.float64))
    args = fixture()
    _, _, predictions, _, _ = evaluate_selected(**args)
    assert predictions.p_raw.dtype == np.float64


def test_calibration_replay_allows_equivalent_optima_but_rejects_wrong_objective():
    from evaluation.calibration import apply_temperature, fit_temperature
    from tasks.ccre.binary import _choose_threshold
    from tasks.transfer.traitgym_report import validate_calibration
    rng = np.random.default_rng(421)
    raw = rng.uniform(.05, .95, size=500)
    y = rng.binomial(1, raw)
    fitted = fit_temperature(y, raw)
    temperature = fitted+1e-7
    p = apply_temperature(raw, temperature)
    check = validate_calibration(y, raw, p, temperature, _choose_threshold(y, p))
    assert abs(check['temperature_refit_delta']) > 1e-10
    wrong = apply_temperature(raw, fitted*2)
    with pytest.raises(ValueError, match='objective'):
        validate_calibration(y, raw, wrong, fitted*2, _choose_threshold(y, wrong))
    with pytest.raises(ValueError, match='does not replay'):
        validate_calibration(y, raw, p+.001, temperature, _choose_threshold(y, p))


def test_test_fusion_must_reuse_validation_selected_weight():
    from tasks.transfer.traitgym_report import validate_test_fusion
    plan = fixture()['plan']
    frame = pd.DataFrame([dict(variant_id='a', feature_set=f, p_raw=p)
                          for f, p in [('V_linear', .2), ('CS_linear', .8), ('blend', .5)]])
    metrics = pd.DataFrame([dict(feature_set='blend', selected_parameter=.5)])
    validate_test_fusion(frame, metrics, plan)
    frame.loc[frame.feature_set.eq('blend'), 'p_raw'] = .2
    with pytest.raises(ValueError, match='validation-selected mixture'):
        validate_test_fusion(frame, metrics, plan)
