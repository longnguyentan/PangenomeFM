import numpy as np
import pandas as pd
import pytest

from tasks.transfer.genotyping_fallback import choose_ridge, derive_run


def test_fallback_rule_ties_and_invalid_validation_errors():
    assert choose_ridge(.1, .2)
    assert not choose_ridge(.2, .1)
    assert not choose_ridge(.1, .1)
    for invalid in [np.nan, np.inf, -.1]:
        with pytest.raises(ValueError, match='Validation MAE'):
            choose_ridge(invalid, .1)


def test_fallback_selection_does_not_read_test_outcomes_and_replays_median():
    loci = pd.DataFrame(dict(chrom=['chr1']*2+['chr2']*2+['chr3']*2, q=[1., 3., 2., 4., 5., 6.]))
    job = dict(fold='a', seed=42, closure='strict', validation=['chr2'], test=['chr3'])
    metrics = pd.DataFrame([dict(target='q', feature_set=f, alpha=a, validation_mae=v)
                            for f, a, v in [('train_median', np.nan, 1.), ('ridge_a', 1., .9), ('ridge_b', 1., 1.)]])
    predictions = pd.DataFrame([dict(target='q', feature_set=f, locus_id=locus, y_true=y, prediction=p)
                                for f, preds in [('train_median', [2., 2.]), ('ridge_a', [4., 6.]), ('ridge_b', [5., 6.])]
                                for locus, y, p in zip(['e', 'f'], [5., 6.], preds)])
    _, output, choice = derive_run(metrics, predictions, loci, job)
    assert choice.selected_feature.tolist() == ['ridge_a', 'train_median']
    np.testing.assert_array_equal(output.loc[output.feature_set.eq('fallback__ridge_b'), 'prediction'], [2., 2.])
    changed = predictions.assign(y_true=-predictions.y_true)
    loci.loc[loci.chrom.eq('chr3'), 'q'] = [-5., -6.]
    _, _, second = derive_run(metrics, changed, loci, job)
    pd.testing.assert_frame_equal(choice, second)
    metrics.loc[0, 'validation_mae'] = .5
    with pytest.raises(ValueError, match='does not replay'):
        derive_run(metrics, predictions, loci, job)
