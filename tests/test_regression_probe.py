import numpy as np
import pytest
from tasks.transfer.regression_probe import regression_metrics, select_ridge


def test_selection_uses_only_training_scaling_and_validation_error():
    x = np.arange(10.)[:, None]
    y = 2*x[:, 0]+3
    val = np.array([[100.], [101.]])
    model, choice, sweep = select_ridge(x, y, val, 2*val[:, 0]+3, [.01, 1, 100])
    np.testing.assert_equal(model[0].mean_, [4.5])
    assert choice['alpha'] == .01
    assert choice['validation_mae'] == min(row['validation_mae'] for row in sweep)
    assert regression_metrics(y, model.predict(x))['mae'] < .02


def test_strongest_penalty_wins_exact_tie_and_constant_baseline_is_explicit():
    x = np.zeros((10, 1))
    model, choice, _ = select_ridge(x, np.ones(10), x[:2], np.ones(2), [.01, 1, 100])
    assert choice['alpha'] == 100
    metrics = regression_metrics([0, 1], model.predict(x[:2]))
    assert np.isnan(metrics['spearman']) and metrics['mae'] == .5


@pytest.mark.parametrize('grid', [[], [0], [-1], [np.nan], [1, 1]])
def test_invalid_penalty_grid_rejected(grid):
    with pytest.raises(ValueError, match='penalties'):
        select_ridge(np.ones((2, 1)), np.ones(2), np.ones((2, 1)), np.ones(2), grid)
