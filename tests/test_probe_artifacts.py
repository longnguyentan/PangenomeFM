import joblib
import numpy as np
import pandas as pd
import pytest

from evaluation.calibration import apply_temperature
from evaluation.probe_artifacts import persist_fitted_probes
from scripts.server.run_ccre_frozen_probe_fold import evaluate_feature_sets


@pytest.mark.parametrize('validation_only', [False, True])
def test_saved_native_probes_replay_without_changing_default_predictions(tmp_path, validation_only):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(180, 5))
    labels = (x[:, 0] + rng.normal(size=180) > 0).astype(np.int8)
    features = {'cs': x, 'random_uniform': x, 'training_prevalence': x}
    kwargs = dict(segids=np.arange(180), chromosomes=np.repeat(['chr1', 'chr2', 'chr3'], 60),
                  labels=labels, features=features, test_chrs={'chr3'}, val_chrs={'chr2'},
                  seed=42, feature_access={k: k for k in features}, validation_only=validation_only)
    original = evaluate_feature_sets(**kwargs)
    fitted = {}
    actual = evaluate_feature_sets(**kwargs, fitted_probes=fitted)
    for expected, observed in zip(original, actual):
        pd.testing.assert_frame_equal(expected, observed, check_exact=True)
    assert set(fitted) == {'cs'}  # synthetic controls never inherit the previous fitted model
    receipt = persist_fitted_probes(fitted, features, tmp_path / 'probes', {'checkpoint_sha256': 'fixture'})
    restored = joblib.load(receipt['cs']['path'])
    predictions = actual[2].query("feature_set == 'cs'")
    raw = restored['model'].predict_proba(x[fitted['cs']['evaluation_mask']])[:, 1]
    np.testing.assert_array_equal(raw, predictions.p_raw)
    calibrated = apply_temperature(raw, restored['temperature'])
    np.testing.assert_array_equal(calibrated, predictions.p_calibrated)
    np.testing.assert_array_equal(calibrated >= restored['threshold'], predictions.y_pred)
    assert restored['feature_dimension'] == 5
    assert restored['training_chromosomes'] == ['chr1']
    assert restored['calibration_partition'] == 'validation'
    assert restored['checkpoint_sha256'] == 'fixture'
    assert receipt['cs']['raw_predictions_exact_after_reload']
    with pytest.raises(FileExistsError):
        persist_fitted_probes(fitted, features, tmp_path / 'probes', {})
