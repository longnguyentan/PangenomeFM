import numpy as np

from evaluation.calibration import apply_temperature, fit_temperature


def test_temperature_scaling_reduces_overconfidence() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    overconfident = np.array([0.01, 0.02, 0.90, 0.20, 0.80, 0.10, 0.98, 0.99])
    temperature = fit_temperature(labels, overconfident)
    calibrated = apply_temperature(overconfident, temperature)
    assert temperature > 1.0
    assert np.max(np.abs(calibrated - 0.5)) < np.max(
        np.abs(overconfident - 0.5)
    )


def test_temperature_one_is_identity() -> None:
    probabilities = np.array([0.1, 0.4, 0.9])
    assert np.allclose(apply_temperature(probabilities, 1.0), probabilities)
