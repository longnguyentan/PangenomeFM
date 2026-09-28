import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from scripts.server import run_ccre_frozen_probe_fold as probe
from tasks.ccre.aligned_baselines import _fit_model


def test_default_fit_predictions_are_unchanged_and_increased_budget_is_explicit():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 5))
    y = (x[:, 0] + rng.normal(size=120) > 0).astype(int)
    original = _fit_model("logistic", 42).fit(x, y)
    actual, audit = probe.fit_logistic_probe(x, y, 42)
    np.testing.assert_array_equal(original.predict_proba(x), actual.predict_proba(x))
    assert audit["probe_max_iter"] == 800 and audit["probe_converged"]
    larger, receipt = probe.fit_logistic_probe(x, y, 42, 4000)
    np.testing.assert_array_equal(original.predict_proba(x), larger.predict_proba(x))
    assert receipt["probe_max_iter"] == 4000 and receipt["probe_converged"]
    with pytest.raises(ValueError, match="only increase"):
        probe.fit_logistic_probe(x, y, 42, 799)


def test_warning_is_recorded_instead_of_claiming_convergence(monkeypatch):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 8))
    y = (x[:, 0] + rng.normal(size=100) > 0).astype(int)
    monkeypatch.setattr(probe, "_fit_model", lambda _, seed: _fit_model("logistic", seed).set_params(
        logisticregression__max_iter=1))
    with pytest.warns(ConvergenceWarning):
        _, audit = probe.fit_logistic_probe(x, y, 42)
    assert not audit["probe_converged"] and audit["probe_iterations"] == 1
    assert "converge" in audit["probe_convergence_messages"]
