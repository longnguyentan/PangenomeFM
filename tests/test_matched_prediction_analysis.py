import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_matched_predictions.py"
SPEC = importlib.util.spec_from_file_location("compare_matched_predictions", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_delong_auc_matches_sklearn() -> None:
    labels = np.array([0, 1, 0, 1, 1, 0])
    first = np.array([0.1, 0.8, 0.2, 0.7, 0.9, 0.3])
    second = np.array([0.2, 0.6, 0.5, 0.4, 0.8, 0.1])
    aucs, covariance = MODULE._delong_covariance(
        np.vstack([first, second]), labels
    )
    assert np.allclose(
        aucs,
        [roc_auc_score(labels, first), roc_auc_score(labels, second)],
    )
    assert covariance.shape == (2, 2)
