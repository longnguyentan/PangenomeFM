"""Small train/validation-only ridge probes for continuous frozen-locus tasks."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def regression_metrics(y, prediction) -> dict:
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.shape != prediction.shape or len(y) < 2 or not np.isfinite([y, prediction]).all():
        raise ValueError("Require finite, aligned regression predictions")
    rho = float(spearmanr(y, prediction).statistic) if len(np.unique(y)) > 1 and len(np.unique(prediction)) > 1 else np.nan
    return dict(mae=float(mean_absolute_error(y, prediction)), rmse=float(np.sqrt(mean_squared_error(y, prediction))),
                r2=float(r2_score(y, prediction)), spearman=rho)


def select_ridge(x_train, y_train, x_val, y_val, alphas):
    """Return the training-only fit selected by validation MAE; never read test y."""
    if not alphas or any(a <= 0 or not np.isfinite(a) for a in alphas) or len(set(alphas)) != len(alphas):
        raise ValueError("Require distinct positive finite ridge penalties")
    candidates, models = [], {}
    for alpha in sorted(alphas):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="cholesky", fit_intercept=True))
        model.fit(x_train, y_train)
        prediction = model.predict(x_val)
        candidates.append(dict(alpha=alpha, validation_mae=float(mean_absolute_error(y_val, prediction))))
        models[alpha] = model
    # A predefined exact-tie preference favors stronger regularization.
    selected = min(candidates, key=lambda item: (item["validation_mae"], -item["alpha"]))
    return models[selected["alpha"]], selected, candidates
