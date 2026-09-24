"""Small-sample paired fold inference; seeds are not independent biological units."""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd


def fold_sign_flip(frame: pd.DataFrame, column: str = "gain") -> float:
    """Exact two-sided sign-flip test of seed-averaged fold differences.

    This is an exploratory symmetry test, not a claim that overlapping CV
    training sets are independent. With five folds, minimum two-sided p=1/16.
    """
    values = frame.groupby("fold")[column].mean().dropna().to_numpy()
    if len(values) < 2:
        return float("nan")
    if len(values) > 20:
        raise ValueError("Exact sign flips limited to 20 fold units")
    signs = np.array(list(itertools.product([-1, 1], repeat=len(values))))
    null = np.abs((signs * values).mean(axis=1))
    return float(np.mean(null >= abs(values.mean()) - 1e-12))


def bh_adjust(values) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    result = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return result
    if ((p[valid] < 0) | (p[valid] > 1)).any():
        raise ValueError("Invalid p-value")
    order = valid[np.argsort(p[valid], kind="stable")]
    adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
    result[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return result
