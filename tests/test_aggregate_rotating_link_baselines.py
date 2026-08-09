from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "server"
    / "aggregate_rotating_link_baselines.py"
)
SPEC = importlib.util.spec_from_file_location("aggregate_link_baselines", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_hierarchical_baseline_summary() -> None:
    rows = []
    for fold, value in (("fold_a", 0.6), ("fold_b", 0.8)):
        for seed in (1, 2):
            rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "closure": "strict",
                    "baseline": "coordinate_logistic",
                    **{metric: value for metric in MODULE.METRICS},
                }
            )
    summary = MODULE.hierarchical_summary(pd.DataFrame(rows), n_bootstrap=100, seed=4)
    auprc = summary.loc[summary["metric"].eq("auprc")].iloc[0]
    assert np.isclose(auprc["mean"], 0.7)
    assert auprc["n_folds"] == 2
    assert auprc["n_seeds"] == 2
