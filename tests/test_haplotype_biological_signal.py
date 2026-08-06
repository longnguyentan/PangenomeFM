import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_haplotype_biological_signal.py"
)
SPEC = importlib.util.spec_from_file_location("haplotype_signal", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_bootstrap_mean_is_reproducible_and_contains_mean():
    values = np.array([0.01, 0.02, 0.03, 0.04])
    first = MODULE._bootstrap_mean(values, seed=7, replicates=2_000)
    second = MODULE._bootstrap_mean(values, seed=7, replicates=2_000)
    assert first == second
    assert first[1] <= first[0] <= first[2]


def test_within_donor_top_decile_is_defined_per_donor():
    frame = pd.DataFrame(
        {
            "donor_id": ["a"] * 10 + ["b"] * 10,
            "methylation_delta": np.r_[np.arange(10), np.arange(10) / 100],
            "node_jaccard": np.linspace(0, 1, 20),
        }
    )
    mask = MODULE._define_strata(frame)["Within-donor top decile"]
    selected = frame.loc[mask]
    assert set(selected["donor_id"]) == {"a", "b"}
    assert len(selected) == 2
