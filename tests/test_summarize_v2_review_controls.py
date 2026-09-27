from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

from scripts.summarize_v2_review_controls import _feature_names, main


def test_summarize_v2_review_controls_writes_paired_deltas(tmp_path: Path, monkeypatch) -> None:
    input_root = tmp_path / "controls"
    for model in ("v1", "random"):
        for task in ("sv", "ccre"):
            names = _feature_names(task)
            for context in ("strict", "1hop"):
                rows = [
                    {"feature_set": feature, "auprc": 0.1 + i / 100, "auroc": 0.5 + i / 100}
                    for i, feature in enumerate(names.values())
                ]
                out = input_root / model / task / context
                out.mkdir(parents=True)
                pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

    summary_path = tmp_path / "summary.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_v2_review_controls", "--input-root", str(input_root), "--output", str(summary_path)],
    )
    assert main() == 0

    summary = pd.read_csv(summary_path)
    assert len(summary) == 8
    assert set(summary["model"]) == {"v1", "random"}
    np.testing.assert_allclose(summary["delta_t_given_cs"], 0.02)
    paired = pd.read_csv(tmp_path / "paired_model_comparison.csv")
    assert len(paired) == 4
