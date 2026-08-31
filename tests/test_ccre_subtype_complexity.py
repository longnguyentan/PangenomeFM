from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.server.analyze_ccre_subtype_complexity import (
    BASELINE_FEATURE,
    FULL_FEATURE,
    prepare_labels,
    run_metrics,
    summarize,
)


def synthetic_labels(tmp_path: Path) -> tuple[Path, Path]:
    labels = pd.DataFrame(
        {
            "segid": np.arange(18),
            "chrom": ["chr1"] * 6 + ["chr2"] * 6 + ["chr3"] * 6,
            "SO": list(range(6)) * 3,
            "ccre_class": [
                "PLS",
                "pELS",
                "dELS",
                "CA-CTCF",
                "background",
                "background",
            ]
            * 3,
            "ccre_label": [1, 1, 1, 1, 0, 0] * 3,
        }
    )
    labels_path = tmp_path / "labels.csv.gz"
    labels.to_csv(labels_path, index=False, compression="gzip")
    complexity = pd.DataFrame(
        {
            "chromosome": ["chr1", "chr2", "chr3"],
            "start": [0, 0, 0],
            "end": [10, 10, 10],
            "context": ["strict"] * 3,
            "locus_complexity_score": [-1.0, 0.0, 1.0],
            "locus_complexity_category": ["low", "medium", "high"],
        }
    )
    complexity_path = tmp_path / "complexity.tsv"
    complexity.to_csv(complexity_path, sep="\t", index=False)
    return labels_path, complexity_path


def prediction_frame(fold: str, seed: int, closure: str) -> pd.DataFrame:
    rows = []
    y = np.array([1, 1, 1, 1, 0, 0] * 3)
    baseline = np.array([0.65, 0.62, 0.58, 0.56, 0.40, 0.35] * 3)
    full = np.clip(baseline + np.array([0.08, 0.07, 0.06, 0.05, -0.03, -0.02] * 3), 0, 1)
    for feature_set, scores in [(BASELINE_FEATURE, baseline), (FULL_FEATURE, full)]:
        for segid, label, score in zip(range(18), y, scores):
            rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "closure": closure,
                    "segid": segid,
                    "chromosome": f"chr{segid // 6 + 1}",
                    "y_true": label,
                    "feature_set": feature_set,
                    "p_calibrated": score,
                }
            )
    return pd.DataFrame(rows)


def test_ccre_subtype_and_complexity_are_independently_defined(tmp_path: Path) -> None:
    labels_path, complexity_path = synthetic_labels(tmp_path)
    labels, audit = prepare_labels(labels_path, complexity_path)
    assert audit["strict_complexity_window_counts"] == {"low": 1, "medium": 1, "high": 1}
    assert labels["complexity_category"].notna().all()

    rows = []
    for fold in ["fold_a", "fold_b"]:
        for seed in [42, 314159]:
            source = tmp_path / f"{fold}_{seed}.csv.gz"
            rows.extend(run_metrics(prediction_frame(fold, seed, "strict"), labels, source))
    metrics = pd.DataFrame(rows)
    pls = metrics.loc[
        (metrics["stratification"] == "ENCODE_cCRE_subtype_vs_background")
        & (metrics["stratum"] == "PLS")
    ]
    # Per run: three PLS positives plus six shared background examples; the
    # other positive cCRE classes are excluded from this one-vs-background row.
    assert set(pls["n_positive"]) == {3}
    assert set(pls["n_negative"]) == {6}
    assert set(pls["n_examples"]) == {9}

    absolute, gains, interactions = summarize(metrics, n_bootstrap=200, seed=7)
    assert not absolute.empty
    assert set(gains["stratum"]) == {"PLS", "pELS", "dELS", "CTCF-only", "low", "medium", "high"}
    assert (gains["mean_gain"] >= 0).all()
    assert len(interactions) == 9
    assert set(interactions["interaction"]) == {"difference_in_topology_gain"}
