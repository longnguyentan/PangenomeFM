from pathlib import Path

import numpy as np
import pandas as pd

from analysis.qtl_overlap import (
    exact_spearman_permutation,
    load_context_windows,
    stream_qtl_overlaps,
)


def test_load_context_windows_averages_seeds(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        {
            "interval_id": ["slice_GRCh38_0_chr8_100_200"],
            "target_sn": ["GRCh38#0#chr8"],
            "core_candidates": [10],
            "expanded_candidates": [20],
            "core_auroc": [0.5],
            "expanded_auroc": [0.9],
        }
    )
    first = tmp_path / "seed1.csv"
    second = tmp_path / "seed2.csv"
    rows.to_csv(first, index=False)
    rows.assign(core_auroc=0.7, expanded_auroc=0.8).to_csv(second, index=False)

    result = load_context_windows([first, second], "chr8")

    assert result.loc[0, "n_seeds"] == 2
    assert result.loc[0, "start"] == 100
    assert np.isclose(result.loc[0, "context_auroc_gain"], 0.25)


def test_stream_qtl_overlaps_uses_coordinate_rule(tmp_path: Path) -> None:
    qtl_path = tmp_path / "qtl.tsv.gz"
    pd.DataFrame(
        {
            "feature_id": ["outside_left", "inside", "inside_end", "outside_right"],
            "snp_id": ["a", "b", "c", "d"],
            "p_value": [1e-9, 1e-9, 1e-6, 1e-9],
            "beta": [1.0, 2.0, 3.0, 4.0],
            "empirical_feature_p_value": [0.01, 0.01, 0.2, 0.01],
            "snp_chromosome": [8, 8, 8, 8],
            "snp_position": [100, 101, 200, 201],
        }
    ).to_csv(qtl_path, sep="\t", index=False, compression="gzip")
    windows = pd.DataFrame(
        {
            "interval_id": ["window"],
            "start": [100],
            "end": [200],
            "window_bp": [100],
        }
    )

    summary, top, processed = stream_qtl_overlaps(
        qtl_path, windows, qtl_type="eQTL", chunksize=2
    )

    assert processed == 4
    assert summary.loc[0, "association_rows"] == 2
    assert summary.loc[0, "genomewide_significant_variant_positions"] == 1
    assert top.iloc[0]["feature_id"] == "inside"


def test_exact_spearman_permutation_detects_perfect_order() -> None:
    rho, p_value, n = exact_spearman_permutation(
        pd.Series([1, 2, 3, 4]), pd.Series([10, 20, 30, 40])
    )
    assert rho == 1.0
    assert np.isclose(p_value, 2 / 24)
    assert n == 4

