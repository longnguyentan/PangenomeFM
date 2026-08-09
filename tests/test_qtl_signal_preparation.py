from pathlib import Path

import pandas as pd

from scripts.server.prepare_qtl_signals import stream_significant_qtls


def test_qtl_signal_preparation_streams_and_deduplicates(tmp_path: Path) -> None:
    source = tmp_path / "qtl.tsv.gz"
    pd.DataFrame(
        {
            "feature_id": ["a", "b", "c", "d"],
            "snp_id": ["rs1", "rs1", "rs2", "rs3"],
            "p_value": [1e-9, 2e-9, 1e-7, 1e-10],
            "empirical_feature_p_value": [0.01, 0.02, 0.01, 0.2],
            "snp_chromosome": [8, 8, 8, "X"],
            "snp_position": [101, 101, 202, 303],
        }
    ).to_csv(source, sep="\t", index=False, compression="gzip")

    signals, audit = stream_significant_qtls(
        source,
        qtl_type="eQTL",
        release="v1",
        p_threshold=5e-8,
        empirical_threshold=0.05,
        chunksize=2,
    )

    assert len(signals) == 1
    assert signals.loc[0, "signal_id"] == "eQTL:chr8:101"
    assert signals.loc[0, "start"] == 100
    assert signals.loc[0, "end"] == 101
    assert signals.loc[0, "n_associations"] == 2
    assert audit["rows_processed"] == 4
    assert audit["association_rows_passing"] == 2
