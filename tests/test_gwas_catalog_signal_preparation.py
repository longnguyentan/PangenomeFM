from pathlib import Path

import pandas as pd

from scripts.server.prepare_gwas_catalog_signals import normalize_catalog


def test_gwas_catalog_normalization_filters_and_reports_ambiguity(tmp_path: Path) -> None:
    source = tmp_path / "catalog.tsv"
    # Deliberately use a different file-column order than the normalizer's
    # requested order; pandas ``usecols`` otherwise preserves source order.
    pd.DataFrame(
        {
            "SNPS": ["rs1", "rs2", "rs3", "rs4"],
            "MAPPED_TRAIT": ["Parkinson disease", "height", "Parkinson disease", "Parkinson disease"],
            "CHR_ID": ["1", "2;3", "X", "4;5"],
            "STUDY ACCESSION": ["GCST1", "GCST2", "GCST3", "GCST4"],
            "CHR_POS": ["101", "201;301", "401", "501"],
            "P-VALUE": [1e-9, 2e-9, 1e-6, 3e-9],
        }
    ).to_csv(source, sep="\t", index=False)

    signals, exclusions, audit = normalize_catalog(
        source,
        snapshot_date="2026-08-09",
        genome_build="GRCh38",
        p_threshold=5e-8,
        trait_regex="Parkinson",
        chunksize=2,
    )

    assert signals["signal_id"].tolist() == ["GWAS:chr1:101"]
    assert signals.loc[0, "start"] == 100
    assert exclusions.set_index("exclusion_reason").loc[
        "ambiguous_coordinate_cardinality", "rows"
    ] == 1
    assert audit["rows_processed"] == 4
    assert audit["unique_signal_positions"] == 1
