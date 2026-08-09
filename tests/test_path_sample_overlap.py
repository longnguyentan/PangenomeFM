from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "server"
    / "audit_path_sample_overlap.py"
)
SPEC = importlib.util.spec_from_file_location("path_overlap", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_overlap_supports_both_metadata_schemas(tmp_path: Path) -> None:
    hprc = tmp_path / "hprc.tsv"
    hgsvc = tmp_path / "hgsvc.csv.gz"
    pd.DataFrame(
        {
            "SAMPLE": ["CHM13", "A", "A", "B"],
            "HAPLOTYPE": [0, 1, 2, 1],
            "LOCUS": ["chr1"] * 4,
        }
    ).to_csv(hprc, sep="\t", index=False)
    pd.DataFrame(
        {
            "sample": ["GRCh38", "B", "B", "C"],
            "haplotype": [0, 1, 2, 1],
            "contig": ["chr1"] * 4,
        }
    ).to_csv(hgsvc, index=False, compression="gzip")
    result = MODULE.audit({"hprc": hprc, "hgsvc": hgsvc}, tmp_path / "out")
    overlap = result["pairwise_overlaps"][0]
    assert overlap["shared_donors"] == 1
    assert overlap["shared_sample_ids"] == "B"
    assert pd.read_csv(tmp_path / "out/shared_samples.csv")["sample"].tolist() == ["B"]
