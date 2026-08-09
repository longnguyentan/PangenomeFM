from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "server"
    / "build_path_resolved_donor_splits.py"
)
SPEC = importlib.util.spec_from_file_location("path_donor_splits", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_donor_split_excludes_shared_samples_and_keeps_haplotypes_together(tmp_path: Path) -> None:
    path = tmp_path / "paths.tsv"
    rows = []
    for donor in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        for haplotype in (1, 2):
            rows.append(
                {
                    "#NAME": f"{donor}#{haplotype}#ctg",
                    "SENSE": "HAPLOTYPE",
                    "SAMPLE": donor,
                    "HAPLOTYPE": haplotype,
                    "LOCUS": "ctg",
                }
            )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    result = MODULE.build(
        path_metadata=path,
        dataset="toy",
        out_dir=tmp_path / "out",
        seed=42,
        train_fraction=0.7,
        validation_fraction=0.15,
        exclude_samples={"B"},
    )
    donors = pd.read_csv(tmp_path / "out/donor_splits.csv")
    paths = pd.read_csv(tmp_path / "out/path_records.csv.gz")
    assert "B" not in set(donors["sample"])
    assert donors["sample"].nunique() == 9
    assert paths.groupby("sample")["donor_split"].nunique().max() == 1
    assert result["status"] == "manifest_complete_examples_not_materialized"
