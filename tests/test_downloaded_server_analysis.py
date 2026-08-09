from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "server"
    / "analyze_downloaded_server_results.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_downloaded_server_results", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_chromosome_normalization_preserves_two_digit_autosomes() -> None:
    assert MODULE.normalize_chromosome("GRCh38#0#chr1") == "chr1"
    assert MODULE.normalize_chromosome("GRCh38#0#chr10") == "chr10"
    assert MODULE.normalize_chromosome("id=CHM13|chr19") == "chr19"
    assert MODULE.normalize_chromosome("slice_chr22_0_5000000") == "chr22"
    assert MODULE.normalize_chromosome("chrX") == "chrX"
    assert MODULE.normalize_chromosome("chrY") == "chrY"


def test_artifact_manifest_recursively_checksums_outputs(tmp_path: Path) -> None:
    (tmp_path / "top.txt").write_text("top", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "result.csv").write_text("x\n1\n", encoding="utf-8")

    MODULE.write_artifact_manifest(tmp_path)

    rows = (tmp_path / "artifact_manifest.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert rows[0] == "file\tbytes\tsha256"
    assert rows[1].startswith("nested/result.csv\t")
    assert rows[2].startswith("top.txt\t")
    assert all("artifact_manifest.tsv" not in row for row in rows[1:])
