from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from scripts.server.audit_sv_truth import audit_vcf


def test_sv_truth_audit_normalizes_and_reports_phasing(tmp_path: Path) -> None:
    vcf = tmp_path / "tiny.vcf.gz"
    with gzip.open(vcf, "wt") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tD1\tD2\n")
        handle.write("chr1\t100\tdel1\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;SVLEN=-100;AF=0.25\tGT\t0|1\t0|0\n")
        handle.write("chr2\t200\tins1\tA\t" + "A" * 61 + "\t.\tPASS\t.\tGT:DP\t1|1:9\t./.:0\n")

    result = audit_vcf(vcf, tmp_path / "audit", dataset="test", release="v1")

    assert result["status"] == "validated_phased_sv_candidate"
    assert result["variant_count"] == 2
    assert result["sample_count"] == 2
    assert result["svtype_counts"] == {"DEL": 1, "INS": 1}
    variants = pd.read_csv(tmp_path / "audit" / "normalized_variants.csv.gz")
    assert variants["svlen"].tolist() == [100, 60]
    assert variants["n_phased_samples"].tolist() == [2, 1]
    samples = pd.read_csv(tmp_path / "audit" / "sample_phasing_summary.csv")
    d1 = samples.set_index("sample").loc["D1"]
    assert d1["called_records"] == 2
    assert d1["phased_records"] == 2
