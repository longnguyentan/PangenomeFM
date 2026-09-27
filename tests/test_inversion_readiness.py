import pandas as pd
import pytest
from scripts.server.audit_inversion_readiness import inversion_intervals


def test_inversion_bed_contract_does_not_copy_vcf_anchor_or_relabel_as_deletion():
    d = pd.DataFrame({"ID": ["chr1-101-INV-50"], "#CHROM": ["chr1"], "POS": [100],
                      "END": [150], "SVTYPE": ["INV"], "SVLEN": [50]})
    result = inversion_intervals(d)
    assert result.start0.iloc[0] == 100 and result.end0.iloc[0] == 150
    assert result.SVTYPE.iloc[0] == "INV" and "binary_svtype_label" not in result
    for field, value in [("POS", 99), ("ID", "chr1-100-INV-50"), ("SVTYPE", "DEL")]:
        bad = d.copy()
        bad.loc[0, field] = value
        with pytest.raises(ValueError):
            inversion_intervals(bad)
