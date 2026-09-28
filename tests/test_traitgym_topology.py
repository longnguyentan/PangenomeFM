import numpy as np
import pandas as pd
import pytest

from scripts.server.audit_traitgym_topology import coverage


def examples():
    return pd.DataFrame(dict(variant_id=["a:C:T", "a:C:A", "b:G:A"], locus_id=["a", "a", "b"],
                             chrom=["chr1"] * 3, label=[1, 1, 0]))


def test_actual_coverage_keeps_distinct_alleles_and_class_dependent_missingness():
    overlaps = pd.DataFrame(dict(locus_id=["a", "b"], segid=[10, 20]))
    result = coverage(examples(), overlaps, np.array([10])).set_index("group")
    assert result.loc["all", "n"] == 3 and result.loc["all", "n_covered"] == 2
    assert result.loc["label=1", "coverage"] == 1 and result.loc["label=0", "coverage"] == 0


def test_topology_coverage_rejects_ambiguous_or_missing_mapping():
    overlaps = pd.DataFrame(dict(locus_id=["a", "b"], segid=[10, 20]))
    for bad in [overlaps.iloc[:1], pd.concat([overlaps, overlaps.iloc[:1]])]:
        with pytest.raises(ValueError, match="one mapped segment"):
            coverage(examples(), bad, np.array([10, 20]))
    with pytest.raises(ValueError, match="Duplicate"):
        coverage(examples(), overlaps, np.array([10, 10]))
