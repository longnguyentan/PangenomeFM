import pandas as pd
import pytest
from scripts.server.audit_traitgym_coverage import normalize, audit


def test_official_rows_labels_boundaries_and_incomplete_cache_are_retained():
    raw = pd.DataFrame(dict(chrom=["1"] * 3, pos=[1, 2, 3], ref=["A"] * 3,
                            alt=["G"] * 3, label=[True, False, False], match_group=["a"] * 3))
    frame = normalize(raw)
    nodes = pd.DataFrame(dict(segid=[5, 6], chrom=["chr1"] * 2, SO=[0, 1], LN=[1, 1], end0=[1, 2]))
    output, overlaps, coverage = audit(frame, nodes, {"S": {5}})
    assert output.n_segments.tolist() == [1, 1, 0]
    assert output.S_covered.tolist() == [True, False, False]
    assert output.label.tolist() == [1, 0, 0] and len(output) == 3
    assert overlaps.segid.tolist() == [5, 6]
    assert coverage.set_index("group").loc["all", "graph_mapped"] == pytest.approx(2 / 3)
    for field, value in [("pos", 0), ("pos", 1.5), ("label", 2), ("alt", "AC"), ("chrom", "scaffold1")]:
        bad = raw.copy()
        bad[field] = bad[field].astype(object)
        bad.loc[0, field] = value
        with pytest.raises(ValueError):
            normalize(bad)


def test_reference_verification_uses_half_open_mapping_and_preserves_mismatches():
    from scripts.server.audit_traitgym_coverage import verify_reference
    examples = pd.DataFrame(dict(dataset=["task"] * 2, segid=[5, 5], start=[10, 11],
                                 ref=["A", "T"], variant_id=["a", "b"]))
    chunks = [pd.DataFrame(dict(id=[5], seq=["AC"], SO=[10], LN=[2]))]
    checked, samples = verify_reference(examples, chunks)
    assert checked.matches.tolist() == [True, False]
    assert samples[0]["sequence"] == "AC" and samples[0]["variant_position_in_record"] == 1
