from __future__ import annotations

import pandas as pd

from scripts.server.analyze_canonical_affected_slice_sensitivity import affected_loci


def test_affected_loci_include_duplicates_and_conflicts() -> None:
    frame = pd.DataFrame(
        {
            "slice_id": ["a", "b", "c"],
            "chromosome": ["chr1", "chr1", "chr2"],
            "start": [0, 10, 0],
            "end": [10, 20, 10],
            "context": ["strict", "1hop", "strict"],
            "reverse_equivalent_candidate_duplicates": [1, 0, 0],
            "orientation_equivalent_candidate_label_conflicts": [0, 2, 0],
        }
    )
    result = affected_loci(frame)
    assert set(result["slice_id"]) == {"a", "b"}
    assert set(result["locus_id"]) == {"chr1:0-10", "chr1:10-20"}
