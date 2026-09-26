import numpy as np
import pandas as pd
import pytest

from tasks.entex.measurement_followups import (
    eligibility,
    identity_digest,
    locus_weights,
    scores,
)


def test_equal_locus_weight_changes_estimand_without_relabeling():
    frame = pd.DataFrame(
        dict(
            measurement_id=list("abcde"),
            locus_id=["A"] * 3 + ["B", "C"],
            y_true=[1, 1, 1, 0, 1],
            p_calibrated=[0.2, 0.2, 0.2, 0.9, 0.8],
        )
    )
    weights = locus_weights(frame)
    np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3, 1, 1])
    result = scores(frame, equal_locus=True)
    assert result["positive_prevalence"] == pytest.approx(2 / 3)
    assert result["auprc"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert scores(frame)["auprc"] == pytest.approx((0.5 + 3 * 0.8) / 4)
    with pytest.raises(ValueError, match="Repeated"):
        locus_weights(pd.concat([frame, frame.iloc[:1]]))


def test_strata_need_power_in_every_fold_and_do_not_use_scores():
    counts = pd.DataFrame(
        dict(
            stratum=["tissue"] * 3,
            group=["a", "a", "b"],
            fold=["fold_a", "fold_b", "fold_a"],
            n=[200, 300, 1000],
            n_positive=[10, 20, 500],
        )
    )
    protocol = dict(
        minimum_measurements_per_test_fold=200,
        minimum_positive_per_test_fold=10,
        minimum_negative_per_test_fold=10,
    )
    result = eligibility(counts, ["fold_a", "fold_b"], protocol).set_index("group")
    assert result.loc["a", "eligible"]
    assert not result.loc["b", "eligible"]
    assert result.loc["b", "min_fold_n"] == 0


def test_metadata_pairing_detects_donor_changes_and_not_row_order():
    frame = pd.DataFrame(
        dict(
            measurement_id=["a", "b"],
            locus_id=["L", "L"],
            chrom=["chr1"] * 2,
            y_true=[0, 1],
            donor=["d1", "d2"],
            tissue=["t"] * 2,
        )
    )
    assert identity_digest(frame) == identity_digest(frame.iloc[::-1])
    assert identity_digest(frame) != identity_digest(frame.assign(donor="d1"))
