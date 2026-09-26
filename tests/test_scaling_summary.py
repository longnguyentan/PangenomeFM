import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from tasks.transfer.scaling_intrinsic import target_digest, validate_scales, summaries


def test_target_identity_includes_labels_but_not_predictions_or_order():
    frame = pd.DataFrame(
        dict(
            dataset=["hprc"] * 2,
            slice=["s"] * 2,
            target_sn=["chr1"] * 2,
            closure=["strict"] * 2,
            split=["heldout_chr_test"] * 2,
            u_local=[1, 1],
            v_local=[2, 3],
            y_true=[0, 1],
            p_edge=[0.2, 0.8],
        )
    )
    digest = target_digest(frame)
    assert digest == target_digest(frame.iloc[::-1].assign(p_edge=[0.1, 0.9]))
    assert digest != target_digest(frame.assign(y_true=[1, 0]))
    with pytest.raises(ValueError, match="Duplicate"):
        target_digest(pd.concat([frame, frame]))


def test_incomplete_or_changed_scaling_comparison_is_rejected():
    frame = pd.DataFrame(
        dict(
            fraction=[0.125, 0.25, 0.5, 1.0],
            fold=["fold_a"] * 4,
            seed=[42] * 4,
            context=["strict"] * 4,
            test_targets_sha256=["t"] * 4,
            validation_targets_sha256=["v"] * 4,
            arguments_sha256=["args"] * 4,
            n_parameters=[100] * 4,
            architecture_sha256=["architecture"] * 4,
            auprc=[0.6, 0.7, 0.8, 0.9],
            auroc=[0.6, 0.7, 0.8, 0.9],
            mean_window_auprc=[0.6, 0.7, 0.8, 0.9],
        )
    )
    validate_scales(frame)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_scales(frame.iloc[:-1])
    changed = frame.copy()
    changed.loc[0, "test_targets_sha256"] = "new_locus"
    with pytest.raises(ValueError, match="test_targets"):
        validate_scales(changed)
    _, pairs = summaries(frame, 20, 42)
    assert pairs.loc[pairs.fraction.eq(0.125), "mean"].to_numpy() == pytest.approx(
        [-0.3] * 3
    )
    assert pairs.ci95_low.isna().all()  # A single fold cannot provide a multi-fold CI.


def test_manuscript_bootstrap_preserves_window_weighted_estimand():
    path = Path(__file__).parents[1] / "manuscript/revision_20260924/build_evidence.py"
    spec = importlib.util.spec_from_file_location("revision_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame(
        dict(
            regime=["r"] * 4,
            dataset=["d"] * 4,
            closure=["strict"] * 4,
            chromosome=["chr1", "chr2"] * 2,
            seed=[42, 42, 314159, 314159],
            mean_auprc=[0.6, 0.9, 0.6, 0.9],
            n_slices=[1, 9, 1, 9],
        )
    )
    result = module.reconstruction_intervals(frame, n_bootstrap=200)
    assert result.auprc_mean.iloc[0] == pytest.approx(0.87)
    assert result.ci95_low.iloc[0] == pytest.approx(0.6)
    assert result.ci95_high.iloc[0] == pytest.approx(0.9)
    assert result.n_chromosomes.iloc[0] == 2
    with pytest.raises(ValueError, match="Duplicate"):
        module.reconstruction_intervals(pd.concat([frame, frame]))
    with pytest.raises(ValueError, match="Incomplete"):
        module.reconstruction_intervals(frame.iloc[:-1])
