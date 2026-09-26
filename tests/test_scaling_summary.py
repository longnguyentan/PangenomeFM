import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from tasks.transfer.scaling_intrinsic import target_digest, validate_scales, summaries
from tasks.transfer.scaling_bio_summary import (
    FEATURE_ALIASES,
    canonical_feature,
    balanced_accuracy,
    validate,
)


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


def test_biological_scaling_summary_requires_all_paired_feature_sets():
    assert canonical_feature("coordinate_plus_frozen_sequence_fm_pair") == "C+S"
    rows = []
    for feature in FEATURE_ALIASES.values():
        rows.append(
            dict(
                fraction=0.125,
                task="sv",
                fold="fold_a",
                seed=42,
                context="strict",
                feature_set=feature,
                n_test=10,
                n_train=30,
                n_val=10,
                positive_prevalence=0.5,
                auprc=0.6,
                auroc=0.7,
                balanced_accuracy=0.6,
                f1=0.6,
                precision=0.6,
                recall=0.6,
            )
        )
    frame = pd.DataFrame(rows)
    matrix = dict(
        fractions=(0.125,),
        tasks=("sv",),
        folds=("fold_a",),
        seeds=(42,),
        contexts=("strict",),
    )
    validate(frame, **matrix)
    with pytest.raises(ValueError, match="Incomplete"):
        validate(frame.iloc[:-1], **matrix)
    with pytest.raises(ValueError, match="Incomplete"):
        validate(frame)  # Entire missing tasks, fractions and runs must fail too.
    with pytest.raises(ValueError, match="unexpected"):
        validate(frame.assign(fold="fold_z"), **matrix)
    frame.loc[0, "n_test"] = 9
    with pytest.raises(ValueError, match="universe"):
        validate(frame, **matrix)


def test_balanced_accuracy_is_not_ordinary_accuracy():
    assert balanced_accuracy(
        dict(positive_fraction=0.1, recall=0, accuracy=0.9)
    ) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="disagrees"):
        balanced_accuracy(
            dict(positive_fraction=0.1, recall=0, accuracy=0.9, balanced_accuracy=0.9)
        )


def test_prediction_digest_rejects_changed_labels_or_equal_size_changed_loci():
    from tasks.transfer.scaling_prediction_audit import target_digest as digest

    frame = pd.DataFrame(dict(id=[1, 2], chrom=["chr1", "chr1"], y_true=[0, 1]))
    expected = digest(frame, "id", "chrom")
    assert digest(frame.iloc[::-1], "id", "chrom") == expected
    assert digest(frame.assign(id=[1, 3]), "id", "chrom") != expected
    assert digest(frame.assign(y_true=[1, 0]), "id", "chrom") != expected


def test_biological_scaling_contrasts_pair_the_same_fold_and_seed():
    from tasks.transfer.scaling_bio_report import contrasts

    rows = []
    for fold, baseline in [("fold_a", 0.5), ("fold_b", 0.8)]:
        for fraction in [0.125, 0.25, 0.5, 1.0]:
            rows.append(
                dict(
                    task="sv",
                    context="strict",
                    fold=fold,
                    seed=42,
                    feature_set="C+S+T",
                    fraction=fraction,
                    auprc=baseline + fraction / 10,
                    auroc=baseline + fraction / 10,
                )
            )
    frame = pd.DataFrame(rows)
    result = contrasts(frame, 20, 42)
    assert result.loc[result.fraction.eq(0.125), "mean"].tolist() == pytest.approx(
        [0.0875, 0.0875]
    )
    with pytest.raises(ValueError, match="Unpaired"):
        contrasts(frame.iloc[:-1], 20, 42)
