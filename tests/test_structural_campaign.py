import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.paired_inference import bh_adjust, fold_sign_flip
from tasks.transfer.scaling import nested_manifests
from tasks.transfer.sv_strata import annotate, run_metrics


def test_nested_scaling_preserves_all_heldout_rows_and_context_pairs():
    rows = []
    for chrom in ["chr1", "chr2", "chr3", "chr4"]:
        for position in range(16):
            for context in ["strict", "1hop"]:
                rows.append(
                    dict(
                        name=f"{chrom}_{position}_{context}",
                        target_sn=f"GRCh38#0#{chrom}",
                        start=position * 10,
                        end=(position + 1) * 10,
                        closure=context,
                    )
                )
    original = pd.DataFrame(rows)
    fold = dict(test=["chr1"], validation=["chr2"])
    first = nested_manifests(original, fold)
    second = nested_manifests(original.sample(frac=1, random_state=1), fold)
    previous = set()
    for fraction, frame in first.items():
        assert set(frame.name) == set(second[fraction].name)
        assert previous.issubset(set(frame.name))
        assert frame.groupby(["target_sn", "start"]).closure.nunique().eq(2).all()
        assert len(frame.loc[frame.target_sn.str.endswith(("chr1", "chr2"))]) == 64
        assert len(frame.loc[frame.target_sn.str.endswith("chr3")]) == 32 * fraction
        previous = set(frame.name)
    pd.testing.assert_frame_equal(first[1.0], original)


def test_scaling_rejects_leaking_fold():
    frame = pd.DataFrame(
        dict(name=["a"], target_sn=["chr1"], start=[0], end=[10], closure=["strict"])
    )
    with pytest.raises(ValueError, match="Overlapping"):
        nested_manifests(frame, dict(test=["chr1"], validation=["chr1"]))


def test_exact_fold_signflip_and_bh():
    frame = pd.DataFrame(dict(fold=np.repeat(list("abcde"), 3), gain=[0.1] * 15))
    assert fold_sign_flip(frame) == 0.0625
    # Repeating seeds does not pretend to increase the number of independent folds.
    assert fold_sign_flip(pd.concat([frame] * 2)) == 0.0625
    np.testing.assert_allclose(
        bh_adjust([0.01, 0.04, 0.03, np.nan])[:3], [0.03, 0.04, 0.04]
    )
    assert np.isnan(bh_adjust([0.01, 0.04, 0.03, np.nan])[-1])


def test_sv_class_is_target_not_a_discrimination_stratum(tmp_path):
    path = tmp_path / "complexity.tsv"
    pd.DataFrame(
        dict(
            context=["strict"] * 2,
            chromosome=["chr1"] * 2,
            start=[0, 10],
            end=[10, 20],
            locus_complexity_category=["low", "high"],
        )
    ).to_csv(path, sep="\t", index=False)
    raw = pd.DataFrame(
        dict(
            example_id=[0, 1],
            chrom=["chr1"] * 2,
            start0=[1, 2],
            end0=[11, 3],
            svtype=["DEL", "INS"],
            binary_svtype_label=[0, 1],
            alt_allele_count=[1, 2],
            length_bin=["bp[50,100)"] * 2,
            af_bin=["af[0,.01)"] * 2,
        )
    )
    meta = annotate(raw, path)
    assert meta.complexity.tolist() == ["high", "low"]
    pred = pd.DataFrame(
        dict(
            example_id=[0, 1],
            fold=["fold_a"] * 2,
            seed=[42] * 2,
            closure=["strict"] * 2,
            feature_set=["C+S"] * 2,
            y_true=[0, 1],
            p_calibrated=[0.1, 0.9],
            threshold=[0.5] * 2,
            y_pred=[0, 1],
        )
    )
    cfg = json.loads(Path("configs/structural_mechanism_v1.json").read_text())
    metrics = run_metrics(pred, meta, cfg)
    by_type = metrics.loc[metrics.family.eq("sv_type")]
    assert by_type.auprc.isna().all() and by_type.auroc.isna().all()
    assert by_type.class_recall.eq(1).all()
    assert not by_type.sufficiently_powered.any()
