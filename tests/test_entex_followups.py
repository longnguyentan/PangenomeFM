import numpy as np
import pandas as pd
import pytest
import json
from itertools import product

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


def test_followup_reproduces_manuscript_clipping_at_extreme_probabilities():
    from scripts.server.run_ccre_frozen_probe_fold import binary_metrics

    frame = pd.DataFrame(
        dict(
            measurement_id=["a", "b", "c", "d"],
            locus_id=["a", "b", "c", "d"],
            y_true=[1, 0, 1, 0],
            p_calibrated=[1e-12, 1e-10, 0.9, 0.5],
        )
    )
    original = binary_metrics(
        frame.y_true.to_numpy(), frame.p_calibrated.to_numpy(), 0.5
    )
    followup = scores(frame)
    assert followup["auprc"] == pytest.approx(original["auprc"], abs=1e-14)
    assert followup["auroc"] == pytest.approx(original["auroc"], abs=1e-14)


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


def test_full_followup_workflow_pairs_all_runs_and_rejects_missing_run(
    tmp_path, monkeypatch
):
    from tasks.entex.analyze import BASE, FULL
    from tasks.entex.measurement_followups import run
    from scripts.server.run_ccre_frozen_probe_fold import binary_metrics

    monkeypatch.chdir(tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    folds = [dict(name=f"fold_{i}", test=[f"chr{i}"]) for i in range(1, 6)]
    (configs / "entex_v1.json").write_text(
        json.dumps(dict(manuscript_config="configs/folds.json"))
    )
    (configs / "folds.json").write_text(
        json.dumps(
            dict(
                rotating_chromosome_folds=folds,
                training=dict(
                    seeds=[42, 314159, 20260806], contexts=["strict", "1hop"]
                ),
            )
        )
    )
    protocol = configs / "followups.json"
    protocol.write_text(
        json.dumps(
            dict(
                prediction_followups=dict(
                    assays=["ctcf"],
                    strata=["donor", "tissue"],
                    minimum_measurements_per_test_fold=4,
                    minimum_positive_per_test_fold=1,
                    minimum_negative_per_test_fold=1,
                    n_bootstrap=20,
                    bootstrap_seed=7,
                    limitations="synthetic test",
                )
            )
        )
    )
    root = tmp_path / "predictions"
    for fold, seed, context in product(
        folds, [42, 314159, 20260806], ["strict", "1hop"]
    ):
        directory = root / fold["name"] / f"seed_{seed}" / context
        directory.mkdir(parents=True)
        frames, metrics = [], []
        for feature, probability in [
            (BASE, [0.1, 0.6, 0.8, 0.7]),
            (FULL, [0.1, 0.2, 0.8, 0.7]),
        ]:
            f = pd.DataFrame(
                dict(
                    measurement_id=[fold["name"] + str(i) for i in range(4)],
                    locus_id=[fold["name"] + str(i) for i in range(4)],
                    chrom=fold["test"] * 4,
                    donor=["d"] * 4,
                    tissue=["t"] * 4,
                    y_true=[0, 0, 1, 1],
                    feature_set=feature,
                    p_calibrated=probability,
                )
            )
            frames.append(f)
            metrics.append(
                dict(
                    feature_set=feature,
                    positive_prevalence=0.5,
                    **binary_metrics(
                        f.y_true.to_numpy(), f.p_calibrated.to_numpy(), 0.5
                    ),
                )
            )
        pd.concat(frames).to_parquet(directory / "predictions.parquet", index=False)
        pd.DataFrame(metrics).to_csv(directory / "metrics.csv", index=False)
    run(root, tmp_path / "analysis", protocol, "ctcf")
    audit = json.loads((tmp_path / "analysis/audit.json").read_text())
    assert audit["n_prediction_runs"] == 30 and audit["n_loci"] == 20
    gains = pd.read_csv(tmp_path / "analysis/paired_gains.csv")
    assert gains.n_runs.eq(15).all() and gains.n_folds.eq(5).all()
    assert set(gains.analysis) == {
        "measurement_weight",
        "equal_locus_weight",
        "donor",
        "tissue",
        "donor_macro",
        "tissue_macro",
    }
    next(root.rglob("predictions.parquet")).unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        run(root, tmp_path / "bad", protocol, "ctcf")
