from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.server.run_masked_chromosome_replication import (
    architecture_template,
    check_development,
)
from tasks.transfer.masked_replication import summarize, validate_matrix


def test_replication_template_transfers_only_architecture_and_declared_partition():
    original = dict(
        args=dict(hidden_dim=48, test_chrs=["chr1"], val_chrs=["chr2"]),
        in_dim=519,
        model_state={"do_not_copy": "trained parameters"},
        predictor_state={"trained": "head"},
    )
    result = architecture_template(original, dict(test=["chr3"], validation=["chr4"]))
    assert result["args"]["test_chrs"] == ["chr3"] and result["args"]["val_chrs"] == [
        "chr4"
    ]
    assert result["args"]["hidden_dim"] == 48 and result["architecture_only"]
    assert "model_state" not in result and "predictor_state" not in result
    assert original["args"]["test_chrs"] == ["chr1"]


def test_development_gate_requires_all_positive_finite_paired_results():
    audit = dict(
        status="eligible_for_chromosome_replication",
        all_declared_seeds_complete=True,
        all_metrics_replayed=True,
        all_probes_converged=True,
        checks=[
            dict(
                task=t,
                seed=s,
                feature="csht",
                metric="auprc",
                comparison="full_trained minus full_random",
                difference=0.001,
            )
            for t, s in product(["sv", "ccre"], [42, 314159, 20260806])
        ],
    )
    check_development(audit)
    for value in [0.0, -0.001, np.nan]:
        audit["checks"][0]["difference"] = value
        with pytest.raises(ValueError, match="gate"):
            check_development(audit)
    audit["checks"][0]["difference"] = 0.001
    audit["all_declared_seeds_complete"] = False
    with pytest.raises(ValueError, match="gate"):
        check_development(audit)


def test_replication_reports_all_folds_but_excludes_development_fold_in_primary():
    plan = json.loads(
        Path("configs/masked_nt_chromosome_replication_20260928.json").read_text()
    )
    plan["n_bootstrap"] = 20
    rows = []
    for fold, seed, model, task, feature in product(
        plan["folds"], plan["seeds"], plan["arms"], plan["tasks"], plan["features"]
    ):
        value = 0.7 if model.endswith("trained") else 0.6
        if model == "full_trained" and fold == "fold_b":
            value = 0.2
        if feature in ["cs", "csh"]:
            value = 0.55
        rows.append(
            dict(
                fold=fold,
                seed=seed,
                model=model,
                task=task,
                feature=feature,
                context="1hop",
                evaluation_partition="test",
                n_train=100,
                n_val=5,
                n_test=20,
                n_evaluated=20,
                targets_sha256=task + fold,
                scores_sha256=feature + task + fold,
                positive_prevalence=0.5,
                **{metric: value for metric in plan["metrics"]},
            )
        )
    frame = pd.DataFrame(rows)
    absolute, paired, contrasts = summarize(frame, plan)
    selected = contrasts.loc[
        contrasts.comparison.eq("full_trained minus full_random")
        & contrasts.metric.eq("auprc")
    ]
    assert selected.loc[
        selected.scope.eq("development_excluded_four_folds"), "mean"
    ].to_numpy() == pytest.approx(0.1)
    assert selected.loc[
        selected.scope.eq("all_five_development_exposed"), "mean"
    ].to_numpy() == pytest.approx(0.0)
    assert (
        selected.loc[selected.scope.eq("development_exposed_fold_only"), "ci95_low"]
        .isna()
        .all()
    )
    assert set(paired.fold) == set(plan["folds"]) and len(absolute) > 0
    with pytest.raises(ValueError, match="matrix"):
        validate_matrix(frame.iloc[1:], plan)
    broken = frame.copy()
    broken.loc[0, "scores_sha256"] = "different"
    with pytest.raises(ValueError, match="baseline"):
        validate_matrix(broken, plan)
    with pytest.raises(ValueError, match="Development-exposed"):
        validate_matrix(frame, dict(plan, primary_summary_folds=plan["folds"]))
