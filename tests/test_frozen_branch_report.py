from itertools import product
import pandas as pd
import pytest
from tasks.transfer.frozen_branch_report import MODELS, FEATURES, compare


def test_composite_gate_requires_both_branches_and_identical_references():
    frame = pd.DataFrame([
        dict(task=task, model=model, feature=feature, evaluation_partition="development_validation",
             fold="fold_a", seed=42, context="1hop", n_train=100, n_val=20, n_evaluated=20, n_test=0,
             positive_prevalence=.5, targets_sha256="targets", scores_sha256="baseline",
             auprc=.7 + (.03 if model == "T_Q" and feature in ["cst", "csht"] else 0),
             auroc=.8, normalized_ap=.4, balanced_accuracy=.6, f1=.5)
        for task, model, feature in product(["sv", "ccre"], MODELS, FEATURES)])
    deltas, gate = compare(frame)
    assert gate["status"] == "eligible_for_replication"
    assert len(deltas) == 100
    assert deltas.query("metric == 'auprc'").difference.tolist() == pytest.approx([.03] * 20)
    bad = frame.copy()
    bad.loc[bad.model.eq("T") & bad.feature.eq("cs"), "scores_sha256"] = "changed"
    with pytest.raises(ValueError, match="Baseline"):
        compare(bad)
    bad = frame.copy()
    bad.loc[bad.task.eq("ccre") & bad.model.eq("T_Rq") & bad.feature.eq("csht"), "auprc"] = .74
    _, gate = compare(bad)
    assert gate["status"] == "not_promoted"
    assert next(c for c in gate["checks"] if c["task"] == "ccre")["after_H_control_gains"]["T_Rq"] < 0
