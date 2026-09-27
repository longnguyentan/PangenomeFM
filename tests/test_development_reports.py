from itertools import product

import pandas as pd
import pytest

from tasks.transfer.development_report import MODELS, FEATURES, contrasts, validate_development_pairs, development_gate
from tasks.transfer.junction_pilot_report import validation_predictions


def development_frame():
    return pd.DataFrame([
        dict(task=task, model=model, feature=feature, evaluation_partition="development_validation",
             fold="fold_a", seed=42, context="1hop", n_train=100, n_val=20, n_test=0, n_evaluated=20,
             positive_prevalence=0.5, targets_sha256="targets", scores_sha256="baseline",
             auprc=0.7 + (0.05 if model == "v2" and feature in ["cst", "csht"] else 0),
             auroc=0.8, normalized_ap=0.4)
        for task, model, feature in product(["sv", "ccre"], MODELS, FEATURES)
    ])


def test_development_comparisons_fail_on_mismatched_loci_or_baselines():
    frame = development_frame()
    validate_development_pairs(frame)
    for column, value, match in [("targets_sha256", "different loci", "universe"),
                                  ("scores_sha256", "different scores", "baseline"),
                                  ("n_test", 1, "test predictions")]:
        bad = frame.copy()
        bad.loc[0, column] = value
        with pytest.raises(ValueError, match=match):
            validate_development_pairs(bad)
    with pytest.raises(ValueError, match="matrix"):
        validate_development_pairs(frame.iloc[:-1])
    delta = contrasts(frame).query('metric == "auprc" and contrast == "v2_minus_random_csht"')
    assert delta.difference.tolist() == pytest.approx([0.05, 0.05])


def test_development_gate_requires_both_contexts_and_random_advantage():
    frame = development_frame()
    result = development_gate(frame)
    assert result["status"] == "not_promoted" and result["missing_required_contexts"] == ["strict"]
    assert all(row["passes_context_task_gate"] for row in result["checks"])
    frame.loc[frame.task.eq("ccre") & frame.model.eq("v2") & frame.feature.eq("cst"), "auprc"] = 0.6999
    row = next(row for row in development_gate(frame)["checks"] if row["task"] == "ccre")
    assert row["within_0_005_of_v1"] and not row["beats_random"]


def test_junction_identity_uses_endpoints_and_labels_and_rejects_test_rows():
    frame = pd.DataFrame(dict(slice=["a"] * 4, target_sn=["chr2"] * 4, closure=["1hop"] * 4,
                              u_local=[1, 1, 2, 2], v_local=[3, 4, 3, 4], y_true=[1, 0, 0, 1],
                              p_edge=[0.7, 0.2, 0.3, 0.8], split=["val_chr_val"] * 4))
    _, digest = validation_predictions(frame)
    assert validation_predictions(frame.iloc[::-1])[1] == digest
    changed = frame.copy()
    changed.loc[0, "u_local"] = 9
    assert validation_predictions(changed)[1] != digest
    changed = frame.copy()
    changed.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="held-out"):
        validation_predictions(changed)
    with pytest.raises(ValueError, match="duplicate"):
        validation_predictions(pd.concat([frame, frame.iloc[:1]]))


def test_audited_development_run_replays_metrics_and_refuses_test_scope(tmp_path):
    import json
    import numpy as np
    from scripts.server.run_ccre_frozen_probe_fold import evaluate_feature_sets
    from scripts.summarize_v2_review_controls import _feature_names
    from tasks.transfer.hr_control_report import audited_run

    names = _feature_names("ccre")
    labels = np.tile([0, 1], 12)
    metrics, _, predictions = evaluate_feature_sets(
        segids=np.arange(24), chromosomes=np.repeat(["chr1", "chr2", "chr3"], 8), labels=labels,
        features={name: np.arange(24).reshape(-1, 1) for name in names.values()},
        test_chrs={"chr3"}, val_chrs={"chr2"}, seed=42, validation_only=True,
        feature_access={name: "synthetic" for name in names.values()},
    )
    metrics["fold"], metrics["seed"], metrics["closure"] = "fold_a", 42, "1hop"
    metrics.to_csv(tmp_path / "metrics.csv", index=False)
    predictions.to_csv(tmp_path / "validation_predictions.csv.gz", index=False)
    (tmp_path / "audit.json").write_text(json.dumps(dict(
        status="complete", fold="fold_a", seed=42, closure="1hop", checkpoint_sha256="checkpoint",
        evaluation_partition="development_validation", heldout_predictions_produced=False,
        validation_chromosomes=["chr2"], test_chromosomes=["chr3"],
    )))
    result = audited_run(tmp_path, "v2", "ccre", "fold_a", 42, "1hop", validation_only=True)
    assert result.n_evaluated.eq(8).all() and result.n_test.eq(0).all()
    with pytest.raises(ValueError, match="partitions differ"):
        audited_run(tmp_path, "v2", "ccre", "fold_a", 42, "1hop")
    (tmp_path / "test_predictions.csv.gz").touch()
    with pytest.raises(ValueError, match="held-out"):
        audited_run(tmp_path, "v2", "ccre", "fold_a", 42, "1hop", validation_only=True)
