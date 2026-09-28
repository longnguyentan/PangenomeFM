import pandas as pd
import pytest

from scripts.server.run_frozen_head_transfer import compare_heads


def test_head_comparison_keeps_negative_results_and_rejects_changed_controls():
    old = pd.DataFrame([dict(model=model, task=task, feature=feature, fold="fold_a", seed=42,
        context="1hop", n_train=20, n_val=10, n_test=0, targets_sha256=task,
        scores_sha256=f"{model}/{task}/{feature}", auprc=.7)
        for model in ["T_Q", "Q"] for task in ["sv", "ccre"] for feature in ["cs", "csh", "cst", "csht"]])
    new = old.copy()
    new.loc[new.model.eq("T_Q") & new.feature.eq("csht"), "auprc"] -= .03
    result = compare_heads(old, new)
    assert result.loc[result.model.eq("T_Q") & result.feature.eq("csht"), "delta_linear_minus_mlp"].lt(0).all()
    new.loc[0, "n_train"] += 1
    with pytest.raises(ValueError, match="examples"):
        compare_heads(old, new)
    new = old.copy()
    new.loc[0, "scores_sha256"] = "modified"
    with pytest.raises(ValueError, match="baseline"):
        compare_heads(old, new)
    with pytest.raises(ValueError, match="scope"):
        compare_heads(old, old.iloc[1:])
