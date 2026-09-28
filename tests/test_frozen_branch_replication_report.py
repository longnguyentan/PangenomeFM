from itertools import product

import pandas as pd
import pytest

from tasks.transfer.frozen_branch_report import FEATURES, MODELS
from tasks.transfer.frozen_branch_replication_report import SEEDS, summarize


def frames():
    return [pd.DataFrame([
        dict(task=task, model=model, feature=feature, evaluation_partition="development_validation",
             fold="fold_a", seed=seed, context="1hop", n_train=100, n_val=20, n_evaluated=20, n_test=0,
             positive_prevalence=.5, targets_sha256="targets", scores_sha256=f"baseline{seed}",
             checkpoint_sha256=f"{model}{seed}",
             auprc=.7 + (.03 if model == "T_Q" and feature in ["cst", "csht"] else 0),
             auroc=.8, normalized_ap=.4, balanced_accuracy=.6, f1=.5)
        for task, model, feature in product(["sv", "ccre"], MODELS, FEATURES)]) for seed in sorted(SEEDS)]


def test_all_seeds_and_unfavorable_contrasts_retained_without_fold_ci():
    sources = frames()
    sources[-1].loc[sources[-1].model.eq("T_Q") & sources[-1].feature.eq("csht"), "auprc"] = .68
    runs, differences, summary, audit = summarize(sources)
    assert len(runs) == 144 and len(differences) == 300
    selected = summary.query("feature == 'csht' and metric == 'auprc'")
    assert selected.n_seeds.eq(3).all() and selected.n_positive.eq(2).all()
    assert selected.minimum.tolist() == pytest.approx([-.02] * 10)
    assert not audit["all_seed_development_gates_pass"]
    assert not audit["confirmatory_ci"] and not audit["independent_chromosome_replication"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "universe", "checkpoint", "test", "context"])
def test_replication_rejects_missing_or_mismatched_evidence(change):
    sources = frames()
    if change == "missing":
        sources.pop()
    elif change == "duplicate":
        sources[1]["seed"] = sources[0].seed.iloc[0]
    elif change == "universe":
        sources[1]["targets_sha256"] = "other"
    elif change == "checkpoint":
        sources[1]["checkpoint_sha256"] = sources[0].checkpoint_sha256
    elif change == "test":
        sources[1]["n_test"] = 20
    else:
        sources[1]["context"] = "strict"
    with pytest.raises(ValueError):
        summarize(sources)
