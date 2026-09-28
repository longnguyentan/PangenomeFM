from itertools import product
from pathlib import Path

import pandas as pd
import pytest

from scripts.server.run_frozen_branch_convergence import MODELS, convergence_command
from tasks.transfer.frozen_branch_replication_report import replay_solver_gate
from tasks.transfer.frozen_branch_report import MODELS as REPORT_MODELS


def command():
    return ["python", "scripts/server/run_v2_review_controls.py", "--fold", "fold_a", "--seed", "314159",
            "--contexts", "1hop", "--extraction-candidate-policy", "manuscript", "--validation-only",
            "--primary-features-only", "--out-root", "original", "--models", *MODELS,
            "--candidate-checkpoint", "v2=original.pt"]


def test_convergence_preserves_every_original_setting_except_budget_and_output():
    old = command()
    new = convergence_command(old, Path("new"), 314159)
    expected = old.copy()
    expected[0] = new[0]
    expected[expected.index("--out-root") + 1] = "new"
    assert new == expected + ["--probe-max-iter", "4000"]
    assert old == command()


@pytest.mark.parametrize("change", ["test", "fold", "arm", "already_overridden"])
def test_convergence_rejects_changed_scope(change):
    old = command()
    if change == "test":
        old.remove("--validation-only")
    elif change == "fold":
        old[old.index("--fold") + 1] = "fold_b"
    elif change == "arm":
        old.remove("Rt_Rq")
    else:
        old += ["--probe-max-iter", "1000"]
    with pytest.raises(ValueError):
        convergence_command(old, Path("new"), 314159)


def optimizer():
    return pd.DataFrame([dict(model=m, task=t, feature_set=f, probe_max_iter=4000,
        probe_solver="lbfgs", probe_converged=True) for m, t, f in
        product(REPORT_MODELS, ["sv", "ccre"], ["cs", "csh", "cst", "csht"])])


def test_numerical_failure_is_retained_instead_of_changing_performance_gate():
    evidence = optimizer()
    evidence.loc[0, "probe_converged"] = False
    saved = dict(status="optimization_incomplete", probe_optimization_fully_recorded=True,
                 all_probes_converged=False)
    gate = replay_solver_gate(dict(status="eligible_for_replication"), saved, evidence)
    assert gate["status"] == "optimization_incomplete"
    assert gate["performance_gate_before_solver_check"] == "eligible_for_replication"
    assert gate["probe_max_iter"] == 4000


def test_legacy_unknown_and_converged_evidence_are_distinguished():
    performance = dict(status="not_promoted")
    assert replay_solver_gate(performance, performance, None)["all_probes_converged"] is None
    saved = dict(**performance, probe_optimization_fully_recorded=True, all_probes_converged=True)
    assert replay_solver_gate(performance, saved, optimizer())["all_probes_converged"]
    with pytest.raises(ValueError, match="Missing optimizer"):
        replay_solver_gate(performance, saved, None)
    with pytest.raises(ValueError, match="Incomplete"):
        replay_solver_gate(performance, saved, optimizer().iloc[:-1])
