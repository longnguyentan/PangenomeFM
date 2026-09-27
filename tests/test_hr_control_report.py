from itertools import product
import json
import sys

import numpy as np
import pandas as pd
import pytest

from tasks.transfer.hr_control_report import summarize, validate_pairs


def comparison_frame():
    rows = []
    for fold, seed, context, task, model, feature in product(
            ["fold_a", "fold_b"], [42], ["strict", "1hop"], ["sv", "ccre"],
            ["v1", "random"], ["cs", "csh", "cst", "csht"]):
        # A large fold baseline difference must cancel under proper pairing.
        value = (0.4 if fold == "fold_a" else 0.8)
        value += (0.03 if model == "v1" else 0.01) if feature in ["cst", "csht"] else 0
        rows.append(dict(fold=fold, seed=seed, context=context, task=task, model=model,
                         feature=feature, targets_sha256="targets", scores_sha256="scores",
                         n_train=200, n_val=50, n_test=50, positive_prevalence=0.2,
                         auprc=value, auroc=value, normalized_ap=value,
                         balanced_accuracy=value, f1=value))
    return pd.DataFrame(rows)


def test_control_report_rejects_missing_cells_and_changed_pairing():
    frame = comparison_frame()
    validate_pairs(frame, ["fold_a", "fold_b"], [42])
    for bad in [frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]])]:
        with pytest.raises(ValueError, match="matrix"):
            validate_pairs(bad, ["fold_a", "fold_b"], [42])
    changed = frame.copy()
    changed.loc[0, "targets_sha256"] = "same size but different loci or labels"
    with pytest.raises(ValueError, match="universe"):
        validate_pairs(changed, ["fold_a", "fold_b"], [42])
    changed = frame.copy()
    changed.loc[0, "scores_sha256"] = "changed frozen baseline"
    with pytest.raises(ValueError, match="baseline"):
        validate_pairs(changed, ["fold_a", "fold_b"], [42])


def test_random_control_gain_pairs_identical_fold_and_seed():
    _, paired = summarize(comparison_frame(), 50)
    selected = paired.loc[paired.contrast.eq("T_minus_R_given_CSH")]
    np.testing.assert_allclose(selected["mean"], 0.02)
    assert selected.n_runs.eq(2).all()
    frame = comparison_frame().query('fold == "fold_a"')
    _, single = summarize(frame, 50)
    assert single.ci95_low.isna().all()


@pytest.mark.parametrize("upstream_complete", [True, False])
def test_report_cli_writes_final_status_and_preserves_validation_error(
    tmp_path, monkeypatch, upstream_complete
):
    from tasks.transfer import hr_control_report as report

    root = tmp_path / "controls"
    root.mkdir()
    (root / "status.json").write_text(json.dumps(dict(
        status="complete" if upstream_complete else "running", completed_commands=10,
        commands=list(range(10)), fold=dict(name="fold_a"), seed=42,
        random_initialization_seed=42, encoders_frozen=True,
    )))
    out = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["report", "--single-root", str(root),
                                      "--out-dir", str(out), "--n-bootstrap", "20"])
    frame = comparison_frame()

    def synthetic_run(directory, model, task, fold, seed, context):
        return frame.loc[(frame.model == model) & (frame.task == task)
                         & (frame.fold == fold) & (frame.seed == seed)
                         & (frame.context == context)].copy()

    monkeypatch.setattr(report, "audited_run", synthetic_run)
    if upstream_complete:
        report.main()
        assert json.loads((out / "audit.json").read_text())["n_runs"] == 8
        assert json.loads((tmp_path / "report_status.json").read_text())["status"] == "complete"
    else:
        with pytest.raises(ValueError, match="Incomplete or mismatched control receipt"):
            report.main()
        assert not out.exists()
        receipt = json.loads((tmp_path / "report_status.json").read_text())
        assert receipt["status"] == "failed" and receipt["error"].startswith("ValueError:")
