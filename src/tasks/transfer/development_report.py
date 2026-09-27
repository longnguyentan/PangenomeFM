"""Audit the fixed biological validation pilot; never report it as a test matrix."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import pandas as pd

from tasks.entex.prepare import fingerprint
from tasks.transfer.hr_control_report import audited_run
from tasks.transfer.report import save_figure

MODELS = ["v1", "random", "v2", "v2_random"]
FEATURES = ["cs", "csh", "cst", "csht"]


def validate_development_pairs(frame: pd.DataFrame) -> None:
    keys = ["task", "model", "feature"]
    expected = set(product(["sv", "ccre"], MODELS, FEATURES))
    if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete or duplicate development comparison matrix")
    if (not frame.evaluation_partition.eq("development_validation").all()
            or not frame.n_test.eq(0).all()):
        raise ValueError("Development analysis cannot include test predictions")
    for key in ["fold", "seed", "context"]:
        if frame[key].nunique() != 1:
            raise ValueError("This report requires one declared development fold/seed/context")
    for task, group in frame.groupby("task"):
        for field in ["targets_sha256", "n_train", "n_val", "n_evaluated", "positive_prevalence"]:
            if group[field].nunique() != 1:
                raise ValueError(f"Paired locus/label universe changed: {task}/{field}")
        for feature in ["cs", "csh"]:
            if group.loc[group.feature.eq(feature), "scores_sha256"].nunique() != 1:
                raise ValueError(f"Non-embedding baseline changed: {task}/{feature}")


def contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    validate_development_pairs(frame)
    rows = []
    for task, group in frame.groupby("task"):
        for metric in ["auprc", "auroc", "normalized_ap"]:
            values = group.set_index(["model", "feature"])[metric]
            comparisons = []
            for model in MODELS:
                comparisons += [(model + "_embedding_given_CS", (model, "cst"), (model, "cs")),
                                (model + "_embedding_given_CSH", (model, "csht"), (model, "csh"))]
            for feature in ["cst", "csht"]:
                comparisons += [("v2_minus_v1_" + feature, ("v2", feature), ("v1", feature)),
                                ("v2_minus_random_" + feature, ("v2", feature), ("v2_random", feature)),
                                ("v1_minus_random_" + feature, ("v1", feature), ("random", feature))]
            for name, left, right in comparisons:
                rows.append(dict(task=task, metric=metric, contrast=name,
                                 difference=float(values[left] - values[right]),
                                 fold=group.fold.iloc[0], seed=int(group.seed.iloc[0]),
                                 context=group.context.iloc[0], scope="development_validation"))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.root / "status.json").read_text())
    if (receipt.get("status") != "complete"
            or receipt["completed_commands"] != len(receipt["commands"])
            or receipt.get("evaluation_partition") != "development_validation"
            or not receipt.get("encoders_frozen")):
        raise ValueError("Incomplete or incorrectly scoped development receipt")
    frame = pd.concat([
        audited_run(args.root / "probes" / model / task / "1hop", model, task,
                    receipt["fold"]["name"], receipt["seed"], "1hop", validation_only=True)
        for model, task in product(MODELS, ["sv", "ccre"])
    ], ignore_index=True)
    differences = contrasts(frame)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.out_dir / "audited_per_run.csv", index=False)
    differences.to_csv(args.out_dir / "paired_differences.csv", index=False)
    (args.out_dir / "audit.json").write_text(json.dumps(dict(
        status="complete", source=fingerprint(args.root / "status.json"),
        n_runs=8, n_folds=1, prediction_identity_and_baseline_invariance="passed",
        scope="development validation; calibration/threshold use these labels; no independent test estimate",
        confidence_intervals="not estimated from a single selected development fold/seed",
        heldout_predictions_produced=False,
    ), indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, task in zip(axes, ["sv", "ccre"]):
        for index, (model, color) in enumerate(zip(MODELS, ["#245a81", "#83afcb", "#a45021", "#e7ad7e"])):
            part = frame.loc[frame.task.eq(task) & frame.model.eq(model)].set_index("feature")
            ax.plot(range(4), part.loc[FEATURES, "auprc"], "o-", color=color, label=model)
        ax.set(xticks=range(4), xticklabels=["C+S", "C+S+H", "C+S+T/R", "C+S+H+T/R"],
               ylim=(0, 1), ylabel="Validation AUPRC", title="SV insertion/deletion" if task == "sv" else "cCRE")
        ax.tick_params(axis="x", rotation=20)
    axes[0].legend(frameon=False)
    fig.suptitle("Frozen probes: single-fold development comparison")
    save_figure(fig, args.out_dir, "biological_validation")
    plt.close(fig)


if __name__ == "__main__":
    main()
