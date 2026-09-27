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
from scripts.summarize_v2_review_controls import _feature_names

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


def development_gate(frame: pd.DataFrame) -> dict:
    """Record the previously specified point-estimate gate; never infer replication."""
    validate_development_pairs(frame)
    checks = []
    for task, group in frame.groupby("task"):
        ap = group.set_index(["model", "feature"]).auprc
        checks.append(dict(task=task, context=group.context.iloc[0],
                           v2_minus_v1=float(ap["v2", "cst"] - ap["v1", "cst"]),
                           v2_minus_random=float(ap["v2", "cst"] - ap["v2_random", "cst"]),
                           v2_given_handcrafted=float(ap["v2", "csht"] - ap["v2", "csh"])))
    for row in checks:
        row["within_0_005_of_v1"] = row["v2_minus_v1"] >= -0.005
        row["beats_random"] = row["v2_minus_random"] > 0
        row["adds_beyond_handcrafted"] = row["v2_given_handcrafted"] > 0
        row["passes_context_task_gate"] = all(row[k] for k in
            ["within_0_005_of_v1", "beats_random", "adds_beyond_handcrafted"])
    missing = sorted({"strict", "1hop"} - set(frame.context))
    return dict(status="not_promoted" if missing or not all(row["passes_context_task_gate"] for row in checks) else "eligible_for_replication",
                checks=checks, missing_required_contexts=missing,
                limitation="single-fold development point estimates; no independent performance claim")


def difference_figure(differences: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    names = ["v2_minus_v1_cst", "v2_minus_random_cst", "v2_minus_v1_csht", "v2_minus_random_csht"]
    labels = ["v2 − v1, C+S+T", "v2 − random, C+S+T/R", "v2 − v1, C+S+H+T", "v2 − random, C+S+H+T/R"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained", sharex=True, sharey=True)
    for ax, task in zip(axes, ["sv", "ccre"]):
        part = differences.loc[differences.task.eq(task) & differences.metric.eq("auprc")].set_index("contrast")
        values = part.loc[names, "difference"]
        ax.scatter(values, range(4), color="#245a81", s=30)
        for y, value in enumerate(values):
            ax.annotate(f"{value:+.6f}", (value, y), xytext=(5, 7), textcoords="offset points", fontsize=9)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set(yticks=range(4), yticklabels=labels, ylim=(3.5, -0.5), xlim=(-0.002, 0.0115),
               xlabel="Paired validation Δ AUPRC", title="SV insertion/deletion" if task == "sv" else "cCRE")
    fig.suptitle("Repaired-model comparison: one development fold/seed; no uncertainty interval")
    save_figure(fig, out, "biological_validation_differences")
    plt.close(fig)


def stratum_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain every inherited bin, including undefined one-class and sparse bins."""
    rows = []
    for (family, value), group in frame.groupby(["stratum", "stratum_value"]):
        if (group.duplicated(["model", "feature"]).any()
                or set(group[["model", "feature"]].itertuples(index=False, name=None)) != set(product(MODELS, FEATURES))):
            raise ValueError("Incomplete stratum comparison matrix")
        if group.n.nunique() != 1 or group.positive_fraction.nunique() != 1:
            raise ValueError("Stratum counts or prevalence changed")
        for metric in ["auprc", "auroc"]:
            indexed = group.set_index(["model", "feature"])[metric]
            for base in ["cs", "csh"]:
                if group.loc[group.feature.eq(base), metric].nunique(dropna=False) != 1:
                    raise ValueError("Stratum non-embedding baseline changed")
            for feature in ["cst", "csht"]:
                for control in ["v1", "v2_random"]:
                    left, right = indexed["v2", feature], indexed[control, feature]
                    if pd.isna(left) != pd.isna(right):
                        raise ValueError("Asymmetric undefined stratum comparison")
                    rows.append(dict(stratum=family, stratum_value=value, n=int(group.n.iloc[0]),
                                     positive_prevalence=float(group.positive_fraction.iloc[0]),
                                     feature=feature, metric=metric, comparison="v2_minus_" + control,
                                     v2=float(left), control=float(right), difference=float(left - right),
                                     scope="single-fold development validation; no CI"))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path,
                        help="Reuse completed v1/random probes; other models in this source are ignored")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.root / "status.json").read_text())
    if (receipt.get("status") != "complete"
            or receipt["completed_commands"] != len(receipt["commands"])
            or receipt.get("evaluation_partition") != "development_validation"
            or not receipt.get("encoders_frozen")):
        raise ValueError("Incomplete or incorrectly scoped development receipt")
    reference_root = args.reference_root or args.root
    reference = json.loads((reference_root / "status.json").read_text())
    if (reference.get("evaluation_partition") != "development_validation"
            or not reference.get("encoders_frozen") or reference["fold"] != receipt["fold"]
            or reference["seed"] != receipt["seed"] or reference["graph_sha256"] != receipt["graph_sha256"]):
        raise ValueError("Reference development source differs in scope, folds or graph")
    frame = pd.concat([
        audited_run((reference_root if model in ["v1", "random"] else args.root) / "probes" / model / task / "1hop", model, task,
                    receipt["fold"]["name"], receipt["seed"], "1hop", validation_only=True)
        for model, task in product(MODELS, ["sv", "ccre"])
    ], ignore_index=True)
    differences = contrasts(frame)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.out_dir / "audited_per_run.csv", index=False)
    differences.to_csv(args.out_dir / "paired_differences.csv", index=False)
    strata = []
    for model in MODELS:
        source = reference_root if model in ["v1", "random"] else args.root
        table = pd.read_csv(source / "probes" / model / "sv/1hop/stratified_metrics.csv", float_precision="round_trip")
        table = table.loc[table.stratum.isin(["length_bin", "allele_frequency_bin", "chromosome"])].copy()
        table["feature"] = table.feature_set.map({raw: short for short, raw in _feature_names("sv").items()})
        table["model"] = model
        strata.append(table.loc[table.feature.notna()])
    strata = pd.concat(strata, ignore_index=True)
    strata.to_csv(args.out_dir / "sv_validation_strata_absolute.csv", index=False)
    stratum_contrasts(strata).to_csv(args.out_dir / "sv_validation_strata_differences.csv", index=False)
    (args.out_dir / "development_gate.json").write_text(json.dumps(development_gate(frame), indent=2) + "\n")
    difference_figure(differences, args.out_dir)
    (args.out_dir / "audit.json").write_text(json.dumps(dict(
        status="complete", source=fingerprint(args.root / "status.json"),
        reference_source=fingerprint(reference_root / "status.json"),
        reference_source_scope="only completed v1/random probes audited; unused candidate probes are excluded",
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
