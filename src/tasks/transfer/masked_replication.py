"""Audit fixed masked-feature chromosome replication with explicit exposure scopes."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.paired_inference import bh_adjust, fold_sign_flip
from tasks.entex.analyze import estimate
from tasks.entex.prepare import fingerprint
from tasks.transfer.hr_control_report import audited_run
from tasks.transfer.report import save_figure
from tasks.transfer.traitgym import write_json


def validate_matrix(frame: pd.DataFrame, plan: dict) -> None:
    keys = ["fold", "seed", "model", "task", "feature"]
    expected = set(
        product(
            plan["folds"], plan["seeds"], plan["arms"], plan["tasks"], plan["features"]
        )
    )
    if (
        frame.duplicated(keys).any()
        or set(frame[keys].itertuples(index=False, name=None)) != expected
    ):
        raise ValueError("Incomplete/duplicate chromosome replication matrix")
    if (
        not frame.context.eq(plan["context"]).all()
        or not frame.evaluation_partition.eq("test").all()
        or not frame.n_test.eq(frame.n_evaluated).all()
        or (frame.n_test <= 0).any()
    ):
        raise ValueError("Unexpected test partition")
    if plan["development_exposed_test_fold"] in plan["primary_summary_folds"]:
        raise ValueError(
            "Development-exposed test chromosomes included in primary aggregate"
        )
    if set(plan["primary_summary_folds"]) != set(plan["folds"]) - {
        plan["development_exposed_test_fold"]
    }:
        raise ValueError("Primary summary must retain every other declared fold")
    if not np.isfinite(frame[plan["metrics"]].to_numpy()).all():
        raise ValueError("Nonfinite complete-task metrics")
    for _, group in frame.groupby(["fold", "seed", "task"]):
        for key in [
            "targets_sha256",
            "n_train",
            "n_val",
            "n_test",
            "positive_prevalence",
        ]:
            if group[key].nunique() != 1:
                raise ValueError("Paired population mismatch")
        for feature in ["cs", "csh"]:
            if group.loc[group.feature.eq(feature), "scores_sha256"].nunique() != 1:
                raise ValueError("Non-embedding baseline changed")


def summarize(frame: pd.DataFrame, plan: dict):
    validate_matrix(frame, plan)
    absolute, paired = [], []
    scopes = {
        "development_excluded_four_folds": plan["primary_summary_folds"],
        "all_five_development_exposed": plan["folds"],
        "development_exposed_fold_only": [plan["development_exposed_test_fold"]],
    }
    for scope, folds in scopes.items():
        part = frame.loc[frame.fold.isin(folds)]
        for (task, model, feature), group in part.groupby(["task", "model", "feature"]):
            for metric in plan["metrics"]:
                absolute.append(
                    dict(
                        scope=scope,
                        task=task,
                        model=model,
                        feature=feature,
                        metric=metric,
                        **estimate(
                            group, metric, plan["n_bootstrap"], plan["statistics_seed"]
                        ),
                    )
                )
        for task in plan["tasks"]:
            task_frame = part.loc[part.task.eq(task)]
            for feature, metric in product(["cst", "csht"], plan["metrics"]):
                wide = task_frame.loc[task_frame.feature.eq(feature)].pivot(
                    index=["fold", "seed"], columns="model", values=metric
                )
                for left, right in [
                    ("full_trained", "full_random"),
                    ("coordinate_trained", "coordinate_random"),
                    ("full_trained", "coordinate_trained"),
                ]:
                    paired.append(
                        (wide[left] - wide[right])
                        .rename("gain")
                        .reset_index()
                        .assign(
                            scope=scope,
                            task=task,
                            feature=feature,
                            metric=metric,
                            comparison=f"{left} minus {right}",
                        )
                    )
                if feature == "csht":
                    base = task_frame.loc[
                        task_frame.model.eq("full_trained")
                        & task_frame.feature.eq("csh")
                    ].set_index(["fold", "seed"])[metric]
                    paired.append(
                        (wide["full_trained"] - base)
                        .rename("gain")
                        .reset_index()
                        .assign(
                            scope=scope,
                            task=task,
                            feature=feature,
                            metric=metric,
                            comparison="full_trained minus C+S+H",
                        )
                    )
    differences = pd.concat(paired, ignore_index=True)
    rows = []
    names = ["scope", "task", "feature", "metric", "comparison"]
    for key, group in differences.groupby(names):
        rows.append(
            dict(
                zip(names, key),
                **estimate(group, "gain", plan["n_bootstrap"], plan["statistics_seed"]),
                sign_flip_p=fold_sign_flip(group),
            )
        )
    contrasts = pd.DataFrame(rows)
    contrasts["bh_q_within_scope_metric"] = contrasts.groupby(
        ["scope", "metric"]
    ).sign_flip_p.transform(bh_adjust)
    return pd.DataFrame(absolute), differences, contrasts



def verify_saved_probes(output: Path, metrics: pd.DataFrame) -> list[dict]:
    """Require every declared fitted artifact, its exact bytes and writer replay."""
    audit = json.loads((output / "audit.json").read_text())
    artifacts = audit.get("fitted_probe_artifacts", {})
    if set(artifacts) != set(metrics.feature_set):
        raise ValueError("Missing or unexpected fitted probe artifacts")
    rows = []
    for feature, artifact in artifacts.items():
        path = Path(artifact["path"])
        if (path.resolve().parent != (output / "fitted_probes").resolve()
                or not artifact.get("raw_predictions_exact_after_reload")
                or fingerprint(path)["sha256"] != artifact["sha256"]):
            raise ValueError("Fitted probe artifact changed or failed serialization replay")
        rows.append(dict(feature_set=feature, **artifact))
    return rows

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    receipt_bytes = (args.root / "status.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    plan = receipt["plan"]
    expected = {(f, s) for f, s in product(plan["folds"], plan["seeds"])}
    observed = [(j["fold"], j["seed"]) for j in receipt["completed_jobs"]]
    if (
        receipt["stage"] not in ["report", "complete"]
        or len(observed) != len(set(observed))
        or set(observed) != expected
    ):
        raise ValueError("Incomplete replication execution")
    rows, artifacts = [], []
    for job in receipt["completed_jobs"]:
        source = Path(job["probes"])
        audit = json.loads((source / "status.json").read_text())
        if (
            audit["status"] != "complete"
            or audit["completed_commands"] != len(audit["commands"])
            or not audit["encoders_frozen"]
        ):
            raise ValueError("Incomplete frozen probe job")
        initial = {}
        for arm in plan["arms"]:
            path = Path(job["checkpoints"][arm])
            pre = json.loads((path.parent / "status.json").read_text())
            if (
                pre["status"] != "complete"
                or pre["biological_labels_used"]
                or pre["test_windows_loaded"]
                or pre["plan"]["fold"] != job["fold"]
                or pre["seed"] != job["seed"]
                or pre["arm"] != arm
                or fingerprint(path)["sha256"] != pre["checkpoint"]["sha256"]
            ):
                raise ValueError("Invalid pretraining checkpoint")
            initial[arm] = pre["initial_encoder_sha256"]
            if arm.endswith("random") and initial[arm] != pre["final_encoder_sha256"]:
                raise ValueError("Random backbone changed")
            for task in plan["tasks"]:
                output = source / "probes" / arm / task / plan["context"]
                m = pd.read_csv(output / "metrics.csv")
                if not m.probe_converged.all() or not m.probe_max_iter.eq(4000).all():
                    raise ValueError("Unconverged probe")
                checked = audited_run(
                    output, arm, task, job["fold"], job["seed"], plan["context"]
                )
                if not checked.checkpoint_sha256.eq(pre["checkpoint"]["sha256"]).all():
                    raise ValueError("Probe checkpoint differs")
                artifacts.extend(dict(fold=job["fold"], seed=job["seed"], arm=arm, task=task, **r)
                                 for r in verify_saved_probes(output, m))
                rows.append(checked)
        if any(
            initial[k + "_trained"] != initial[k + "_random"]
            for k in ["full", "coordinate"]
        ):
            raise ValueError("Random/trained initializations differ")
    frame = pd.concat(rows, ignore_index=True)
    frame["embedding_representation"] = "sequence_conditioned_masked_features"
    frame["embedding"] = np.where(frame.model.str.endswith("random"), "E_random", "E")
    absolute, paired, contrasts = summarize(frame, plan)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    # The driver later updates its live status; preserve the exact audited input.
    source_snapshot = args.out_dir / "execution_receipt.json"
    source_snapshot.write_bytes(receipt_bytes)
    pd.DataFrame(artifacts).to_csv(args.out_dir / "fitted_probe_artifacts.csv", index=False)
    for name, data in [
        ("audited_per_run", frame),
        ("absolute", absolute),
        ("paired_per_run", paired),
        ("contrasts", contrasts),
    ]:
        data.to_csv(args.out_dir / (name + ".csv"), index=False)
    write_json(
        args.out_dir / "audit.json",
        dict(
            status="complete",
            plan=plan,
            n_probe_runs=len(frame) // 4,
            n_feature_evaluations=len(frame),
            all_saved_predictions_replayed=True,
            all_probes_converged=True,
            scope=plan["scope"],
            n_verified_fitted_probe_artifacts=len(artifacts),
            receipt=fingerprint(source_snapshot),
            live_status_path=str((args.root / "status.json").resolve()),
            implementation=fingerprint(Path(__file__)),
        ),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, task in zip(axes, plan["tasks"]):
        chosen = contrasts.loc[
            contrasts.task.eq(task)
            & contrasts.scope.eq("development_excluded_four_folds")
            & contrasts.feature.eq("csht")
            & contrasts.metric.eq("auprc")
        ]
        for i, r in enumerate(chosen.itertuples()):
            ax.plot(r.mean, i, "o")
            ax.hlines(i, r.ci95_low, r.ci95_high, color=".4")
        ax.axvline(0, color=".5", lw=0.8)
        ax.set(
            yticks=range(len(chosen)),
            yticklabels=chosen.comparison,
            title=task,
            xlabel="Test Δ AUPRC after C+S+H; exploratory 95% CI",
        )
    fig.suptitle(
        "Masked-feature representation: four development-excluded chromosome folds"
    )
    save_figure(fig, args.out_dir, "masked_feature_replication")
    plt.close(fig)


if __name__ == "__main__":
    main()
