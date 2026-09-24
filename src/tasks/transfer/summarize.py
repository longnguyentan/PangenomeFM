"""Summarize the complete HG008 paired matrix without treating seeds as genomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.server.run_ccre_frozen_probe_matrix import build_jobs
from tasks.entex.analyze import BASE, FULL, estimate, paired, ranking_metrics


def validate_matrix(metrics: pd.DataFrame, expected: set[tuple]) -> None:
    if metrics.duplicated(["context", "fold", "seed", "scope", "feature_set"]).any():
        raise ValueError("Duplicate run/feature result")
    primary = metrics.loc[metrics.scope.eq("all")]
    for feature in [BASE, FULL]:
        actual = set(
            primary.loc[
                primary.feature_set.eq(feature), ["fold", "seed", "context"]
            ].itertuples(index=False, name=None)
        )
        if actual != expected:
            raise ValueError(
                "Incomplete or extra folds/seeds/contexts; no full-matrix summary"
            )
    a = primary.pivot(
        index=["fold", "seed", "context"], columns="feature_set", values="n_test"
    )
    if not a[BASE].eq(a[FULL]).all():
        raise ValueError("Paired comparators use different test counts")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(Path("configs/entex_v1.json").read_text())
    jobs = build_jobs(json.loads(Path(config["manuscript_config"]).read_text()))
    paths = sorted(args.probe_root.glob("fold_*/seed_*/*/metrics.csv"))
    if not paths:
        raise ValueError("No completed HG008 results")
    metrics = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    validate_matrix(metrics, {(j.fold, j.seed, j.closure) for j in jobs})
    predictions, regression, coverages = [], [], []
    for path in paths:
        audit = json.loads((path.parent / "audit.json").read_text())
        if audit["status"] != "complete" or audit["hg008_training_or_calibration"]:
            raise ValueError("Incomplete or non-zero-shot run")
        predictions.append(pd.read_parquet(path.parent / "predictions.parquet"))
        r = pd.read_csv(path.parent / "original_probe_regression.csv")
        r["fold"], r["seed"], r["context"] = (
            path.parts[-4],
            int(path.parts[-3][5:]),
            path.parts[-2],
        )
        regression.append(r)
        coverages.append(audit["external_feature_coverage"])
    p = pd.concat(predictions, ignore_index=True)
    if p.duplicated(["variant_id", "seed", "context", "feature_set"]).any():
        raise ValueError("HG008 variant occurs in multiple test folds")
    nfeature = p.groupby(["variant_id", "seed", "context"]).feature_set.nunique()
    if not nfeature.eq(2).all():
        raise ValueError("Missing paired variant prediction")
    universe = p.groupby(["context", "seed", "feature_set"]).variant_id.apply(
        lambda x: frozenset(x)
    )
    if len(set(universe)) != 1:
        raise ValueError("Feature universe differs across runs")
    summaries, gains, pooled = [], [], []
    for (context, scope), frame in metrics.groupby(["context", "scope"]):
        for name, group in frame.groupby("feature_set"):
            for metric in ["auprc", "auroc", "balanced_accuracy", "f1"]:
                summaries.append(
                    dict(
                        context=context,
                        scope=scope,
                        feature_set=name,
                        metric=metric,
                        **estimate(group, metric, 10000, 20260922),
                    )
                )
        for metric in ["auprc", "auroc"]:
            gains.append(
                dict(
                    context=context,
                    scope=scope,
                    metric=metric,
                    comparison="Delta_T_given_C_S",
                    **paired(frame, BASE, 10000, 20260922, metric=metric),
                )
            )
    for (context, seed, feature), frame in p.groupby(
        ["context", "seed", "feature_set"]
    ):
        pooled.append(
            dict(
                context=context,
                seed=seed,
                feature_set=feature,
                n=len(frame),
                prevalence=float(frame.y_true.mean()),
                **ranking_metrics(frame.y_true, frame.p_calibrated),
            )
        )
    summary, gain = pd.DataFrame(summaries), pd.DataFrame(gains)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in [
        ("per_run", metrics),
        ("summary", summary),
        ("paired_gains", gain),
        ("pooled_crossfit", pd.DataFrame(pooled)),
        ("original_probe_regression", pd.concat(regression)),
    ]:
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout="constrained")
    for i, context in enumerate(["strict", "1hop"]):
        frame = summary.loc[
            summary.context.eq(context)
            & summary.scope.eq("all")
            & summary.metric.eq("auprc")
        ]
        for j, feature in enumerate([BASE, FULL]):
            row = frame.loc[frame.feature_set.eq(feature)].iloc[0]
            x = i * 3 + j
            axes[0].plot(x, row["mean"], "o", color=["#7b8794", "#245a81"][j])
            axes[0].vlines(
                x, row.ci95_low, row.ci95_high, color=["#7b8794", "#245a81"][j]
            )
        row = gain.loc[
            gain.context.eq(context) & gain.scope.eq("all") & gain.metric.eq("auprc")
        ].iloc[0]
        axes[1].plot(i, row["mean"], "o", color="#245a81")
        axes[1].vlines(i, row.ci95_low, row.ci95_high, color="#245a81")
    axes[0].set(
        xticks=[0, 1, 3, 4],
        xticklabels=["C+S\nstrict", "C+S+T\nstrict", "C+S\n1-hop", "C+S+T\n1-hop"],
        ylim=(0, 1),
        ylabel="Fold-mean AUPRC",
    )
    axes[1].set(xticks=[0, 1], xticklabels=["strict", "1-hop"], ylabel="Paired Δ AUPRC")
    axes[1].axhline(0, color="0.5", linewidth=0.8)
    fig.suptitle("HG008 clonal insertion vs deletion — one external genome")
    for extension in ["png", "svg"]:
        fig.savefig(args.out_dir / f"hg008_transfer.{extension}", dpi=300)
    plt.close(fig)
    (args.out_dir / "audit.json").write_text(
        json.dumps(
            dict(
                status="complete",
                n_runs=len(paths),
                external_variants=int(p.variant_id.nunique()),
                minimum_feature_coverage=min(coverages),
                primary_context="strict",
                primary_metric="auprc",
                uncertainty="paired chromosome-fold then seed bootstrap, 10000 draws; descriptive one-genome result",
                pooled_scores="secondary cross-fitted scores; fold-specific calibrations can affect pooled ranking",
                null_or_negative_results_retained=True,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
