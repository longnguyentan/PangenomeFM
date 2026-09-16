"""Paired fold/seed summaries and fixed native-complexity P0 strata."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from scripts.server.aggregate_ccre_frozen_probes import _hierarchical_draws

BASE = "coordinate_plus_frozen_sequence_fm"
FULL = BASE + "_plus_frozen_pangenomefm"
CT = "coordinate_plus_frozen_pangenomefm"
LABELS = {
    "coordinate": "C",
    "sequence_kmer": "K",
    "frozen_sequence_fm": "S",
    "frozen_pangenomefm": "T",
    BASE: "C+S",
    CT: "C+T",
    FULL: "C+S+T",
}


def estimate(group: pd.DataFrame, column: str, n_bootstrap: int, seed: int) -> dict:
    valid = group.loc[np.isfinite(group[column])]
    values = valid[column].to_numpy()
    if not len(values):
        return dict(
            mean=np.nan,
            std=np.nan,
            ci95_low=np.nan,
            ci95_high=np.nan,
            n_runs=0,
            n_folds=0,
            undefined_runs=len(group),
        )
    arrays = [x[column].to_numpy() for _, x in valid.groupby("fold", sort=True)]
    # One fold is a smoke test, not a population uncertainty estimate.
    draws = (
        _hierarchical_draws(
            arrays, n_bootstrap=n_bootstrap, rng=np.random.default_rng(seed)
        )
        if len(arrays) > 1
        else np.array([np.nan])
    )
    return dict(
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        ci95_low=float(np.quantile(draws, 0.025)),
        ci95_high=float(np.quantile(draws, 0.975)),
        n_runs=len(values),
        n_folds=len(arrays),
        undefined_runs=len(group) - len(valid),
    )


def paired(
    frame: pd.DataFrame,
    baseline: str,
    n_bootstrap: int,
    seed: int,
    *,
    metric: str = "auprc",
) -> dict:
    if frame.duplicated(["fold", "seed", "feature_set"]).any():
        raise ValueError("Duplicate fold/seed/feature result")
    wide = frame.pivot(index=["fold", "seed"], columns="feature_set", values=metric)
    if baseline not in wide or FULL not in wide:
        raise ValueError("Missing paired comparator")
    if wide[[baseline, FULL]].isna().any(axis=1).any():
        # Both undefined is legitimate in a single-class stratum; one missing is not.
        if (wide[baseline].isna() != wide[FULL].isna()).any():
            raise ValueError("Unpaired missing scores")
    gain = (wide[FULL] - wide[baseline]).rename("gain").reset_index()
    return estimate(gain, "gain", n_bootstrap, seed)


def ranking_metrics(labels: pd.Series, scores: pd.Series) -> dict:
    prevalence = float(labels.mean())
    if labels.nunique() != 2:
        return dict(auprc=np.nan, auroc=np.nan, normalized_ap=np.nan)
    ap = float(average_precision_score(labels, scores))
    return dict(
        auprc=ap,
        auroc=float(roc_auc_score(labels, scores)),
        normalized_ap=(ap - prevalence) / (1 - prevalence),
    )


def validate_comparator_loci(predictions: pd.DataFrame) -> None:
    selected = predictions.loc[predictions.feature_set.isin([BASE, FULL])]
    identity = "measurement_id" if "measurement_id" in selected else "locus_id"
    if selected.duplicated([identity, "feature_set"]).any():
        raise ValueError("Duplicate locus predictions within comparator")
    if not selected.groupby(identity).feature_set.nunique().eq(2).all():
        raise ValueError("Comparators do not share exactly the same loci")
    if not selected.groupby(identity).y_true.nunique().eq(1).all():
        raise ValueError("Comparator labels differ")


def complexity_labels(loci: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Use existing native v2 strict categories at the locus start, without rebinning."""
    c = pd.read_csv(path, sep="\t")
    c = c.loc[c.context.eq("strict")].copy()
    widths = c.end - c.start
    if widths.nunique() != 1 or widths.iloc[0] <= 0:
        raise ValueError("Expected fixed-width native complexity windows")
    if c.duplicated(["chromosome", "start"]).any():
        raise ValueError("Duplicate native complexity windows")
    result = loci.copy()
    result["window_start"] = (result.start // int(widths.iloc[0])) * int(widths.iloc[0])
    result = result.merge(
        c[["chromosome", "start", "locus_complexity_category"]].rename(
            columns={
                "chromosome": "chrom",
                "start": "window_start",
                "locus_complexity_category": "complexity",
            }
        ),
        on=["chrom", "window_start"],
        how="left",
        validate="many_to_one",
    )
    result["complexity"] = result.complexity.fillna("unassigned")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--complexity", type=Path, required=True)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    args = ap.parse_args()
    paths = sorted(args.probe_root.glob("fold_*/seed_*/*/metrics.csv"))
    if not paths:
        raise FileNotFoundError("No completed P0 runs")
    metrics = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if metrics.duplicated(["fold", "seed", "closure", "feature_set"]).any():
        raise ValueError("Duplicate runs")
    if "subtask" in metrics and metrics.subtask.nunique() != 1:
        raise ValueError("Analyze each prespecified sensitivity in a separate root")
    task = str(metrics.task.iloc[0]) if "task" in metrics else "p0"
    if "task" in metrics and metrics.task.nunique() != 1:
        raise ValueError("Mixed biological tasks in analysis root")
    rows = []
    gains = []
    strata = []
    for (context, feature), g in metrics.groupby(["closure", "feature_set"]):
        for metric in [
            "auprc",
            "auroc",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            *(["normalized_ap"] if "normalized_ap" in metrics else []),
        ]:
            rows.append(
                dict(
                    context=context,
                    feature_set=feature,
                    metric=metric,
                    **estimate(g, metric, args.n_bootstrap, 20260806),
                )
            )
    for context, g in metrics.groupby("closure"):
        for base, name in [(BASE, "Delta_T_given_C_S"), (CT, "Delta_S_given_C_T")]:
            gains.append(
                dict(
                    context=context,
                    comparison=name,
                    **paired(g, base, args.n_bootstrap, 20260806),
                )
            )
    for path in paths:
        p = pd.read_parquet(path.parent / "predictions.parquet")
        validate_comparator_loci(p)
        unique = p[["locus_id", "chrom", "start", "end"]].drop_duplicates()
        c = complexity_labels(unique, args.complexity)
        p = p.merge(
            c[["locus_id", "complexity"]], on="locus_id", validate="many_to_one"
        )
        for (context, fold, seed, complexity, feature), g in p.groupby(
            ["closure", "fold", "seed", "complexity", "feature_set"]
        ):
            if feature not in [BASE, FULL]:
                continue
            strata.append(
                dict(
                    closure=context,
                    fold=fold,
                    seed=seed,
                    stratum=complexity,
                    feature_set=feature,
                    n=len(g),
                    positive_prevalence=float(g.y_true.mean()),
                    **ranking_metrics(g.y_true, g.p_calibrated),
                )
            )
    strata = pd.DataFrame(strata)
    stratum_summary = []
    stratum_metric_summary = []
    for (context, stratum), g in strata.groupby(["closure", "stratum"]):
        row = dict(
            context=context,
            stratum=stratum,
            **paired(g, BASE, args.n_bootstrap, 20260806),
        )
        for feature, short in [(BASE, "C_S"), (FULL, "C_S_T")]:
            row["auprc_" + short] = g.loc[g.feature_set.eq(feature), "auprc"].mean()
        counts = g.loc[g.feature_set.eq(BASE)]
        row["n_test_mean"] = counts.n.mean()
        row["positive_prevalence_mean"] = counts.positive_prevalence.mean()
        stratum_summary.append(row)
        for metric in ["auprc", "auroc", "normalized_ap"]:
            common = dict(
                context=context,
                stratum=stratum,
                metric=metric,
                n_test_mean=counts.n.mean(),
                positive_prevalence_mean=counts.positive_prevalence.mean(),
            )
            stratum_metric_summary.append(
                dict(
                    **common,
                    comparison="Delta_T_given_C_S",
                    **paired(g, BASE, args.n_bootstrap, 20260806, metric=metric),
                )
            )
            for feature in [BASE, FULL]:
                stratum_metric_summary.append(
                    dict(
                        **common,
                        comparison=feature,
                        **estimate(
                            g.loc[g.feature_set.eq(feature)],
                            metric,
                            args.n_bootstrap,
                            20260806,
                        ),
                    )
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    gains = pd.DataFrame(gains)
    ss = pd.DataFrame(stratum_summary)
    for name, frame in [
        ("per_run", metrics),
        ("summary", summary),
        ("paired_gains", gains),
        ("complexity_per_run", strata),
        ("complexity_summary", ss),
        ("complexity_metric_summary", pd.DataFrame(stratum_metric_summary)),
    ]:
        frame.to_csv(args.out_dir / (name + ".csv"), index=False)
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
    for context in metrics.closure.unique():
        for name, frame, label_col, title, zero_axis in [
            (
                "auprc",
                summary.loc[summary.context.eq(context) & summary.metric.eq("auprc")],
                "feature_set",
                "Active/repressed dELS AUPRC"
                if task == "p1"
                else (
                    "Allele-specific SNV AUPRC"
                    if task == "p2"
                    else "AS-prone cCRE AUPRC"
                ),
                True,
            ),
            (
                "gain",
                gains.loc[gains.context.eq(context)],
                "comparison",
                "Paired modality gain",
                False,
            ),
            (
                "complexity_gain",
                ss.loc[ss.context.eq(context)],
                "stratum",
                "Topology gain by native graph complexity",
                False,
            ),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
            for i, row in enumerate(frame.to_dict("records")):
                ax.plot(i, row["mean"], "o", color="#245a81")
                if np.isfinite(row["ci95_low"]):
                    ax.vlines(i, row["ci95_low"], row["ci95_high"], color="#245a81")
            ax.set_xticks(
                range(len(frame)),
                [LABELS.get(x, x) for x in frame[label_col]],
                rotation=20,
            )
            ax.set_title(f"{title} ({context})")
            ax.set_ylabel("AUPRC" if zero_axis else "Δ AUPRC")
            if zero_axis:
                ax.set_ylim(0, 1)
            else:
                ax.axhline(0, color="0.5", linewidth=0.8)
            for ext in ["png", "svg"]:
                fig.savefig(args.out_dir / f"{name}_{context}.{ext}", dpi=300)
            plt.close(fig)
    (args.out_dir / "audit.json").write_text(
        json.dumps(
            dict(
                task=task,
                runs=len(paths),
                resampling="chromosome fold then seed; paired differences before bootstrap",
                single_fold_ci="undefined",
                subtask=str(metrics.subtask.iloc[0])
                if "subtask" in metrics
                else "unspecified",
                normalized_ap="(AP - prevalence) / (1 - prevalence); undefined for single-class strata; not full prevalence invariance",
                complexity_assignment="existing strict native v2 category at locus start",
                complexity_source=str(args.complexity.resolve()),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
