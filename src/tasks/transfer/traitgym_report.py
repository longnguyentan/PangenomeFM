"""Replay all TraitGym predictions and summarize the prespecified paired contrasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from evaluation.paired_inference import bh_adjust, fold_sign_flip
from scripts.server.run_ccre_frozen_probe_fold import binary_metrics
from tasks.entex.analyze import BASE, CT, FULL, LABELS, estimate
from tasks.entex.prepare import fingerprint
from tasks.transfer.report import save_figure
from tasks.transfer.traitgym import IDENTITY, verified_fingerprint, weighted_chromosome_ap, write_json

CSH = BASE + "_plus_topology_control"
CSHT = CSH + "_plus_frozen_pangenomefm"
METRICS = ["auprc", "chromosome_weighted_auprc", "auroc", "normalized_ap", "balanced_accuracy", "f1",
           "precision", "recall"]


def replay_run(directory: Path, plan: dict, test_chromosomes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = json.loads((directory / "audit.json").read_text())
    if audit.get("status") != "complete" or audit.get("n_excluded") != 0:
        raise ValueError("Incomplete run or excluded variants")
    verified_fingerprint(directory / "predictions.parquet", audit["predictions"]["sha256"])
    frame = pd.read_parquet(directory / "predictions.parquet")
    metrics = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
    if (metrics.feature_set.duplicated().any() or set(metrics.feature_set) != set(plan["feature_sets"])
            or set(frame.feature_set) != set(plan["feature_sets"])):
        raise ValueError("Missing or duplicated feature sets")
    metrics = metrics.set_index("feature_set")
    identity, chrom_rows = None, []
    for feature in plan["feature_sets"]:
        p = frame.loc[frame.feature_set.eq(feature)].sort_values("variant_id")
        m = metrics.loc[feature]
        if (p.variant_id.duplicated().any() or not p.label.eq(p.y_true).all()
                or set(p.chromosome) != set(test_chromosomes) or not p.chrom.eq(p.chromosome).all()
                or len(p) != m.n_test or not p.threshold.eq(m.threshold).all()
                or not np.isfinite(p.p_calibrated).all() or not p.p_calibrated.between(0, 1).all()
                or m.probe_max_iter != plan["probe_max_iter"]):
            raise ValueError("Prediction identities, chromosome partition, numerical budget or labels differ")
        current = p[IDENTITY].reset_index(drop=True)
        if identity is not None:
            pd.testing.assert_frame_equal(identity, current)
        identity = current
        for key in ["task", "dataset", "fold", "seed", "context"]:
            if not p[key].eq(m[key]).all():
                raise ValueError("Metric/prediction run identity differs")
        replay = binary_metrics(p.y_true.to_numpy(), p.p_calibrated.to_numpy(), float(m.threshold))
        for chrom, group in p.groupby("chromosome"):
            chrom_rows.append(dict(feature_set=feature, chromosome=chrom,
                **binary_metrics(group.y_true.to_numpy(), group.p_calibrated.to_numpy(), float(m.threshold))))
        chromosome_frame = pd.DataFrame(chrom_rows)
        replay.update(chromosome_weighted_auprc=weighted_chromosome_ap(
            chromosome_frame.loc[chromosome_frame.feature_set.eq(feature)]),
            balanced_accuracy=float(balanced_accuracy_score(p.y_true, p.y_pred)),
            normalized_ap=(replay["auprc"] - replay["positive_fraction"]) / (1 - replay["positive_fraction"]))
        if not p.y_pred.eq((p.p_calibrated >= m.threshold).astype(int)).all():
            raise ValueError("Saved classifications differ from thresholded probabilities")
        for key in [*METRICS, "positive_fraction"]:
            if not np.isclose(m[key], replay[key], atol=1e-10, rtol=0):
                raise ValueError(f"Stored {key} differs from replay: {directory}/{feature}")
    return metrics.reset_index(), identity


def summarize(metrics: pd.DataFrame, plan: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    absolute, paired = [], []
    if metrics.duplicated(["dataset", "context", "fold", "seed", "feature_set"]).any():
        raise ValueError("Duplicate runs")
    n_boot, seed = plan["n_bootstrap"], plan["statistics_seed"]
    for (dataset, context), group in metrics.groupby(["dataset", "context"]):
        for metric in METRICS:
            for feature, values in group.groupby("feature_set"):
                absolute.append(dict(dataset=dataset, context=context, metric=metric, feature_set=feature,
                                     **estimate(values, metric, n_boot, seed)))
            if metric not in ["auprc", "chromosome_weighted_auprc", "auroc", "normalized_ap"]:
                continue
            wide = group.pivot(index=["fold", "seed"], columns="feature_set", values=metric)
            if wide.isna().any().any():
                raise ValueError("Missing paired result")
            for name, left, right in [("T_given_CS", FULL, BASE), ("T_given_CSH", CSHT, CSH),
                                      ("S_given_CT", FULL, CT)]:
                paired.append((wide[left] - wide[right]).rename("gain").reset_index().assign(
                    dataset=dataset, context=context, metric=metric, contrast=name))
    paired = pd.concat(paired, ignore_index=True)
    contrasts = []
    for keys, group in paired.groupby(["dataset", "context", "metric", "contrast"]):
        contrasts.append(dict(zip(["dataset", "context", "metric", "contrast"], keys),
            **estimate(group, "gain", n_boot, seed), sign_flip_p=fold_sign_flip(group)))
    contrasts = pd.DataFrame(contrasts)
    contrasts["bh_q_within_metric"] = contrasts.groupby("metric").sign_flip_p.transform(bh_adjust)
    return pd.DataFrame(absolute), contrasts, paired


def expected_test_support(qc: dict, chromosomes: list[str]) -> tuple[list[str], int]:
    """A held-out chromosome may legitimately have zero variants in the source."""
    counts = qc["chromosome_counts"]
    observed = [chrom for chrom in chromosomes if counts.get(chrom, 0) > 0]
    return observed, sum(counts[chrom] for chrom in observed)


def figures(absolute: pd.DataFrame, contrasts: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    colors = {"strict": "#245a81", "1hop": "#be6831"}
    for ax, dataset in zip(axes, ["complex_traits", "mendelian_traits"]):
        subset = absolute.loc[absolute.dataset.eq(dataset) & absolute.metric.eq("auprc")]
        features = list(dict.fromkeys([*LABELS, CSH, CSHT]))
        for i, (context, color) in enumerate(colors.items()):
            indexed = subset.loc[subset.context.eq(context)].set_index("feature_set")
            for j, feature in enumerate(features):
                if feature not in indexed.index:
                    continue
                r = indexed.loc[feature]
                y = j + (i - .5) * .18
                ax.plot(r["mean"], y, "o", color=color, label=context if j == 0 else None)
                ax.hlines(y, r.ci95_low, r.ci95_high, color=color)
        names = {**LABELS, CSH: "C+S+H", CSHT: "C+S+H+T"}
        ax.set(yticks=range(len(features)), yticklabels=[names[f] for f in features], xlim=(0, 1),
               xlabel="Mean fold AUPRC (95% bootstrap interval)", title=dataset.replace("_", " "))
        ax.axvline(.1, color=".6", ls=":", label="prevalence 0.10")
        ax.invert_yaxis()
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("TraitGym locus priors: five-fold adaptation; frozen v1 encoders")
    save_figure(fig, out, "traitgym_feature_ap")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    data = contrasts.loc[contrasts.metric.eq("auprc") & contrasts.contrast.isin(["T_given_CS", "T_given_CSH"])]
    for i, (_, r) in enumerate(data.iterrows()):
        ax.plot(r["mean"], i, "o", color=colors[r.context])
        ax.hlines(i, r.ci95_low, r.ci95_high, color=colors[r.context])
    ax.set(yticks=range(len(data)), yticklabels=[f"{r.dataset} · {r.context} · {r.contrast}"
        for r in data.itertuples()], xlabel="Paired Δ AUPRC (95% bootstrap interval)")
    ax.axvline(0, color=".5", lw=.8)
    ax.invert_yaxis()
    save_figure(fig, out, "traitgym_topology_gains")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    receipt = json.loads((args.root / "status.json").read_text())
    if receipt["status"] != "complete" or receipt["completed_runs"] != receipt["planned_runs"]:
        raise ValueError("Only summarize the complete declared run matrix")
    plan, rows, identities, sources = receipt["plan"], [], {}, []
    qc = json.loads((args.root / "qc.json").read_text())
    for job in receipt["jobs"]:
        for name in plan["datasets"]:
            path = args.root / name / job["fold"] / f"seed_{job['seed']}" / job["closure"]
            observed_chromosomes, expected_n = expected_test_support(qc[name], job["test"])
            metrics, identity = replay_run(path, plan, observed_chromosomes)
            if len(identity) != expected_n:
                raise ValueError("Test predictions dropped original source rows")
            for key, expected in dict(dataset=name, fold=job["fold"], seed=job["seed"], context=job["closure"]).items():
                if not metrics[key].eq(expected).all():
                    raise ValueError("Declared job differs from actual result")
            identity_key = (name, job["fold"])
            if identity_key in identities:
                pd.testing.assert_frame_equal(identities[identity_key], identity)
            identities[identity_key] = identity
            rows.append(metrics)
            sources.extend(fingerprint(path / f) for f in ["audit.json", "metrics.csv", "predictions.parquet"])
    if receipt["scope"] == "full_matrix":
        for name in plan["datasets"]:
            universe = pd.concat([v for (dataset, _), v in identities.items() if dataset == name])
            if (universe.variant_id.duplicated().any() or len(universe) != qc[name]["n"]
                    or universe.label.sum() != qc[name]["positives"]):
                raise ValueError("Test folds do not retain the full original variant universe")
    metrics = pd.concat(rows, ignore_index=True)
    absolute, contrasts, paired = summarize(metrics, plan)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in [("per_run", metrics), ("absolute", absolute), ("contrasts", contrasts), ("paired", paired)]:
        frame.to_csv(args.out_dir / (name + ".csv"), index=False)
    figures(absolute, contrasts, args.out_dir)
    write_json(args.out_dir / "audit.json", dict(status="complete", scope=receipt["scope"],
        all_probes_converged=bool(metrics.probe_converged.all()), n_runs=len(rows), n_fits=len(metrics),
        numerical_gate="pass" if metrics.probe_converged.all() else "optimization_incomplete",
        source_root=str(args.root), sources=sources, plan=plan,
        interpretation="Exploratory pointwise fold/seed intervals; seed repetitions are not independent biological samples. No performance-based subset selection."))
    (args.out_dir / "README.md").write_text(
        "# TraitGym frozen locus-prior results\n\n" + plan["interpretation"] + "\n\n"
        + f"Completed {len(rows)} declared runs; {len(metrics)} fits. All converged: {bool(metrics.probe_converged.all())}.\n\n"
        + "## All primary and handcrafted-control AUPRC contrasts\n\n"
        + contrasts.loc[contrasts.metric.eq("auprc")].to_markdown(index=False) + "\n\n"
        + "Intervals are pointwise hierarchical fold/seed bootstrap. Exact fold sign-flip p-values and BH-adjusted q-values are also supplied; five folds limit inferential resolution.\n")


if __name__ == "__main__":
    main()
