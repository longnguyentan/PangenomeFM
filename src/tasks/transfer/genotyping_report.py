"""Replay and summarize held-out genotyping-quality predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.paired_inference import bh_adjust, fold_sign_flip
from tasks.entex.analyze import estimate
from tasks.transfer.regression_probe import regression_metrics
from tasks.transfer.report import save_figure
from tasks.transfer.traitgym import verified_fingerprint, write_json


def replay(directory: Path, loci: pd.DataFrame, job: dict, plan: dict) -> pd.DataFrame:
    audit = json.loads((directory / "audit.json").read_text())
    if audit.get("status") != "complete" or audit.get("n_excluded") != 0 or audit.get("feature_coverage") != 1:
        raise ValueError("Incomplete or filtered regression run")
    verified_fingerprint(directory / "predictions.parquet", audit["predictions"]["sha256"])
    frame = pd.read_parquet(directory / "predictions.parquet")
    metrics = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
    sweep = pd.read_csv(directory / "validation_selection.csv", float_precision="round_trip")
    keys = ["target", "feature_set"]
    expected = {(t, f) for t in plan["targets"] for f in plan["feature_sets"]}
    if (metrics.duplicated(keys).any() or set(metrics[keys].itertuples(index=False, name=None)) != expected
            or set(frame[keys].itertuples(index=False, name=None)) != expected):
        raise ValueError("Incomplete feature/target factorial")
    truth = loci.loc[loci.chrom.isin(job["test"])].sort_values("locus_id")
    n_val = int(loci.chrom.isin(job["validation"]).sum())
    for m in metrics.itertuples():
        p = frame.loc[frame.target.eq(m.target) & frame.feature_set.eq(m.feature_set)].sort_values("locus_id")
        if (p.locus_id.duplicated().any() or p.locus_id.tolist() != truth.locus_id.tolist()
                or not np.array_equal(p.y_true, truth[m.target])
                or not np.array_equal(p.chrom, truth.chrom) or m.n_test != len(truth)
                or m.n_val != n_val or m.n_train != len(loci)-n_val-len(truth)):
            raise ValueError("Regression outcome universe or partition differs")
        for key, value in dict(fold=job["fold"], seed=job["seed"], context=job["closure"], task=plan["task"]).items():
            if not p[key].eq(value).all() or getattr(m, key) != value:
                raise ValueError("Incorrect run identity")
        for key, value in regression_metrics(p.y_true, p.prediction).items():
            if not np.isclose(getattr(m, key), value, atol=1e-10, rtol=0, equal_nan=True):
                raise ValueError("Metric differs from saved prediction replay")
        if m.feature_set == "train_median":
            train = ~loci.chrom.isin([*job["test"], *job["validation"]])
            if not np.allclose(p.prediction, np.median(loci.loc[train, m.target]), rtol=0, atol=1e-12):
                raise ValueError("Constant reference is not the training median")
        else:
            part = sweep.loc[sweep.target.eq(m.target) & sweep.feature_set.eq(m.feature_set)]
            if part.alpha.duplicated().any() or set(part.alpha) != set(plan["ridge_alphas"]):
                raise ValueError("Incomplete validation penalty selection")
            selected = part.sort_values(["validation_mae", "alpha"], ascending=[True, False]).iloc[0]
            if m.alpha != selected.alpha or m.validation_mae != selected.validation_mae:
                raise ValueError("Saved model was not selected by predefined validation rule")
    return metrics


def summarize(frame: pd.DataFrame, plan: dict):
    if frame.duplicated(["target", "context", "fold", "seed", "feature_set"]).any():
        raise ValueError("Duplicate regression runs")
    absolute, paired = [], []
    for (target, context), group in frame.groupby(["target", "context"]):
        for metric in plan["metrics"]:
            for feature, part in group.groupby("feature_set"):
                absolute.append(dict(target=target, context=context, metric=metric, feature_set=feature,
                    **estimate(part, metric, plan["n_bootstrap"], plan["statistics_seed"])))
            wide = group.pivot(index=["fold", "seed"], columns="feature_set", values=metric)
            for contrast, full, base in plan["comparisons"]:
                sign = -1 if metric in ["mae", "rmse"] else 1
                if wide[[full, base]].isna().any().any():
                    raise ValueError("Undefined primary paired model contrast")
                paired.append((sign*(wide[full]-wide[base])).rename("gain").reset_index().assign(
                    target=target, context=context, metric=metric, contrast=contrast))
    pairs = pd.concat(paired, ignore_index=True)
    rows = []
    for keys, group in pairs.groupby(["target", "context", "metric", "contrast"]):
        rows.append(dict(zip(["target", "context", "metric", "contrast"], keys),
            **estimate(group, "gain", plan["n_bootstrap"], plan["statistics_seed"]), sign_flip_p=fold_sign_flip(group)))
    contrasts = pd.DataFrame(rows)
    contrasts["bh_q_within_metric"] = contrasts.groupby("metric").sign_flip_p.transform(bh_adjust)
    return pd.DataFrame(absolute), contrasts, pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--loci", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    receipt = json.loads((args.root / "status.json").read_text())
    if receipt["status"] != "complete" or receipt["completed_runs"] != len(receipt["jobs"]):
        raise ValueError("Incomplete source experiment")
    plan = receipt["plan"]
    verified_fingerprint(args.loci, plan["loci_sha256"])
    loci = pd.read_parquet(args.loci)
    records = [replay(args.root/j["fold"]/f"seed_{j['seed']}"/j["closure"], loci, j, plan) for j in receipt["jobs"]]
    metrics = pd.concat(records, ignore_index=True)
    absolute, contrasts, pairs = summarize(metrics, plan)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in [("per_run", metrics), ("absolute", absolute), ("contrasts", contrasts), ("paired", pairs)]:
        frame.to_csv(args.out_dir / (name+".csv"), index=False)
    write_json(args.out_dir / "audit.json", dict(status="complete", n_runs=len(receipt["jobs"]),
        n_evaluations=len(metrics), n_loci=len(loci), n_excluded=0, all_metrics_replayed=True,
        all_selected_alphas_match_validation_minimum=True,
        scope="five-fold exploratory evaluation; small number of loci; no random-encoder superiority claim"))
    lines = ["# Measured genotyping quality: held-out regression", "", "Positive gain means lower MAE after adding T.", "",
             "| Target | Context | Contrast | MAE gain | 95% CI | Fold p |", "|---|---|---|---:|---|---:|"]
    for row in contrasts.loc[contrasts.metric.eq("mae")].itertuples():
        lines.append(f"| {row.target} | {row.context} | {row.contrast} | {row.mean:+.6f} | "
                     f"[{row.ci95_low:+.6f}, {row.ci95_high:+.6f}] | {row.sign_flip_p:.4f} |")
    (args.out_dir / "README.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, target in zip(axes, plan["targets"]):
        part = contrasts.loc[contrasts.target.eq(target) & contrasts.metric.eq("mae")]
        for i, row in enumerate(part.itertuples()):
            ax.scatter(row.mean, i, color="#245a81" if row.context == "strict" else "#be6831")
            ax.hlines(i, row.ci95_low, row.ci95_high, color=".4")
        ax.axvline(0, color=".6", lw=.8)
        ax.set(yticks=range(len(part)), yticklabels=[f"{r.context}: {r.contrast}" for r in part.itertuples()],
               title=target, xlabel="MAE baseline − MAE with T (95% CI)")
        ax.invert_yaxis()
    fig.suptitle("Measured genotyping quality: 265 loci, chromosome-held-out")
    save_figure(fig, args.out_dir, "genotyping_quality_gains")
    plt.close(fig)


if __name__ == "__main__":
    main()
