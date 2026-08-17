#!/usr/bin/env python3
"""Aggregate cross-fitted SV probes and their prespecified strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.server.aggregate_ccre_frozen_probes import (
    LOWER_IS_BETTER,
    METRICS,
    PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST,
    _hierarchical_gain_interval,
    hierarchical_summary,
    modality_contrasts,
    paired_context_summary,
    paired_modality_contribution_summary,
    validate_exact_feature_universe,
)


def stratified_interpretation_status(group: pd.DataFrame, *, stratum_levels: int) -> str:
    """Describe whether a stratum supports ranking interpretation."""

    if stratum_levels < 2:
        return "non_informative_single_level"
    finite_ranking = int(group[["auroc", "auprc"]].notna().all(axis=1).sum())
    if finite_ranking == 0:
        return "single_class_ranking_undefined"
    if float(group["n"].mean()) < 100:
        return "underpowered_mean_n_lt_100"
    return "descriptive_ranking_supported"


def summarize_strata(strata: pd.DataFrame) -> pd.DataFrame:
    """Summarize strata while making single-class and tiny-bin limits explicit."""

    level_counts = strata.groupby("stratum")["stratum_value"].nunique(dropna=False)
    rows: list[dict[str, object]] = []
    for keys, group in strata.groupby(
        ["closure", "feature_set", "stratum", "stratum_value"],
        dropna=False,
        sort=True,
    ):
        closure, feature_set, stratum, stratum_value = keys
        positive_fraction = (
            group["positive_fraction"]
            if "positive_fraction" in group
            else pd.Series(np.nan, index=group.index, dtype=float)
        )
        rows.append(
            {
                "closure": closure,
                "feature_set": feature_set,
                "stratum": stratum,
                "stratum_value": stratum_value,
                "n": int(group["n"].sum()),
                "mean_n_per_run": float(group["n"].mean()),
                "minimum_n_per_run": int(group["n"].min()),
                "maximum_n_per_run": int(group["n"].max()),
                "mean_positive_fraction": float(positive_fraction.mean()),
                "minimum_positive_fraction": float(positive_fraction.min()),
                "maximum_positive_fraction": float(positive_fraction.max()),
                "mean_auroc": float(group["auroc"].mean()),
                "mean_auprc": float(group["auprc"].mean()),
                "mean_accuracy": float(group["accuracy"].mean()),
                "mean_f1": float(group["f1"].mean()),
                "ranking_metric_valid_runs": int(
                    group[["auroc", "auprc"]].notna().all(axis=1).sum()
                ),
                "runs": int(len(group)),
                "folds": int(group["fold"].nunique()),
                "seeds": int(group["seed"].nunique()),
                "interpretation_status": stratified_interpretation_status(
                    group,
                    stratum_levels=int(level_counts.loc[stratum]),
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_primary_stratum_contributions(
    strata: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Estimate the primary topology-over-sequence-FM gain within SV strata."""

    suffix = "_pair"
    larger, smaller = modality_contrasts(suffix=suffix)[
        PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST
    ]
    available = set(strata["feature_set"])
    if larger not in available or smaller not in available:
        return pd.DataFrame()
    keys = ["fold", "seed", "closure", "stratum", "stratum_value"]
    left = strata.loc[strata["feature_set"].eq(larger), keys + ["n", *METRICS]]
    right = strata.loc[strata["feature_set"].eq(smaller), keys + ["n", *METRICS]]
    paired = left.merge(
        right,
        on=keys,
        how="inner",
        suffixes=("_larger", "_smaller"),
        validate="one_to_one",
    )
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError("Unpaired primary sequence/topology SV stratum rows")
    level_counts = strata.groupby("stratum")["stratum_value"].nunique(dropna=False)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for keys_value, group in paired.groupby(
        ["closure", "stratum", "stratum_value"],
        dropna=False,
        sort=True,
    ):
        closure, stratum, stratum_value = keys_value
        for metric in METRICS:
            raw_delta = (
                group[f"{metric}_larger"].to_numpy(float)
                - group[f"{metric}_smaller"].to_numpy(float)
            )
            gain = -raw_delta if metric in LOWER_IS_BETTER else raw_delta
            finite = np.isfinite(gain)
            valid = group.loc[finite, ["fold", "seed"]].copy()
            valid["gain"] = gain[finite]
            if valid.empty:
                continue
            ci95_low, ci95_high = _hierarchical_gain_interval(
                valid,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            # Positive fraction is not part of METRICS, so retrieve complete
            # source rows when assigning the interpretation safeguard.
            source_group = strata.loc[
                strata["feature_set"].eq(larger)
                & strata["closure"].eq(closure)
                & strata["stratum"].eq(stratum)
                & strata["stratum_value"].astype(str).eq(str(stratum_value))
            ]
            status = stratified_interpretation_status(
                source_group,
                stratum_levels=int(level_counts.loc[stratum]),
            )
            rows.append(
                {
                    "closure": closure,
                    "stratum": stratum,
                    "stratum_value": stratum_value,
                    "contrast": PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST,
                    "larger_feature_set": larger,
                    "smaller_feature_set": smaller,
                    "metric": metric,
                    "mean_gain": float(valid["gain"].mean()),
                    "ci95_low": ci95_low,
                    "ci95_high": ci95_high,
                    "n_paired_runs": int(len(valid)),
                    "n_folds": int(valid["fold"].nunique()),
                    "n_seeds": int(valid["seed"].nunique()),
                    "mean_examples_per_run": float(source_group["n"].mean()),
                    "interpretation_status": status,
                    "gain_definition": (
                        "smaller_minus_larger"
                        if metric in LOWER_IS_BETTER
                        else "larger_minus_smaller"
                    ),
                    "positive_means_topology_improves": True,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path)
    parser.add_argument(
        "--fold-metrics",
        type=Path,
        help="Combined fold metrics CSV; useful for evidence-package reanalysis",
    )
    parser.add_argument(
        "--stratified-fold-metrics",
        type=Path,
        help="Combined stratified metrics CSV paired with --fold-metrics",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    combined = args.fold_metrics is not None or args.stratified_fold_metrics is not None
    if combined:
        if args.probe_root is not None or args.fold_metrics is None or args.stratified_fold_metrics is None:
            parser.error(
                "use either --probe-root or both --fold-metrics and "
                "--stratified-fold-metrics"
            )
        metric_paths = [args.fold_metrics]
        stratum_paths = [args.stratified_fold_metrics]
        metrics = pd.read_csv(args.fold_metrics)
        strata = pd.read_csv(args.stratified_fold_metrics)
    else:
        if args.probe_root is None:
            parser.error("provide --probe-root or the two combined metrics inputs")
        metric_paths = sorted(args.probe_root.glob("fold_*/seed_*/*/metrics.csv"))
        stratum_paths = sorted(args.probe_root.glob("fold_*/seed_*/*/stratified_metrics.csv"))
        if not metric_paths or len(metric_paths) != len(stratum_paths):
            raise FileNotFoundError(
                f"Incomplete SV probe outputs: metrics={len(metric_paths)}, strata={len(stratum_paths)}"
            )
        metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
        strata = pd.concat([pd.read_csv(path) for path in stratum_paths], ignore_index=True)
    summary = hierarchical_summary(metrics, n_bootstrap=args.n_bootstrap, seed=args.seed)
    contributions = paired_modality_contribution_summary(
        metrics,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        suffix="_pair",
    )
    context = paired_context_summary(
        metrics,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    stratum_summary = summarize_strata(strata)
    stratum_contributions = paired_primary_stratum_contributions(
        strata,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    exact_universe = validate_exact_feature_universe(metrics)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "sv_fold_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "sv_summary.csv", index=False)
    contributions.to_csv(args.out_dir / "sv_modality_contributions.csv", index=False)
    context.to_csv(args.out_dir / "sv_context_contributions.csv", index=False)
    strata.to_csv(args.out_dir / "sv_stratified_fold_metrics.csv", index=False)
    stratum_summary.to_csv(args.out_dir / "sv_stratified_summary.csv", index=False)
    stratum_contributions.to_csv(
        args.out_dir / "sv_primary_stratified_contributions.csv",
        index=False,
    )
    run_count = int(metrics[["fold", "seed", "closure"]].drop_duplicates().shape[0])
    audit = {
        "schema_version": 2,
        "status": "complete",
        "files": run_count,
        "source_metric_files": len(metric_paths),
        "folds": sorted(metrics["fold"].unique()),
        "seeds": sorted(int(value) for value in metrics["seed"].unique()),
        "contexts": sorted(metrics["closure"].unique()),
        "feature_sets": sorted(metrics["feature_set"].unique()),
        "modality_contrasts": sorted(contributions["contrast"].unique()) if not contributions.empty else [],
        "primary_sequence_topology_contrast": PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST,
        "context_comparisons": sorted(context["contrast"].unique()) if not context.empty else [],
        "strata": sorted(strata["stratum"].unique()),
        "stratum_interpretation_statuses": sorted(
            stratum_summary["interpretation_status"].unique()
        ),
        "exact_feature_universe": exact_universe,
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.seed,
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
