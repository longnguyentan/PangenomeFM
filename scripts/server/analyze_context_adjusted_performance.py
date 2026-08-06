#!/usr/bin/env python3
"""Quantify one-hop gains after controlling/matching graph-context exposure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


EXPOSURES = {
    "visible_nodes": "visible_nodes",
    "visible_edges": "visible_link_rows",
    "sequence_bp": "materialized_sequence_bp",
    "candidate_targets": "candidate_targets",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-slice", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--relative-caliper", type=float, default=0.10)
    args = parser.parse_args()
    performance = pd.read_csv(args.per_slice, compression="infer")
    contexts = []
    for path in sorted(args.context_root.glob("*/context_statistics.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "dataset", path.parent.name)
        contexts.append(frame)
    if not contexts:
        raise FileNotFoundError(f"No context_statistics.csv files below {args.context_root}")
    context = pd.concat(contexts, ignore_index=True)
    context = context.rename(columns={"closure_legacy_label": "closure", "slice_id": "slice"})
    columns = ["dataset", "slice", "closure", *EXPOSURES.values()]
    merged = performance.merge(
        context[columns], on=["dataset", "slice", "closure"], how="left", validate="many_to_one"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / "slice_performance_with_context.csv.gz", index=False, compression="gzip")

    merged["slice_pair_key"] = merged["slice"].str.replace(r"_(?:strict|1hop)$", "", regex=True)
    index = [
        "experiment_type", "regime", "dataset", "split", "fold", "transfer_pair", "seed",
        "slice_pair_key", "chromosome",
    ]
    for column in index:
        merged[column] = merged[column].fillna("NA")
    values = ["auroc", "auprc", "brier", *EXPOSURES.values()]
    paired = merged.pivot_table(
        index=index, columns="closure", values=values, aggfunc="first", dropna=True
    )
    paired.columns = [f"{value}_{closure}" for value, closure in paired.columns]
    paired = paired.reset_index()
    required = ["auprc_strict", "auprc_1hop"]
    if set(required) <= set(paired):
        paired = paired.dropna(subset=required)
    else:
        paired = paired.iloc[0:0].copy()
    for metric in ["auroc", "auprc", "brier"]:
        strict, expanded = f"{metric}_strict", f"{metric}_1hop"
        if strict in paired and expanded in paired:
            paired[f"delta_{metric}"] = paired[expanded] - paired[strict]
    exposure_delta_columns = []
    matched_columns = []
    for label, column in EXPOSURES.items():
        strict, expanded = f"{column}_strict", f"{column}_1hop"
        if strict not in paired or expanded not in paired:
            continue
        delta = f"delta_{label}"
        relative = f"relative_delta_{label}"
        matched = f"matched_{label}"
        paired[delta] = paired[expanded] - paired[strict]
        paired[relative] = paired[delta].abs() / paired[strict].abs().clip(lower=1)
        paired[matched] = paired[relative] <= args.relative_caliper
        exposure_delta_columns.append(delta)
        matched_columns.append(matched)
    paired["exposure_matched"] = paired[matched_columns].all(axis=1) if matched_columns else False
    paired.to_csv(args.out_dir / "paired_context_exposure_performance.csv", index=False)

    effect_rows = []
    for label, group in paired.groupby(["experiment_type", "regime", "dataset", "split"], dropna=False):
        for subset_name, subset in [
            ("all_matched_intervals", group),
            ("exposure_caliper_matched", group[group["exposure_matched"]]),
        ]:
            for metric in ["delta_auroc", "delta_auprc", "delta_brier"]:
                values_metric = subset.get(metric, pd.Series(dtype=float)).dropna()
                effect_rows.append(
                    {
                        "experiment_type": label[0],
                        "regime": label[1],
                        "dataset": label[2],
                        "split": label[3],
                        "subset": subset_name,
                        "metric": metric,
                        "pairs": int(len(values_metric)),
                        "mean_delta": float(values_metric.mean()) if len(values_metric) else np.nan,
                        "median_delta": float(values_metric.median()) if len(values_metric) else np.nan,
                        "sd_delta": float(values_metric.std()) if len(values_metric) > 1 else np.nan,
                    }
                )
    pd.DataFrame(effect_rows).to_csv(args.out_dir / "exposure_matched_context_effects.csv", index=False)

    regression_rows = []
    predictors = exposure_delta_columns
    for label, group in paired.groupby(["experiment_type", "regime", "dataset", "split"], dropna=False):
        if "delta_auprc" not in group:
            continue
        columns_needed = ["delta_auprc", *predictors]
        selected = group[columns_needed].replace([np.inf, -np.inf], np.nan).dropna()
        if len(selected) < max(12, len(predictors) + 3):
            continue
        x = selected[predictors].copy()
        for column in predictors:
            scale = x[column].std()
            x[column] = (x[column] - x[column].mean()) / scale if scale and np.isfinite(scale) else 0.0
        x = sm.add_constant(x, has_constant="add")
        fit = sm.OLS(selected["delta_auprc"], x).fit(cov_type="HC3")
        for term in fit.params.index:
            regression_rows.append(
                {
                    "experiment_type": label[0],
                    "regime": label[1],
                    "dataset": label[2],
                    "split": label[3],
                    "outcome": "delta_auprc_1hop_minus_strict",
                    "term": term,
                    "coefficient": fit.params[term],
                    "standard_error_hc3": fit.bse[term],
                    "p_value": fit.pvalues[term],
                    "n_pairs": int(len(selected)),
                    "r_squared": fit.rsquared,
                }
            )
    pd.DataFrame(regression_rows).to_csv(args.out_dir / "context_adjusted_regression.csv", index=False)
    summary = {
        "paired_intervals": int(len(paired)),
        "exposure_caliper": args.relative_caliper,
        "exposure_matched_pairs": int(paired["exposure_matched"].sum()),
        "exposure_matched_fraction": float(paired["exposure_matched"].mean()) if len(paired) else None,
        "regression_coefficients": len(regression_rows),
        "interpretation": (
            "The caliper subset is the direct context-matched stress test. The regression tests whether "
            "one-hop AUPRC differences persist after adjusting for changes in visible nodes, edges, "
            "materialized sequence, and candidate count. A nonzero intercept is not causal evidence."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
