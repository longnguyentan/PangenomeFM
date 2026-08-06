#!/usr/bin/env python3
"""Test graph-scale/complexity correlates of HPRC-versus-HGSVC performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FEATURES = [
    "nodes",
    "within_chromosome_edges",
    "sequence_bp",
    "coordinate_span_bp",
    "mean_node_length",
    "median_node_length",
    "max_node_length",
    "mean_incident_degree",
    "edge_node_ratio",
    "nodes_per_megabase",
    "edges_per_megabase",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-statistics", type=Path, required=True)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    graph = pd.read_csv(args.graph_statistics)
    performance = pd.read_csv(args.performance)
    performance = performance[performance["experiment_type"] == "final_models"].copy()
    if "split" in performance:
        performance = performance[
            performance["split"].astype(str).str.contains("test|all", regex=True)
        ]
    performance["dataset"] = performance["regime"].map(
        {
            "hprc_r2": "hprc_r2",
            "hgsvc3": "hgsvc3",
            "official_hgsvc3_hprc1_integrated": "hgsvc3_hprc1_combined",
        }
    )
    performance = performance[performance["dataset"].notna()]
    merged = performance.merge(graph, on=["dataset", "chromosome"], how="left", validate="many_to_one")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / "performance_with_graph_statistics.csv", index=False)

    correlation_rows = []
    for (dataset, closure), group in merged.groupby(["dataset", "closure"]):
        for metric in ["mean_auroc", "mean_auprc", "mean_brier"]:
            for feature in FEATURES:
                selected = group[[metric, feature]].dropna()
                if len(selected) < 5 or selected[feature].nunique() < 2:
                    continue
                statistic, pvalue = spearmanr(selected[feature], selected[metric])
                correlation_rows.append(
                    {
                        "dataset": dataset,
                        "closure": closure,
                        "performance_metric": metric,
                        "graph_feature": feature,
                        "n_chromosomes": len(selected),
                        "spearman_rho": statistic,
                        "p_value": pvalue,
                    }
                )
    correlations = pd.DataFrame(correlation_rows)
    if not correlations.empty:
        correlations["bh_q_value"] = np.nan
        order = correlations["p_value"].sort_values().index
        ranks = np.arange(1, len(order) + 1)
        adjusted = np.minimum.accumulate(
            (correlations.loc[order, "p_value"].to_numpy() * len(order) / ranks)[::-1]
        )[::-1]
        correlations.loc[order, "bh_q_value"] = np.minimum(adjusted, 1.0)
    correlations.to_csv(args.out_dir / "graph_performance_correlations.csv", index=False)

    pair = merged[merged["dataset"].isin(["hprc_r2", "hgsvc3"])].pivot_table(
        index=["chromosome", "closure"],
        columns="dataset",
        values=["mean_auroc", "mean_auprc", *FEATURES],
        aggfunc="first",
    )
    pair.columns = [f"{feature}_{dataset}" for feature, dataset in pair.columns]
    pair = pair.reset_index()
    for metric in ["mean_auroc", "mean_auprc", *FEATURES]:
        left, right = f"{metric}_hgsvc3", f"{metric}_hprc_r2"
        if left in pair and right in pair:
            pair[f"{metric}_hgsvc3_minus_hprc_r2"] = pair[left] - pair[right]
    pair.to_csv(args.out_dir / "matched_chromosome_hprc_hgsvc_comparison.csv", index=False)

    summary = {
        "matched_dataset_chromosome_rows": int(len(merged)),
        "matched_hprc_hgsvc_chromosome_contexts": int(len(pair)),
        "correlation_tests": int(len(correlations)),
        "interpretation_guardrail": (
            "Associations with graph statistics are explanatory diagnostics, not causal proof. "
            "Cohort size, graph construction, coordinate reference, and four overlapping donors remain confounded."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
