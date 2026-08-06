#!/usr/bin/env python3
"""Apply a preregistered held-out-donor gate before acquiring more modalities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    metric_source = (
        "macro_metrics" if "macro_metrics" in summary else "metrics"
    )
    metrics = {row["model"]: row for row in summary[metric_source]}
    baseline = metrics["frozen_sequence_ridge"]
    candidate_name = (
        "hierarchical_graph_context_residual"
        if "hierarchical_graph_context_residual" in metrics
        else "hierarchical_graph_residual"
    )
    candidate = metrics[candidate_name]
    checks = {
        "spearman_gain_at_least_0.02": (
            candidate["spearman"] - baseline["spearman"] >= 0.02
        ),
        "direction_gain_at_least_0.01": (
            candidate["direction_accuracy"] - baseline["direction_accuracy"] >= 0.01
        ),
        "mae_not_worse_than_2_percent": (
            candidate["mae"] <= baseline["mae"] * 1.02
        ),
        "r2_not_worse": candidate["r2"] >= baseline["r2"],
        "at_least_8_heldout_donors": int(summary["n_donors"]) >= 8,
    }
    passed = all(checks.values())
    result = {
        "gate": "methylation held-out-donor threshold before modality expansion",
        "metric_aggregation": (
            "macro-average across heldout groups"
            if metric_source == "macro_metrics"
            else "pooled pairs"
        ),
        "passed": passed,
        "checks": checks,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_model": candidate_name,
        "decision": (
            "Proceed to expression, accessibility, and 3D-contact acquisition."
            if passed
            else "Do not expand large functional modalities yet; improve methylation generalization."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
