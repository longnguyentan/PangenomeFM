#!/usr/bin/env python3
"""Compare graph-context regimes on identical target intervals.

Candidate links differ between core-node-induced and endpoint-expanded graphs,
so this script does not pretend that edge predictions can be paired one-to-one.
Instead, it pairs target intervals, resamples those intervals jointly, pools the
candidate predictions within each resample, and reports the metric difference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_prediction_statistics import METRICS, _load, metrics


def _interval_id(slice_id: str, closure: str) -> str:
    suffix = f"_{closure}"
    value = str(slice_id)
    if not value.endswith(suffix):
        raise ValueError(f"Slice {value!r} does not end with closure suffix {suffix!r}")
    return value[: -len(suffix)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", required=True, help="LABEL=core prediction CSV")
    parser.add_argument("--expanded", required=True, help="LABEL=expanded prediction CSV")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-pattern", default="^heldout_chr_test$")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--context-statistics", type=Path)
    args = parser.parse_args()

    core_label, core, core_path = _load(args.core, args.split_pattern)
    expanded_label, expanded, expanded_path = _load(args.expanded, args.split_pattern)
    if str(core["closure"].iloc[0]) != "strict":
        raise ValueError("--core predictions must have closure=strict")
    if str(expanded["closure"].iloc[0]) != "1hop":
        raise ValueError("--expanded predictions must have closure=1hop")

    core["interval_id"] = [
        _interval_id(slice_id, closure)
        for slice_id, closure in zip(core["slice"], core["closure"])
    ]
    expanded["interval_id"] = [
        _interval_id(slice_id, closure)
        for slice_id, closure in zip(expanded["slice"], expanded["closure"])
    ]
    core_intervals = set(core["interval_id"])
    expanded_intervals = set(expanded["interval_id"])
    matched = sorted(core_intervals & expanded_intervals)
    if not matched:
        raise ValueError("No identical target intervals occur in both contexts")
    core_only_intervals = sorted(core_intervals - expanded_intervals)
    expanded_only_intervals = sorted(expanded_intervals - core_intervals)
    core = core[core["interval_id"].isin(matched)].reset_index(drop=True)
    expanded = expanded[expanded["interval_id"].isin(matched)].reset_index(drop=True)

    core_groups = {
        name: group.index.to_numpy() for name, group in core.groupby("interval_id")
    }
    expanded_groups = {
        name: group.index.to_numpy() for name, group in expanded.groupby("interval_id")
    }
    point_core = metrics(core["y_true"].to_numpy(), core["p_edge"].to_numpy())
    point_expanded = metrics(
        expanded["y_true"].to_numpy(), expanded["p_edge"].to_numpy()
    )

    rng = np.random.default_rng(args.seed)
    replicate_rows: list[dict[str, object]] = []
    names = np.asarray(matched, dtype=object)
    for replicate in range(args.bootstrap_replicates):
        sample_names = rng.choice(names, size=len(names), replace=True)
        core_indices = np.concatenate([core_groups[name] for name in sample_names])
        expanded_indices = np.concatenate([expanded_groups[name] for name in sample_names])
        core_metrics = metrics(
            core.loc[core_indices, "y_true"].to_numpy(),
            core.loc[core_indices, "p_edge"].to_numpy(),
        )
        expanded_metrics = metrics(
            expanded.loc[expanded_indices, "y_true"].to_numpy(),
            expanded.loc[expanded_indices, "p_edge"].to_numpy(),
        )
        replicate_rows.extend(
            {
                "replicate": replicate,
                "metric": metric,
                "expanded_minus_core": expanded_metrics[metric] - core_metrics[metric],
            }
            for metric in METRICS
        )

    replicates = pd.DataFrame(replicate_rows)
    summaries: list[dict[str, object]] = []
    for metric, group in replicates.groupby("metric"):
        deltas = group["expanded_minus_core"].to_numpy()
        summaries.append(
            {
                "metric": metric,
                "core_value": point_core[metric],
                "expanded_value": point_expanded[metric],
                "expanded_minus_core": point_expanded[metric] - point_core[metric],
                "ci95_low": float(np.quantile(deltas, 0.025)),
                "ci95_high": float(np.quantile(deltas, 0.975)),
                "bootstrap_two_sided_p": float(
                    min(
                        1.0,
                        2
                        * min(
                            (np.sum(deltas <= 0) + 1) / (len(deltas) + 1),
                            (np.sum(deltas >= 0) + 1) / (len(deltas) + 1),
                        ),
                    )
                ),
                "matched_intervals": len(matched),
                "core_candidates": len(core),
                "expanded_candidates": len(expanded),
            }
        )

    interval_rows = []
    for name in matched:
        core_group = core.loc[core_groups[name]]
        expanded_group = expanded.loc[expanded_groups[name]]
        interval_rows.append(
            {
                "interval_id": name,
                "target_sn": core_group["target_sn"].iloc[0],
                "core_candidates": len(core_group),
                "expanded_candidates": len(expanded_group),
                **{
                    f"core_{metric}": value
                    for metric, value in metrics(
                        core_group["y_true"].to_numpy(), core_group["p_edge"].to_numpy()
                    ).items()
                },
                **{
                    f"expanded_{metric}": value
                    for metric, value in metrics(
                        expanded_group["y_true"].to_numpy(),
                        expanded_group["p_edge"].to_numpy(),
                    ).items()
                },
            }
        )
    intervals = pd.DataFrame(interval_rows)

    exposure_summary: dict[str, object] = {}
    if args.context_statistics:
        context = pd.read_csv(args.context_statistics)
        context = context[context["split"] == "test"].copy()
        if not context.empty:
            exposure_summary = {
                closure: {
                    column: float(group[column].mean())
                    for column in [
                        "visible_nodes",
                        "visible_link_rows",
                        "materialized_sequence_bp",
                        "candidate_targets",
                    ]
                }
                for closure, group in context.groupby("closure_legacy_label")
            }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.out_dir / "paired_context_comparison.csv", index=False)
    replicates.to_csv(
        args.out_dir / "paired_context_bootstrap.csv.gz", index=False, compression="gzip"
    )
    intervals.to_csv(args.out_dir / "per_interval_metrics.csv", index=False)
    summary = {
        "core_label": core_label,
        "core_path": core_path,
        "expanded_label": expanded_label,
        "expanded_path": expanded_path,
        "matched_intervals": len(matched),
        "evaluated_core_intervals_before_intersection": len(core_intervals),
        "evaluated_expanded_intervals_before_intersection": len(expanded_intervals),
        "core_only_intervals": core_only_intervals,
        "expanded_only_intervals": expanded_only_intervals,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "resampling_unit": "matched target interval",
        "candidate_universe_note": (
            "Contexts share loci but not candidate links or exposure counts; this is a "
            "paired-locus comparison, not an edge- or exposure-matched comparison."
        ),
        "mean_test_exposure": exposure_summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
