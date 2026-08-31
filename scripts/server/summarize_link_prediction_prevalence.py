#!/usr/bin/env python3
"""Recover class prevalence and chance AUPRC for reconstruction/transfer runs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def select_primary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    transfer = frame.loc[
        frame["experiment_type"].isin(["cohort_transfer", "release_transfer"])
        & frame["metric_scope"].eq("split")
        & frame["split"].eq("all")
    ].copy()
    transfer["analysis_scope"] = "cross_resource_or_release_transfer"
    transfer["analysis_name"] = transfer["transfer_pair"].astype(str)

    reconstruction = frame.loc[
        frame["experiment_type"].eq("rotating_folds")
        & frame["metric_scope"].eq("split")
        & frame["split"].eq("heldout_chr_test")
    ].copy()
    reconstruction["analysis_scope"] = "chromosome_held_out_reconstruction"
    reconstruction["analysis_name"] = reconstruction["regime"].astype(str)
    selected = pd.concat([reconstruction, transfer], ignore_index=True)
    keys = ["analysis_scope", "analysis_name", "closure", "fold", "seed"]
    if selected.duplicated(keys).any():
        duplicated = selected.loc[selected.duplicated(keys, keep=False), keys].head()
        raise ValueError(f"duplicate primary prevalence rows:\n{duplicated}")
    return selected


def summarize(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in selected.groupby(
        ["analysis_scope", "analysis_name", "closure"], sort=True
    ):
        targets = group["n_targets"].to_numpy(np.int64)
        prevalence = group["positive_fraction"].to_numpy(float)
        positives = np.rint(targets * prevalence).astype(np.int64)
        pooled = float(positives.sum() / targets.sum())
        rows.append(
            {
                "analysis_scope": keys[0],
                "analysis_name": keys[1],
                "closure": keys[2],
                "runs": int(len(group)),
                "folds": int(group["fold"].nunique(dropna=True)),
                "seeds": int(group["seed"].nunique()),
                "total_targets": int(targets.sum()),
                "total_connected": int(positives.sum()),
                "pooled_positive_fraction": pooled,
                "chance_auprc": pooled,
                "minimum_run_positive_fraction": float(prevalence.min()),
                "maximum_run_positive_fraction": float(prevalence.max()),
                "definition": "AUPRC expected from an uninformative ranking equals positive-class prevalence",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-seed-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.experiment_seed_metrics)
    required = {
        "experiment_type",
        "metric_scope",
        "split",
        "regime",
        "fold",
        "transfer_pair",
        "closure",
        "seed",
        "n_targets",
        "positive_fraction",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"experiment metrics are missing columns: {missing}")
    selected = select_primary_rows(frame)
    if selected.empty:
        raise RuntimeError("no primary reconstruction or transfer rows were selected")
    summary = summarize(selected)
    if not summary["chance_auprc"].between(0, 1).all():
        raise RuntimeError("invalid chance AUPRC")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out_dir / "link_prediction_prevalence_runs.csv", index=False)
    summary.to_csv(args.out_dir / "link_prediction_prevalence_summary.csv", index=False)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": str(args.experiment_seed_metrics.resolve()),
        "selected_runs": int(len(selected)),
        "summary_rows": int(len(summary)),
        "analysis_scopes": sorted(summary["analysis_scope"].unique()),
        "contexts": sorted(summary["closure"].unique()),
        "chance_auprc_definition": "positive-class prevalence for the exact evaluated candidate set",
        "outputs": {
            "runs": "link_prediction_prevalence_runs.csv",
            "summary": "link_prediction_prevalence_summary.csv",
        },
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
