#!/usr/bin/env python3
"""Aggregate cross-fitted SV probes and their prespecified strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.server.aggregate_ccre_frozen_probes import (
    hierarchical_summary,
    paired_modality_contribution_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
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
    stratum_summary = (
        strata.groupby(["closure", "feature_set", "stratum", "stratum_value"], dropna=False)
        .agg(
            n=("n", "sum"),
            mean_auroc=("auroc", "mean"),
            mean_auprc=("auprc", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_f1=("f1", "mean"),
            runs=("fold", "size"),
            folds=("fold", "nunique"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "sv_fold_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "sv_summary.csv", index=False)
    contributions.to_csv(args.out_dir / "sv_modality_contributions.csv", index=False)
    strata.to_csv(args.out_dir / "sv_stratified_fold_metrics.csv", index=False)
    stratum_summary.to_csv(args.out_dir / "sv_stratified_summary.csv", index=False)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "files": len(metric_paths),
        "folds": sorted(metrics["fold"].unique()),
        "seeds": sorted(int(value) for value in metrics["seed"].unique()),
        "contexts": sorted(metrics["closure"].unique()),
        "feature_sets": sorted(metrics["feature_set"].unique()),
        "modality_contrasts": sorted(contributions["contrast"].unique()) if not contributions.empty else [],
        "strata": sorted(strata["stratum"].unique()),
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.seed,
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
