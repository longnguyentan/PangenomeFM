#!/usr/bin/env python3
"""Aggregate frozen cCRE probe folds with hierarchical confidence intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["auroc", "auprc", "accuracy", "precision", "recall", "f1", "nll", "brier", "ece"]


def hierarchical_summary(frame: pd.DataFrame, *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (closure, feature_set), group in frame.groupby(["closure", "feature_set"], sort=True):
        folds = sorted(group["fold"].unique())
        for metric in METRICS:
            arrays = [group.loc[group["fold"].eq(fold), metric].dropna().to_numpy(float) for fold in folds]
            observed = np.concatenate(arrays)
            draws = np.empty(n_bootstrap, dtype=float)
            for index in range(n_bootstrap):
                sampled_fold_indices = rng.integers(0, len(folds), len(folds))
                sampled = [rng.choice(arrays[fold_index], len(arrays[fold_index]), replace=True) for fold_index in sampled_fold_indices]
                draws[index] = np.concatenate(sampled).mean()
            rows.append(
                {
                    "closure": closure,
                    "feature_set": feature_set,
                    "metric": metric,
                    "mean": float(observed.mean()),
                    "std_across_runs": float(observed.std(ddof=1)) if len(observed) > 1 else 0.0,
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "n_runs": int(len(observed)),
                    "n_folds": int(len(folds)),
                    "n_seeds": int(group["seed"].nunique()),
                    "resampling_unit": "chromosome_fold_then_seed",
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    paths = sorted(args.probe_root.glob("fold_*/seed_*/*/metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No completed metrics under {args.probe_root}")
    frames = [pd.read_csv(path) for path in paths]
    metrics = pd.concat(frames, ignore_index=True)
    expected = {"fold", "seed", "closure", "feature_set", *METRICS}
    missing = expected - set(metrics)
    if missing:
        raise ValueError(f"Probe metrics miss columns: {sorted(missing)}")
    summary = hierarchical_summary(metrics, n_bootstrap=args.n_bootstrap, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "ccre_fold_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "ccre_summary.csv", index=False)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "files": len(paths),
        "folds": sorted(metrics["fold"].unique()),
        "seeds": sorted(int(value) for value in metrics["seed"].unique()),
        "contexts": sorted(metrics["closure"].unique()),
        "feature_sets": sorted(metrics["feature_set"].unique()),
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.seed,
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
