#!/usr/bin/env python3
"""Aggregate identical-fold link baselines with fold/seed bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["auroc", "auprc", "accuracy", "precision", "recall", "f1", "nll", "brier", "ece"]


def parse_path(path: Path, root: Path) -> tuple[str, int, str]:
    relative = path.relative_to(root)
    if len(relative.parts) < 5:
        raise ValueError(f"Unexpected baseline path: {relative}")
    fold = relative.parts[1]
    match = re.fullmatch(r"seed_(\d+)", relative.parts[2])
    if match is None:
        raise ValueError(f"Could not parse seed from {relative}")
    return fold, int(match.group(1)), relative.parts[3]


def hierarchical_summary(frame: pd.DataFrame, *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (closure, baseline), group in frame.groupby(["closure", "baseline"], sort=True):
        folds = sorted(group["fold"].unique())
        for metric in METRICS:
            arrays = [group.loc[group["fold"].eq(fold), metric].dropna().to_numpy(float) for fold in folds]
            observed = np.concatenate(arrays)
            draws = np.empty(n_bootstrap, dtype=float)
            for draw in range(n_bootstrap):
                sampled_folds = rng.integers(0, len(folds), len(folds))
                values = [rng.choice(arrays[index], len(arrays[index]), replace=True) for index in sampled_folds]
                draws[draw] = np.concatenate(values).mean()
            rows.append(
                {
                    "closure": closure,
                    "baseline": baseline,
                    "metric": metric,
                    "mean": float(observed.mean()),
                    "std_across_runs": float(observed.std(ddof=1)) if len(observed) > 1 else 0.0,
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "n_runs": int(len(observed)),
                    "n_folds": len(folds),
                    "n_seeds": int(group["seed"].nunique()),
                    "resampling_unit": "chromosome_fold_then_seed",
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    paths = sorted(args.baseline_root.glob("hprc_r2/fold_*/seed_*/*/metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No baseline metrics below {args.baseline_root}")
    frames = []
    for path in paths:
        fold, seed, closure = parse_path(path, args.baseline_root)
        frame = pd.read_csv(path)
        frame = frame.loc[frame["chromosome"].isna()].copy()
        frame["fold"] = fold
        frame["seed"] = seed
        if not frame["closure"].astype(str).eq(closure).all():
            raise ValueError(f"Closure mismatch in {path}")
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    summary = hierarchical_summary(metrics, n_bootstrap=args.n_bootstrap, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "baseline_fold_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "baseline_summary.csv", index=False)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "source_files": len(paths),
        "runs": len(metrics),
        "baselines": sorted(metrics["baseline"].unique()),
        "folds": sorted(metrics["fold"].unique()),
        "seeds": sorted(int(value) for value in metrics["seed"].unique()),
        "contexts": sorted(metrics["closure"].unique()),
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.seed,
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
