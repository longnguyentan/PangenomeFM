#!/usr/bin/env python3
"""Paired bootstrap comparison of two models across held-out groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare(
    *,
    metrics_path: str | Path,
    baseline: str,
    candidate: str,
    out_path: str | Path,
    seed: int = 42,
    bootstrap_replicates: int = 10_000,
) -> list[dict[str, object]]:
    metrics = pd.read_csv(metrics_path)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for metric in ("spearman", "mae", "r2", "direction_accuracy"):
        pivot = metrics.pivot(
            index="heldout_group", columns="model", values=metric
        )[[baseline, candidate]].dropna()
        if metric == "mae":
            differences = (
                pivot[baseline] - pivot[candidate]
            ).to_numpy(dtype=float)
            orientation = "baseline_minus_candidate"
        else:
            differences = (
                pivot[candidate] - pivot[baseline]
            ).to_numpy(dtype=float)
            orientation = "candidate_minus_baseline"
        draws = rng.choice(
            differences,
            size=(bootstrap_replicates, len(differences)),
            replace=True,
        ).mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "baseline": baseline,
                "candidate": candidate,
                "n_groups": int(len(differences)),
                "difference": float(differences.mean()),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "positive_groups": int((differences > 0).sum()),
                "orientation": orientation,
                "bootstrap_replicates": bootstrap_replicates,
            }
        )
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    print(
        json.dumps(
            compare(
                metrics_path=args.metrics,
                baseline=args.baseline,
                candidate=args.candidate,
                out_path=args.out,
                seed=args.seed,
                bootstrap_replicates=args.bootstrap_replicates,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
