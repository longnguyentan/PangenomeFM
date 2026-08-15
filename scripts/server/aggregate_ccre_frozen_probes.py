#!/usr/bin/env python3
"""Aggregate frozen cCRE probe folds with hierarchical confidence intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["auroc", "auprc", "accuracy", "precision", "recall", "f1", "nll", "brier", "ece"]
LOWER_IS_BETTER = {"nll", "brier", "ece"}


def modality_contrasts(*, suffix: str = "") -> dict[str, tuple[str, str]]:
    """Return prespecified paired feature contrasts as (first, reference)."""

    def name(value: str) -> str:
        return f"{value}{suffix}"

    return {
        "topology_given_coordinate_and_sequence": (
            name("coordinate_plus_sequence_plus_frozen_pangenomefm"),
            name("coordinate_plus_sequence_kmer"),
        ),
        "sequence_given_coordinate_and_topology": (
            name("coordinate_plus_sequence_plus_frozen_pangenomefm"),
            name("coordinate_plus_frozen_pangenomefm"),
        ),
        "coordinate_given_sequence_and_topology": (
            name("coordinate_plus_sequence_plus_frozen_pangenomefm"),
            name("sequence_plus_frozen_pangenomefm"),
        ),
        "topology_given_coordinate": (
            name("coordinate_plus_frozen_pangenomefm"),
            name("coordinate"),
        ),
        "topology_given_sequence": (
            name("sequence_plus_frozen_pangenomefm"),
            name("sequence_kmer"),
        ),
        "sequence_given_coordinate": (
            name("coordinate_plus_sequence_kmer"),
            name("coordinate"),
        ),
        "coordinate_given_sequence": (
            name("coordinate_plus_sequence_kmer"),
            name("sequence_kmer"),
        ),
        "frozen_sequence_fm_vs_kmer_composition": (
            name("frozen_sequence_fm"),
            name("sequence_kmer"),
        ),
        "frozen_sequence_fm_vs_kmer_given_topology": (
            name("frozen_sequence_fm_plus_frozen_pangenomefm"),
            name("sequence_plus_frozen_pangenomefm"),
        ),
        "frozen_sequence_fm_vs_kmer_given_coordinate_and_topology": (
            name("coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm"),
            name("coordinate_plus_sequence_plus_frozen_pangenomefm"),
        ),
    }


def paired_modality_contribution_summary(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    suffix: str = "",
) -> pd.DataFrame:
    """Estimate modality gains with a run-paired hierarchical bootstrap."""

    keys = ["fold", "seed", "closure"]
    if frame.duplicated(keys + ["feature_set"]).any():
        raise ValueError("Expected one all-test metric row per run and feature set")
    available = set(frame["feature_set"])
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for contrast, (larger, smaller) in modality_contrasts(suffix=suffix).items():
        if larger not in available or smaller not in available:
            continue
        left = frame.loc[frame["feature_set"].eq(larger), keys + METRICS]
        right = frame.loc[frame["feature_set"].eq(smaller), keys + METRICS]
        paired = left.merge(
            right,
            on=keys,
            how="inner",
            suffixes=("_larger", "_smaller"),
            validate="one_to_one",
        )
        expected_larger = int((frame["feature_set"] == larger).sum())
        expected_smaller = int((frame["feature_set"] == smaller).sum())
        if len(paired) != expected_larger or len(paired) != expected_smaller:
            raise ValueError(f"Unpaired modality runs for contrast {contrast}")
        for closure, group in paired.groupby("closure", sort=True):
            folds = sorted(group["fold"].unique())
            for metric in METRICS:
                larger_values = group[f"{metric}_larger"].to_numpy(float)
                smaller_values = group[f"{metric}_smaller"].to_numpy(float)
                raw_delta = larger_values - smaller_values
                gain = -raw_delta if metric in LOWER_IS_BETTER else raw_delta
                finite = np.isfinite(gain)
                valid = group.loc[finite, keys].copy()
                valid["gain"] = gain[finite]
                if valid.empty:
                    continue
                fold_arrays = [
                    valid.loc[valid["fold"].eq(fold), "gain"].to_numpy(float)
                    for fold in folds
                    if valid["fold"].eq(fold).any()
                ]
                draws = np.empty(n_bootstrap, dtype=float)
                for index in range(n_bootstrap):
                    sampled_fold_indices = rng.integers(0, len(fold_arrays), len(fold_arrays))
                    sampled = [
                        rng.choice(fold_arrays[fold_index], len(fold_arrays[fold_index]), replace=True)
                        for fold_index in sampled_fold_indices
                    ]
                    draws[index] = np.concatenate(sampled).mean()
                rows.append(
                    {
                        "closure": closure,
                        "contrast": contrast,
                        "larger_feature_set": larger,
                        "smaller_feature_set": smaller,
                        "metric": metric,
                        "mean_gain": float(valid["gain"].mean()),
                        "ci95_low": float(np.quantile(draws, 0.025)),
                        "ci95_high": float(np.quantile(draws, 0.975)),
                        "n_paired_runs": int(len(valid)),
                        "n_folds": int(valid["fold"].nunique()),
                        "n_seeds": int(valid["seed"].nunique()),
                        "gain_definition": (
                            "smaller_minus_larger"
                            if metric in LOWER_IS_BETTER
                            else "larger_minus_smaller"
                        ),
                        "positive_means_first_feature_set_improves": True,
                    }
                )
    return pd.DataFrame(rows)


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
    contributions = paired_modality_contribution_summary(
        metrics,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "ccre_fold_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "ccre_summary.csv", index=False)
    contributions.to_csv(args.out_dir / "ccre_modality_contributions.csv", index=False)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "files": len(paths),
        "folds": sorted(metrics["fold"].unique()),
        "seeds": sorted(int(value) for value in metrics["seed"].unique()),
        "contexts": sorted(metrics["closure"].unique()),
        "feature_sets": sorted(metrics["feature_set"].unique()),
        "modality_contrasts": sorted(contributions["contrast"].unique()) if not contributions.empty else [],
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.seed,
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
