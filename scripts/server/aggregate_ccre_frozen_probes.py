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
PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST = (
    "topology_given_coordinate_and_frozen_sequence_fm"
)


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
        "topology_given_frozen_sequence_fm": (
            name("frozen_sequence_fm_plus_frozen_pangenomefm"),
            name("frozen_sequence_fm"),
        ),
        PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST: (
            name("coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm"),
            name("coordinate_plus_frozen_sequence_fm"),
        ),
        "frozen_sequence_fm_given_coordinate_and_topology": (
            name("coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm"),
            name("coordinate_plus_frozen_pangenomefm"),
        ),
        "coordinate_given_frozen_sequence_fm_and_topology": (
            name("coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm"),
            name("frozen_sequence_fm_plus_frozen_pangenomefm"),
        ),
    }


def _hierarchical_gain_interval(
    valid: pd.DataFrame,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap a paired gain by sampling folds and then runs within folds."""

    fold_arrays = [
        valid.loc[valid["fold"].eq(fold), "gain"].to_numpy(float)
        for fold in sorted(valid["fold"].unique())
    ]
    draws = _hierarchical_draws(
        fold_arrays,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )
    return (
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _hierarchical_draws(
    fold_arrays: list[np.ndarray],
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw fold-then-run bootstrap means, vectorizing balanced fold arrays."""

    if not fold_arrays or any(len(values) == 0 for values in fold_arrays):
        raise ValueError("Hierarchical bootstrap needs non-empty fold arrays")
    lengths = {len(values) for values in fold_arrays}
    if len(lengths) == 1:
        matrix = np.stack(fold_arrays)
        n_folds, n_runs = matrix.shape
        sampled_folds = rng.integers(
            0,
            n_folds,
            size=(n_bootstrap, n_folds),
        )
        sampled_runs = rng.integers(
            0,
            n_runs,
            size=(n_bootstrap, n_folds, n_runs),
        )
        selected = matrix[sampled_folds[:, :, None], sampled_runs]
        return selected.mean(axis=(1, 2))
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled_fold_indices = rng.integers(0, len(fold_arrays), len(fold_arrays))
        sampled = [
            rng.choice(fold_arrays[fold_index], len(fold_arrays[fold_index]), replace=True)
            for fold_index in sampled_fold_indices
        ]
        draws[index] = np.concatenate(sampled).mean()
    return draws


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
                ci95_low, ci95_high = _hierarchical_gain_interval(
                    valid,
                    n_bootstrap=n_bootstrap,
                    rng=rng,
                )
                rows.append(
                    {
                        "closure": closure,
                        "contrast": contrast,
                        "larger_feature_set": larger,
                        "smaller_feature_set": smaller,
                        "metric": metric,
                        "mean_gain": float(valid["gain"].mean()),
                        "ci95_low": ci95_low,
                        "ci95_high": ci95_high,
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


def paired_context_summary(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Compare 1-hop with strict on paired downstream runs and interactions.

    The first block compares the same feature set between contexts.  The second
    block is a difference-in-differences: it asks whether an added-modality gain
    is larger in 1-hop than strict.  These are downstream probe comparisons and
    do not convert the upstream closure tasks into a causal exposure contrast.
    """

    keys = ["fold", "seed", "feature_set"]
    if frame.duplicated(keys + ["closure"]).any():
        raise ValueError("Expected one all-test metric row per run, feature, and context")
    contexts = set(frame["closure"].astype(str))
    if not {"strict", "1hop"}.issubset(contexts):
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    strict = frame.loc[frame["closure"].eq("strict"), keys + METRICS]
    one_hop = frame.loc[frame["closure"].eq("1hop"), keys + METRICS]
    paired_context = one_hop.merge(
        strict,
        on=keys,
        how="inner",
        suffixes=("_1hop", "_strict"),
        validate="one_to_one",
    )
    for feature_set, group in paired_context.groupby("feature_set", sort=True):
        for metric in METRICS:
            raw_delta = (
                group[f"{metric}_1hop"].to_numpy(float)
                - group[f"{metric}_strict"].to_numpy(float)
            )
            gain = -raw_delta if metric in LOWER_IS_BETTER else raw_delta
            finite = np.isfinite(gain)
            valid = group.loc[finite, ["fold", "seed"]].copy()
            valid["gain"] = gain[finite]
            if valid.empty:
                continue
            ci95_low, ci95_high = _hierarchical_gain_interval(
                valid,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            rows.append(
                {
                    "comparison_type": "feature_set_context",
                    "contrast": "one_hop_vs_strict",
                    "feature_set": feature_set,
                    "modality_contrast": None,
                    "metric": metric,
                    "mean_gain": float(valid["gain"].mean()),
                    "ci95_low": ci95_low,
                    "ci95_high": ci95_high,
                    "n_paired_runs": int(len(valid)),
                    "n_folds": int(valid["fold"].nunique()),
                    "n_seeds": int(valid["seed"].nunique()),
                    "gain_definition": (
                        "strict_minus_1hop"
                        if metric in LOWER_IS_BETTER
                        else "1hop_minus_strict"
                    ),
                    "positive_means_1hop_improves": True,
                    "interpretation_scope": "paired_downstream_probe_context",
                }
            )

    suffix = "_pair" if any(
        str(value).endswith("_pair") for value in frame["feature_set"].unique()
    ) else ""
    run_keys = ["fold", "seed", "closure"]
    available = set(frame["feature_set"])
    for contrast, (larger, smaller) in modality_contrasts(suffix=suffix).items():
        if larger not in available or smaller not in available:
            continue
        left = frame.loc[frame["feature_set"].eq(larger), run_keys + METRICS]
        right = frame.loc[frame["feature_set"].eq(smaller), run_keys + METRICS]
        paired = left.merge(
            right,
            on=run_keys,
            how="inner",
            suffixes=("_larger", "_smaller"),
            validate="one_to_one",
        )
        for metric in METRICS:
            raw_delta = (
                paired[f"{metric}_larger"].to_numpy(float)
                - paired[f"{metric}_smaller"].to_numpy(float)
            )
            paired = paired.copy()
            paired["modality_gain"] = (
                -raw_delta if metric in LOWER_IS_BETTER else raw_delta
            )
            gains = paired.pivot(
                index=["fold", "seed"],
                columns="closure",
                values="modality_gain",
            ).reset_index()
            if not {"strict", "1hop"}.issubset(gains.columns):
                continue
            gains["gain"] = gains["1hop"] - gains["strict"]
            valid = gains.loc[
                np.isfinite(gains["gain"]), ["fold", "seed", "gain"]
            ].copy()
            if valid.empty:
                continue
            ci95_low, ci95_high = _hierarchical_gain_interval(
                valid,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            rows.append(
                {
                    "comparison_type": "modality_gain_context_interaction",
                    "contrast": f"one_hop_minus_strict__{contrast}",
                    "feature_set": None,
                    "modality_contrast": contrast,
                    "metric": metric,
                    "mean_gain": float(valid["gain"].mean()),
                    "ci95_low": ci95_low,
                    "ci95_high": ci95_high,
                    "n_paired_runs": int(len(valid)),
                    "n_folds": int(valid["fold"].nunique()),
                    "n_seeds": int(valid["seed"].nunique()),
                    "gain_definition": "1hop_modality_gain_minus_strict_modality_gain",
                    "positive_means_1hop_improves": True,
                    "interpretation_scope": "paired_downstream_probe_context_interaction",
                }
            )
    return pd.DataFrame(rows)


def validate_exact_feature_universe(frame: pd.DataFrame) -> dict[str, object]:
    """Verify that feature sets share exact train/validation/test counts per run."""

    required = {"fold", "seed", "closure", "feature_set", "n_train", "n_validation", "n_test"}
    missing = required - set(frame)
    if missing:
        return {
            "status": "not_available",
            "missing_columns": sorted(missing),
            "runs_checked": 0,
        }
    counts = frame.groupby(["fold", "seed", "closure"])[
        ["n_train", "n_validation", "n_test"]
    ].nunique()
    exact = bool(counts.le(1).all().all())
    if not exact:
        bad = counts.loc[counts.gt(1).any(axis=1)].head().to_dict("index")
        raise ValueError(f"Feature sets use different evaluation counts: {bad}")
    return {
        "status": "pass",
        "runs_checked": int(len(counts)),
        "maximum_distinct_counts_per_run": int(counts.to_numpy().max()),
    }


def hierarchical_summary(frame: pd.DataFrame, *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (closure, feature_set), group in frame.groupby(["closure", "feature_set"], sort=True):
        folds = sorted(group["fold"].unique())
        for metric in METRICS:
            arrays = [group.loc[group["fold"].eq(fold), metric].dropna().to_numpy(float) for fold in folds]
            observed = np.concatenate(arrays)
            draws = _hierarchical_draws(
                arrays,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
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
    parser.add_argument("--probe-root", type=Path)
    parser.add_argument(
        "--fold-metrics",
        type=Path,
        help="Combined fold metrics CSV; useful for evidence-package reanalysis",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if (args.probe_root is None) == (args.fold_metrics is None):
        parser.error("provide exactly one of --probe-root or --fold-metrics")
    if args.fold_metrics is not None:
        paths = [args.fold_metrics]
        metrics = pd.read_csv(args.fold_metrics)
    else:
        assert args.probe_root is not None
        paths = sorted(args.probe_root.glob("fold_*/seed_*/*/metrics.csv"))
        if not paths:
            raise FileNotFoundError(f"No completed metrics under {args.probe_root}")
        metrics = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
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
    context = paired_context_summary(
        metrics,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    exact_universe = validate_exact_feature_universe(metrics)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "ccre_fold_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "ccre_summary.csv", index=False)
    contributions.to_csv(args.out_dir / "ccre_modality_contributions.csv", index=False)
    context.to_csv(args.out_dir / "ccre_context_contributions.csv", index=False)
    run_count = int(metrics[["fold", "seed", "closure"]].drop_duplicates().shape[0])
    audit = {
        "schema_version": 2,
        "status": "complete",
        "files": run_count,
        "source_metric_files": len(paths),
        "folds": sorted(metrics["fold"].unique()),
        "seeds": sorted(int(value) for value in metrics["seed"].unique()),
        "contexts": sorted(metrics["closure"].unique()),
        "feature_sets": sorted(metrics["feature_set"].unique()),
        "modality_contrasts": sorted(contributions["contrast"].unique()) if not contributions.empty else [],
        "primary_sequence_topology_contrast": PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST,
        "context_comparisons": sorted(context["contrast"].unique()) if not context.empty else [],
        "exact_feature_universe": exact_universe,
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.seed,
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
