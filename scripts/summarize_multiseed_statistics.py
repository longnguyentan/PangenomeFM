#!/usr/bin/env python3
"""Aggregate fixed-split prediction audits across model-training seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


METRICS = ["auroc", "auprc", "accuracy", "precision", "recall", "f1", "brier", "ece"]
KEY_COLUMNS = [
    "slice",
    "target_sn",
    "closure",
    "u_local",
    "v_local",
    "y_true",
]


def _parse_specification(specification: str) -> tuple[int, Path]:
    if "=" not in specification:
        raise ValueError("Each --statistics value must be SEED=DIRECTORY")
    seed, directory = specification.split("=", 1)
    return int(seed), Path(directory)


def _parse_context_specification(specification: str) -> tuple[str, int, Path]:
    if ":" not in specification or "=" not in specification:
        raise ValueError("Each --context-comparison value must be MODEL:SEED=CSV")
    model, remainder = specification.split(":", 1)
    seed, path = remainder.split("=", 1)
    return model, int(seed), Path(path)


def _evaluation_universe_hash(path: Path, split_pattern: str) -> tuple[str, int, float]:
    frame = pd.read_csv(path, compression="infer")
    if "split" not in frame:
        frame["split"] = "external_all"
    missing = set(KEY_COLUMNS + ["split"]) - set(frame)
    if missing:
        raise ValueError(f"{path} is missing universe-key columns: {sorted(missing)}")
    frame = frame[frame["split"].astype(str).str.contains(split_pattern, regex=True)].copy()
    frame["occurrence"] = frame.groupby(KEY_COLUMNS).cumcount()
    ordered = frame[KEY_COLUMNS + ["occurrence"]].sort_values(
        KEY_COLUMNS + ["occurrence"], kind="stable"
    )
    digest = hashlib.sha256()
    for row in ordered.itertuples(index=False, name=None):
        digest.update("\t".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), len(ordered), float(ordered["y_true"].mean())


def _aggregate(frame: pd.DataFrame, groups: list[str], values: list[str]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {"n_seeds": ("seed", "nunique")}
    for value in values:
        aggregations[f"{value}_mean"] = (value, "mean")
        aggregations[f"{value}_sd"] = (value, "std")
        aggregations[f"{value}_min"] = (value, "min")
        aggregations[f"{value}_max"] = (value, "max")
    return frame.groupby(groups, as_index=False).agg(**aggregations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", action="append", required=True, help="SEED=DIRECTORY")
    parser.add_argument(
        "--context-comparison",
        action="append",
        default=[],
        help="Optional MODEL:SEED=paired_context_comparison.csv",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    overall_frames: list[pd.DataFrame] = []
    paired_frames: list[pd.DataFrame] = []
    chromosome_frames: list[pd.DataFrame] = []
    universe_rows: list[dict[str, object]] = []
    seeds: list[int] = []

    for specification in args.statistics:
        seed, directory = _parse_specification(specification)
        seeds.append(seed)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))

        overall = pd.read_csv(directory / "overall_metrics.csv")
        overall.insert(0, "seed", seed)
        overall_frames.append(overall)

        paired = pd.read_csv(directory / "paired_comparisons.csv")
        paired.insert(0, "seed", seed)
        paired_frames.append(paired)

        chromosome = pd.read_csv(directory / "per_chromosome_metrics.csv")
        chromosome.insert(0, "seed", seed)
        chromosome_frames.append(chromosome)

        manifest = pd.read_csv(directory / "prediction_manifest.csv")
        for row in manifest.itertuples(index=False):
            digest, n_predictions, positive_fraction = _evaluation_universe_hash(
                Path(row.path), summary["split_pattern"]
            )
            universe_rows.append(
                {
                    "seed": seed,
                    "model": row.model,
                    "closure": row.closure,
                    "n_predictions": n_predictions,
                    "positive_fraction": positive_fraction,
                    "universe_sha256": digest,
                    "prediction_path": row.path,
                }
            )

    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seed specifications: {seeds}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overall_all = pd.concat(overall_frames, ignore_index=True)
    paired_all = pd.concat(paired_frames, ignore_index=True)
    chromosome_all = pd.concat(chromosome_frames, ignore_index=True)
    universes = pd.DataFrame(universe_rows)

    for closure, group in universes.groupby("closure"):
        if group["universe_sha256"].nunique() != 1:
            raise ValueError(
                f"Evaluation candidate universe differs across seeds/models for {closure}."
            )

    aggregate = _aggregate(overall_all, ["model", "closure"], METRICS)
    paired_aggregate = _aggregate(
        paired_all,
        ["reference", "comparator", "closure", "metric"],
        ["delta_reference_minus_comparator"],
    )
    paired_signs = (
        paired_all.assign(
            positive=paired_all["delta_reference_minus_comparator"] > 0,
            negative=paired_all["delta_reference_minus_comparator"] < 0,
        )
        .groupby(["reference", "comparator", "closure", "metric"], as_index=False)
        .agg(positive_seeds=("positive", "sum"), negative_seeds=("negative", "sum"))
    )
    paired_aggregate = paired_aggregate.merge(
        paired_signs, on=["reference", "comparator", "closure", "metric"], validate="one_to_one"
    )
    chromosome_aggregate = _aggregate(
        chromosome_all, ["model", "closure", "chromosome"], ["auroc", "auprc"]
    )

    overall_all.to_csv(args.out_dir / "per_seed_metrics.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    paired_all.to_csv(args.out_dir / "per_seed_paired_comparisons.csv", index=False)
    paired_aggregate.to_csv(args.out_dir / "aggregate_paired_comparisons.csv", index=False)
    chromosome_all.to_csv(args.out_dir / "per_seed_chromosome_metrics.csv", index=False)
    chromosome_aggregate.to_csv(args.out_dir / "aggregate_chromosome_metrics.csv", index=False)
    universes.to_csv(args.out_dir / "evaluation_universe_audit.csv", index=False)
    if args.context_comparison:
        context_frames: list[pd.DataFrame] = []
        for specification in args.context_comparison:
            model, seed, path = _parse_context_specification(specification)
            frame = pd.read_csv(path)
            frame.insert(0, "seed", seed)
            frame.insert(0, "model", model)
            context_frames.append(frame)
        context_all = pd.concat(context_frames, ignore_index=True)
        context_aggregate = _aggregate(
            context_all,
            ["model", "metric"],
            ["core_value", "expanded_value", "expanded_minus_core"],
        )
        context_all.to_csv(args.out_dir / "per_seed_context_comparisons.csv", index=False)
        context_aggregate.to_csv(
            args.out_dir / "aggregate_context_comparisons.csv", index=False
        )
    summary = {
        "seeds": sorted(seeds),
        "n_seeds": len(seeds),
        "evaluation_universe_identical_across_seeds_and_models": True,
        "uncertainty_policy": (
            "Report mean and sample standard deviation across training seeds; "
            "within-seed slice-bootstrap intervals remain in each statistics directory."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
