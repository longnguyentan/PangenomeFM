#!/usr/bin/env python3
"""Score native HPRC benchmark tiles using chromosome-held-out predictions.

The rotating benchmark is tiled at 5 Mb.  This program deliberately preserves
that native resolution: subdividing an already-evaluated tile would not create
new held-out predictions and would leave most subwindows without enough query
edges.  For every tile, the score is the validation-calibrated binary log-loss
advantage of PangenomeFM over one prespecified, validation-calibrated baseline.

No QTL, GWAS, cCRE, gene, or other downstream annotation is read here.  The
result is therefore a model-derived prioritization input rather than a region
set selected after inspecting external biological signals.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evaluation.calibration import apply_temperature, fit_temperature
from evaluation.splits import normalize_chrom
from scripts.run_full_multicohort_all_chromosomes import load_config


CANONICAL_CHROMOSOMES = tuple(
    [f"chr{index}" for index in range(1, 23)] + ["chrX", "chrY"]
)
REQUIRED_NEURAL_COLUMNS = {
    "dataset", "slice", "target_sn", "split", "y_true", "p_edge",
}
REQUIRED_BASELINE_COLUMNS = {
    "slice", "chromosome", "fold_evaluation_split", "candidate_row_index",
    "u_oid", "v_oid", "y_true", "baseline", "p_calibrated",
}
REQUIRED_MANIFEST_COLUMNS = {
    "name", "target_sn", "closure", "start", "end", "n_segments", "n_links",
}


def progress(message: str) -> None:
    print(f"[dense-region {time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}", flush=True)


def sha256sum(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chromosome_sort_key(value: object) -> tuple[int, str]:
    chrom = normalize_chrom(str(value))
    order = {name: index for index, name in enumerate(CANONICAL_CHROMOSOMES)}
    return order.get(chrom, len(order)), chrom


def validate_fold_map(
    folds: Iterable[dict[str, object]],
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    """Return chromosome-to-fold and fold metadata after leakage checks."""

    chromosome_to_fold: dict[str, str] = {}
    fold_metadata: dict[str, dict[str, object]] = {}
    for raw in folds:
        name = str(raw["name"])
        test = tuple(normalize_chrom(value) for value in raw.get("test", []))
        validation = tuple(
            normalize_chrom(value) for value in raw.get("validation", [])
        )
        overlap = set(test) & set(validation)
        if overlap:
            raise ValueError(f"Fold {name} has test/validation overlap: {sorted(overlap)}")
        for chrom in test:
            if chrom in chromosome_to_fold:
                raise ValueError(
                    f"Chromosome {chrom} is held out by both "
                    f"{chromosome_to_fold[chrom]} and {name}"
                )
            chromosome_to_fold[chrom] = name
        fold_metadata[name] = {
            "name": name,
            "test": test,
            "validation": validation,
        }
    return chromosome_to_fold, fold_metadata


def load_tile_inventory(
    manifest_path: Path,
    *,
    closure: str,
    chromosome_to_fold: dict[str, str],
    chromosomes: set[str],
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest)
    if missing:
        raise ValueError(f"Benchmark manifest misses columns: {sorted(missing)}")
    manifest = manifest.loc[manifest["closure"].astype(str).eq(closure)].copy()
    manifest["chromosome"] = manifest["target_sn"].map(normalize_chrom)
    manifest = manifest.loc[manifest["chromosome"].isin(chromosomes)].copy()
    if manifest.empty:
        raise ValueError("No benchmark tiles remain after context/chromosome filtering")
    if manifest["name"].astype(str).duplicated().any():
        duplicates = manifest.loc[
            manifest["name"].astype(str).duplicated(keep=False), "name"
        ].astype(str).tolist()
        raise ValueError(f"Duplicate native tile names: {duplicates[:10]}")
    for column in ("start", "end", "n_segments", "n_links"):
        manifest[column] = pd.to_numeric(manifest[column], errors="raise")
    if (manifest["start"] < 0).any() or (manifest["end"] <= manifest["start"]).any():
        raise ValueError("Benchmark manifest contains invalid tile intervals")
    manifest["fold"] = manifest["chromosome"].map(chromosome_to_fold)
    if manifest["fold"].isna().any():
        missing_chroms = sorted(manifest.loc[manifest["fold"].isna(), "chromosome"].unique())
        raise ValueError(f"No held-out fold assignment for chromosomes: {missing_chroms}")
    for chrom, group in manifest.groupby("chromosome", sort=False):
        ordered = group.sort_values(["start", "end", "name"])
        starts = ordered["start"].to_numpy(np.int64)
        ends = ordered["end"].to_numpy(np.int64)
        if len(ordered) > 1 and np.any(starts[1:] < ends[:-1]):
            raise ValueError(f"Native benchmark tiles overlap on {chrom}")
    manifest["region_id"] = manifest["name"].astype(str)
    manifest["region_length"] = (
        manifest["end"].astype("int64") - manifest["start"].astype("int64")
    )
    manifest["graph_complexity"] = (
        2.0 * manifest["n_links"].astype(float)
        / manifest["n_segments"].clip(lower=1).astype(float)
    )
    return manifest[
        [
            "region_id", "name", "chromosome", "start", "end", "region_length",
            "fold", "target_sn", "n_segments", "n_links", "graph_complexity",
        ]
    ].sort_values(["chromosome", "start", "end", "region_id"]).reset_index(drop=True)


def align_heldout_predictions(
    neural: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    baseline_name: str,
    chromosomes: set[str],
) -> pd.DataFrame:
    """Align identical held-out candidates without using their labels as keys."""

    missing_neural = REQUIRED_NEURAL_COLUMNS - set(neural)
    missing_baseline = REQUIRED_BASELINE_COLUMNS - set(baseline)
    if missing_neural:
        raise ValueError(f"Neural predictions miss columns: {sorted(missing_neural)}")
    if missing_baseline:
        raise ValueError(f"Baseline predictions miss columns: {sorted(missing_baseline)}")

    neural_test = neural.loc[neural["split"].astype(str).eq("heldout_chr_test")].copy()
    neural_test["chromosome"] = neural_test["target_sn"].map(normalize_chrom)
    neural_test = neural_test.loc[neural_test["chromosome"].isin(chromosomes)].copy()
    base_test = baseline.loc[
        baseline["baseline"].astype(str).eq(baseline_name)
        & baseline["fold_evaluation_split"].astype(str).eq("heldout")
    ].copy()
    base_test["chromosome"] = base_test["chromosome"].map(normalize_chrom)
    base_test = base_test.loc[base_test["chromosome"].isin(chromosomes)].copy()
    if neural_test.empty or base_test.empty:
        raise ValueError(
            f"Empty held-out alignment: neural={len(neural_test)}, baseline={len(base_test)}"
        )

    # Both producers traverse the fixed split-seed candidate indices in the
    # same manifest order.  candidate_order is intentionally independent of
    # y_true, so labels are available only for the subsequent identity audit.
    neural_test["candidate_order"] = neural_test.groupby("slice", sort=False).cumcount()
    base_test["candidate_order"] = base_test.groupby("slice", sort=False).cumcount()
    merged = neural_test.merge(
        base_test[
            [
                "slice", "candidate_order", "candidate_row_index", "u_oid", "v_oid",
                "y_true", "p_calibrated", "chromosome",
            ]
        ],
        on=["slice", "candidate_order"],
        how="outer",
        validate="one_to_one",
        suffixes=("_neural", "_baseline"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(f"Neural/baseline candidate coverage differs: {counts}")
    if not merged["chromosome_neural"].eq(merged["chromosome_baseline"]).all():
        raise ValueError("Neural/baseline chromosome assignments differ")
    if not np.array_equal(
        merged["y_true_neural"].to_numpy(np.int8),
        merged["y_true_baseline"].to_numpy(np.int8),
    ):
        raise ValueError(
            "Neural/baseline held-out labels differ; refusing a non-identical comparison"
        )
    return merged.drop(columns="_merge")


def score_one_run(
    neural_path: Path,
    baseline_path: Path,
    *,
    baseline_name: str,
    chromosomes: set[str],
    dataset: str,
    expected_test_chromosomes: set[str] | None = None,
    expected_validation_chromosomes: set[str] | None = None,
) -> tuple[pd.DataFrame, float]:
    neural = pd.read_csv(neural_path, compression="infer")
    if "dataset" in neural:
        neural = neural.loc[neural["dataset"].astype(str).eq(dataset)].copy()
    neural["_chromosome"] = neural["target_sn"].map(normalize_chrom)
    if expected_test_chromosomes is not None:
        observed_test = set(
            neural.loc[
                neural["split"].astype(str).eq("heldout_chr_test"), "_chromosome"
            ].unique()
        )
        if observed_test != expected_test_chromosomes:
            raise ValueError(
                f"Neural held-out chromosomes {sorted(observed_test)} differ from "
                f"fold definition {sorted(expected_test_chromosomes)}"
            )
    if expected_validation_chromosomes is not None:
        observed_validation = set(
            neural.loc[
                neural["split"].astype(str).eq("val_chr_test"), "_chromosome"
            ].unique()
        )
        if observed_validation != expected_validation_chromosomes:
            raise ValueError(
                f"Neural validation chromosomes {sorted(observed_validation)} differ "
                f"from fold definition {sorted(expected_validation_chromosomes)}"
            )
    validation = neural.loc[neural["split"].astype(str).eq("val_chr_test")]
    if validation.empty or validation["y_true"].nunique() < 2:
        raise ValueError(f"Validation calibration is impossible for {neural_path}")
    temperature = fit_temperature(
        validation["y_true"].to_numpy(np.int8),
        validation["p_edge"].to_numpy(float),
    )
    baseline = pd.read_csv(baseline_path, compression="infer")
    if expected_test_chromosomes is not None:
        selected_baseline = baseline.loc[
            baseline["baseline"].astype(str).eq(baseline_name)
            & baseline["fold_evaluation_split"].astype(str).eq("heldout")
        ].copy()
        observed_baseline = set(
            selected_baseline["chromosome"].map(normalize_chrom).unique()
        )
        if observed_baseline != expected_test_chromosomes:
            raise ValueError(
                f"Baseline held-out chromosomes {sorted(observed_baseline)} differ "
                f"from fold definition {sorted(expected_test_chromosomes)}"
            )
    aligned = align_heldout_predictions(
        neural,
        baseline,
        baseline_name=baseline_name,
        chromosomes=chromosomes,
    )
    model_probability = apply_temperature(
        aligned["p_edge"].to_numpy(float), temperature
    )
    baseline_probability = np.clip(
        aligned["p_calibrated"].to_numpy(float), 1e-7, 1 - 1e-7
    )
    raw_labels = aligned["y_true_neural"].to_numpy(float)
    if not set(np.unique(raw_labels)).issubset({0.0, 1.0}):
        raise ValueError("Held-out labels must be binary")
    labels = raw_labels.astype(np.int8)
    if not np.isfinite(model_probability).all() or not np.isfinite(
        baseline_probability
    ).all():
        raise ValueError("Held-out probabilities contain non-finite values")
    model_probability = np.clip(model_probability, 1e-7, 1 - 1e-7)
    model_loss = -(
        labels * np.log(model_probability)
        + (1 - labels) * np.log1p(-model_probability)
    )
    baseline_loss = -(
        labels * np.log(baseline_probability)
        + (1 - labels) * np.log1p(-baseline_probability)
    )
    aligned["model_probability"] = model_probability
    aligned["baseline_probability"] = baseline_probability
    aligned["model_log_loss"] = model_loss
    aligned["baseline_log_loss"] = baseline_loss
    aligned["log_loss_advantage"] = baseline_loss - model_loss
    rows = []
    for region_id, group in aligned.groupby("slice", sort=True):
        rows.append(
            {
                "region_id": str(region_id),
                "n_test_candidates": int(len(group)),
                "n_positive_test_candidates": int(group["y_true_neural"].sum()),
                "positive_fraction": float(group["y_true_neural"].mean()),
                "model_nll": float(group["model_log_loss"].mean()),
                "baseline_nll": float(group["baseline_log_loss"].mean()),
                "log_loss_advantage": float(group["log_loss_advantage"].mean()),
                "mean_model_probability": float(group["model_probability"].mean()),
                "mean_baseline_probability": float(group["baseline_probability"].mean()),
            }
        )
    return pd.DataFrame(rows), float(temperature)


def find_one(pattern: Path, label: str) -> Path:
    matches = [Path(value) for value in sorted(glob.glob(str(pattern)))]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {label} matching {pattern}; found {len(matches)}"
        )
    return matches[0]


def aggregate_region_scores(
    inventory: pd.DataFrame,
    seed_scores: pd.DataFrame,
    *,
    expected_seeds: list[int],
    minimum_test_candidates: int,
) -> pd.DataFrame:
    if seed_scores.empty:
        raise ValueError("No per-seed tile scores were produced")
    aggregates = []
    for region_id, group in seed_scores.groupby("region_id", sort=True):
        counts = group["n_test_candidates"].astype(int)
        positives = group["n_positive_test_candidates"].astype(int)
        aggregates.append(
            {
                "region_id": str(region_id),
                "n_seeds": int(group["seed"].nunique()),
                "seeds": ",".join(str(value) for value in sorted(group["seed"].unique())),
                "n_test_candidates_min": int(counts.min()),
                "n_test_candidates_max": int(counts.max()),
                "n_positive_test_candidates_min": int(positives.min()),
                "n_positive_test_candidates_max": int(positives.max()),
                "model_nll": float(group["model_nll"].mean()),
                "baseline_nll": float(group["baseline_nll"].mean()),
                "model_score": float(group["log_loss_advantage"].mean()),
                "model_score_seed_std": (
                    float(group["log_loss_advantage"].std(ddof=1))
                    if len(group) > 1 else 0.0
                ),
                "mean_model_probability": float(group["mean_model_probability"].mean()),
                "mean_baseline_probability": float(
                    group["mean_baseline_probability"].mean()
                ),
            }
        )
    result = inventory.merge(pd.DataFrame(aggregates), on="region_id", how="left")
    reasons = []
    expected = len(expected_seeds)
    for row in result.itertuples(index=False):
        if pd.isna(row.model_score):
            reasons.append("missing_heldout_score")
        elif int(row.n_seeds) != expected:
            reasons.append("incomplete_seed_coverage")
        elif int(row.n_test_candidates_min) != int(row.n_test_candidates_max):
            reasons.append("candidate_count_differs_across_seeds")
        elif int(row.n_test_candidates_min) < minimum_test_candidates:
            reasons.append("insufficient_heldout_candidates")
        elif int(row.n_positive_test_candidates_min) <= 0:
            reasons.append("no_positive_heldout_candidates")
        elif int(row.n_positive_test_candidates_max) >= int(row.n_test_candidates_min):
            reasons.append("no_negative_heldout_candidates")
        else:
            reasons.append("eligible")
    result["eligibility_reason"] = reasons
    result["is_eligible"] = result["eligibility_reason"].eq("eligible")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--regime", default="hprc_r2")
    parser.add_argument("--dataset", default="hprc_r2")
    parser.add_argument("--closure", choices=["strict"], default="strict")
    parser.add_argument("--baseline", default="sequence_composition_sgd")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--chromosomes", nargs="+")
    parser.add_argument("--minimum-test-candidates", type=int, default=10)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.out_dir}")
    if args.minimum_test_candidates < 2:
        parser.error("--minimum-test-candidates must be at least 2")

    started = time.monotonic()
    config = load_config(args.config)
    dataset_config = config["datasets"][args.dataset]
    seeds = args.seeds or [int(value) for value in config["training"]["seeds"]]
    if len(seeds) != len(set(seeds)):
        parser.error("--seeds contains duplicates")
    chromosome_to_fold, fold_metadata = validate_fold_map(
        config["rotating_chromosome_folds"]
    )
    chromosomes = (
        {normalize_chrom(value) for value in args.chromosomes}
        if args.chromosomes
        else set(CANONICAL_CHROMOSOMES)
    )
    unknown = chromosomes - set(chromosome_to_fold)
    if unknown:
        parser.error(f"No chromosome-held-out fold for: {sorted(unknown)}")
    manifest_path = Path(dataset_config["pretrain_benchmark"]) / "manifest.csv"
    inventory = load_tile_inventory(
        manifest_path,
        closure=args.closure,
        chromosome_to_fold=chromosome_to_fold,
        chromosomes=chromosomes,
    )
    progress(
        f"loaded {len(inventory)} native tiles across "
        f"{inventory['chromosome'].nunique()} chromosomes"
    )

    seed_frames = []
    sources = []
    temperatures = []
    required_folds = sorted(inventory["fold"].unique())
    for fold in required_folds:
        fold_chromosomes = set(
            inventory.loc[inventory["fold"].eq(fold), "chromosome"].unique()
        )
        for seed in seeds:
            neural_pattern = (
                args.results_root / "rotating_folds" / args.regime / fold
                / f"seed_{seed}" / args.closure / "run_*" / "*pooled_predictions*.csv.gz"
            )
            neural_path = find_one(neural_pattern, "neural prediction file")
            baseline_path = (
                args.baseline_root / args.dataset / fold / f"seed_{seed}"
                / args.closure / "heldout_predictions.csv.gz"
            )
            if not baseline_path.is_file():
                raise FileNotFoundError(f"Missing baseline predictions: {baseline_path}")
            progress(f"scoring fold={fold} seed={seed} chromosomes={sorted(fold_chromosomes)}")
            frame, temperature = score_one_run(
                neural_path,
                baseline_path,
                baseline_name=args.baseline,
                chromosomes=fold_chromosomes,
                dataset=args.dataset,
                expected_test_chromosomes=set(fold_metadata[fold]["test"]),
                expected_validation_chromosomes=set(
                    fold_metadata[fold]["validation"]
                ),
            )
            frame["fold"] = fold
            frame["seed"] = seed
            frame["closure"] = args.closure
            frame["baseline"] = args.baseline
            frame["temperature"] = temperature
            seed_frames.append(frame)
            temperatures.append({"fold": fold, "seed": seed, "temperature": temperature})
            for role, path in (("neural", neural_path), ("baseline", baseline_path)):
                sources.append(
                    {
                        "role": role,
                        "fold": fold,
                        "seed": seed,
                        "path": str(path.resolve()),
                        "sha256": sha256sum(path),
                    }
                )

    seed_scores = pd.concat(seed_frames, ignore_index=True)
    progress("aggregating per-seed scores and applying eligibility gates")
    region_scores = aggregate_region_scores(
        inventory,
        seed_scores,
        expected_seeds=seeds,
        minimum_test_candidates=args.minimum_test_candidates,
    )
    order = {chrom: index for index, chrom in enumerate(CANONICAL_CHROMOSOMES)}
    region_scores["_chrom_order"] = region_scores["chromosome"].map(order)
    region_scores = region_scores.sort_values(
        ["_chrom_order", "start", "end", "region_id"]
    ).drop(columns="_chrom_order")
    seed_scores = seed_scores.sort_values(["fold", "seed", "region_id"])

    args.out_dir.mkdir(parents=True, exist_ok=False)
    progress(f"writing outputs to {args.out_dir}")
    seed_scores.to_csv(
        args.out_dir / "region_seed_scores.csv.gz", index=False, compression="gzip"
    )
    region_scores.to_csv(args.out_dir / "region_scores.csv", index=False)
    exclusions = region_scores.loc[~region_scores["is_eligible"]].copy()
    exclusions.to_csv(args.out_dir / "region_score_exclusions.csv", index=False)
    output_paths = [
        args.out_dir / "region_seed_scores.csv.gz",
        args.out_dir / "region_scores.csv",
        args.out_dir / "region_score_exclusions.csv",
    ]
    audit = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "complete",
        "config": str(args.config.resolve()),
        "config_sha256": sha256sum(args.config),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256sum(manifest_path),
        "results_root": str(args.results_root.resolve()),
        "baseline_root": str(args.baseline_root.resolve()),
        "regime": args.regime,
        "dataset": args.dataset,
        "closure": args.closure,
        "baseline": args.baseline,
        "seeds": seeds,
        "chromosomes": sorted(chromosomes, key=chromosome_sort_key),
        "folds": required_folds,
        "fold_definitions": {name: fold_metadata[name] for name in required_folds},
        "native_tile_resolution": "deterministic non-overlapping 5-Mb benchmark tiles",
        "score_definition": (
            "mean across seeds of baseline held-out binary log loss minus "
            "validation-temperature-calibrated PangenomeFM held-out binary log loss"
        ),
        "model_calibration": "temperature fit only on val_chr_test for the same fold/seed",
        "baseline_calibration": "existing Platt calibration fit only on validation chromosomes",
        "minimum_test_candidates": args.minimum_test_candidates,
        "tiles": int(len(region_scores)),
        "eligible_tiles": int(region_scores["is_eligible"].sum()),
        "exclusion_reason_counts": {
            str(key): int(value)
            for key, value in region_scores["eligibility_reason"].value_counts().items()
        },
        "temperatures": temperatures,
        "input_prediction_files": sources,
        "output_sha256": {
            path.name: sha256sum(path) for path in output_paths
        },
        "downstream_signal_access": "none; QTL, GWAS, cCRE, and gene annotations are not read",
        "leakage_control": (
            "every tile is scored only by the rotating checkpoint whose upstream "
            "pretraining excluded that tile's chromosome"
        ),
        "important_limitation": (
            "scores measure held-out link-prediction loss advantage at native 5-Mb "
            "benchmark resolution; they are not biological effect estimates"
        ),
        "wall_seconds": time.monotonic() - started,
    }
    audit_path = args.out_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    progress("writing output checksums")
    with (args.out_dir / "SHA256SUMS").open("w") as handle:
        for path in [*output_paths, audit_path]:
            handle.write(f"{sha256sum(path)}  {path.name}\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
