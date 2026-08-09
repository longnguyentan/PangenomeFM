#!/usr/bin/env python3
"""Run one cross-fitted frozen PangenomeFM insertion/deletion probe.

The upstream checkpoint held out exactly the downstream test chromosomes.  A
single checkpoint embeds train, validation, and test breakpoint nodes, keeping
the latent space aligned.  Every baseline uses the identical mapped-variant
universe and validation-only calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.server.run_ccre_frozen_probe_fold import (
    _canonical_chrom,
    binary_metrics,
    evaluate_feature_sets,
    validate_checkpoint_holdout,
)
from tasks.ccre.embedding_baseline import _extract_embeddings


FEATURE_ACCESS = {
    "coordinate_pair": "both breakpoint reference coordinates and node lengths; no graph or nucleotide input",
    "graph_pair": "both breakpoint node degrees; no coordinates or nucleotide input",
    "structural_pair": "both breakpoint coordinate/length/degree/orientation features",
    "sequence_kmer_pair": "both breakpoint mono/di/tri-nucleotide composition; no graph adjacency",
    "frozen_pangenomefm_pair": "both frozen topology-pretrained breakpoint embeddings; current encoder has no nucleotide input",
    "sequence_plus_frozen_pangenomefm_pair": "sequence-composition and frozen topology embeddings at both breakpoints",
    "training_prevalence": "constant insertion prevalence estimated from downstream training chromosomes",
    "random_uniform": "seeded U(0,1) score independent of all inputs",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pair_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([left, right, np.abs(left - right), left * right], axis=1).astype(np.float32)


def build_pair_matrix(
    examples: pd.DataFrame,
    *,
    node_values: np.ndarray,
    positions: dict[int, int],
) -> np.ndarray:
    left = node_values[[positions[int(value)] for value in examples["start_segid"]]]
    right = node_values[[positions[int(value)] for value in examples["end_segid"]]]
    return pair_features(left, right)


def stratified_metrics(
    predictions: pd.DataFrame,
    examples: pd.DataFrame,
) -> pd.DataFrame:
    metadata_columns = [
        "example_id", "svtype", "svlen", "length_bin", "allele_frequency", "af_bin",
        "start_mapping_rule", "end_mapping_rule", "start_mapping_distance", "end_mapping_distance",
    ]
    merged = predictions.merge(examples[metadata_columns], on="example_id", how="left", validate="many_to_one")
    max_mapping_distance = merged[["start_mapping_distance", "end_mapping_distance"]].max(axis=1)
    merged["mapping_distance_bin"] = pd.cut(
        max_mapping_distance,
        bins=[-1, 0, 100, 1_000, 10_000, np.inf],
        labels=["overlap", "1-100", "101-1000", "1001-10000", ">10000"],
    ).astype(str)
    rows: list[dict[str, object]] = []
    strata = {
        "chromosome": "chromosome",
        "sv_class": "svtype",
        "length_bin": "length_bin",
        "allele_frequency_bin": "af_bin",
        "breakpoint_mapping_distance": "mapping_distance_bin",
    }
    for feature_set, feature_frame in merged.groupby("feature_set", sort=True):
        threshold = float(feature_frame["threshold"].iloc[0])
        for stratum, column in strata.items():
            for value, group in feature_frame.groupby(column, dropna=False, sort=True):
                rows.append(
                    {
                        "feature_set": feature_set,
                        "stratum": stratum,
                        "stratum_value": value,
                        **binary_metrics(
                            group["y_true"].to_numpy(int),
                            group["p_calibrated"].to_numpy(float),
                            threshold,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def run_probe(
    *,
    checkpoint: Path,
    manifest: Path,
    full_segments: Path,
    examples_path: Path,
    feature_cache: Path,
    out_dir: Path,
    fold: str,
    test_chrs: set[str],
    val_chrs: set[str],
    closure: str,
    device: str,
    seed: int,
    max_slices: int | None,
) -> dict[str, object]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    started = time.monotonic()
    checkpoint_validation = validate_checkpoint_holdout(
        checkpoint,
        test_chrs=test_chrs,
        closure=closure,
        seed=seed,
    )
    examples = pd.read_csv(examples_path, compression="infer")
    required = {"example_id", "chrom", "start_segid", "end_segid", "binary_svtype_label"}
    missing = required - set(examples)
    if missing:
        raise ValueError(f"SV examples miss columns: {sorted(missing)}")
    examples["chrom"] = examples["chrom"].map(_canonical_chrom)
    required_nodes = set(examples["start_segid"].astype(int)) | set(examples["end_segid"].astype(int))
    embeddings, occurrences = _extract_embeddings(
        checkpoint=checkpoint,
        manifest=manifest,
        full_segments=full_segments,
        labeled_segids=required_nodes,
        closure=closure,
        device_name=device,
        seed=seed,
        max_slices=max_slices,
    )
    with np.load(feature_cache, allow_pickle=False) as cache:
        cache_segids = cache["segid"].astype(np.int64)
        cache_positions = {int(segid): index for index, segid in enumerate(cache_segids)}
        keep = examples["start_segid"].astype(int).isin(embeddings).to_numpy()
        keep &= examples["end_segid"].astype(int).isin(embeddings).to_numpy()
        keep &= examples["start_segid"].astype(int).isin(cache_positions).to_numpy()
        keep &= examples["end_segid"].astype(int).isin(cache_positions).to_numpy()
        examples = examples.loc[keep].sort_values("example_id").reset_index(drop=True)
        if examples.empty:
            raise RuntimeError("No SV examples have both breakpoint embeddings and cached features")
        cached = {
            name: build_pair_matrix(examples, node_values=cache[name], positions=cache_positions)
            for name in ("coordinate", "graph", "structural", "sequence_kmer")
        }
    embedding_segids = np.asarray(sorted(embeddings), dtype=np.int64)
    embedding_positions = {int(segid): index for index, segid in enumerate(embedding_segids)}
    embedding_values = np.stack([embeddings[int(segid)] for segid in embedding_segids]).astype(np.float32)
    frozen = build_pair_matrix(examples, node_values=embedding_values, positions=embedding_positions)
    features = {
        "coordinate_pair": cached["coordinate"],
        "graph_pair": cached["graph"],
        "structural_pair": cached["structural"],
        "sequence_kmer_pair": cached["sequence_kmer"],
        "frozen_pangenomefm_pair": frozen,
        "sequence_plus_frozen_pangenomefm_pair": np.concatenate([cached["sequence_kmer"], frozen], axis=1),
        "training_prevalence": np.empty((len(examples), 0), dtype=np.float32),
        "random_uniform": np.empty((len(examples), 0), dtype=np.float32),
    }
    metrics, per_chromosome, predictions = evaluate_feature_sets(
        segids=examples["example_id"].to_numpy(np.int64),
        chromosomes=examples["chrom"].to_numpy(),
        labels=examples["binary_svtype_label"].to_numpy(np.int8),
        features=features,
        test_chrs=test_chrs,
        val_chrs=val_chrs,
        seed=seed,
        feature_access=FEATURE_ACCESS,
    )
    predictions = predictions.rename(columns={"segid": "example_id"})
    strata = stratified_metrics(predictions, examples)
    for frame in (metrics, per_chromosome, predictions, strata):
        frame.insert(0, "fold", fold)
        frame.insert(1, "seed", seed)
        frame.insert(2, "closure", closure)
    out_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    per_chromosome.to_csv(out_dir / "per_chromosome_metrics.csv", index=False)
    strata.to_csv(out_dir / "stratified_metrics.csv", index=False)
    predictions.to_csv(out_dir / "test_predictions.csv.gz", index=False, compression="gzip")
    embedded_nodes = set(examples["start_segid"].astype(int)) | set(examples["end_segid"].astype(int))
    audit = {
        "schema_version": 1,
        "status": "complete",
        "fold": fold,
        "seed": seed,
        "closure": closure,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "examples": str(examples_path.resolve()),
        "examples_sha256": sha256_file(examples_path),
        "feature_cache": str(feature_cache.resolve()),
        "test_chromosomes": sorted({_canonical_chrom(value) for value in test_chrs}),
        "validation_chromosomes": sorted({_canonical_chrom(value) for value in val_chrs}),
        "n_examples_with_complete_features": int(len(examples)),
        "n_unique_embedded_breakpoint_nodes": int(len(embedded_nodes)),
        "mean_embedding_occurrences": float(np.mean([occurrences[node] for node in embedded_nodes])),
        "feature_sets": FEATURE_ACCESS,
        "target": "insertion versus deletion among versioned SV records of length >=50 bp",
        "fairness_policy": "all feature sets use identical mapped variants and chromosome splits",
        "calibration_policy": "temperature and F1 threshold fit on validation chromosomes only",
        "upstream_leakage_control": "checkpoint pretraining excluded every downstream test chromosome",
        "important_limitation": "variant records are graph-derived/assembly-derived truth, not a donor-held-out molecular phenotype",
        "checkpoint_validation": checkpoint_validation,
        "max_slices": max_slices,
        "wall_seconds": time.monotonic() - started,
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-segments", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--test-chrs", nargs="+", required=True)
    parser.add_argument("--val-chrs", nargs="+", required=True)
    parser.add_argument("--closure", choices=["strict", "1hop"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-slices", type=int)
    args = parser.parse_args()
    run_probe(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        full_segments=args.full_segments,
        examples_path=args.examples,
        feature_cache=args.feature_cache,
        out_dir=args.out_dir,
        fold=args.fold,
        test_chrs=set(args.test_chrs),
        val_chrs=set(args.val_chrs),
        closure=args.closure,
        device=args.device,
        seed=args.seed,
        max_slices=args.max_slices,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
