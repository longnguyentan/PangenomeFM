#!/usr/bin/env python3
"""Evaluate one chromosome-held-out frozen PangenomeFM checkpoint on cCREs.

One upstream rotating-fold checkpoint is used for the entire downstream fold,
so training, validation, and test embeddings share a single latent coordinate
system.  The checkpoint was pretrained without the downstream test
chromosomes.  All learned and non-learned baselines are evaluated on exactly
the labeled nodes that received frozen embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from evaluation.calibration import apply_temperature, expected_calibration_error, fit_temperature
from evaluation.modality_factorial import (
    build_modality_factorial,
    factorial_feature_access,
    load_frozen_node_embedding_cache,
    select_feature_sets,
)
from tasks.ccre.aligned_baselines import _fit_model
from tasks.ccre.binary import _choose_threshold
from tasks.ccre.embedding_baseline import _extract_embeddings
from tasks.ccre.embedding_baseline import _manifest_target_chromosome as _canonical_chrom


FEATURE_ACCESS = {
    **factorial_feature_access(),
    "coordinate": "reference coordinate, node length, and constant orientation",
    "graph": "global node degree only",
    "structural": "coordinate, node length, degree, and orientation; excludes SR and reference identity",
    "sequence_kmer": "mono/di/tri-nucleotide composition; no graph adjacency",
    "frozen_pangenomefm": "frozen topology-pretrained node embedding; current encoder has no nucleotide input",
    "training_prevalence": "constant probability estimated from downstream training chromosomes",
    "random_uniform": "seeded U(0,1) score independent of all inputs",
}


def validate_checkpoint_holdout(
    checkpoint: Path,
    *,
    test_chrs: set[str],
    closure: str,
    seed: int,
) -> dict[str, object]:
    """Verify that checkpoint metadata encodes the requested upstream holdout."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arguments = payload.get("args", {})
    checkpoint_test = {_canonical_chrom(value) for value in arguments.get("test_chrs", [])}
    requested_test = {_canonical_chrom(value) for value in test_chrs}
    if checkpoint_test != requested_test:
        raise ValueError(
            f"Checkpoint held out {sorted(checkpoint_test)}, requested {sorted(requested_test)}"
        )
    checkpoint_closure = str(payload.get("closure", ""))
    if checkpoint_closure != closure:
        raise ValueError(f"Checkpoint closure={checkpoint_closure!r}, requested {closure!r}")
    checkpoint_seed = int(arguments.get("seed", -1))
    if checkpoint_seed != seed:
        raise ValueError(f"Checkpoint seed={checkpoint_seed}, requested {seed}")
    return {
        "checkpoint_test_chromosomes": sorted(checkpoint_test),
        "checkpoint_validation_chromosomes": sorted(
            _canonical_chrom(value) for value in arguments.get("val_chrs", [])
        ),
        "checkpoint_closure": checkpoint_closure,
        "checkpoint_seed": checkpoint_seed,
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def binary_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    *,
    n_bins: int = 15,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    predicted = probability >= threshold
    two_classes = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "positive_fraction": float(y_true.mean()),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, probability)) if two_classes else np.nan,
        "auprc": float(average_precision_score(y_true, probability)) if two_classes else np.nan,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "nll": float(log_loss(y_true, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece": float(expected_calibration_error(y_true, probability, n_bins=n_bins)),
    }


def evaluate_feature_sets(
    *,
    segids: np.ndarray,
    chromosomes: np.ndarray,
    labels: np.ndarray,
    features: dict[str, np.ndarray],
    test_chrs: set[str],
    val_chrs: set[str],
    seed: int,
    feature_access: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit/calibrate on train/validation chromosomes and score test only."""

    test_chrs = {_canonical_chrom(chrom) for chrom in test_chrs}
    val_chrs = {_canonical_chrom(chrom) for chrom in val_chrs}
    chromosome_norm = np.asarray([_canonical_chrom(chrom) for chrom in chromosomes])
    is_test = np.isin(chromosome_norm, sorted(test_chrs))
    is_val = np.isin(chromosome_norm, sorted(val_chrs))
    is_train = ~(is_test | is_val)
    if not is_train.any() or not is_val.any() or not is_test.any():
        raise ValueError(
            f"Empty downstream split: train={is_train.sum()}, val={is_val.sum()}, test={is_test.sum()}"
        )
    if any(len(np.unique(labels[mask])) < 2 for mask in (is_train, is_val)):
        raise ValueError("Training and validation splits must each contain both classes.")

    access = FEATURE_ACCESS if feature_access is None else feature_access
    metrics_rows: list[dict[str, object]] = []
    per_chromosome_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for feature_index, (feature_name, matrix) in enumerate(features.items()):
        if len(matrix) != len(labels):
            raise ValueError(f"Feature length mismatch for {feature_name}")
        if feature_name == "training_prevalence":
            raw_val = np.full(is_val.sum(), labels[is_train].mean(), dtype=float)
            raw_test = np.full(is_test.sum(), labels[is_train].mean(), dtype=float)
        elif feature_name == "random_uniform":
            rng = np.random.default_rng(seed + feature_index * 1_000_003)
            scores = rng.random(len(labels))
            raw_val = scores[is_val]
            raw_test = scores[is_test]
        else:
            model = _fit_model("logistic", seed)
            model.fit(matrix[is_train], labels[is_train])
            raw_val = model.predict_proba(matrix[is_val])[:, 1]
            raw_test = model.predict_proba(matrix[is_test])[:, 1]

        temperature = fit_temperature(labels[is_val], raw_val)
        calibrated_val = apply_temperature(raw_val, temperature)
        calibrated_test = apply_temperature(raw_test, temperature)
        threshold = _choose_threshold(labels[is_val], calibrated_val)
        common = {
            "feature_set": feature_name,
            "feature_access": access[feature_name],
            "temperature": float(temperature),
            "validation_f1_threshold": float(threshold),
            "n_train": int(is_train.sum()),
            "n_validation": int(is_val.sum()),
            "n_test": int(is_test.sum()),
        }
        metrics_rows.append(
            {
                **common,
                "scope": "all_test_chromosomes",
                "chromosome": "all",
                **binary_metrics(labels[is_test], calibrated_test, threshold),
            }
        )
        for chrom in sorted(set(chromosome_norm[is_test])):
            selected = chromosome_norm[is_test] == chrom
            per_chromosome_rows.append(
                {
                    **common,
                    "scope": "test_chromosome",
                    "chromosome": chrom,
                    **binary_metrics(labels[is_test][selected], calibrated_test[selected], threshold),
                }
            )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "segid": segids[is_test],
                    "chromosome": chromosome_norm[is_test],
                    "y_true": labels[is_test],
                    "feature_set": feature_name,
                    "p_raw": raw_test,
                    "p_calibrated": calibrated_test,
                    "threshold": threshold,
                    "y_pred": (calibrated_test >= threshold).astype(np.int8),
                }
            )
        )
    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(per_chromosome_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def run_probe(
    *,
    checkpoint: Path,
    manifest: Path,
    full_segments: Path,
    node_labels: Path,
    feature_cache: Path,
    out_dir: Path,
    fold: str,
    test_chrs: set[str],
    val_chrs: set[str],
    closure: str,
    device: str,
    seed: int,
    max_slices: int | None,
    external_sequence_cache: Path | None = None,
    minimum_external_coverage: float = 0.95,
    feature_sets: list[str] | None = None,
    canonical_conflict_policy: str = "exclude",
) -> dict[str, object]:
    if (out_dir / "audit.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {out_dir}")
    started = time.monotonic()
    checkpoint_validation = validate_checkpoint_holdout(
        checkpoint,
        test_chrs=test_chrs,
        closure=closure,
        seed=seed,
    )
    labels_frame = pd.read_csv(node_labels, compression="infer")
    labels_frame["chrom"] = labels_frame["chrom"].astype(str).map(_canonical_chrom)
    labels_frame["binary_label"] = (labels_frame["ccre_label"].astype(int) != 0).astype(np.int8)
    if labels_frame["segid"].duplicated().any():
        raise ValueError("node_labels contains duplicate segid values")

    labeled_segids = set(labels_frame["segid"].astype(int))
    embedding_map, occurrence_map, canonical_candidate_audit = _extract_embeddings(
        checkpoint=checkpoint,
        manifest=manifest,
        full_segments=full_segments,
        labeled_segids=labeled_segids,
        closure=closure,
        device_name=device,
        seed=seed,
        max_slices=max_slices,
        canonical_conflict_policy=canonical_conflict_policy,
        return_canonical_audit=True,
    )
    keep = labels_frame["segid"].astype(int).isin(embedding_map)
    labels_frame = labels_frame.loc[keep].sort_values("segid").reset_index(drop=True)
    if labels_frame.empty:
        raise RuntimeError("No cCRE-labeled reference nodes received embeddings")

    external_values: np.ndarray | None = None
    external_positions: dict[int, int] = {}
    external_audit: dict[str, object] | None = None
    n_embedded_before_external_filter = len(labels_frame)
    if external_sequence_cache is not None:
        external_segids, external_values, external_audit = load_frozen_node_embedding_cache(
            external_sequence_cache
        )
        external_positions = {
            int(segid): index for index, segid in enumerate(external_segids)
        }
        has_external = labels_frame["segid"].astype(int).isin(external_positions)
        external_coverage = float(has_external.mean())
        if external_coverage < minimum_external_coverage:
            raise ValueError(
                "External sequence-model coverage is below the prespecified gate: "
                f"{external_coverage:.6f} < {minimum_external_coverage:.6f}"
            )
        labels_frame = labels_frame.loc[has_external].reset_index(drop=True)

    with np.load(feature_cache, allow_pickle=False) as cache:
        cache_segids = cache["segid"].astype(np.int64)
        cache_position = {int(segid): index for index, segid in enumerate(cache_segids)}
        missing_cache = [int(segid) for segid in labels_frame["segid"] if int(segid) not in cache_position]
        if missing_cache:
            raise KeyError(f"Feature cache misses {len(missing_cache)} embedded nodes")
        indices = np.asarray([cache_position[int(segid)] for segid in labels_frame["segid"]])
        cached_features = {
            name: cache[name][indices].astype(np.float32)
            for name in ("coordinate", "graph", "structural", "sequence_kmer")
        }
    frozen = np.stack(
        [embedding_map[int(segid)] for segid in labels_frame["segid"]], axis=0
    ).astype(np.float32)
    components = {
        "coordinate": cached_features["coordinate"],
        "sequence_kmer": cached_features["sequence_kmer"],
        "frozen_pangenomefm": frozen,
    }
    if external_values is not None:
        external_indices = np.asarray(
            [external_positions[int(segid)] for segid in labels_frame["segid"]]
        )
        components["frozen_sequence_fm"] = external_values[external_indices]
    features = build_modality_factorial(
        components,
        include_external_sequence=external_values is not None,
    )
    features["graph"] = cached_features["graph"]
    features["structural"] = cached_features["structural"]
    # Empty matrices are deliberate sentinels; these baselines do not fit features.
    features["training_prevalence"] = np.empty((len(labels_frame), 0), dtype=np.float32)
    features["random_uniform"] = np.empty((len(labels_frame), 0), dtype=np.float32)
    features = select_feature_sets(features, feature_sets)
    selected_feature_access = {name: FEATURE_ACCESS[name] for name in features}

    metrics, per_chromosome, predictions = evaluate_feature_sets(
        segids=labels_frame["segid"].to_numpy(np.int64),
        chromosomes=labels_frame["chrom"].to_numpy(),
        labels=labels_frame["binary_label"].to_numpy(np.int8),
        features=features,
        test_chrs=test_chrs,
        val_chrs=val_chrs,
        seed=seed,
        feature_access=selected_feature_access,
    )
    for frame in (metrics, per_chromosome, predictions):
        frame.insert(0, "fold", fold)
        frame.insert(1, "seed", seed)
        frame.insert(2, "closure", closure)

    out_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    per_chromosome.to_csv(out_dir / "per_chromosome_metrics.csv", index=False)
    predictions.to_csv(out_dir / "test_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(
        {
            "segid": labels_frame["segid"].to_numpy(np.int64),
            "chromosome": labels_frame["chrom"].to_numpy(),
            "embedding_occurrences": [occurrence_map[int(segid)] for segid in labels_frame["segid"]],
        }
    ).to_csv(out_dir / "embedding_coverage.csv.gz", index=False, compression="gzip")
    audit = {
        "schema_version": 1,
        "status": "complete",
        "fold": fold,
        "seed": seed,
        "closure": closure,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest": str(manifest.resolve()),
        "node_labels": str(node_labels.resolve()),
        "feature_cache": str(feature_cache.resolve()),
        "test_chromosomes": sorted({_canonical_chrom(chrom) for chrom in test_chrs}),
        "validation_chromosomes": sorted({_canonical_chrom(chrom) for chrom in val_chrs}),
        "n_labeled_nodes_total": int(len(labeled_segids)),
        "n_labeled_nodes_embedded": int(len(labels_frame)),
        "embedding_coverage_fraction": float(len(labels_frame) / len(labeled_segids)),
        "feature_sets": selected_feature_access,
        "feature_dimensions": {
            name: int(matrix.shape[1]) for name, matrix in features.items()
        },
        "fairness_policy": "all feature sets use the identical embedded-node universe and chromosome split",
        "upstream_leakage_control": "checkpoint pretraining excluded every downstream test chromosome",
        "checkpoint_validation": checkpoint_validation,
        "canonical_candidate_audit": canonical_candidate_audit,
        "canonical_conflict_policy": canonical_conflict_policy,
        "canonical_conflict_interpretation": "all representations of a conflicting canonical identity are excluded before frozen embedding extraction; remaining same-label equivalents are collapsed",
        "calibration_policy": "temperature and F1 threshold fit on validation chromosomes only",
        "sequence_note": "PangenomeFM checkpoint itself has no nucleotide input; sequence is supplied only to explicit downstream baselines",
        "modality_factorial": {
            "C": "coordinate",
            "S": "sequence_kmer",
            "T": "frozen_pangenomefm",
            "comparisons": ["C", "S", "T", "C+S", "C+T", "S+T", "C+S+T"],
        },
        "external_sequence_cache": (
            str(external_sequence_cache.resolve()) if external_sequence_cache else None
        ),
        "external_sequence_cache_audit": external_audit,
        "minimum_external_coverage": minimum_external_coverage,
        "n_embedded_before_external_filter": n_embedded_before_external_filter,
        "external_coverage_fraction": (
            float(len(labels_frame) / n_embedded_before_external_filter)
            if external_sequence_cache is not None
            else None
        ),
        "max_slices": max_slices,
        "wall_seconds": time.monotonic() - started,
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-segments", type=Path, required=True)
    parser.add_argument("--node-labels", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--test-chrs", nargs="+", required=True)
    parser.add_argument("--val-chrs", nargs="+", required=True)
    parser.add_argument("--closure", choices=["strict", "1hop"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-slices", type=int)
    parser.add_argument("--external-sequence-cache", type=Path)
    parser.add_argument("--minimum-external-coverage", type=float, default=0.95)
    parser.add_argument("--feature-sets", nargs="+")
    parser.add_argument(
        "--canonical-conflict-policy",
        choices=["error", "exclude"],
        default="exclude",
    )
    args = parser.parse_args()
    run_probe(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        full_segments=args.full_segments,
        node_labels=args.node_labels,
        feature_cache=args.feature_cache,
        out_dir=args.out_dir,
        fold=args.fold,
        test_chrs=set(args.test_chrs),
        val_chrs=set(args.val_chrs),
        closure=args.closure,
        device=args.device,
        seed=args.seed,
        max_slices=args.max_slices,
        external_sequence_cache=args.external_sequence_cache,
        minimum_external_coverage=args.minimum_external_coverage,
        feature_sets=args.feature_sets,
        canonical_conflict_policy=args.canonical_conflict_policy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
