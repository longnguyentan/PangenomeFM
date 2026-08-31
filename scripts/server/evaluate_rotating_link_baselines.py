#!/usr/bin/env python3
"""Evaluate non-neural link baselines on the exact rotating chromosome folds.

Coordinate and sequence-composition classifiers are trained only on training
chromosomes. Topology heuristics never fit labels. All threshold calibration is
fit on validation chromosomes and applied once to held-out chromosomes. The
per-slice candidate split is identical to ``training.pretrain`` via the shared
fixed split seed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation.link_heuristics import (
    _build_undirected_adjacency,
    _score_candidates,
    _split_indices,
)
from evaluation.splits import normalize_chrom
from graph.neg_sampling import oriented_ids_from_links
from graph.slicing import build_global_index
from scripts.server.calibrate_rotating_fold_predictions import (
    choose_f1_threshold,
    classification_metrics,
)


HEURISTICS = [
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "resource_allocation",
    "preferential_attachment",
    "degree_sum",
    "shortest_path",
]
BASELINE_NAMES = [
    "coordinate_sgd",
    "sequence_composition_sgd",
    "training_frequency",
    "random_uniform",
    *[f"topology_{name}" for name in HEURISTICS],
]


def resolve_path(value: str | Path, manifest_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    candidate = manifest_dir / path
    return candidate if candidate.exists() else path


def sequence_vector(sequence: str) -> np.ndarray:
    """Orientation-invariant, fixed-dimensional composition features."""

    seq = str(sequence).upper()
    length = max(len(seq), 1)
    bases = "ACGT"
    mono = np.array([seq.count(base) / length for base in bases], dtype=np.float32)
    denom = max(length - 1, 1)
    di = np.array(
        [seq.count(a + b) / denom for a in bases for b in bases], dtype=np.float32
    )
    nonzero = mono[mono > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum()) if len(nonzero) else 0.0
    return np.concatenate(
        [mono, di, np.array([(mono[1] + mono[2]), entropy / 2.0, math.log1p(length) / 15.0])]
    ).astype(np.float32)


def coordinate_features(
    u: np.ndarray,
    v: np.ndarray,
    *,
    so: np.ndarray,
    sn: np.ndarray,
) -> np.ndarray:
    u_seg = u // 2
    v_seg = v // 2
    u_so = so[u_seg].astype(np.float64)
    v_so = so[v_seg].astype(np.float64)
    delta = np.abs(u_so - v_so)
    return np.column_stack(
        [
            u_so / 250_000_000.0,
            v_so / 250_000_000.0,
            np.log1p(delta) / 20.0,
            delta / 250_000_000.0,
            (sn[u_seg] == sn[v_seg]).astype(np.float32),
            (u % 2).astype(np.float32),
            (v % 2).astype(np.float32),
        ]
    ).astype(np.float32)


def sequence_features(
    u: np.ndarray,
    v: np.ndarray,
    *,
    sequences: np.ndarray,
    cache: dict[int, np.ndarray],
) -> np.ndarray:
    def lookup(segid: int) -> np.ndarray:
        segid = int(segid)
        if segid not in cache:
            cache[segid] = sequence_vector(sequences[segid])
        return cache[segid]

    u_features = np.stack([lookup(value) for value in u // 2])
    v_features = np.stack([lookup(value) for value in v // 2])
    return np.concatenate(
        [u_features, v_features, u_features * v_features, np.abs(u_features - v_features)],
        axis=1,
    ).astype(np.float32)


def load_candidates(row: pd.Series, manifest_dir: Path) -> pd.DataFrame:
    return pd.read_csv(resolve_path(row["edge_pred_path"], manifest_dir), compression="infer")


def candidate_arrays(frame: pd.DataFrame, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = frame.iloc[indices]
    return (
        selected["u_oid"].to_numpy(np.int64),
        selected["v_oid"].to_numpy(np.int64),
        selected["label"].to_numpy(np.int8),
    )


def fit_platt(validation_scores: np.ndarray, validation_y: np.ndarray):
    if len(np.unique(validation_y)) < 2 or np.nanstd(validation_scores) == 0:
        prevalence = float(np.mean(validation_y))
        return lambda values: np.full(len(values), prevalence, dtype=float)
    model = LogisticRegression(C=1.0, solver="lbfgs", random_state=0)
    model.fit(validation_scores.reshape(-1, 1), validation_y)
    return lambda values: model.predict_proba(np.asarray(values).reshape(-1, 1))[:, 1]


def evaluate(
    *,
    manifest_path: Path,
    full_segments_path: Path,
    out_dir: Path,
    closure: str,
    test_chrs: set[str],
    val_chrs: set[str],
    seed: int,
    split_seed: int,
    shortest_path_cutoff: int,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    manifest = manifest.loc[manifest["closure"].astype(str).eq(closure)].copy()
    manifest["chromosome"] = manifest["target_sn"].map(normalize_chrom)
    train = manifest.loc[~manifest["chromosome"].isin(test_chrs | val_chrs)].copy()
    validation = manifest.loc[manifest["chromosome"].isin(val_chrs)].copy()
    heldout = manifest.loc[manifest["chromosome"].isin(test_chrs)].copy()
    if train.empty or validation.empty or heldout.empty:
        raise ValueError(
            f"Empty split: train={len(train)}, validation={len(validation)}, heldout={len(heldout)}"
        )

    segments = pd.read_csv(
        full_segments_path,
        compression="infer",
        usecols=["name", "seq", "SN", "SO"],
    )
    seg_index, seg_u = build_global_index(segments)
    sequences = seg_u["seq"].fillna("").astype(str).to_numpy()
    sn = seg_u["SN"].fillna("").astype(str).to_numpy()
    so = seg_u["SO"].fillna(0).to_numpy(np.int64)
    sequence_cache: dict[int, np.ndarray] = {}
    coordinate_model = SGDClassifier(
        loss="log_loss", alpha=1e-4, max_iter=1, tol=None, random_state=seed,
        learning_rate="optimal", average=True,
    )
    sequence_model = SGDClassifier(
        loss="log_loss", alpha=1e-4, max_iter=1, tol=None, random_state=seed,
        learning_rate="optimal", average=True,
    )
    classes = np.array([0, 1], dtype=np.int8)
    fitted = False
    train_counts: Counter = Counter()
    rng = np.random.default_rng(seed)
    train_order = rng.permutation(len(train))
    for row_index in train_order:
        row = train.iloc[int(row_index)]
        candidates = load_candidates(row, manifest_path.parent)
        indices = _split_indices(len(candidates), "train", split_seed)
        u, v, y = candidate_arrays(candidates, indices)
        if len(y) < 2 or len(np.unique(y)) < 2:
            continue
        order = rng.permutation(len(y))
        u, v, y = u[order], v[order], y[order]
        coordinate_model.partial_fit(
            coordinate_features(u, v, so=so, sn=sn), y, classes=classes
        )
        sequence_model.partial_fit(
            sequence_features(u, v, sequences=sequences, cache=sequence_cache),
            y,
            classes=classes,
        )
        train_counts.update(y.tolist())
        fitted = True
    if not fitted:
        raise RuntimeError("No eligible training candidate batches")
    training_prevalence = train_counts[1] / sum(train_counts.values())

    prediction_rows: list[pd.DataFrame] = []
    for split_name, split_frame in [("validation", validation), ("heldout", heldout)]:
        for _, row in split_frame.iterrows():
            candidates = load_candidates(row, manifest_path.parent)
            indices = _split_indices(len(candidates), "test", split_seed)
            u, v, y = candidate_arrays(candidates, indices)
            if len(y) == 0:
                continue
            scores: dict[str, np.ndarray] = {
                "coordinate_sgd": coordinate_model.predict_proba(
                    coordinate_features(u, v, so=so, sn=sn)
                )[:, 1],
                "sequence_composition_sgd": sequence_model.predict_proba(
                    sequence_features(u, v, sequences=sequences, cache=sequence_cache)
                )[:, 1],
                "training_frequency": np.full(len(y), training_prevalence),
                "random_uniform": rng.random(len(y)),
            }
            links = pd.read_csv(
                resolve_path(row["links_path"], manifest_path.parent), compression="infer"
            )
            structural_u, structural_v = oriented_ids_from_links(links, seg_index)
            adjacency = _build_undirected_adjacency(zip(structural_u, structural_v))
            heuristic_frame = _score_candidates(
                adjacency,
                candidates,
                indices,
                mask_query_edges=True,
                shortest_path_cutoff=shortest_path_cutoff,
            )
            for name in HEURISTICS:
                scores[f"topology_{name}"] = heuristic_frame[name].to_numpy(float)
            base = {
                "slice": str(row["name"]),
                "target_sn": str(row["target_sn"]),
                "chromosome": str(row["chromosome"]),
                "closure": closure,
                "fold_evaluation_split": split_name,
            }
            for baseline, values in scores.items():
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            **base,
                            "candidate_row_index": indices,
                            "u_oid": u,
                            "v_oid": v,
                            "y_true": y,
                            "baseline": baseline,
                            "raw_score": values,
                        }
                    )
                )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    calibrated_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for baseline, group in predictions.groupby("baseline", sort=True):
        val = group.loc[group["fold_evaluation_split"].eq("validation")]
        test = group.loc[group["fold_evaluation_split"].eq("heldout")].copy()
        calibrator = fit_platt(
            val["raw_score"].to_numpy(float), val["y_true"].to_numpy(int)
        )
        val_probability = calibrator(val["raw_score"].to_numpy(float))
        threshold = choose_f1_threshold(val["y_true"].to_numpy(int), val_probability)
        test["p_calibrated"] = calibrator(test["raw_score"].to_numpy(float))
        test["validation_f1_threshold"] = threshold
        calibrated_frames.append(test)
        metric_rows.append(
            {
                "baseline": baseline,
                "closure": closure,
                "seed": seed,
                "split_seed": split_seed,
                "n_training_candidates": int(sum(train_counts.values())),
                "n_validation_candidates": int(len(val)),
                "calibration": "Platt scaling on validation chromosomes only",
                **classification_metrics(
                    test["y_true"].to_numpy(int),
                    test["p_calibrated"].to_numpy(float),
                    threshold=threshold,
                    n_bins=15,
                ),
            }
        )
        for chromosome, chrom_group in test.groupby("chromosome", sort=True):
            y_chrom = chrom_group["y_true"].to_numpy(int)
            p_chrom = chrom_group["p_calibrated"].to_numpy(float)
            metric_rows.append(
                {
                    "baseline": baseline,
                    "closure": closure,
                    "seed": seed,
                    "split_seed": split_seed,
                    "chromosome": chromosome,
                    "n_training_candidates": int(sum(train_counts.values())),
                    "n_validation_candidates": int(len(val)),
                    "calibration": "Platt scaling on validation chromosomes only",
                    **classification_metrics(
                        y_chrom, p_chrom, threshold=threshold, n_bins=15
                    ),
                }
            )

    heldout_predictions = pd.concat(calibrated_frames, ignore_index=True)
    heldout_predictions.to_csv(
        out_dir / "heldout_predictions.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)
    audit = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "manifest": str(manifest_path.resolve()),
        "full_segments": str(full_segments_path.resolve()),
        "closure": closure,
        "seed": seed,
        "split_seed": split_seed,
        "training_chromosomes": sorted(set(train["chromosome"])),
        "validation_chromosomes": sorted(val_chrs),
        "heldout_chromosomes": sorted(test_chrs),
        "training_slices": int(len(train)),
        "validation_slices": int(len(validation)),
        "heldout_slices": int(len(heldout)),
        "training_candidate_counts": {str(k): int(v) for k, v in train_counts.items()},
        "sequence_feature_definition": "endpoint mono/di-nucleotide composition, GC, entropy, and length interactions; no topology",
        "coordinate_feature_definition": "endpoint positions, distance, chromosome equality, and orientation; no topology or sequence",
        "topology_query_masking": "positive direct query edge removed independently before each score",
        "topology_adjacency_source": "closure-specific links_path from each manifest row",
        "topology_degree_definition": "endpoint degree recomputed on the strict or one-hop graph visible to the corresponding baseline before independent positive query-edge masking",
        "baselines": BASELINE_NAMES,
        "status": "complete",
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-segments", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--closure", choices=["strict", "1hop"], required=True)
    parser.add_argument("--test-chrs", nargs="+", required=True)
    parser.add_argument("--val-chrs", nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=20260806)
    parser.add_argument("--shortest-path-cutoff", type=int, default=4)
    args = parser.parse_args()
    payload = evaluate(
        manifest_path=args.manifest,
        full_segments_path=args.full_segments,
        out_dir=args.out_dir,
        closure=args.closure,
        test_chrs={normalize_chrom(value) for value in args.test_chrs},
        val_chrs={normalize_chrom(value) for value in args.val_chrs},
        seed=args.seed,
        split_seed=args.split_seed,
        shortest_path_cutoff=args.shortest_path_cutoff,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
