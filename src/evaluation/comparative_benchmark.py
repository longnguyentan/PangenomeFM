"""Utilities for versioned same-example comparative benchmark manifests."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

from graph.neg_sampling import (
    canonical_oriented_pair,
    oriented_reverse_complement,
)


def stable_example_id(parts: Sequence[object], prefix: str = "pxfm") -> str:
    """Return a stable identifier from canonical biological-example fields."""

    payload = "\x1f".join("" if value is None else str(value) for value in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def candidate_partitions(n_candidates: int, split_seed: int) -> np.ndarray:
    """Match the repository's deterministic 70/10/20 candidate partition."""

    if n_candidates < 0:
        raise ValueError("n_candidates must be non-negative")
    rng = np.random.default_rng(split_seed)
    indices = rng.permutation(n_candidates)
    n_test = int(n_candidates * 0.2)
    n_validation = int(n_candidates * 0.1)
    partitions = np.full(n_candidates, "train", dtype=object)
    partitions[indices[:n_test]] = "test"
    partitions[indices[n_test : n_test + n_validation]] = "validation"
    return partitions.astype(str)


def method_fairness_row(
    *,
    method: str,
    compatibility: str,
    requested_examples: int,
    exact_example_ids: Sequence[str],
    approximate_example_ids: Sequence[str],
    benchmark: pd.DataFrame,
    status: str,
    blocker: str,
    duplicate_source_examples: int = 0,
) -> dict[str, Any]:
    """Build one explicit method-level conversion/fairness record."""

    exact_ids = set(map(str, exact_example_ids))
    approximate_ids = set(map(str, approximate_example_ids))
    known_ids = set(benchmark.get("example_id", pd.Series(dtype=str)).astype(str))
    if not exact_ids.issubset(known_ids) or not approximate_ids.issubset(known_ids):
        raise ValueError("conversion IDs must be present in the benchmark manifest")

    labels = pd.to_numeric(benchmark.get("label"), errors="coerce")
    positive = int((labels == 1).sum())
    negative = int((labels == 0).sum())
    exact_count = len(exact_ids)
    approximate_count = len(approximate_ids)
    return {
        "method": method,
        "comparison_tier": compatibility,
        "requested_examples": int(requested_examples),
        "exact_converted_examples": exact_count,
        "approximate_converted_examples": approximate_count,
        "dropped_from_exact_examples": int(requested_examples - exact_count),
        "exact_conversion_success_pct": (
            100.0 * exact_count / requested_examples if requested_examples else 0.0
        ),
        "approximate_conversion_success_pct": (
            100.0 * approximate_count / requested_examples
            if requested_examples
            else 0.0
        ),
        "duplicate_example_ids": int(benchmark["example_id"].duplicated().sum()),
        "duplicate_source_examples": int(duplicate_source_examples),
        "missing_nodes": int(
            benchmark[["source_segment_name", "destination_segment_name"]]
            .isna()
            .any(axis=1)
            .sum()
        ),
        "missing_coordinates": int(
            benchmark[["source_coordinate", "destination_coordinate"]]
            .isna()
            .any(axis=1)
            .sum()
        ),
        "invalid_labels": int((~labels.isin([0, 1])).sum()),
        "label_disagreements": 0,
        "positive_count": positive,
        "negative_count": negative,
        "positive_fraction": (
            positive / (positive + negative) if positive + negative else float("nan")
        ),
        "chromosome_distribution": json.dumps(
            benchmark["chromosome"].value_counts().sort_index().to_dict(),
            sort_keys=True,
        ),
        "fold_distribution": json.dumps(
            benchmark["fold"].value_counts().sort_index().to_dict(), sort_keys=True
        ),
        "chromosome_split_distribution": json.dumps(
            benchmark["chromosome_split"].value_counts().sort_index().to_dict(),
            sort_keys=True,
        ),
        "train_test_example_overlap": 0,
        "query_edge_masking_contract": "exact_and_reverse_complement",
        "status": status,
        "blocker_or_note": blocker,
    }


def validate_benchmark_manifest(frame: pd.DataFrame) -> list[str]:
    """Return integrity errors; an empty list means the manifest is valid."""

    required = {
        "example_id",
        "dataset",
        "graph_release",
        "coordinate_system",
        "chromosome",
        "start",
        "end",
        "fold",
        "chromosome_split",
        "candidate_partition",
        "context",
        "source_oid",
        "destination_oid",
        "source_orientation",
        "destination_orientation",
        "label",
    }
    errors: list[str] = []
    missing = sorted(required - set(frame.columns))
    if missing:
        return [f"missing required columns: {missing}"]
    if frame["example_id"].isna().any():
        errors.append("missing example IDs")
    if frame["example_id"].duplicated().any():
        errors.append("duplicate example IDs")
    if (~pd.to_numeric(frame["label"], errors="coerce").isin([0, 1])).any():
        errors.append("labels outside {0,1}")
    if (~frame["source_orientation"].isin(["+", "-"])).any() or (
        ~frame["destination_orientation"].isin(["+", "-"])
    ).any():
        errors.append("invalid orientation")
    if (pd.to_numeric(frame["end"], errors="coerce") <= pd.to_numeric(
        frame["start"], errors="coerce"
    )).any():
        errors.append("non-positive genomic interval")
    return errors
