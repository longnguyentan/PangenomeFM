"""Utilities for exact-universe modality-factorial downstream probes.

The experiment treats coordinate (C), sequence (S), and topology (T) as
separate information channels.  Keeping construction here prevents the cCRE
and SV probes from silently using different definitions for the same
factorial comparison.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


CORE_MODALITY_COMBINATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("coordinate", ("coordinate",)),
    ("sequence_kmer", ("sequence_kmer",)),
    ("frozen_pangenomefm", ("frozen_pangenomefm",)),
    ("coordinate_plus_sequence_kmer", ("coordinate", "sequence_kmer")),
    (
        "coordinate_plus_frozen_pangenomefm",
        ("coordinate", "frozen_pangenomefm"),
    ),
    (
        "sequence_plus_frozen_pangenomefm",
        ("sequence_kmer", "frozen_pangenomefm"),
    ),
    (
        "coordinate_plus_sequence_plus_frozen_pangenomefm",
        ("coordinate", "sequence_kmer", "frozen_pangenomefm"),
    ),
)


EXTERNAL_SEQUENCE_COMBINATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("frozen_sequence_fm", ("frozen_sequence_fm",)),
    (
        "coordinate_plus_frozen_sequence_fm",
        ("coordinate", "frozen_sequence_fm"),
    ),
    (
        "frozen_sequence_fm_plus_frozen_pangenomefm",
        ("frozen_sequence_fm", "frozen_pangenomefm"),
    ),
    (
        "coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm",
        ("coordinate", "frozen_sequence_fm", "frozen_pangenomefm"),
    ),
)


def concatenate_modalities(
    components: Mapping[str, np.ndarray],
    names: Sequence[str],
) -> np.ndarray:
    """Validate and concatenate modality matrices without changing rows."""

    matrices: list[np.ndarray] = []
    row_count: int | None = None
    for name in names:
        if name not in components:
            raise KeyError(f"Missing modality component: {name}")
        matrix = np.asarray(components[name])
        if matrix.ndim != 2:
            raise ValueError(f"Modality {name} must be two-dimensional, got {matrix.shape}")
        if row_count is None:
            row_count = len(matrix)
        elif len(matrix) != row_count:
            raise ValueError(
                f"Modality row mismatch for {name}: {len(matrix)} != {row_count}"
            )
        if matrix.shape[1] == 0:
            raise ValueError(f"Modality {name} has zero columns")
        if not np.isfinite(matrix).all():
            raise ValueError(f"Modality {name} contains non-finite values")
        matrices.append(matrix.astype(np.float32, copy=False))
    if not matrices:
        raise ValueError("At least one modality is required")
    if len(matrices) == 1:
        return matrices[0]
    return np.concatenate(matrices, axis=1).astype(np.float32, copy=False)


def build_modality_factorial(
    components: Mapping[str, np.ndarray],
    *,
    suffix: str = "",
    include_external_sequence: bool = False,
) -> dict[str, np.ndarray]:
    """Build C/S/T singles, pairs, and triple on one fixed row universe."""

    combinations = list(CORE_MODALITY_COMBINATIONS)
    if include_external_sequence:
        combinations.extend(EXTERNAL_SEQUENCE_COMBINATIONS)
    return {
        f"{feature_name}{suffix}": concatenate_modalities(components, names)
        for feature_name, names in combinations
    }


def factorial_feature_access(*, suffix: str = "") -> dict[str, str]:
    """Return explicit information-access statements for every combination."""

    descriptions = {
        "coordinate": "C: reference coordinate, node length, and constant orientation only",
        "sequence_kmer": "S: mono/di/tri-nucleotide composition only; no graph adjacency",
        "frozen_pangenomefm": "T: frozen topology-pretrained PangenomeFM embedding only; no nucleotide input",
        "coordinate_plus_sequence_kmer": "C+S: coordinate and sequence-composition features; no topology embedding",
        "coordinate_plus_frozen_pangenomefm": "C+T: coordinate and frozen topology embedding; no nucleotide input",
        "sequence_plus_frozen_pangenomefm": "S+T: sequence composition and frozen topology embedding",
        "coordinate_plus_sequence_plus_frozen_pangenomefm": "C+S+T: coordinate, sequence composition, and frozen topology embedding",
        "frozen_sequence_fm": "S-FM: frozen pretrained sequence-model embedding only; no graph adjacency",
        "coordinate_plus_frozen_sequence_fm": "C+S-FM: coordinate and frozen sequence-model embedding",
        "frozen_sequence_fm_plus_frozen_pangenomefm": "S-FM+T: frozen sequence-model and topology embeddings",
        "coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm": "C+S-FM+T: coordinate, frozen sequence-model, and topology embeddings",
    }
    return {f"{name}{suffix}": description for name, description in descriptions.items()}


def select_feature_sets(
    features: Mapping[str, np.ndarray],
    requested: Sequence[str] | None,
) -> dict[str, np.ndarray]:
    """Select features in requested order and fail on misspellings."""

    if requested is None:
        return dict(features)
    duplicates = sorted({name for name in requested if requested.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate --feature-sets values: {duplicates}")
    missing = [name for name in requested if name not in features]
    if missing:
        raise ValueError(
            f"Unknown feature sets {missing}; available={sorted(features)}"
        )
    return {name: features[name] for name in requested}


def load_frozen_node_embedding_cache(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load a label-blind external node-embedding cache and its audit."""

    audit_path = Path(f"{path}.audit.json")
    if not audit_path.exists():
        raise FileNotFoundError(
            f"External sequence cache requires an audit sidecar: {audit_path}"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "complete":
        raise ValueError(f"External sequence cache audit is not complete: {audit_path}")
    if audit.get("downstream_label_access") not in {"none", False}:
        raise ValueError(
            "External sequence cache audit must declare downstream_label_access='none'"
        )
    with np.load(path, allow_pickle=False) as cache:
        if "segid" not in cache or "embeddings" not in cache:
            raise KeyError("External sequence cache needs arrays 'segid' and 'embeddings'")
        segids = cache["segid"].astype(np.int64)
        embeddings = cache["embeddings"].astype(np.float32)
    if segids.ndim != 1 or embeddings.ndim != 2 or len(segids) != len(embeddings):
        raise ValueError(
            f"Invalid external cache shapes: segid={segids.shape}, embeddings={embeddings.shape}"
        )
    if len(np.unique(segids)) != len(segids):
        raise ValueError("External sequence cache contains duplicate segid values")
    if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
        raise ValueError("External sequence cache embeddings are empty or non-finite")
    return segids, embeddings, audit
