"""Performance-independent graph-window complexity features and scoring.

The functions in this module deliberately accept graph slice tables rather
than model predictions.  Complexity is a property of the materialized graph
window; it must be frozen before any comparison of model performance.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from graph.neg_sampling import canonical_oriented_pair


FEATURE_DEFINITIONS: dict[str, dict[str, str]] = {
    "node_count": {
        "definition": "Number of unique materialized segment names in the window.",
        "units": "nodes",
        "meaning": "Graph segmentation and local structural content.",
    },
    "link_row_count": {
        "definition": "Number of oriented link rows in the materialized slice.",
        "units": "link rows",
        "meaning": "Traversal-level adjacency exposure.",
    },
    "unique_edge_count": {
        "definition": "Number of unique undirected segment-name pairs after removing duplicate rows.",
        "units": "edges",
        "meaning": "Simple-graph connectivity independent of duplicate orientation rows.",
    },
    "nodes_per_kb": {
        "definition": "Unique nodes divided by target-window length in kilobases.",
        "units": "nodes/kb",
        "meaning": "Local segmentation density.",
    },
    "edges_per_kb": {
        "definition": "Unique undirected edges divided by target-window length in kilobases.",
        "units": "edges/kb",
        "meaning": "Local adjacency density.",
    },
    "mean_degree": {
        "definition": "Mean simple undirected degree across all materialized nodes, including isolates.",
        "units": "neighbors/node",
        "meaning": "Average local connectivity.",
    },
    "max_degree": {
        "definition": "Maximum simple undirected node degree.",
        "units": "neighbors",
        "meaning": "Strongest local branch point.",
    },
    "branching_node_count": {
        "definition": "Number of materialized nodes with simple undirected degree greater than two.",
        "units": "nodes",
        "meaning": "Count of non-linear junctions.",
    },
    "branching_fraction": {
        "definition": "Fraction of materialized nodes with degree greater than two.",
        "units": "fraction",
        "meaning": "Prevalence of non-linear junctions.",
    },
    "connected_components": {
        "definition": "Number of connected components in the simple undirected graph, including isolates.",
        "units": "components",
        "meaning": "Fragmentation of the materialized context.",
    },
    "largest_component_fraction": {
        "definition": "Nodes in the largest component divided by all materialized nodes.",
        "units": "fraction",
        "meaning": "Extent to which visible structure forms one connected context.",
    },
    "local_density": {
        "definition": "Simple undirected density 2E/[N(N-1)].",
        "units": "fraction",
        "meaning": "Connectivity relative to a complete graph; size-dependent and reported descriptively.",
    },
    "cycle_rank": {
        "definition": "Cyclomatic number E - N + C for the simple undirected graph.",
        "units": "independent cycles",
        "meaning": "Number of independent alternative cycles in the materialized topology.",
    },
    "cycle_rank_per_kb": {
        "definition": "Cyclomatic number divided by target-window length in kilobases.",
        "units": "independent cycles/kb",
        "meaning": "Cycle-related complexity normalized for window length.",
    },
    "reference_node_fraction": {
        "definition": "Fraction of nodes whose numeric GFA SR tag equals zero.",
        "units": "fraction",
        "meaning": "Release-specific reference-ranked material; this is not a universal biological reference label.",
    },
    "alternate_node_fraction": {
        "definition": "Fraction of nodes whose numeric GFA SR tag is nonzero.",
        "units": "fraction",
        "meaning": "Release-specific non-reference-ranked graph material.",
    },
    "alternate_to_reference_ratio": {
        "definition": "Count(SR != 0) / Count(SR == 0); undefined when no SR==0 node exists.",
        "units": "ratio",
        "meaning": "Relative amount of alternate-ranked material.",
    },
    "mean_node_length_bp": {
        "definition": "Mean numeric LN tag among materialized nodes.",
        "units": "bp",
        "meaning": "Average graph segment granularity.",
    },
    "median_node_length_bp": {
        "definition": "Median numeric LN tag among materialized nodes.",
        "units": "bp",
        "meaning": "Typical graph segment granularity.",
    },
    "p95_node_length_bp": {
        "definition": "95th percentile of numeric LN tags.",
        "units": "bp",
        "meaning": "Upper-tail segment length.",
    },
    "max_node_length_bp": {
        "definition": "Maximum numeric LN tag.",
        "units": "bp",
        "meaning": "Largest materialized segment.",
    },
    "materialized_sequence_bp": {
        "definition": "Sum of numeric LN tags.",
        "units": "bp",
        "meaning": "Total segment sequence exposed by the context.",
    },
    "node_length_iqr_over_median": {
        "definition": "Interquartile range of numeric LN tags divided by their median.",
        "units": "ratio",
        "meaning": "Robust heterogeneity of graph segmentation lengths.",
    },
    "positive_candidate_count": {
        "definition": "Number of candidate rows with label 1.",
        "units": "examples",
        "meaning": "Positive reconstruction targets.",
    },
    "negative_candidate_count": {
        "definition": "Number of candidate rows with label 0.",
        "units": "examples",
        "meaning": "Sampled non-edge targets.",
    },
}


REQUIRED_SEGMENT_COLUMNS = {"name", "LN", "SR"}
REQUIRED_LINK_COLUMNS = {"from_seg", "to_seg", "from_orient", "to_orient"}


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def compute_slice_complexity(
    segments: pd.DataFrame,
    links: pd.DataFrame,
    candidates: pd.DataFrame | None,
    *,
    window_bp: int,
) -> dict[str, float | int]:
    """Compute deterministic, interpretable metrics for one graph slice."""

    missing_segments = REQUIRED_SEGMENT_COLUMNS - set(segments.columns)
    missing_links = REQUIRED_LINK_COLUMNS - set(links.columns)
    if missing_segments:
        raise ValueError(f"segments missing required columns: {sorted(missing_segments)}")
    if missing_links:
        raise ValueError(f"links missing required columns: {sorted(missing_links)}")
    if int(window_bp) <= 0:
        raise ValueError("window_bp must be positive")

    duplicate_segment_ids = int(segments["name"].astype(str).duplicated().sum())
    node_names = pd.Index(segments["name"].astype(str).unique())
    node_set = set(node_names)
    missing_endpoint_rows = int(
        (
            ~links["from_seg"].astype(str).isin(node_set)
            | ~links["to_seg"].astype(str).isin(node_set)
        ).sum()
    )
    valid_orientation = links["from_orient"].isin(["+", "-"]) & links[
        "to_orient"
    ].isin(["+", "-"])
    invalid_orientation_rows = int((~valid_orientation).sum())

    graph = nx.Graph()
    graph.add_nodes_from(node_names)
    valid_links = links.loc[
        links["from_seg"].astype(str).isin(node_set)
        & links["to_seg"].astype(str).isin(node_set)
    ]
    graph.add_edges_from(
        zip(
            valid_links["from_seg"].astype(str),
            valid_links["to_seg"].astype(str),
        )
    )

    node_count = int(graph.number_of_nodes())
    edge_count = int(graph.number_of_edges())
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    components = list(nx.connected_components(graph)) if node_count else []
    connected_components = len(components)
    largest_component = max((len(c) for c in components), default=0)
    cycle_rank = max(0, edge_count - node_count + connected_components)
    kb = float(window_bp) / 1000.0

    sr = pd.to_numeric(
        segments.drop_duplicates("name")["SR"], errors="coerce"
    ).to_numpy(dtype=float)
    known_sr = sr[np.isfinite(sr)]
    reference_count = int(np.sum(known_sr == 0))
    alternate_count = int(np.sum(known_sr != 0))

    lengths = pd.to_numeric(
        segments.drop_duplicates("name")["LN"], errors="coerce"
    ).to_numpy(dtype=float)
    finite_lengths = lengths[np.isfinite(lengths)]

    oriented_key_columns = ["from_seg", "from_orient", "to_seg", "to_orient"]
    duplicate_link_rows = int(links.duplicated(oriented_key_columns).sum())
    nonforward_rows = int(
        ((links["from_orient"] != "+") | (links["to_orient"] != "+")).sum()
    )

    positive_count = negative_count = duplicate_candidate_pairs = 0
    reverse_equivalent_candidate_duplicates = 0
    orientation_equivalent_candidate_label_conflicts = 0
    invalid_candidate_labels = 0
    if candidates is not None:
        if not {"u_oid", "v_oid", "label"}.issubset(candidates.columns):
            raise ValueError("candidates require u_oid, v_oid, and label columns")
        labels = pd.to_numeric(candidates["label"], errors="coerce")
        positive_count = int((labels == 1).sum())
        negative_count = int((labels == 0).sum())
        invalid_candidate_labels = int((~labels.isin([0, 1])).sum())
        duplicate_candidate_pairs = int(
            candidates.duplicated(["u_oid", "v_oid"]).sum()
        )
        canonical_pairs = [
            canonical_oriented_pair(u, v)
            for u, v in zip(candidates["u_oid"], candidates["v_oid"])
        ]
        reverse_equivalent_candidate_duplicates = int(
            pd.Series(canonical_pairs, dtype=object).duplicated().sum()
        )
        canonical_frame = pd.DataFrame(
            {
                "canonical_u": [pair[0] for pair in canonical_pairs],
                "canonical_v": [pair[1] for pair in canonical_pairs],
                "label": labels,
            }
        )
        orientation_equivalent_candidate_label_conflicts = int(
            (
                canonical_frame.groupby(
                    ["canonical_u", "canonical_v"], sort=False
                )["label"].nunique(dropna=True)
                > 1
            ).sum()
        )

    median_length = (
        float(np.median(finite_lengths)) if len(finite_lengths) else float("nan")
    )
    length_iqr = (
        float(np.quantile(finite_lengths, 0.75) - np.quantile(finite_lengths, 0.25))
        if len(finite_lengths)
        else float("nan")
    )

    return {
        "node_count": node_count,
        "link_row_count": int(len(links)),
        "unique_edge_count": edge_count,
        "nodes_per_kb": node_count / kb,
        "edges_per_kb": edge_count / kb,
        "mean_degree": float(degrees.mean()) if len(degrees) else float("nan"),
        "max_degree": int(degrees.max()) if len(degrees) else 0,
        "branching_node_count": int(np.sum(degrees > 2)),
        "branching_fraction": _safe_fraction(np.sum(degrees > 2), node_count),
        "connected_components": int(connected_components),
        "largest_component_fraction": _safe_fraction(largest_component, node_count),
        "local_density": float(nx.density(graph)) if node_count > 1 else 0.0,
        "cycle_rank": int(cycle_rank),
        "cycle_rank_per_kb": cycle_rank / kb,
        "reference_node_fraction": _safe_fraction(reference_count, len(known_sr)),
        "alternate_node_fraction": _safe_fraction(alternate_count, len(known_sr)),
        "alternate_to_reference_ratio": _safe_fraction(alternate_count, reference_count),
        "mean_node_length_bp": float(np.mean(finite_lengths)) if len(finite_lengths) else float("nan"),
        "median_node_length_bp": median_length,
        "p95_node_length_bp": float(np.quantile(finite_lengths, 0.95)) if len(finite_lengths) else float("nan"),
        "max_node_length_bp": float(np.max(finite_lengths)) if len(finite_lengths) else float("nan"),
        "materialized_sequence_bp": int(np.sum(finite_lengths)) if len(finite_lengths) else 0,
        "node_length_iqr_over_median": _safe_fraction(length_iqr, median_length),
        "positive_candidate_count": positive_count,
        "negative_candidate_count": negative_count,
        "positive_candidate_fraction": _safe_fraction(
            positive_count, positive_count + negative_count
        ),
        "duplicate_segment_ids": duplicate_segment_ids,
        "duplicate_link_rows": duplicate_link_rows,
        "duplicate_candidate_pairs": duplicate_candidate_pairs,
        "reverse_equivalent_candidate_duplicates": reverse_equivalent_candidate_duplicates,
        "orientation_equivalent_candidate_label_conflicts": (
            orientation_equivalent_candidate_label_conflicts
        ),
        "missing_link_endpoint_rows": missing_endpoint_rows,
        "invalid_orientation_rows": invalid_orientation_rows,
        "invalid_candidate_labels": invalid_candidate_labels,
        "nonforward_link_fraction": _safe_fraction(nonforward_rows, len(links)),
    }


def _transformed(values: pd.Series, transform: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if transform == "identity":
        return numeric
    if transform == "log1p":
        if (numeric.dropna() < 0).any():
            raise ValueError("log1p complexity features must be non-negative")
        return np.log1p(numeric)
    raise ValueError(f"unsupported complexity transform: {transform}")


def _score_from_parameters(
    features: pd.DataFrame,
    feature_specs: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Mapping[str, float]],
    winsorize: tuple[float, float],
) -> pd.DataFrame:
    scored = features.copy()
    weighted: list[np.ndarray] = []
    weights: list[float] = []
    for spec in feature_specs:
        source = str(spec["source_feature"])
        output_name = str(spec["name"])
        if source not in features.columns:
            raise ValueError(f"complexity source feature is missing: {source}")
        transformed = _transformed(features[source], str(spec["transform"]))
        center = float(parameters[output_name]["center"])
        scale = float(parameters[output_name]["scale"])
        normalized = (
            pd.Series(np.zeros(len(transformed)), index=transformed.index, dtype=float)
            if scale == 0.0
            else (transformed - center) / scale
        )
        normalized = normalized.clip(winsorize[0], winsorize[1]).fillna(0.0)
        scored[f"complexity_component__{output_name}"] = normalized
        weight = float(spec.get("weight", 1.0))
        weighted.append(normalized.to_numpy(dtype=float) * weight)
        weights.append(weight)
    if not weighted or sum(weights) == 0:
        raise ValueError("complexity definition requires at least one nonzero weight")
    scored["complexity_score"] = np.sum(weighted, axis=0) / sum(weights)
    return scored


def fit_complexity_definition(
    reference_features: pd.DataFrame,
    definition: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit robust normalization and tertiles without reading performance."""

    feature_specs = definition["composite"]["features"]
    lo, hi = map(float, definition["normalization"]["winsorize"])
    parameters: dict[str, dict[str, float]] = {}
    for spec in feature_specs:
        transformed = _transformed(
            reference_features[str(spec["source_feature"])], str(spec["transform"])
        )
        center = float(transformed.median())
        mad = float((transformed - center).abs().median())
        scale = float(1.4826 * mad) if mad > 0 else 0.0
        parameters[str(spec["name"])] = {"center": center, "scale": scale}

    scored = _score_from_parameters(
        reference_features, feature_specs, parameters, (lo, hi)
    )
    lower, upper = np.quantile(
        scored["complexity_score"].to_numpy(dtype=float), [1.0 / 3.0, 2.0 / 3.0]
    )
    labels = list(definition["categories"]["labels"])
    scored["complexity_category"] = pd.cut(
        scored["complexity_score"],
        bins=[-np.inf, float(lower), float(upper), np.inf],
        labels=labels,
        include_lowest=True,
    ).astype(str)

    frozen = {
        "schema_version": 1,
        "definition_name": definition["name"],
        "independence_rule": definition["independence_rule"],
        "reference_context": definition["fit_population"]["reference_context"],
        "feature_specs": [dict(spec) for spec in feature_specs],
        "normalization_parameters": parameters,
        "winsorize": [lo, hi],
        "category_method": definition["categories"]["method"],
        "category_labels": labels,
        "thresholds": {"low_to_medium": float(lower), "medium_to_high": float(upper)},
        "reference_window_count": int(len(reference_features)),
    }
    return scored, frozen


def apply_complexity_definition(
    features: pd.DataFrame,
    frozen: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply a previously frozen complexity definition."""

    scored = _score_from_parameters(
        features,
        frozen["feature_specs"],
        frozen["normalization_parameters"],
        tuple(map(float, frozen["winsorize"])),
    )
    thresholds = frozen["thresholds"]
    scored["complexity_category"] = pd.cut(
        scored["complexity_score"],
        bins=[
            -np.inf,
            float(thresholds["low_to_medium"]),
            float(thresholds["medium_to_high"]),
            np.inf,
        ],
        labels=list(frozen["category_labels"]),
        include_lowest=True,
    ).astype(str)
    return scored
