"""Quantify construction-domain shift between cleaned pangenome graph tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["log_node_length", "log_degree", "branching", "log_coordinate", "sr"]
FEATURE_SETS = {
    "all": FEATURES,
    "structure_only": ["log_node_length", "log_degree", "branching"],
    "topology_only": ["log_degree", "branching"],
}


def _parse_graph(specification: str) -> tuple[str, Path, Path]:
    parts = specification.split("=", 1)
    if len(parts) != 2 or "," not in parts[1]:
        raise ValueError(
            "Graph inputs must use LABEL=SEGMENTS,LINKS syntax; received "
            f"{specification!r}."
        )
    segments, links = parts[1].split(",", 1)
    return parts[0], Path(segments), Path(links)


def _sample_segments(
    path: Path, *, max_nodes: int, seed: int
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        compression="infer",
        usecols=["name", "LN", "SO", "SR"],
    )
    if len(frame) > max_nodes:
        frame = frame.sample(max_nodes, random_state=seed)
    return frame.reset_index(drop=True)


def _degrees_for_nodes(path: Path, names: set[str]) -> dict[str, int]:
    degree = {name: 0 for name in names}
    for chunk in pd.read_csv(
        path,
        compression="infer",
        usecols=["from_seg", "to_seg"],
        chunksize=250_000,
    ):
        for column in ("from_seg", "to_seg"):
            counts = chunk.loc[chunk[column].isin(names), column].value_counts()
            for name, count in counts.items():
                degree[str(name)] += int(count)
    return degree


def graph_feature_sample(
    *,
    segments_path: Path,
    links_path: Path,
    max_nodes: int,
    seed: int,
) -> pd.DataFrame:
    frame = _sample_segments(segments_path, max_nodes=max_nodes, seed=seed)
    degree = _degrees_for_nodes(links_path, set(frame["name"].astype(str)))
    lengths = pd.to_numeric(frame["LN"], errors="coerce").fillna(0).clip(lower=0)
    coordinates = (
        pd.to_numeric(frame["SO"], errors="coerce").fillna(0).abs().clip(lower=0)
    )
    sr = pd.to_numeric(frame["SR"], errors="coerce").fillna(-1)
    node_degree = frame["name"].astype(str).map(degree).fillna(0).astype(float)
    return pd.DataFrame(
        {
            "log_node_length": np.log1p(lengths.to_numpy(float)),
            "log_degree": np.log1p(node_degree.to_numpy(float)),
            "branching": (node_degree.to_numpy(float) > 2).astype(float),
            "log_coordinate": np.log1p(coordinates.to_numpy(float)),
            "sr": sr.to_numpy(float),
        }
    )


def _summaries(label: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        values = frame[feature].to_numpy(float)
        rows.append(
            {
                "graph": label,
                "feature": feature,
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "q05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "q95": float(np.quantile(values, 0.95)),
            }
        )
    return rows


def pairwise_domain_shift(
    left_label: str,
    left: pd.DataFrame,
    right_label: str,
    right: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    balanced_n = min(len(left), len(right))
    left_indices = rng.choice(len(left), balanced_n, replace=False)
    right_indices = rng.choice(len(right), balanced_n, replace=False)
    features = pd.concat(
        [left.iloc[left_indices], right.iloc[right_indices]], ignore_index=True
    )
    labels = np.concatenate(
        [np.zeros(balanced_n, dtype=int), np.ones(balanced_n, dtype=int)]
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    domain_auc: dict[str, float] = {}
    for feature_set, columns in FEATURE_SETS.items():
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2_000, class_weight="balanced"),
        )
        probability = cross_val_predict(
            classifier,
            features[columns],
            labels,
            cv=folds,
            method="predict_proba",
        )[:, 1]
        domain_auc[feature_set] = float(roc_auc_score(labels, probability))
    ks = {
        feature: {
            "statistic": float(
                ks_2samp(left[feature], right[feature]).statistic
            ),
            "pvalue": float(ks_2samp(left[feature], right[feature]).pvalue),
        }
        for feature in FEATURES
    }
    return {
        "left": left_label,
        "right": right_label,
        "balanced_nodes_per_graph": int(balanced_n),
        "domain_classifier_auroc": domain_auc["all"],
        "domain_classifier_auroc_by_feature_set": domain_auc,
        "ks": ks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graphs",
        nargs="+",
        required=True,
        help="One or more LABEL=SEGMENTS,LINKS specifications.",
    )
    parser.add_argument("--compare", nargs=2, action="append")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-nodes", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames: dict[str, pd.DataFrame] = {}
    graph_sources: dict[str, dict[str, str]] = {}
    summary_rows: list[dict[str, Any]] = []
    for index, specification in enumerate(args.graphs):
        label, segments, links = _parse_graph(specification)
        if label in frames:
            raise ValueError(f"Duplicate graph label: {label!r}")
        frame = graph_feature_sample(
            segments_path=segments,
            links_path=links,
            max_nodes=args.max_nodes,
            seed=args.seed + index,
        )
        frames[label] = frame
        graph_sources[label] = {"segments": str(segments), "links": str(links)}
        summary_rows.extend(_summaries(label, frame))

    comparisons = args.compare or []
    pairwise = []
    for left, right in comparisons:
        if left not in frames or right not in frames:
            raise ValueError(f"Unknown comparison labels: {left!r}, {right!r}")
        pairwise.append(
            pairwise_domain_shift(
                left,
                frames[left],
                right,
                frames[right],
                seed=args.seed,
            )
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_dir / "feature_summaries.csv", index=False)
    result = {
        "graphs": graph_sources,
        "max_nodes": int(args.max_nodes),
        "features": FEATURES,
        "feature_sets": FEATURE_SETS,
        "comparisons": pairwise,
        "interpretation": (
            "Domain-classifier AUROC near 0.5 indicates similar sampled feature "
            "distributions; high AUROC indicates construction-domain signal."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
