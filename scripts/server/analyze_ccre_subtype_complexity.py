#!/usr/bin/env python3
"""Stratify frozen cCRE predictions by ENCODE class and graph complexity.

The analysis is deliberately post hoc with respect to model fitting: neither
the ENCODE subtype labels nor the graph-complexity categories are used to train
or select a downstream classifier.  Subtype analyses compare one named cCRE
class with the common background class and exclude other positive cCRE classes.
Complexity analyses retain the original binary cCRE outcome within independently
defined native-graph complexity tertiles.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from scripts.server.aggregate_ccre_frozen_probes import _hierarchical_draws


BASELINE_FEATURE = "coordinate_plus_frozen_sequence_fm"
FULL_FEATURE = "coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm"
SUBTYPE_MAP = {
    "PLS": "PLS",
    "pELS": "pELS",
    "dELS": "dELS",
    "CTCF-only": "CA-CTCF",
}
VALID_COMPLEXITY = ("low", "medium", "high")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_chromosome(value: object) -> str:
    text = str(value).strip()
    if "#" in text:
        text = text.split("#")[-1]
    if text.lower().startswith("chr"):
        return "chr" + text[3:]
    return "chr" + text


def prepare_labels(node_labels: Path, complexity_features: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    labels = pd.read_csv(
        node_labels,
        compression="infer",
        usecols=["segid", "chrom", "SO", "ccre_class", "ccre_label"],
    )
    if labels["segid"].duplicated().any():
        raise ValueError("node-label table contains duplicate segid values")
    labels["segid"] = labels["segid"].astype(np.int64)
    labels["chromosome"] = labels["chrom"].map(normalize_chromosome)
    labels["SO"] = pd.to_numeric(labels["SO"], errors="coerce")
    # ``ccre_label`` is the canonical nine-class integer label written by
    # ``tasks.ccre.label_nodes``.  The frozen-probe factorial uses the binary
    # task (any cCRE class versus background), so retain the canonical index
    # and derive the binary target explicitly instead of treating every
    # non-zero class index as though it were already the value 1.
    labels["ccre_label_index"] = pd.to_numeric(
        labels["ccre_label"], errors="raise"
    ).astype(np.int16)
    labels["ccre_binary_label"] = labels["ccre_class"].ne("background").astype(
        np.int8
    )

    complexity = pd.read_csv(complexity_features, sep="\t")
    required = {
        "chromosome",
        "start",
        "end",
        "context",
        "locus_complexity_score",
        "locus_complexity_category",
    }
    missing = sorted(required - set(complexity.columns))
    if missing:
        raise ValueError(f"complexity table is missing columns: {missing}")
    strict = complexity.loc[
        complexity["context"].astype(str).eq("strict"),
        list(required),
    ].copy()
    strict["chromosome"] = strict["chromosome"].map(normalize_chromosome)
    strict["start"] = strict["start"].astype(np.int64)
    strict["end"] = strict["end"].astype(np.int64)
    if strict.duplicated(["chromosome", "start"]).any():
        raise ValueError("strict complexity windows are not unique by chromosome/start")
    widths = strict["end"] - strict["start"]
    if widths.nunique() != 1:
        raise ValueError("complexity windows do not have a single fixed width")
    window_bp = int(widths.iloc[0])
    if window_bp <= 0:
        raise ValueError("invalid complexity window size")

    labels["window_start"] = (
        np.floor_divide(labels["SO"].fillna(-1).astype(np.int64), window_bp) * window_bp
    )
    strict = strict.rename(
        columns={
            "start": "window_start",
            "locus_complexity_category": "complexity_category",
            "locus_complexity_score": "complexity_score",
        }
    )
    labels = labels.merge(
        strict[["chromosome", "window_start", "complexity_category", "complexity_score"]],
        on=["chromosome", "window_start"],
        how="left",
        validate="many_to_one",
    )
    labels.loc[labels["SO"].lt(0), ["complexity_category", "complexity_score"]] = np.nan
    counts = strict["complexity_category"].value_counts().reindex(VALID_COMPLEXITY, fill_value=0)
    audit = {
        "node_labels": str(node_labels.resolve()),
        "node_labels_sha256": sha256(node_labels),
        "node_label_rows": int(len(labels)),
        "canonical_label_values": sorted(
            int(value) for value in labels["ccre_label_index"].unique()
        ),
        "binary_target_definition": "ccre_class != background",
        "binary_positive_labels": int(labels["ccre_binary_label"].sum()),
        "binary_background_labels": int(
            (labels["ccre_binary_label"] == 0).sum()
        ),
        "complexity_features": str(complexity_features.resolve()),
        "complexity_features_sha256": sha256(complexity_features),
        "complexity_window_bp": window_bp,
        "strict_complexity_window_counts": {key: int(value) for key, value in counts.items()},
        "labels_with_complexity_assignment": int(labels["complexity_category"].notna().sum()),
        "subtype_source_labels": SUBTYPE_MAP,
    }
    return labels, audit


def load_selected_predictions(path: Path) -> pd.DataFrame:
    columns = [
        "fold",
        "seed",
        "closure",
        "segid",
        "chromosome",
        "y_true",
        "feature_set",
        "p_calibrated",
    ]
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        compression="infer",
        usecols=columns,
        chunksize=500_000,
    ):
        keep = chunk["feature_set"].isin([BASELINE_FEATURE, FULL_FEATURE])
        if keep.any():
            selected.append(chunk.loc[keep].copy())
    if not selected:
        raise ValueError(f"required feature sets were not found in {path}")
    frame = pd.concat(selected, ignore_index=True)
    available = set(frame["feature_set"].unique())
    expected = {BASELINE_FEATURE, FULL_FEATURE}
    if available != expected:
        raise ValueError(f"feature-set mismatch in {path}: {sorted(available)}")
    keys = ["fold", "seed", "closure"]
    if any(frame[key].nunique() != 1 for key in keys):
        raise ValueError(f"prediction file mixes run identifiers: {path}")
    counts = frame.groupby("feature_set")["segid"].nunique()
    if counts.nunique() != 1:
        raise ValueError(f"feature sets do not cover the same segments in {path}")
    return frame


def safe_average_precision(y_true: np.ndarray, probability: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) != 2:
        return float("nan")
    return float(average_precision_score(y_true, probability))


def run_metrics(predictions: pd.DataFrame, labels: pd.DataFrame, source_file: Path) -> list[dict[str, object]]:
    merged = predictions.merge(
        labels[
            [
                "segid",
                "ccre_class",
                "ccre_label_index",
                "ccre_binary_label",
                "complexity_category",
                "complexity_score",
            ]
        ],
        on="segid",
        how="left",
        validate="many_to_one",
    )
    if merged["ccre_binary_label"].isna().any():
        missing = int(merged["ccre_binary_label"].isna().sum())
        raise ValueError(f"{missing} prediction rows lack cCRE labels in {source_file}")
    if not np.array_equal(
        merged["y_true"].to_numpy(np.int8),
        merged["ccre_binary_label"].to_numpy(np.int8),
    ):
        raise ValueError(f"binary labels disagree with prediction y_true in {source_file}")
    metadata = {
        "fold": str(merged["fold"].iloc[0]),
        "seed": int(merged["seed"].iloc[0]),
        "closure": str(merged["closure"].iloc[0]),
        "prediction_file": str(source_file.resolve()),
    }
    rows: list[dict[str, object]] = []
    for feature_set, feature_frame in merged.groupby("feature_set", sort=True):
        for display_name, source_label in SUBTYPE_MAP.items():
            selected = feature_frame.loc[
                feature_frame["ccre_class"].eq(source_label)
                | feature_frame["ccre_binary_label"].eq(0)
            ].copy()
            y = selected["ccre_class"].eq(source_label).to_numpy(np.int8)
            rows.append(
                {
                    **metadata,
                    "stratification": "ENCODE_cCRE_subtype_vs_background",
                    "stratum": display_name,
                    "source_label": source_label,
                    "feature_set": feature_set,
                    "n_examples": int(len(selected)),
                    "n_positive": int(y.sum()),
                    "n_negative": int(len(y) - y.sum()),
                    "positive_fraction": float(y.mean()) if len(y) else np.nan,
                    "auprc": safe_average_precision(
                        y, selected["p_calibrated"].to_numpy(float)
                    ),
                }
            )
        assigned = feature_frame.loc[feature_frame["complexity_category"].notna()]
        for category in VALID_COMPLEXITY:
            selected = assigned.loc[assigned["complexity_category"].eq(category)]
            y = selected["ccre_binary_label"].to_numpy(np.int8)
            rows.append(
                {
                    **metadata,
                    "stratification": "native_graph_complexity_tertile",
                    "stratum": category,
                    "source_label": category,
                    "feature_set": feature_set,
                    "n_examples": int(len(selected)),
                    "n_positive": int(y.sum()),
                    "n_negative": int(len(y) - y.sum()),
                    "positive_fraction": float(y.mean()) if len(y) else np.nan,
                    "auprc": safe_average_precision(
                        y, selected["p_calibrated"].to_numpy(float)
                    ),
                }
            )
    return rows


def summarize(
    metrics: pd.DataFrame, n_bootstrap: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    absolute_rows: list[dict[str, object]] = []
    group_keys = ["stratification", "stratum", "closure", "feature_set"]
    for keys, group in metrics.groupby(group_keys, sort=True):
        fold_arrays = [
            fold_group["auprc"].dropna().to_numpy(float)
            for _, fold_group in group.groupby("fold", sort=True)
        ]
        if any(len(values) == 0 for values in fold_arrays):
            continue
        draws = _hierarchical_draws(fold_arrays, n_bootstrap=n_bootstrap, rng=rng)
        values = np.concatenate(fold_arrays)
        absolute_rows.append(
            {
                **dict(zip(group_keys, keys)),
                "metric": "auprc",
                "mean": float(values.mean()),
                "std_across_runs": float(values.std(ddof=1)),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "n_runs": int(len(values)),
                "n_folds": int(group["fold"].nunique()),
                "n_seeds": int(group["seed"].nunique()),
                "mean_examples_per_run": float(group["n_examples"].mean()),
                "mean_positives_per_run": float(group["n_positive"].mean()),
                "resampling_unit": "chromosome_fold_then_seed",
            }
        )
    absolute = pd.DataFrame(absolute_rows)

    pair_keys = ["fold", "seed", "closure", "stratification", "stratum"]
    full = metrics.loc[metrics["feature_set"].eq(FULL_FEATURE), pair_keys + ["auprc", "n_examples", "n_positive"]]
    base = metrics.loc[metrics["feature_set"].eq(BASELINE_FEATURE), pair_keys + ["auprc"]]
    paired = full.merge(base, on=pair_keys, suffixes=("_full", "_baseline"), validate="one_to_one")
    paired["gain"] = paired["auprc_full"] - paired["auprc_baseline"]
    gain_rows: list[dict[str, object]] = []
    for keys, group in paired.groupby(
        ["stratification", "stratum", "closure"], sort=True
    ):
        finite = group.loc[np.isfinite(group["gain"])].copy()
        fold_arrays = [
            fold_group["gain"].to_numpy(float)
            for _, fold_group in finite.groupby("fold", sort=True)
        ]
        if not fold_arrays or any(len(values) == 0 for values in fold_arrays):
            continue
        draws = _hierarchical_draws(fold_arrays, n_bootstrap=n_bootstrap, rng=rng)
        gain_rows.append(
            {
                **dict(zip(["stratification", "stratum", "closure"], keys)),
                "contrast": "topology_given_coordinates_and_frozen_sequence_fm",
                "larger_feature_set": FULL_FEATURE,
                "smaller_feature_set": BASELINE_FEATURE,
                "metric": "auprc",
                "mean_gain": float(finite["gain"].mean()),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "n_paired_runs": int(len(finite)),
                "n_folds": int(finite["fold"].nunique()),
                "n_seeds": int(finite["seed"].nunique()),
                "mean_examples_per_run": float(finite["n_examples"].mean()),
                "mean_positives_per_run": float(finite["n_positive"].mean()),
                "gain_definition": "larger_minus_smaller",
                "resampling_unit": "chromosome_fold_then_seed",
            }
        )
    interactions: list[dict[str, object]] = []
    stratum_orders = {
        "ENCODE_cCRE_subtype_vs_background": list(SUBTYPE_MAP),
        "native_graph_complexity_tertile": list(VALID_COMPLEXITY),
    }
    for (stratification, closure), group in paired.groupby(
        ["stratification", "closure"], sort=True
    ):
        available = [
            item
            for item in stratum_orders[stratification]
            if item in set(group["stratum"])
        ]
        for first, second in itertools.combinations(available, 2):
            left = group.loc[
                group["stratum"].eq(first), ["fold", "seed", "gain"]
            ]
            right = group.loc[
                group["stratum"].eq(second), ["fold", "seed", "gain"]
            ]
            comparison = left.merge(
                right,
                on=["fold", "seed"],
                how="inner",
                suffixes=("_first", "_second"),
                validate="one_to_one",
            )
            comparison["gain_difference"] = (
                comparison["gain_first"] - comparison["gain_second"]
            )
            fold_arrays = [
                fold_group["gain_difference"].to_numpy(float)
                for _, fold_group in comparison.groupby("fold", sort=True)
            ]
            draws = _hierarchical_draws(
                fold_arrays,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            interactions.append(
                {
                    "stratification": stratification,
                    "closure": closure,
                    "first_stratum": first,
                    "second_stratum": second,
                    "interaction": "difference_in_topology_gain",
                    "mean_gain_difference": float(
                        comparison["gain_difference"].mean()
                    ),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "n_paired_runs": int(len(comparison)),
                    "n_folds": int(comparison["fold"].nunique()),
                    "n_seeds": int(comparison["seed"].nunique()),
                    "positive_means_first_stratum_has_larger_graph_gain": True,
                    "resampling_unit": "chromosome_fold_then_seed",
                }
            )
    return absolute, pd.DataFrame(gain_rows), pd.DataFrame(interactions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--node-labels", type=Path, required=True)
    parser.add_argument("--complexity-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--expected-files", type=int, default=30)
    args = parser.parse_args()
    if args.n_bootstrap < 100:
        parser.error("--n-bootstrap must be at least 100")

    paths = sorted(args.probe_root.glob("fold_*/seed_*/*/test_predictions.csv.gz"))
    if len(paths) != args.expected_files:
        raise ValueError(
            f"expected {args.expected_files} prediction files below {args.probe_root}, found {len(paths)}"
        )
    labels, input_audit = prepare_labels(args.node_labels, args.complexity_features)
    rows: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        print(f"[ccre-strata] {index}/{len(paths)} {path}", flush=True)
        rows.extend(run_metrics(load_selected_predictions(path), labels, path))
    metrics = pd.DataFrame(rows)
    expected_per_file = 2 * (len(SUBTYPE_MAP) + len(VALID_COMPLEXITY))
    if len(metrics) != len(paths) * expected_per_file:
        raise RuntimeError("unexpected stratum metric count")
    summary, gains, interactions = summarize(metrics, args.n_bootstrap, args.seed)
    if gains.empty or gains["mean_gain"].isna().any():
        raise RuntimeError("cCRE stratum gain summary is incomplete")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "ccre_stratum_run_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "ccre_stratum_absolute_summary.csv", index=False)
    gains.to_csv(args.out_dir / "ccre_stratum_topology_gains.csv", index=False)
    interactions.to_csv(
        args.out_dir / "ccre_stratum_gain_interactions.csv", index=False
    )
    audit = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "probe_root": str(args.probe_root.resolve()),
        "prediction_files": len(paths),
        "folds": sorted(metrics["fold"].unique()),
        "seeds": sorted(int(value) for value in metrics["seed"].unique()),
        "contexts": sorted(metrics["closure"].unique()),
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.seed,
        "subtype_comparison": "named ENCODE cCRE subtype versus common background; other positive cCRE classes excluded",
        "complexity_comparison": "original binary cCRE outcome within strict-context native-graph complexity tertiles fixed independently of model performance",
        "required_feature_sets": [BASELINE_FEATURE, FULL_FEATURE],
        "outputs": {
            "run_metrics": "ccre_stratum_run_metrics.csv",
            "absolute_summary": "ccre_stratum_absolute_summary.csv",
            "topology_gains": "ccre_stratum_topology_gains.csv",
            "gain_interactions": "ccre_stratum_gain_interactions.csv",
        },
        **input_audit,
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
