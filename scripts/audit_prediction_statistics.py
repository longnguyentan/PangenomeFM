#!/usr/bin/env python3
"""Compute reproducible held-out metrics, slice-bootstrap CIs, and paired tests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


METRICS = ["auroc", "auprc", "accuracy", "precision", "recall", "f1", "brier", "ece"]
PAIRED_SUMMARY_COLUMNS = [
    "reference",
    "comparator",
    "closure",
    "metric",
    "n_matched_predictions",
    "n_resampling_slices",
    "reference_value",
    "comparator_value",
    "delta_reference_minus_comparator",
    "ci95_low",
    "ci95_high",
    "bootstrap_two_sided_p",
]


def _chromosome(value: str) -> str:
    matches = re.findall(r"chr(?:[0-9]+|X|Y|M|MT)", str(value), flags=re.IGNORECASE)
    return matches[-1] if matches else str(value)


def _ece(y_true: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (score >= lower) & (score < upper if index < bins - 1 else score <= upper)
        if mask.any():
            result += mask.mean() * abs(float(y_true[mask].mean()) - float(score[mask].mean()))
    return float(result) if total else np.nan


def metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    prediction = score >= 0.5
    positive = y_true == 1
    negative = ~positive
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive and n_negative:
        ranks = rankdata(score, method="average")
        auroc = float(
            (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
            / (n_positive * n_negative)
        )
        order = np.argsort(-score, kind="stable")
        ordered_positive = positive[order]
        ordered_score = score[order]
        group_ends = np.r_[np.flatnonzero(ordered_score[:-1] != ordered_score[1:]), len(score) - 1]
        true_positive_at_end = np.cumsum(ordered_positive)[group_ends]
        recall_increments = np.diff(np.r_[0, true_positive_at_end]) / n_positive
        precision_at_end = true_positive_at_end / (group_ends + 1)
        auprc = float(np.sum(recall_increments * precision_at_end))
    else:
        auroc = np.nan
        auprc = np.nan
    true_positive = int(np.sum(prediction & positive))
    false_positive = int(np.sum(prediction & negative))
    false_negative = int(np.sum(~prediction & positive))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": float(np.mean(prediction == positive)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "brier": float(np.mean((score - y_true) ** 2)),
        "ece": _ece(y_true, score),
    }


def _summarize_frame(frame: pd.DataFrame, label: str) -> dict[str, object]:
    values = metrics(frame["y_true"].to_numpy(), frame["p_edge"].to_numpy())
    return {
        "model": label,
        "closure": str(frame["closure"].iloc[0]),
        "n_predictions": len(frame),
        "n_slices": frame["slice"].nunique(),
        "n_chromosomes": frame["chromosome"].nunique(),
        "positive_fraction": float(frame["y_true"].mean()),
        **values,
    }


def _bootstrap(
    frame: pd.DataFrame,
    label: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    grouped = {name: group.index.to_numpy() for name, group in frame.groupby("slice")}
    names = np.asarray(list(grouped), dtype=object)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for replicate in range(replicates):
        sampled_names = rng.choice(names, size=len(names), replace=True)
        indices = np.concatenate([grouped[name] for name in sampled_names])
        sample = frame.loc[indices]
        rows.append(
            {
                "model": label,
                "closure": str(frame["closure"].iloc[0]),
                "replicate": replicate,
                **metrics(sample["y_true"].to_numpy(), sample["p_edge"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def _load(specification: str, split_pattern: str) -> tuple[str, pd.DataFrame, str]:
    if "=" not in specification:
        raise ValueError("Each --input must be LABEL=PATH")
    label, path_string = specification.split("=", 1)
    path = Path(path_string)
    frame = pd.read_csv(path, compression="infer")
    if "split" not in frame:
        frame["split"] = "external_all"
    required = {"slice", "target_sn", "closure", "split", "u_local", "v_local", "y_true", "p_edge"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    frame = frame[frame["split"].astype(str).str.contains(split_pattern, regex=True)].copy()
    if frame.empty:
        raise ValueError(f"{path} has no rows matching split pattern {split_pattern!r}")
    if frame["closure"].nunique() != 1:
        raise ValueError(f"{path} mixes closures: {sorted(frame['closure'].unique())}")
    frame["chromosome"] = frame["target_sn"].map(_chromosome)
    return label, frame.reset_index(drop=True), str(path)


def _paired_bootstrap(
    reference: pd.DataFrame,
    comparator: pd.DataFrame,
    reference_label: str,
    comparator_label: str,
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    key = ["slice", "target_sn", "closure", "u_local", "v_local", "y_true"]
    left = reference[key + ["p_edge"]].rename(columns={"p_edge": "p_reference"})
    right = comparator[key + ["p_edge"]].rename(columns={"p_edge": "p_comparator"})
    left["occurrence"] = left.groupby(key).cumcount()
    right["occurrence"] = right.groupby(key).cumcount()
    merged = left.merge(right, on=key + ["occurrence"], validate="one_to_one")
    if len(merged) != len(reference) or len(merged) != len(comparator):
        raise ValueError(
            f"Prediction universes differ for {reference_label} and {comparator_label}: "
            f"{len(reference)}, {len(comparator)}, intersection {len(merged)}"
        )
    grouped = {name: group.index.to_numpy() for name, group in merged.groupby("slice")}
    names = np.asarray(list(grouped), dtype=object)
    rng = np.random.default_rng(seed)
    replicate_rows: list[dict[str, object]] = []
    for replicate in range(replicates):
        sampled_names = rng.choice(names, size=len(names), replace=True)
        indices = np.concatenate([grouped[name] for name in sampled_names])
        sample = merged.loc[indices]
        reference_metrics = metrics(sample["y_true"].to_numpy(), sample["p_reference"].to_numpy())
        comparator_metrics = metrics(sample["y_true"].to_numpy(), sample["p_comparator"].to_numpy())
        replicate_rows.extend(
            {
                "reference": reference_label,
                "comparator": comparator_label,
                "closure": str(reference["closure"].iloc[0]),
                "replicate": replicate,
                "metric": metric,
                "delta_reference_minus_comparator": reference_metrics[metric] - comparator_metrics[metric],
            }
            for metric in METRICS
        )
    replicate_frame = pd.DataFrame(replicate_rows)
    point_reference = metrics(merged["y_true"].to_numpy(), merged["p_reference"].to_numpy())
    point_comparator = metrics(merged["y_true"].to_numpy(), merged["p_comparator"].to_numpy())
    summaries: list[dict[str, object]] = []
    for metric, values in replicate_frame.groupby("metric"):
        deltas = values["delta_reference_minus_comparator"].to_numpy()
        summaries.append(
            {
                "reference": reference_label,
                "comparator": comparator_label,
                "closure": str(reference["closure"].iloc[0]),
                "metric": metric,
                "n_matched_predictions": len(merged),
                "n_resampling_slices": len(names),
                "reference_value": point_reference[metric],
                "comparator_value": point_comparator[metric],
                "delta_reference_minus_comparator": point_reference[metric] - point_comparator[metric],
                "ci95_low": float(np.quantile(deltas, 0.025)),
                "ci95_high": float(np.quantile(deltas, 0.975)),
                "bootstrap_two_sided_p": float(
                    min(
                        1.0,
                        2
                        * min(
                            (np.sum(deltas <= 0) + 1) / (len(deltas) + 1),
                            (np.sum(deltas >= 0) + 1) / (len(deltas) + 1),
                        ),
                    )
                ),
            }
        )
    return summaries, replicate_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="LABEL=prediction.csv.gz")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-prefix", default="full_")
    parser.add_argument(
        "--split-pattern",
        default="^heldout_chr_test$",
        help="Regular expression selecting the locked evaluation subset exactly.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    loaded = [_load(specification, args.split_pattern) for specification in args.input]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prediction_manifest = []
    overall_rows = []
    subgroup_rows = []
    slice_rows = []
    bootstrap_frames = []
    frames: dict[str, pd.DataFrame] = {}
    for index, (label, frame, path) in enumerate(loaded):
        frames[label] = frame
        prediction_manifest.append(
            {
                "model": label,
                "path": path,
                "closure": frame["closure"].iloc[0],
                "selected_split_pattern": args.split_pattern,
                "selected_predictions": len(frame),
            }
        )
        overall_rows.append(_summarize_frame(frame, label))
        for chromosome, group in frame.groupby("chromosome"):
            subgroup_rows.append({"chromosome": chromosome, **_summarize_frame(group, label)})
        for slice_id, group in frame.groupby("slice"):
            slice_rows.append({"slice": slice_id, **_summarize_frame(group, label)})
        bootstrap_frames.append(
            _bootstrap(frame, label, args.bootstrap_replicates, args.seed + index * 1009)
        )

    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    overall = pd.DataFrame(overall_rows)
    for metric in METRICS:
        quantiles = bootstrap.groupby("model")[metric].quantile([0.025, 0.975]).unstack()
        lookup = quantiles.to_dict("index")
        overall[f"{metric}_ci95_low"] = overall["model"].map(lambda name: lookup[name][0.025])
        overall[f"{metric}_ci95_high"] = overall["model"].map(lambda name: lookup[name][0.975])

    paired_summaries: list[dict[str, object]] = []
    paired_replicates: list[pd.DataFrame] = []
    for closure, closure_labels in overall.groupby("closure")["model"]:
        candidates = [label for label in closure_labels if label.startswith(args.reference_prefix)]
        if len(candidates) != 1:
            continue
        reference_label = candidates[0]
        for comparator_label in closure_labels:
            if comparator_label == reference_label:
                continue
            summaries, replicates = _paired_bootstrap(
                frames[reference_label],
                frames[comparator_label],
                reference_label,
                comparator_label,
                args.bootstrap_replicates,
                args.seed + len(paired_summaries) * 101,
            )
            paired_summaries.extend(summaries)
            paired_replicates.append(replicates)

    overall.to_csv(args.out_dir / "overall_metrics.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(args.out_dir / "per_chromosome_metrics.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(args.out_dir / "per_slice_metrics.csv", index=False)
    pd.DataFrame(prediction_manifest).to_csv(args.out_dir / "prediction_manifest.csv", index=False)
    bootstrap.to_csv(args.out_dir / "bootstrap_metrics.csv.gz", index=False, compression="gzip")
    pd.DataFrame(paired_summaries, columns=PAIRED_SUMMARY_COLUMNS).to_csv(
        args.out_dir / "paired_comparisons.csv", index=False
    )
    if paired_replicates:
        pd.concat(paired_replicates, ignore_index=True).to_csv(
            args.out_dir / "paired_bootstrap_replicates.csv.gz", index=False, compression="gzip"
        )
    summary = {
        "inputs": len(loaded),
        "bootstrap_replicates": args.bootstrap_replicates,
        "resampling_unit": "benchmark slice",
        "split_pattern": args.split_pattern,
        "seed": args.seed,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
