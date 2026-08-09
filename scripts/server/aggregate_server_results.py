#!/usr/bin/env python3
"""Aggregate full-run predictions into paper-source tables and diagnostic figures."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from evaluation.splits import normalize_chrom


def metric_row(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame["y_true"].to_numpy(dtype=int)
    probability = frame["p_edge"].to_numpy(dtype=float)
    predicted = probability >= 0.5
    bins = np.linspace(0, 1, 11)
    bin_index = np.clip(np.digitize(probability, bins) - 1, 0, 9)
    ece = 0.0
    for index in range(10):
        selected = bin_index == index
        if selected.any():
            ece += selected.mean() * abs(probability[selected].mean() - y[selected].mean())
    two_classes = len(np.unique(y)) == 2
    return {
        "n_targets": int(len(frame)),
        "positive_fraction": float(y.mean()) if len(y) else np.nan,
        "auroc": float(roc_auc_score(y, probability)) if two_classes else np.nan,
        "auprc": float(average_precision_score(y, probability)) if two_classes else np.nan,
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "ece_10bin": float(ece),
    }


def parse_run_metadata(path: Path, root: Path) -> dict[str, object]:
    parts = path.relative_to(root).parts
    text = "/".join(parts)
    seed_match = re.search(r"seed_(\d+)", text)
    metadata: dict[str, object] = {
        "prediction_file": str(path),
        "seed": int(seed_match.group(1)) if seed_match else None,
        "closure": "1hop" if "1hop" in path.name or "/1hop/" in f"/{text}/" else "strict",
        "experiment_type": parts[0] if parts else "unknown",
        "regime": None,
        "fold": None,
        "transfer_pair": None,
    }
    if len(parts) > 1 and parts[0] in {"final_models", "rotating_folds", "release_models"}:
        metadata["regime"] = parts[1]
    if parts and parts[0] == "rotating_folds" and len(parts) > 2:
        metadata["fold"] = parts[2]
    if parts and parts[0] in {"cohort_transfer", "release_transfer"} and len(parts) > 1:
        metadata["transfer_pair"] = parts[1]
        metadata["regime"] = parts[1]
    return metadata


def bootstrap_slices(per_slice: pd.DataFrame, n_boot: int, seed: int) -> list[dict[str, object]]:
    if per_slice.empty:
        return []
    rng = np.random.default_rng(seed)
    metrics = [column for column in ["auroc", "auprc", "accuracy", "precision", "recall", "f1", "brier", "ece_10bin"] if column in per_slice]
    rows = []
    for metric in metrics:
        values = per_slice[metric].dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        draws = np.empty(n_boot)
        for index in range(n_boot):
            draws[index] = rng.choice(values, size=len(values), replace=True).mean()
        rows.append(
            {
                "metric": metric,
                "estimate_mean_across_slices": float(values.mean()),
                "ci_lower_95": float(np.quantile(draws, 0.025)),
                "ci_upper_95": float(np.quantile(draws, 0.975)),
                "resampling_unit": "graph_slice",
                "n_slices": int(len(values)),
                "bootstrap_replicates": n_boot,
                "bootstrap_seed": seed,
            }
        )
    return rows


def collect_execution_steps(root: Path) -> tuple[list[dict], list[dict]]:
    """Collect runner steps from JSON payloads containing a ``steps`` map.

    Server state basenames are experiment names (for example,
    ``folds_hprc_r2_fold_a_seed42_strict.json``), so a ``*state*.json``
    basename glob misses them even though they live in ``*_states``
    directories.
    """

    rows: list[dict] = []
    failures: list[dict] = []
    for state_path in root.glob("**/*.json"):
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            steps = payload.get("steps")
            if not isinstance(steps, dict):
                continue
            for name, step in steps.items():
                if isinstance(step, dict):
                    rows.append(
                        {"state_file": str(state_path), "step": name, **step}
                    )
        except Exception as error:
            failures.append(
                {"prediction_file": str(state_path), "error": str(error)}
            )
    return rows, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    root = args.results_root.resolve()
    out_dir = (args.out_dir or root / "paper_source_data").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_files = sorted(root.glob("**/*pooled_predictions*.csv.gz"))
    run_rows = []
    chromosome_rows = []
    slice_rows = []
    ci_rows = []
    failures = []
    for file_index, prediction_file in enumerate(prediction_files):
        try:
            frame = pd.read_csv(
                prediction_file,
                usecols=lambda column: column in {"slice", "target_sn", "closure", "split", "dataset", "y_true", "p_edge"},
                compression="infer",
            )
            if not {"y_true", "p_edge"} <= set(frame):
                raise ValueError("prediction file lacks y_true/p_edge")
            metadata = parse_run_metadata(prediction_file, root)
            run_id = f"run_{file_index:04d}"
            run_rows.append({"run_id": run_id, "metric_scope": "all_saved_splits", "split": "all_saved", **metadata, **metric_row(frame)})
            if "split" in frame:
                for split_name, split_group in frame.groupby("split", sort=True):
                    run_rows.append(
                        {
                            "run_id": run_id,
                            "metric_scope": "split",
                            "split": split_name,
                            **metadata,
                            **metric_row(split_group),
                        }
                    )
            if "target_sn" in frame:
                frame["chromosome"] = frame["target_sn"].astype(str).map(normalize_chrom)
            else:
                frame["chromosome"] = "unknown"
            slice_key = "slice" if "slice" in frame else "target_sn"
            slice_group_keys = (
                (["dataset"] if "dataset" in frame else [])
                + (["split"] if "split" in frame else [])
                + [slice_key]
            )
            per_slice = []
            slice_grouper = slice_group_keys[0] if len(slice_group_keys) == 1 else slice_group_keys
            for key, group in frame.groupby(slice_grouper, sort=False):
                key_values = key if isinstance(key, tuple) else (key,)
                cursor = 0
                dataset_value = key_values[cursor] if "dataset" in frame else None
                cursor += int("dataset" in frame)
                split_value = key_values[cursor] if "split" in frame else "all"
                cursor += int("split" in frame)
                slice_value = key_values[cursor]
                row = {"run_id": run_id, "dataset": dataset_value, "split": split_value, "slice": str(slice_value), **metadata, **metric_row(group)}
                if "target_sn" in group:
                    row["target_sn"] = str(group["target_sn"].iloc[0])
                    row["chromosome"] = normalize_chrom(row["target_sn"])
                per_slice.append(row)
            per_slice_frame = pd.DataFrame(per_slice)
            slice_rows.extend(per_slice)
            chromosome_group_keys = (
                (["dataset"] if "dataset" in frame else [])
                + (["split"] if "split" in frame else [])
                + ["chromosome"]
            )
            chromosome_grouper = chromosome_group_keys[0] if len(chromosome_group_keys) == 1 else chromosome_group_keys
            for key, group in frame.groupby(chromosome_grouper, sort=True):
                key_values = key if isinstance(key, tuple) else (key,)
                cursor = 0
                dataset_value = key_values[cursor] if "dataset" in frame else None
                cursor += int("dataset" in frame)
                split_value = key_values[cursor] if "split" in frame else "all"
                cursor += int("split" in frame)
                chrom = key_values[cursor]
                chromosome_rows.append({"run_id": run_id, "dataset": dataset_value, "split": split_value, "chromosome": chrom, **metadata, **metric_row(group)})
            for split_value, split_frame in per_slice_frame.groupby("split", dropna=False):
                for row in bootstrap_slices(split_frame, args.n_boot, args.seed + file_index):
                    ci_rows.append({"run_id": run_id, "split": split_value, **metadata, **row})
        except Exception as error:
            failures.append({"prediction_file": str(prediction_file), "error": str(error)})

    runs = pd.DataFrame(run_rows)
    chromosomes = pd.DataFrame(chromosome_rows)
    slices = pd.DataFrame(slice_rows)
    intervals = pd.DataFrame(ci_rows)
    runs.to_csv(out_dir / "experiment_seed_metrics.csv", index=False)
    chromosomes.to_csv(out_dir / "per_chromosome_metrics.csv", index=False)
    slices.to_csv(out_dir / "per_slice_metrics.csv.gz", index=False, compression="gzip")
    intervals.to_csv(out_dir / "bootstrap_confidence_intervals.csv", index=False)

    if not chromosomes.empty:
        summary = (
            chromosomes.groupby(["experiment_type", "regime", "dataset", "split", "closure", "chromosome"], dropna=False)
            .agg(
                seeds=("seed", "nunique"),
                runs=("run_id", "nunique"),
                n_targets=("n_targets", "sum"),
                mean_auroc=("auroc", "mean"),
                sd_auroc=("auroc", "std"),
                mean_auprc=("auprc", "mean"),
                sd_auprc=("auprc", "std"),
                mean_brier=("brier", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(out_dir / "chromosome_summary_across_seeds.csv", index=False)
        failure_ranking = summary.sort_values(["mean_auprc", "mean_auroc"], ascending=True)
        failure_ranking.to_csv(out_dir / "chromosome_failure_ranking.csv", index=False)

        plot = summary[summary["experiment_type"] == "final_models"].copy()
        if not plot.empty:
            plot = plot[plot["split"].astype(str).str.contains("test|all", regex=True)]
            labels = plot["regime"].fillna("unknown") + ":" + plot["dataset"].fillna("unknown") + ":" + plot["split"] + ":" + plot["closure"]
            plot["series"] = labels
            pivot = plot.pivot_table(index="series", columns="chromosome", values="mean_auprc")
            ordered = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
            pivot = pivot.reindex(columns=[column for column in ordered if column in pivot])
            figure, axis = plt.subplots(figsize=(14, max(3, 0.5 * len(pivot))))
            image = axis.imshow(pivot.to_numpy(), aspect="auto", vmin=0.5, vmax=1.0, cmap="viridis")
            axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
            axis.set_yticks(range(len(pivot.index)), pivot.index)
            axis.set_title("All-chromosome AUPRC by training regime")
            figure.colorbar(image, ax=axis, label="Mean AUPRC")
            figure.tight_layout()
            figure.savefig(out_dir / "chromosome_auprc_heatmap.png", dpi=220)
            figure.savefig(out_dir / "chromosome_auprc_heatmap.pdf")
            plt.close(figure)

    state_rows, state_failures = collect_execution_steps(root)
    failures.extend(state_failures)
    state_frame = pd.DataFrame(state_rows)
    state_frame.to_csv(out_dir / "execution_steps.csv", index=False)
    if not state_frame.empty and "wall_seconds" in state_frame:
        state_frame["phase"] = state_frame["step"].astype(str).str.extract(
            r"^(parse|index_paths|validate|benchmark_full_coverage|benchmark|train_final|train_release|train|transfer|release_transfer)"
        )[0].fillna("other")
        state_frame.groupby(["phase", "status"], dropna=False).agg(
            steps=("step", "size"), wall_seconds=("wall_seconds", "sum")
        ).reset_index().to_csv(out_dir / "execution_runtime_summary.csv", index=False)
    failed_steps = [row for row in state_rows if row.get("status") == "failed"]
    pd.DataFrame(
        failed_steps,
        columns=(
            list(state_frame.columns)
            if not state_frame.empty
            else ["state_file", "step", "status"]
        ),
    ).to_csv(out_dir / "failed_execution_steps.csv", index=False)
    pd.DataFrame(
        failures,
        columns=["prediction_file", "error"],
    ).to_csv(out_dir / "aggregation_failures.csv", index=False)

    storage_rows = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        top_level = relative.parts[0] if relative.parts else "."
        if path.suffix == ".pt":
            category = "checkpoint_or_recovery"
        elif "predictions" in path.name:
            category = "predictions"
        elif path.suffix in {".csv", ".gz", ".parquet"}:
            category = "tables"
        elif path.suffix in {".png", ".pdf", ".svg"}:
            category = "figures"
        elif path.suffix == ".log":
            category = "logs"
        else:
            category = "other"
        storage_rows.append(
            {
                "top_level": top_level,
                "category": category,
                "relative_path": str(relative),
                "bytes": path.stat().st_size,
            }
        )
    storage = pd.DataFrame(storage_rows)
    storage.to_csv(out_dir / "result_storage_inventory.csv.gz", index=False, compression="gzip")
    if not storage.empty:
        storage.groupby(["top_level", "category"], dropna=False).agg(
            files=("relative_path", "size"), bytes=("bytes", "sum")
        ).reset_index().to_csv(out_dir / "result_storage_summary.csv", index=False)
        checkpoints = storage[storage["category"] == "checkpoint_or_recovery"]
        checkpoint_summary = {
            "files": int(len(checkpoints)),
            "total_bytes": int(checkpoints["bytes"].sum()),
            "median_bytes": float(checkpoints["bytes"].median()) if len(checkpoints) else None,
            "maximum_bytes": int(checkpoints["bytes"].max()) if len(checkpoints) else None,
        }
    else:
        checkpoint_summary = {"files": 0, "total_bytes": 0, "median_bytes": None, "maximum_bytes": None}

    if not slices.empty and {"slice", "closure"} <= set(slices):
        slices["slice_pair_key"] = slices["slice"].str.replace(
            r"_(?:strict|1hop)$", "", regex=True
        )
        index_columns = [
            "experiment_type", "regime", "dataset", "split", "fold", "transfer_pair", "seed",
            "slice_pair_key", "chromosome",
        ]
        for column in index_columns:
            slices[column] = slices[column].fillna("NA")
        matched = slices.pivot_table(
            index=index_columns,
            columns="closure",
            values=["auroc", "auprc", "brier", "ece_10bin", "n_targets"],
            aggfunc="first",
            dropna=True,
        )
        matched.columns = [f"{metric}_{closure}" for metric, closure in matched.columns]
        matched = matched.reset_index()
        required = {"auprc_strict", "auprc_1hop"}
        if required <= set(matched):
            matched = matched.dropna(subset=list(required), how="any")
            for metric in ["auroc", "auprc", "brier", "ece_10bin"]:
                strict, expanded = f"{metric}_strict", f"{metric}_1hop"
                if strict in matched and expanded in matched:
                    matched[f"delta_{metric}_1hop_minus_strict"] = matched[expanded] - matched[strict]
            matched.to_csv(out_dir / "matched_context_slice_deltas.csv", index=False)
            delta_columns = [column for column in matched if column.startswith("delta_")]
            context_summary = (
                matched.groupby(["experiment_type", "regime", "dataset", "split", "fold", "transfer_pair"], dropna=False)[delta_columns]
                .agg(["count", "mean", "std", "median"])
            )
            context_summary.columns = ["_".join(column) for column in context_summary.columns]
            context_summary.reset_index().to_csv(
                out_dir / "matched_context_summary.csv", index=False
            )
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results_root": str(root),
        "prediction_files": len(prediction_files),
        "prediction_files_aggregated": int(runs["run_id"].nunique()) if not runs.empty else 0,
        "aggregation_failures": len(failures),
        "execution_steps": len(state_rows),
        "failed_execution_steps": len(failed_steps),
        "checkpoint_storage": checkpoint_summary,
        "bootstrap_unit": "graph slice",
        "bootstrap_replicates": args.n_boot,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
