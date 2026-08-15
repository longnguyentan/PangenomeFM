#!/usr/bin/env python3
"""Consolidate verified rotating and directional transfer results."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def context_name(value: str) -> str:
    return {"strict": "core", "1hop": "expanded"}.get(str(value), str(value))


def split_transfer_pair(pair: str) -> tuple[str, str]:
    if "_to_" not in pair:
        raise ValueError(f"cannot parse directional transfer pair: {pair}")
    return tuple(pair.rsplit("_to_", 1))  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary",
        type=Path,
        default=ROOT
        / "analysis/full_multicohort_20260809/primary_cross_chromosome_per_seed_chromosome.csv",
    )
    parser.add_argument(
        "--transfer",
        type=Path,
        default=ROOT / "analysis/full_multicohort_20260809/transfer_per_seed.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/server_full_multicohort_20260806.json",
    )
    parser.add_argument(
        "--out-tsv", type=Path, default=ROOT / "results/canonical_primary_results.tsv"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_parquet = args.out_tsv.with_suffix(".parquet")
    audit_path = args.out_tsv.with_name("canonical_primary_results_audit.json")
    if args.out_tsv.exists() and not args.overwrite:
        raise FileExistsError(
            f"canonical table already exists at {args.out_tsv}; use --overwrite explicitly"
        )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_seeds = {int(seed) for seed in config["training"]["seeds"]}
    fold_by_test_chromosome: dict[str, str] = {}
    for fold in config["rotating_chromosome_folds"]:
        for chromosome in fold["test"]:
            if chromosome in fold_by_test_chromosome:
                raise ValueError(f"test chromosome occurs in multiple folds: {chromosome}")
            fold_by_test_chromosome[chromosome] = fold["name"]

    primary = pd.read_csv(args.primary)
    primary_rows: list[dict[str, object]] = []
    for row in primary.itertuples(index=False):
        chromosome = str(row.chromosome)
        if chromosome not in fold_by_test_chromosome:
            raise ValueError(f"no held-out fold for chromosome {chromosome}")
        primary_rows.append(
            {
                "experiment_type": "rotating_chromosome_generalization",
                "regime": str(row.regime),
                "dataset": str(row.dataset),
                "source_graph": str(row.regime),
                "target_graph": str(row.dataset),
                "direction": "within_or_shared_training_to_dataset",
                "context": context_name(row.closure),
                "fold": fold_by_test_chromosome[chromosome],
                "seed": int(row.seed),
                "chromosome": chromosome,
                "n_slices": int(row.n_slices),
                "n_examples": int(row.n_targets),
                "n_positive": pd.NA,
                "n_negative": pd.NA,
                "AUROC": float(row.mean_auroc),
                "AUPRC": float(row.mean_auprc),
                "ECE": float(row.mean_ece_10bin),
                "threshold": pd.NA,
                "F1": float(row.mean_f1),
                "precision": float(row.mean_precision),
                "recall": float(row.mean_recall),
                "checkpoint": pd.NA,
                "result_path": str(args.primary),
                "provenance_note": "Per-recorded-seed/per-chromosome aggregate; historical trainer did not seed PyTorch initialization/DropEdge, so the numeric seed is not an exactly reproducible model-initialization seed. Class counts, threshold and checkpoint are not present in this source table.",
            }
        )

    transfer = pd.read_csv(args.transfer)
    transfer_rows: list[dict[str, object]] = []
    for row in transfer.itertuples(index=False):
        source, target = split_transfer_pair(str(row.transfer_pair))
        transfer_rows.append(
            {
                "experiment_type": str(row.experiment_type),
                "regime": str(row.regime),
                "dataset": target,
                "source_graph": source,
                "target_graph": target,
                "direction": f"{source} -> {target}",
                "context": context_name(row.closure),
                "fold": pd.NA,
                "seed": int(row.seed),
                "chromosome": "all",
                "n_slices": pd.NA,
                "n_examples": int(row.n_targets),
                "n_positive": pd.NA,
                "n_negative": pd.NA,
                "AUROC": float(row.auroc),
                "AUPRC": float(row.auprc),
                "ECE": float(row.ece_10bin),
                "threshold": pd.NA,
                "F1": float(row.f1),
                "precision": float(row.precision),
                "recall": float(row.recall),
                "checkpoint": pd.NA,
                "result_path": str(args.transfer),
                "provenance_note": "Directional all-chromosome transfer aggregate; do not average opposite directions. Historical numeric seed labels did not seed PyTorch initialization/DropEdge.",
            }
        )

    canonical = pd.DataFrame(primary_rows + transfer_rows)
    errors: list[str] = []
    primary_key = [
        "regime",
        "dataset",
        "context",
        "fold",
        "seed",
        "chromosome",
    ]
    primary_part = canonical.loc[
        canonical["experiment_type"].eq("rotating_chromosome_generalization")
    ]
    if primary_part.duplicated(primary_key).any():
        errors.append("duplicate rotating result keys")
    for key, group in primary.groupby(["regime", "dataset", "closure", "chromosome"]):
        seeds = set(map(int, group["seed"]))
        if seeds != expected_seeds:
            errors.append(f"missing/unexpected rotating seeds for {key}: {sorted(seeds)}")
    transfer_key = ["experiment_type", "direction", "context", "seed"]
    transfer_part = canonical.loc[
        ~canonical["experiment_type"].eq("rotating_chromosome_generalization")
    ]
    if transfer_part.duplicated(transfer_key).any():
        errors.append("duplicate transfer result keys")
    for key, group in transfer.groupby(["experiment_type", "transfer_pair", "closure"]):
        seeds = set(map(int, group["seed"]))
        if seeds != expected_seeds:
            errors.append(f"missing/unexpected transfer seeds for {key}: {sorted(seeds)}")
    if canonical[["AUROC", "AUPRC"]].isna().any().any():
        errors.append("missing primary AUROC/AUPRC")

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(args.out_tsv, sep="\t", index=False)
    canonical.to_parquet(out_parquet, index=False)
    audit = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "rows": int(len(canonical)),
        "rotating_rows": int(len(primary_rows)),
        "transfer_rows": int(len(transfer_rows)),
        "seeds": sorted(expected_seeds),
        "folds": sorted(set(fold_by_test_chromosome.values())),
        "inputs": {
            str(args.primary): sha256(args.primary),
            str(args.transfer): sha256(args.transfer),
            str(args.config): sha256(args.config),
        },
        "unknown_by_design": [
            "n_positive",
            "n_negative",
            "threshold",
            "checkpoint",
        ],
        "seed_interpretation": "Historical rows are completed repeated executions grouped by recorded seed labels, but the trainer did not seed PyTorch initialization or DropEdge. Rerun for claims about exact named initialization seeds.",
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
