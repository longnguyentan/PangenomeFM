#!/usr/bin/env python3
"""Test whether canonical edge-identity affected loci drive Figure 4 results.

The exact analysis already canonicalizes reverse-equivalent candidates and
excludes contradictory canonical labels.  This conservative sensitivity goes
one step further: it removes every genomic locus whose strict or 1-hop slice
contained either condition, then recomputes chromosome-block summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("auprc_advantage", "auroc_advantage", "model_score")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def affected_loci(complexity: pd.DataFrame) -> pd.DataFrame:
    required = {
        "slice_id",
        "chromosome",
        "start",
        "end",
        "context",
        "reverse_equivalent_candidate_duplicates",
        "orientation_equivalent_candidate_label_conflicts",
    }
    missing = required - set(complexity)
    if missing:
        raise ValueError(f"Complexity table misses columns: {sorted(missing)}")
    reverse = pd.to_numeric(
        complexity["reverse_equivalent_candidate_duplicates"], errors="raise"
    )
    conflicts = pd.to_numeric(
        complexity["orientation_equivalent_candidate_label_conflicts"], errors="raise"
    )
    selected = complexity.loc[(reverse > 0) | (conflicts > 0)].copy()
    selected["reverse_equivalent_candidate_duplicates"] = reverse.loc[selected.index]
    selected["orientation_equivalent_candidate_label_conflicts"] = conflicts.loc[
        selected.index
    ]
    selected["locus_id"] = (
        selected["chromosome"].astype(str)
        + ":"
        + selected["start"].astype(int).astype(str)
        + "-"
        + selected["end"].astype(int).astype(str)
    )
    columns = [
        "locus_id",
        "slice_id",
        "chromosome",
        "start",
        "end",
        "context",
        "reverse_equivalent_candidate_duplicates",
        "orientation_equivalent_candidate_label_conflicts",
    ]
    return selected[columns].sort_values(
        ["chromosome", "start", "context"], kind="stable"
    )


def chromosome_means(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        frame.groupby(
            ["baseline", "context", "locus_complexity_category", "chromosome"],
            dropna=False,
            sort=True,
        )[metric]
        .mean()
        .rename("value")
        .reset_index()
    )


def compare_policies(
    all_regions: pd.DataFrame,
    retained_regions: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grouping = ["baseline", "context", "locus_complexity_category"]
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        all_chrom = chromosome_means(all_regions, metric).rename(
            columns={"value": "all_value"}
        )
        drop_chrom = chromosome_means(retained_regions, metric).rename(
            columns={"value": "drop_value"}
        )
        paired = all_chrom.merge(
            drop_chrom,
            on=grouping + ["chromosome"],
            how="inner",
            validate="one_to_one",
        )
        for keys, group in paired.groupby(grouping, dropna=False, sort=True):
            difference = group["drop_value"].to_numpy(float) - group[
                "all_value"
            ].to_numpy(float)
            difference = difference[np.isfinite(difference)]
            if not len(difference):
                continue
            draws = np.empty(n_bootstrap, dtype=float)
            for index in range(n_bootstrap):
                draws[index] = rng.choice(
                    difference, size=len(difference), replace=True
                ).mean()
            baseline, context, category = keys
            all_mean = float(group["all_value"].mean())
            drop_mean = float(group["drop_value"].mean())
            rows.append(
                {
                    "baseline": baseline,
                    "context": context,
                    "locus_complexity_category": category,
                    "metric": metric,
                    "audited_all_loci_mean": all_mean,
                    "drop_affected_loci_mean": drop_mean,
                    "policy_shift_drop_minus_all": drop_mean - all_mean,
                    "policy_shift_ci95_low": float(np.quantile(draws, 0.025)),
                    "policy_shift_ci95_high": float(np.quantile(draws, 0.975)),
                    "n_paired_chromosomes": int(len(difference)),
                    "audited_all_region_rows": int(
                        len(
                            all_regions.loc[
                                all_regions["baseline"].eq(baseline)
                                & all_regions["context"].eq(context)
                                & all_regions["locus_complexity_category"].eq(category)
                            ]
                        )
                    ),
                    "drop_affected_region_rows": int(
                        len(
                            retained_regions.loc[
                                retained_regions["baseline"].eq(baseline)
                                & retained_regions["context"].eq(context)
                                & retained_regions["locus_complexity_category"].eq(category)
                            ]
                        )
                    ),
                    "resampling_unit": "paired_chromosome_block",
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-level", type=Path, required=True)
    parser.add_argument("--complexity-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--primary-baseline", default="sequence_composition_sgd")
    parser.add_argument("--max-auprc-shift", type=float, default=0.01)
    parser.add_argument("--max-auroc-shift", type=float, default=0.01)
    parser.add_argument("--max-model-score-shift", type=float, default=0.01)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap must be positive")

    started = time.monotonic()
    regions = pd.read_csv(args.region_level, compression="infer")
    complexity = pd.read_csv(args.complexity_features, sep="\t")
    required_regions = {
        "region_id",
        "chromosome",
        "start",
        "end",
        "baseline",
        "context",
        "locus_complexity_category",
        *METRICS,
    }
    missing = required_regions - set(regions)
    if missing:
        raise ValueError(f"Region-level table misses columns: {sorted(missing)}")
    affected = affected_loci(complexity)
    if affected.empty:
        raise ValueError("No canonical-identity affected slices were found")
    affected_keys = set(
        affected[["chromosome", "start", "end"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    region_keys = list(
        regions[["chromosome", "start", "end"]].itertuples(index=False, name=None)
    )
    regions = regions.copy()
    regions["canonical_identity_affected_locus"] = [
        key in affected_keys for key in region_keys
    ]
    retained = regions.loc[~regions["canonical_identity_affected_locus"]].copy()
    summary = compare_policies(
        regions,
        retained,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    tolerances = {
        "auprc_advantage": args.max_auprc_shift,
        "auroc_advantage": args.max_auroc_shift,
        "model_score": args.max_model_score_shift,
    }
    summary["absolute_shift_tolerance"] = summary["metric"].map(tolerances)
    summary["within_shift_tolerance"] = summary[
        "policy_shift_drop_minus_all"
    ].abs().le(summary["absolute_shift_tolerance"])
    same_sign = np.sign(summary["audited_all_loci_mean"]) == np.sign(
        summary["drop_affected_loci_mean"]
    )
    substantively_zero = (
        summary[["audited_all_loci_mean", "drop_affected_loci_mean"]]
        .abs()
        .max(axis=1)
        .le(summary["absolute_shift_tolerance"])
    )
    summary["direction_stable"] = same_sign | substantively_zero
    summary["primary_gate_row"] = summary["baseline"].eq(args.primary_baseline)
    summary["gate_pass"] = (
        summary["within_shift_tolerance"] & summary["direction_stable"]
    )
    primary = summary.loc[summary["primary_gate_row"]]
    if primary.empty:
        raise ValueError(
            f"Primary baseline {args.primary_baseline!r} is absent from region-level results"
        )
    gate_passed = bool(primary["gate_pass"].all())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    affected.to_csv(args.out_dir / "canonical_identity_affected_slices.csv", index=False)
    regions.loc[regions["canonical_identity_affected_locus"]].to_csv(
        args.out_dir / "affected_region_rows.csv.gz", index=False, compression="gzip"
    )
    summary.to_csv(args.out_dir / "policy_sensitivity_summary.csv", index=False)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "sensitivity_gate": "pass" if gate_passed else "fail",
        "region_level": str(args.region_level.resolve()),
        "region_level_sha256": sha256_file(args.region_level),
        "complexity_features": str(args.complexity_features.resolve()),
        "complexity_features_sha256": sha256_file(args.complexity_features),
        "canonical_identity_affected_context_slices": int(len(affected)),
        "canonical_identity_affected_genomic_loci": int(len(affected_keys)),
        "reverse_equivalent_candidate_duplicates": int(
            affected["reverse_equivalent_candidate_duplicates"].sum()
        ),
        "orientation_equivalent_candidate_label_conflicts": int(
            affected["orientation_equivalent_candidate_label_conflicts"].sum()
        ),
        "region_rows_all": int(len(regions)),
        "region_rows_removed": int(regions["canonical_identity_affected_locus"].sum()),
        "primary_baseline": args.primary_baseline,
        "primary_gate_rows": int(len(primary)),
        "failed_primary_gate_rows": int((~primary["gate_pass"]).sum()),
        "shift_tolerances": tolerances,
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.seed,
        "policy": "remove both strict and 1-hop rows at every genomic locus with a reverse-equivalent duplicate or canonical label conflict in either context",
        "interpretation": "pass supports robustness of the audited exact analysis; fail requires a corrected end-to-end canonical rerun before manuscript claims",
        "wall_seconds": time.monotonic() - started,
    }
    (args.out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    if args.fail_on_gate and not gate_passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
