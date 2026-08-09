#!/usr/bin/env python3
"""Test signal enrichment in prioritized loci using matched genomic controls.

Cases must be prespecified with a boolean priority column.  Each case is
matched to non-prioritized loci on chromosome and user-declared covariates.
All match assignments and exclusions are retained.  Inference resamples or
permutes complete matched sets, avoiding the false precision of treating
overlapping signal records as independent biological replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


REQUIRED_REGION_COLUMNS = {"region_id", "chromosome", "start", "end"}
REQUIRED_SIGNAL_COLUMNS = {"signal_id", "chromosome", "start", "end"}


def sha256sum(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_chromosome(value: object) -> str:
    text = str(value).strip()
    return text if text.lower().startswith("chr") else f"chr{text}"


def validate_intervals(frame: pd.DataFrame, required: set[str], label: str) -> pd.DataFrame:
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{label} table misses columns: {sorted(missing)}")
    result = frame.copy()
    result["chromosome"] = result["chromosome"].map(normalize_chromosome)
    result["start"] = pd.to_numeric(result["start"], errors="raise").astype("int64")
    result["end"] = pd.to_numeric(result["end"], errors="raise").astype("int64")
    if (result["start"] < 0).any() or (result["end"] <= result["start"]).any():
        raise ValueError(f"{label} contains invalid zero-based half-open intervals")
    identifier = "region_id" if label == "region" else "signal_id"
    if result[identifier].astype(str).duplicated().any():
        raise ValueError(f"{label} contains duplicate {identifier} values")
    return result


def annotate_overlaps(regions: pd.DataFrame, signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate region hit/count and return a direct-overlap join table."""

    rows: list[dict[str, object]] = []
    for chrom, region_group in regions.groupby("chromosome", sort=False):
        signal_group = signals.loc[signals["chromosome"].eq(chrom)].sort_values(
            ["start", "end", "signal_id"]
        )
        if signal_group.empty:
            continue
        signal_start = signal_group["start"].to_numpy("int64")
        signal_end = signal_group["end"].to_numpy("int64")
        maximum_signal_span = int((signal_end - signal_start).max())
        for region in region_group.itertuples(index=False):
            # Any interval starting at or before start-max_span cannot overlap;
            # any interval starting at/after end cannot overlap. This bounded
            # search avoids an O(regions × all signals) genome-wide scan while
            # remaining exact for variable-length signal intervals.
            left = int(
                np.searchsorted(
                    signal_start,
                    int(region.start) - maximum_signal_span,
                    side="right",
                )
            )
            right = int(np.searchsorted(signal_start, int(region.end), side="left"))
            candidates = signal_group.iloc[left:right]
            hit = candidates["end"].to_numpy("int64") > int(region.start)
            for signal in candidates.loc[hit].itertuples(index=False):
                rows.append(
                    {
                        "region_id": str(region.region_id),
                        "signal_id": str(signal.signal_id),
                        "chromosome": chrom,
                        "region_start": int(region.start),
                        "region_end": int(region.end),
                        "signal_start": int(signal.start),
                        "signal_end": int(signal.end),
                    }
                )
    overlaps = pd.DataFrame(
        rows,
        columns=[
            "region_id", "signal_id", "chromosome", "region_start", "region_end",
            "signal_start", "signal_end",
        ],
    )
    counts = overlaps.groupby("region_id").size() if not overlaps.empty else pd.Series(dtype=int)
    annotated = regions.copy()
    annotated["signal_count"] = annotated["region_id"].map(counts).fillna(0).astype(int)
    annotated["signal_hit"] = annotated["signal_count"].gt(0).astype(int)
    return annotated, overlaps


def coerce_priority(values: pd.Series, column: str) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    normalized = values.astype(str).str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError(f"{column} must contain boolean values")
    return normalized.isin({"true", "1"})


def build_matched_sets(
    regions: pd.DataFrame,
    *,
    priority_column: str,
    match_columns: list[str],
    controls_per_case: int,
    relative_caliper: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nearest-neighbor matching with exact chromosome and robust scaling."""

    if priority_column not in regions:
        raise ValueError(f"Missing priority column: {priority_column}")
    missing = [column for column in match_columns if column not in regions]
    if missing:
        raise ValueError(
            "Required matching covariates are absent; refusing an unmatched enrichment: "
            + ", ".join(missing)
        )
    work = regions.copy()
    work[priority_column] = coerce_priority(work[priority_column], priority_column)
    for column in match_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if work[match_columns].isna().any().any():
        bad = work.loc[work[match_columns].isna().any(axis=1), "region_id"].astype(str).tolist()
        raise ValueError(f"Missing/non-numeric matching covariates for regions: {bad[:10]}")
    cases = work.loc[work[priority_column]].copy()
    controls = work.loc[~work[priority_column]].copy()
    if cases.empty or controls.empty:
        raise ValueError("Both prioritized cases and non-prioritized controls are required")

    assignments: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for chrom, chrom_cases in cases.groupby("chromosome", sort=True):
        chrom_controls = controls.loc[controls["chromosome"].eq(chrom)].copy()
        if chrom_controls.empty:
            exclusions.extend(
                {"case_region_id": str(case.region_id), "reason": "no_same_chromosome_controls"}
                for case in chrom_cases.itertuples(index=False)
            )
            continue
        combined = pd.concat([chrom_cases, chrom_controls], ignore_index=True)
        center = combined[match_columns].median()
        scale = (combined[match_columns] - center).abs().median().replace(0, 1.0)
        z_controls = (chrom_controls[match_columns] - center) / scale
        for case in chrom_cases.itertuples(index=False):
            case_values = pd.Series(
                {column: getattr(case, column) for column in match_columns}
            )
            distances = np.sqrt(((z_controls - (case_values - center) / scale) ** 2).sum(axis=1))
            relative_differences = []
            for column in match_columns:
                denominator = max(abs(float(case_values[column])), float(scale[column]), 1e-12)
                relative_differences.append(
                    (chrom_controls[column] - float(case_values[column])).abs() / denominator
                )
            maximum_relative = pd.concat(relative_differences, axis=1).max(axis=1)
            eligible = maximum_relative.le(relative_caliper)
            ranked = distances.loc[eligible].sort_values(kind="mergesort").head(controls_per_case)
            if len(ranked) < controls_per_case:
                exclusions.append(
                    {
                        "case_region_id": str(case.region_id),
                        "reason": "insufficient_controls_within_caliper",
                        "eligible_controls": int(eligible.sum()),
                    }
                )
                continue
            case_row = chrom_cases.loc[chrom_cases["region_id"].eq(case.region_id)].iloc[0]
            for rank, (control_index, distance) in enumerate(ranked.items(), start=1):
                control_row = chrom_controls.loc[control_index]
                row = {
                    "matched_set_id": f"set:{case.region_id}",
                    "case_region_id": str(case.region_id),
                    "control_region_id": str(control_row["region_id"]),
                    "chromosome": chrom,
                    "control_rank": rank,
                    "standardized_distance": float(distance),
                    "case_signal_hit": int(case_row["signal_hit"]),
                    "control_signal_hit": int(control_row["signal_hit"]),
                    "case_signal_count": int(case_row["signal_count"]),
                    "control_signal_count": int(control_row["signal_count"]),
                }
                for column in match_columns:
                    row[f"case_{column}"] = float(case_row[column])
                    row[f"control_{column}"] = float(control_row[column])
                assignments.append(row)
    return pd.DataFrame(assignments), pd.DataFrame(exclusions)


def matched_inference(
    assignments: pd.DataFrame,
    *,
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if assignments.empty:
        raise ValueError("No complete matched sets; enrichment cannot be estimated")
    sets = []
    for set_id, group in assignments.groupby("matched_set_id", sort=True):
        case_hit = int(group["case_signal_hit"].iloc[0])
        control_hits = group["control_signal_hit"].to_numpy(int)
        sets.append((set_id, case_hit, control_hits))
    differences = np.asarray([case - controls.mean() for _, case, controls in sets])
    case_hits = np.asarray([case for _, case, _ in sets], dtype=int)
    control_hits = np.concatenate([controls for _, _, controls in sets])
    table = np.array(
        [
            [case_hits.sum(), len(case_hits) - case_hits.sum()],
            [control_hits.sum(), len(control_hits) - control_hits.sum()],
        ]
    )
    odds_ratio, fisher_p = fisher_exact(table, alternative="greater")
    rng = np.random.default_rng(seed)
    permutation_draws = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        permuted_differences = []
        for _, case, controls in sets:
            values = np.concatenate([[case], controls])
            selected = int(rng.integers(0, len(values)))
            pseudo_case = values[selected]
            pseudo_controls = np.delete(values, selected)
            permuted_differences.append(pseudo_case - pseudo_controls.mean())
        permutation_draws[index] = np.mean(permuted_differences)
    observed = float(differences.mean())
    permutation_p_greater = float(
        (1 + np.sum(permutation_draws >= observed - 1e-12)) / (n_permutations + 1)
    )
    bootstrap_draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled = rng.integers(0, len(differences), len(differences))
        bootstrap_draws[index] = differences[sampled].mean()
    summary = pd.DataFrame(
        [
            {
                "matched_sets": len(sets),
                "controls_per_case": int(len(assignments) / len(sets)),
                "case_hit_fraction": float(case_hits.mean()),
                "control_hit_fraction": float(control_hits.mean()),
                "matched_risk_difference": observed,
                "risk_difference_ci95_low": float(np.quantile(bootstrap_draws, 0.025)),
                "risk_difference_ci95_high": float(np.quantile(bootstrap_draws, 0.975)),
                "permutation_p_greater": permutation_p_greater,
                "pooled_odds_ratio": float(odds_ratio),
                "pooled_fisher_p_greater": float(fisher_p),
                "primary_inference": "matched_set_permutation",
            }
        ]
    )
    draws = pd.DataFrame(
        {
            "draw": np.arange(n_permutations),
            "permuted_matched_risk_difference": permutation_draws,
        }
    )
    return summary, draws


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--priority-column", default="is_prioritized")
    parser.add_argument(
        "--match-columns",
        nargs="+",
        default=[
            "region_length", "gc_content", "mappability", "graph_complexity",
            "variant_density", "distance_to_gene",
        ],
    )
    parser.add_argument("--controls-per-case", type=int, default=5)
    parser.add_argument("--relative-caliper", type=float, default=0.25)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    regions = validate_intervals(pd.read_csv(args.regions), REQUIRED_REGION_COLUMNS, "region")
    signals = validate_intervals(pd.read_csv(args.signals), REQUIRED_SIGNAL_COLUMNS, "signal")
    annotated, overlaps = annotate_overlaps(regions, signals)
    if args.priority_column not in annotated:
        raise ValueError(f"Missing priority column: {args.priority_column}")
    annotated[args.priority_column] = coerce_priority(
        annotated[args.priority_column], args.priority_column
    )
    assignments, exclusions = build_matched_sets(
        annotated,
        priority_column=args.priority_column,
        match_columns=args.match_columns,
        controls_per_case=args.controls_per_case,
        relative_caliper=args.relative_caliper,
    )
    summary, permutation_draws = matched_inference(
        assignments,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(args.out_dir / "region_signal_summary.csv", index=False)
    overlaps.to_csv(args.out_dir / "direct_overlaps.csv.gz", index=False)
    assignments.to_csv(args.out_dir / "matched_assignments.csv", index=False)
    exclusions.to_csv(args.out_dir / "match_exclusions.csv", index=False)
    summary.to_csv(args.out_dir / "enrichment_summary.csv", index=False)
    permutation_draws.to_csv(args.out_dir / "permutation_draws.csv.gz", index=False)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "region_source": str(args.regions.resolve()),
        "region_source_sha256": sha256sum(args.regions),
        "signal_source": str(args.signals.resolve()),
        "signal_source_sha256": sha256sum(args.signals),
        "regions": len(regions),
        "signals": len(signals),
        "direct_overlaps": len(overlaps),
        "prioritized_regions": int(annotated[args.priority_column].astype(bool).sum()),
        "complete_matched_sets": int(assignments["matched_set_id"].nunique()),
        "excluded_cases": len(exclusions),
        "priority_column": args.priority_column,
        "match_columns": args.match_columns,
        "controls_per_case": args.controls_per_case,
        "relative_caliper": args.relative_caliper,
        "n_permutations": args.n_permutations,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "coordinate_rule": "zero-based half-open interval overlap",
        "interpretation": "Association-signal enrichment only; not causality or colocalization.",
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
