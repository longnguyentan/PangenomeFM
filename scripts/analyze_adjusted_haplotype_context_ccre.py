#!/usr/bin/env python3
"""Confounder-adjusted cCRE association with alternate-haplotype context.

The Cochran--Mantel--Haenszel analysis stratifies by chromosome-local
coordinate, node length, total degree, and local sequence composition. It is a
conservative association audit, not a causal analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable


csv.field_size_limit(sys.maxsize)


def _sequence_features(sequence: str) -> tuple[float, float, float]:
    sequence = sequence.upper()
    counts = np.asarray(
        [sequence.count(base) for base in "ACGT"], dtype=float
    )
    valid = counts.sum()
    if valid == 0:
        return 0.0, 0.0, 0.0
    frequencies = counts / valid
    nonzero = frequencies[frequencies > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    gc = float((counts[1] + counts[2]) / valid)
    cpg = float(sequence.count("CG") / max(len(sequence) - 1, 1))
    return gc, cpg, entropy


def _load_sequence_features(
    segments_path: str | Path, segids: set[int]
) -> pd.DataFrame:
    rows = []
    with Path(segments_path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if row_index not in segids:
                continue
            gc, cpg, entropy = _sequence_features(row["seq"])
            rows.append(
                {
                    "segid": row_index,
                    "gc_fraction": gc,
                    "cpg_density": cpg,
                    "base_entropy": entropy,
                }
            )
    if len(rows) != len(segids):
        raise ValueError(
            f"Resolved {len(rows):,} of {len(segids):,} labeled segment IDs."
        )
    return pd.DataFrame(rows)


def _quantile_bin(values: pd.Series, bins: int) -> pd.Series:
    ranks = values.rank(method="average", pct=True)
    return np.minimum((ranks * bins).astype(int), bins - 1)


def _raw_association(frame: pd.DataFrame, target: str) -> dict[str, float | int]:
    table = pd.crosstab(frame["has_alt_neighbor"], frame[target])
    table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    odds_ratio, pvalue = fisher_exact(
        [
            [int(table.loc[1, 1]), int(table.loc[1, 0])],
            [int(table.loc[0, 1]), int(table.loc[0, 0])],
        ]
    )
    return {
        "raw_odds_ratio": float(odds_ratio),
        "raw_fisher_pvalue": float(pvalue),
        "positive_rate_alt_context": float(
            table.loc[1, 1] / max(table.loc[1].sum(), 1)
        ),
        "positive_rate_no_alt_context": float(
            table.loc[0, 1] / max(table.loc[0].sum(), 1)
        ),
    }


def _stratified_association(
    frame: pd.DataFrame, target: str
) -> dict[str, float | int]:
    tables = []
    included_nodes = 0
    for _, group in frame.groupby("adjustment_stratum", sort=False):
        table = pd.crosstab(group["has_alt_neighbor"], group[target])
        table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        if (
            table.loc[0].sum() == 0
            or table.loc[1].sum() == 0
            or table[0].sum() == 0
            or table[1].sum() == 0
        ):
            continue
        # Rows are alt-context yes/no and columns are target yes/no.
        tables.append(
            np.asarray(
                [
                    [table.loc[1, 1], table.loc[1, 0]],
                    [table.loc[0, 1], table.loc[0, 0]],
                ],
                dtype=float,
            )
        )
        included_nodes += int(len(group))
    if not tables:
        raise ValueError(f"No informative adjustment strata for {target}.")
    stratified = StratifiedTable(
        np.stack(tables, axis=2), shift_zeros=True
    )
    lower, upper = stratified.oddsratio_pooled_confint()
    test = stratified.test_null_odds()
    return {
        "adjusted_common_odds_ratio": float(stratified.oddsratio_pooled),
        "adjusted_ci95_lower": float(lower),
        "adjusted_ci95_upper": float(upper),
        "adjusted_pvalue": float(test.pvalue),
        "n_informative_strata": int(len(tables)),
        "n_nodes_in_informative_strata": included_nodes,
    }


def analyze(
    *,
    context_path: str | Path,
    segments_path: str | Path,
    out_dir: str | Path,
    coordinate_bin_bp: int = 10_000_000,
) -> dict[str, object]:
    frame = pd.read_csv(context_path, compression="infer", low_memory=False)
    sequence = _load_sequence_features(
        segments_path, set(frame["segid"].astype(int))
    )
    frame = frame.merge(sequence, on="segid", validate="one_to_one")
    frame["coordinate_bin"] = (
        frame["SO"].clip(lower=0).astype(np.int64) // coordinate_bin_bp
    )
    frame["length_bin"] = frame.groupby("chrom")["LN"].transform(
        lambda values: _quantile_bin(values, 3)
    )
    frame["gc_bin"] = frame.groupby("chrom")["gc_fraction"].transform(
        lambda values: _quantile_bin(values, 3)
    )
    frame["cpg_bin"] = frame.groupby("chrom")["cpg_density"].transform(
        lambda values: _quantile_bin(values, 3)
    )
    frame["degree_bin"] = np.minimum(frame["degree_total"].astype(int), 5)
    frame["adjustment_stratum"] = (
        frame["chrom"].astype(str)
        + ":"
        + frame["coordinate_bin"].astype(str)
        + ":"
        + frame["length_bin"].astype(str)
        + ":"
        + frame["degree_bin"].astype(str)
        + ":"
        + frame["gc_bin"].astype(str)
        + ":"
        + frame["cpg_bin"].astype(str)
    )
    targets = ["is_ccre", "is_enhancer_like", "is_dels"]
    rows = []
    for target in targets:
        rows.append(
            {
                "target": target,
                "n": int(len(frame)),
                **_raw_association(frame, target),
                **_stratified_association(frame, target),
            }
        )
    results = pd.DataFrame(rows)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "adjusted_associations.csv", index=False)
    sequence.to_csv(
        output / "labeled_node_sequence_covariates.csv.gz",
        index=False,
        compression="gzip",
    )
    summary = {
        "task": "confounder_adjusted_haplotype_context_ccre_association",
        "scope": (
            "reference-projected cCRE labels near alternate-haplotype graph "
            "context; not haplotype-specific functional labels"
        ),
        "causal_interpretation": False,
        "n_nodes": int(len(frame)),
        "coordinate_bin_bp": coordinate_bin_bp,
        "adjustment": [
            "chromosome",
            "coordinate bin",
            "node-length chromosome tertile",
            "total degree clipped at 5",
            "GC-fraction chromosome tertile",
            "CpG-density chromosome tertile",
        ],
        "results": results.to_dict("records"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--coordinate-bin-bp", type=int, default=10_000_000)
    args = parser.parse_args()
    analyze(
        context_path=args.context,
        segments_path=args.segments,
        out_dir=args.out_dir,
        coordinate_bin_bp=args.coordinate_bin_bp,
    )


if __name__ == "__main__":
    main()
