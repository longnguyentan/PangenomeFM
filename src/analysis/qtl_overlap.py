"""Streaming overlap analysis for graph-model windows and molecular-QTL tables.

The HGSVC2 summary-statistic files are large even for one chromosome.  This
module intentionally reads them in chunks and retains only counters and a
small set of top associations for the supplied, non-overlapping graph windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


_INTERVAL_RE = re.compile(r"_(chr(?:[0-9]+|X|Y))_(\d+)_(\d+)$")


@dataclass
class RegionAccumulator:
    """Memory-bounded summary of QTL associations intersecting one window."""

    association_rows: int = 0
    significant_rows: int = 0
    positions: set[int] = field(default_factory=set)
    significant_positions: set[int] = field(default_factory=set)
    features: set[str] = field(default_factory=set)
    empirical_significant_features: set[str] = field(default_factory=set)
    min_p_value: float = np.inf
    top_rows: list[dict[str, object]] = field(default_factory=list)

    def update(
        self,
        rows: pd.DataFrame,
        *,
        p_threshold: float,
        empirical_threshold: float,
        top_n: int,
    ) -> None:
        if rows.empty:
            return
        self.association_rows += len(rows)
        positions = rows["snp_position"].dropna().astype(int)
        self.positions.update(positions.tolist())
        self.features.update(rows["feature_id"].dropna().astype(str).tolist())

        finite_p = rows.loc[np.isfinite(rows["p_value"]), "p_value"]
        if not finite_p.empty:
            self.min_p_value = min(self.min_p_value, float(finite_p.min()))

        significant = rows[rows["p_value"] <= p_threshold]
        self.significant_rows += len(significant)
        self.significant_positions.update(
            significant["snp_position"].dropna().astype(int).tolist()
        )

        empirical = rows[rows["empirical_feature_p_value"] <= empirical_threshold]
        self.empirical_significant_features.update(
            empirical["feature_id"].dropna().astype(str).tolist()
        )

        candidate_rows = rows.nsmallest(top_n, "p_value")
        self.top_rows.extend(candidate_rows.to_dict(orient="records"))
        self.top_rows = sorted(
            self.top_rows,
            key=lambda row: float(row.get("p_value", np.inf)),
        )[:top_n]


def load_context_windows(
    metric_paths: Iterable[Path], chromosome: str
) -> pd.DataFrame:
    """Average per-window context metrics across seeds and parse coordinates."""

    frames: list[pd.DataFrame] = []
    for metric_path in metric_paths:
        frame = pd.read_csv(metric_path)
        required = {
            "interval_id",
            "target_sn",
            "core_candidates",
            "expanded_candidates",
            "core_auroc",
            "expanded_auroc",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{metric_path} is missing columns: {sorted(missing)}")
        frame = frame[frame["target_sn"].astype(str).str.endswith(f"#{chromosome}")]
        if not frame.empty:
            frame = frame.copy()
            frame["metric_source"] = str(metric_path)
            frames.append(frame)
    if not frames:
        raise ValueError(f"No context windows found for {chromosome}")

    combined = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        column
        for column in combined.columns
        if column not in {"interval_id", "target_sn", "metric_source"}
        and pd.api.types.is_numeric_dtype(combined[column])
    ]
    averaged = (
        combined.groupby(["interval_id", "target_sn"], as_index=False)[numeric_columns]
        .mean()
        .sort_values("interval_id")
    )
    averaged["n_seeds"] = combined.groupby("interval_id").size().reindex(
        averaged["interval_id"]
    ).to_numpy()

    parsed = averaged["interval_id"].str.extract(_INTERVAL_RE)
    if parsed.isna().any().any():
        bad = averaged.loc[parsed.isna().any(axis=1), "interval_id"].tolist()
        raise ValueError(f"Could not parse coordinates from interval IDs: {bad}")
    averaged["chromosome"] = parsed[0]
    averaged["start"] = parsed[1].astype(int)
    averaged["end"] = parsed[2].astype(int)
    averaged["window_bp"] = averaged["end"] - averaged["start"]
    averaged["context_auroc_gain"] = (
        averaged["expanded_auroc"] - averaged["core_auroc"]
    )
    gain_rank = averaged["context_auroc_gain"].rank(method="first", pct=True)
    averaged["context_recovery_group"] = np.where(
        gain_rank > 0.75, "top_quartile", "other"
    )
    return averaged.reset_index(drop=True)


def stream_qtl_overlaps(
    qtl_path: Path,
    windows: pd.DataFrame,
    *,
    qtl_type: str,
    chunksize: int = 500_000,
    p_threshold: float = 5e-8,
    empirical_threshold: float = 0.05,
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Stream a gzipped tabular QTL file and summarize overlaps per window.

    Graph windows are treated as zero-based half-open intervals and the QTL
    position as GRCh38 one-based, so a hit satisfies ``start < pos <= end``.
    """

    usecols = [
        "feature_id",
        "snp_id",
        "p_value",
        "beta",
        "empirical_feature_p_value",
        "snp_chromosome",
        "snp_position",
    ]
    accumulators = {
        row.interval_id: RegionAccumulator() for row in windows.itertuples(index=False)
    }
    rows_processed = 0
    reader = pd.read_csv(
        qtl_path,
        sep="\t",
        usecols=usecols,
        chunksize=chunksize,
        compression="infer",
        low_memory=False,
    )
    for chunk in reader:
        rows_processed += len(chunk)
        for column in ("p_value", "beta", "empirical_feature_p_value", "snp_position"):
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        for window in windows.itertuples(index=False):
            hit = chunk[
                (chunk["snp_position"] > window.start)
                & (chunk["snp_position"] <= window.end)
            ]
            accumulators[window.interval_id].update(
                hit,
                p_threshold=p_threshold,
                empirical_threshold=empirical_threshold,
                top_n=top_n,
            )

    summaries: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for window in windows.itertuples(index=False):
        accumulator = accumulators[window.interval_id]
        min_p = accumulator.min_p_value
        summaries.append(
            {
                "interval_id": window.interval_id,
                "qtl_type": qtl_type,
                "association_rows": accumulator.association_rows,
                "unique_variant_positions": len(accumulator.positions),
                "unique_features": len(accumulator.features),
                "genomewide_significant_rows": accumulator.significant_rows,
                "genomewide_significant_variant_positions": len(
                    accumulator.significant_positions
                ),
                "empirical_significant_features": len(
                    accumulator.empirical_significant_features
                ),
                "min_p_value": min_p if np.isfinite(min_p) else np.nan,
                "neg_log10_min_p": (
                    -np.log10(max(min_p, np.finfo(float).tiny))
                    if np.isfinite(min_p)
                    else np.nan
                ),
                "significant_variant_positions_per_mb": (
                    len(accumulator.significant_positions) / window.window_bp * 1e6
                ),
            }
        )
        for rank, row in enumerate(accumulator.top_rows, start=1):
            top_rows.append(
                {
                    "interval_id": window.interval_id,
                    "qtl_type": qtl_type,
                    "rank_within_window": rank,
                    **row,
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(top_rows), rows_processed


def exact_spearman_permutation(
    x: pd.Series, y: pd.Series, *, max_exact_n: int = 9
) -> tuple[float, float, int]:
    """Return Spearman rho and an exact two-sided permutation p-value."""

    valid = x.notna() & y.notna()
    x_valid = x[valid].astype(float).to_numpy()
    y_valid = y[valid].astype(float).to_numpy()
    n = len(x_valid)
    if n < 3:
        return np.nan, np.nan, n
    x_rank = pd.Series(x_valid).rank(method="average").to_numpy()
    y_rank = pd.Series(y_valid).rank(method="average").to_numpy()
    observed = float(np.corrcoef(x_rank, y_rank)[0, 1])
    if n > max_exact_n:
        return observed, np.nan, n
    permuted = np.asarray(
        [np.corrcoef(x_rank, permuted_y)[0, 1] for permuted_y in permutations(y_rank)]
    )
    p_value = float((np.abs(permuted) >= abs(observed) - 1e-12).mean())
    return observed, p_value, n

