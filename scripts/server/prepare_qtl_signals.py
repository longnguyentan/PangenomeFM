#!/usr/bin/env python3
"""Stream HGSVC molecular-QTL tables into compact, versioned signal tables.

The source summary-statistic tables are too large to load eagerly.  This
utility retains unique genome-wide significant variant positions and records
the exact input checksum, filtering rule, row counts, and coordinate
convention.  The result is suitable for direct-overlap or matched-background
analyses; it is not a colocalization result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd


def sha256sum(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_chromosome(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    if text.endswith(".0"):
        text = text[:-2]
    return f"chr{text}"


def stream_significant_qtls(
    source: Path,
    *,
    qtl_type: str,
    release: str,
    p_threshold: float,
    empirical_threshold: float | None,
    chunksize: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return one row per unique significant variant position."""

    required = [
        "feature_id",
        "snp_id",
        "p_value",
        "empirical_feature_p_value",
        "snp_chromosome",
        "snp_position",
    ]
    rows_processed = 0
    rows_passing = 0
    retained: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        source,
        sep="\t",
        usecols=required,
        compression="infer",
        chunksize=chunksize,
        low_memory=False,
    ):
        rows_processed += len(chunk)
        chunk["p_value"] = pd.to_numeric(chunk["p_value"], errors="coerce")
        chunk["empirical_feature_p_value"] = pd.to_numeric(
            chunk["empirical_feature_p_value"], errors="coerce"
        )
        chunk["snp_position"] = pd.to_numeric(chunk["snp_position"], errors="coerce")
        keep = chunk["p_value"].le(p_threshold) & chunk["snp_position"].notna()
        if empirical_threshold is not None:
            keep &= chunk["empirical_feature_p_value"].le(empirical_threshold)
        selected = chunk.loc[keep].copy()
        rows_passing += len(selected)
        if not selected.empty:
            retained.append(selected)

    if retained:
        associations = pd.concat(retained, ignore_index=True)
        associations["chromosome"] = associations["snp_chromosome"].map(
            normalize_chromosome
        )
        associations["position"] = associations["snp_position"].astype("int64")
        associations = associations.sort_values(
            ["chromosome", "position", "p_value", "feature_id", "snp_id"]
        )
        grouped = associations.groupby(["chromosome", "position"], sort=True)
        signals = grouped.agg(
            min_p_value=("p_value", "min"),
            n_associations=("p_value", "size"),
            n_features=("feature_id", "nunique"),
            n_variant_ids=("snp_id", "nunique"),
            lead_snp_id=("snp_id", "first"),
            lead_feature_id=("feature_id", "first"),
        ).reset_index()
    else:
        signals = pd.DataFrame(
            columns=[
                "chromosome", "position", "min_p_value", "n_associations",
                "n_features", "n_variant_ids", "lead_snp_id", "lead_feature_id",
            ]
        )
    signals.insert(0, "signal_id", [f"{qtl_type}:{c}:{p}" for c, p in zip(signals["chromosome"], signals["position"])])
    signals["start"] = signals["position"].astype("int64") - 1
    signals["end"] = signals["position"].astype("int64")
    signals["signal_type"] = qtl_type
    signals["release"] = release
    audit = {
        "schema_version": 1,
        "status": "complete",
        "source": str(source.resolve()),
        "source_sha256": sha256sum(source),
        "release": release,
        "qtl_type": qtl_type,
        "p_threshold": p_threshold,
        "empirical_threshold": empirical_threshold,
        "rows_processed": rows_processed,
        "association_rows_passing": rows_passing,
        "unique_signal_positions": len(signals),
        "chromosomes": sorted(signals["chromosome"].unique()),
        "coordinate_rule": "input positions are GRCh38 one-based; output intervals are zero-based half-open [position-1, position)",
        "interpretation": "Significant-position normalization only; not enrichment, colocalization, fine-mapping, or causality.",
    }
    return signals, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--qtl-type", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--p-threshold", type=float, default=5e-8)
    parser.add_argument("--empirical-threshold", type=float)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    started = time.monotonic()
    signals, audit = stream_significant_qtls(
        args.input,
        qtl_type=args.qtl_type,
        release=args.release,
        p_threshold=args.p_threshold,
        empirical_threshold=args.empirical_threshold,
        chunksize=args.chunksize,
    )
    audit["wall_seconds"] = time.monotonic() - started
    args.out_dir.mkdir(parents=True, exist_ok=True)
    signals.to_csv(args.out_dir / "signals.csv.gz", index=False)
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
