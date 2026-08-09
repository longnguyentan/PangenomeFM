#!/usr/bin/env python3
"""Normalize a pinned GWAS Catalog association snapshot to GRCh38 loci.

The caller must supply the snapshot date and genome build explicitly.  Rows
with ambiguous chromosome/position cardinality are excluded and reported,
rather than guessed.  Optional trait filtering supports a prespecified disease
analysis (for example Parkinson disease), but the output remains an
association-locus table and does not imply causality or colocalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def sha256sum(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_field(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in re.split(r"[;,]", str(value)) if item.strip()]


def normalize_chromosome(value: str) -> str | None:
    text = value.removeprefix("chr").upper()
    if text in {str(index) for index in range(1, 23)} | {"X", "Y"}:
        return f"chr{text}"
    return None


def normalize_catalog(
    source: Path,
    *,
    snapshot_date: str,
    genome_build: str,
    p_threshold: float,
    trait_regex: str | None,
    chunksize: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required = [
        "CHR_ID", "CHR_POS", "P-VALUE", "MAPPED_TRAIT", "STUDY ACCESSION", "SNPS"
    ]
    rows_processed = 0
    reasons: Counter[str] = Counter()
    signal_rows: list[dict[str, object]] = []
    pattern = re.compile(trait_regex, flags=re.IGNORECASE) if trait_regex else None
    for chunk in pd.read_csv(
        source,
        sep="\t",
        usecols=required,
        compression="infer",
        chunksize=chunksize,
        dtype=str,
        low_memory=False,
    ):
        rows_processed += len(chunk)
        # pandas preserves source-file column order for ``usecols`` rather than
        # the requested list order. Reorder explicitly before positional tuple
        # unpacking so Catalog schema changes cannot silently swap fields.
        chunk = chunk[required]
        chunk["P-VALUE"] = pd.to_numeric(chunk["P-VALUE"], errors="coerce")
        chunk = chunk.loc[chunk["P-VALUE"].le(p_threshold)]
        if pattern is not None:
            chunk = chunk.loc[
                chunk["MAPPED_TRAIT"].fillna("").str.contains(pattern, na=False)
            ]
        for row in chunk.itertuples(index=False, name=None):
            chrom_text, position_text, p_value, trait, accession, snps = row
            chromosomes = split_field(chrom_text)
            positions = split_field(position_text)
            if not chromosomes or not positions:
                reasons["missing_coordinate"] += 1
                continue
            if len(chromosomes) != len(positions):
                reasons["ambiguous_coordinate_cardinality"] += 1
                continue
            emitted = 0
            for chrom_item, position_item in zip(chromosomes, positions):
                chromosome = normalize_chromosome(chrom_item)
                try:
                    position = int(float(position_item))
                except (TypeError, ValueError):
                    reasons["invalid_position"] += 1
                    continue
                if chromosome is None or position < 1:
                    reasons["noncanonical_coordinate"] += 1
                    continue
                emitted += 1
                signal_rows.append(
                    {
                        "chromosome": chromosome,
                        "position": position,
                        "start": position - 1,
                        "end": position,
                        "p_value": float(p_value),
                        "mapped_trait": str(trait),
                        "study_accession": str(accession),
                        "reported_snps": str(snps),
                    }
                )
            if emitted == 0:
                reasons["no_valid_coordinate"] += 1
    raw = pd.DataFrame(signal_rows)
    if raw.empty:
        signals = pd.DataFrame(
            columns=[
                "signal_id", "chromosome", "position", "start", "end", "min_p_value",
                "n_associations", "n_studies", "mapped_traits", "study_accessions",
                "reported_snps", "signal_type", "snapshot_date", "genome_build",
            ]
        )
    else:
        raw = raw.sort_values(
            ["chromosome", "position", "p_value", "study_accession"]
        )
        signals = (
            raw.groupby(["chromosome", "position", "start", "end"], sort=True)
            .agg(
                min_p_value=("p_value", "min"),
                n_associations=("p_value", "size"),
                n_studies=("study_accession", "nunique"),
                mapped_traits=("mapped_trait", lambda x: "|".join(sorted(set(x)))),
                study_accessions=("study_accession", lambda x: "|".join(sorted(set(x)))),
                reported_snps=("reported_snps", lambda x: "|".join(sorted(set(x)))),
            )
            .reset_index()
        )
        signals.insert(
            0,
            "signal_id",
            [f"GWAS:{chrom}:{position}" for chrom, position in zip(signals["chromosome"], signals["position"])],
        )
        signals["signal_type"] = "GWAS_Catalog_association"
        signals["snapshot_date"] = snapshot_date
        signals["genome_build"] = genome_build
    exclusions = pd.DataFrame(
        [{"exclusion_reason": reason, "rows": count} for reason, count in sorted(reasons.items())]
    )
    audit = {
        "schema_version": 1,
        "status": "complete",
        "source": str(source.resolve()),
        "source_sha256": sha256sum(source),
        "snapshot_date": snapshot_date,
        "genome_build_asserted_by_caller": genome_build,
        "p_threshold": p_threshold,
        "trait_regex": trait_regex,
        "rows_processed": rows_processed,
        "unique_signal_positions": len(signals),
        "exclusion_reason_counts": dict(reasons),
        "coordinate_rule": "catalog one-based positions converted to zero-based half-open [position-1, position)",
        "interpretation": "Catalog association loci only; not credible sets, fine-mapping, colocalization, or causality.",
    }
    return signals, exclusions, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--snapshot-date", required=True, help="YYYY-MM-DD release/snapshot date")
    parser.add_argument("--genome-build", choices=["GRCh38"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--p-threshold", type=float, default=5e-8)
    parser.add_argument("--trait-regex")
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.snapshot_date):
        raise ValueError("--snapshot-date must use YYYY-MM-DD")
    signals, exclusions, audit = normalize_catalog(
        args.input,
        snapshot_date=args.snapshot_date,
        genome_build=args.genome_build,
        p_threshold=args.p_threshold,
        trait_regex=args.trait_regex,
        chunksize=args.chunksize,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    signals.to_csv(args.out_dir / "signals.csv.gz", index=False)
    exclusions.to_csv(args.out_dir / "exclusions.csv", index=False)
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
