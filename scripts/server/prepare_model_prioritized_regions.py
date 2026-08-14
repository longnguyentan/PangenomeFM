#!/usr/bin/env python3
"""Build a prespecified model-prioritized region universe with covariates.

The input scores must have been created without downstream biological signals.
This program adds independently sourced genomic matching covariates and marks
the top score fraction within each chromosome.  It never reads QTL, GWAS, or
other enrichment signals, so the case/control universe is fixed before those
signals are evaluated.

Coordinates are zero-based and half-open.  GC excludes ambiguous reference
bases from its denominator.  Umap/Bismap omits zero-valued intervals, so bases
absent from the bedGraph contribute zero to the full region-length denominator.
Variant density counts normalized HPRC records by their one-based POS anchor.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd

from evaluation.splits import normalize_chrom


CANONICAL_CHROMOSOMES = tuple(
    [f"chr{index}" for index in range(1, 23)] + ["chrX", "chrY"]
)
REQUIRED_SCORE_COLUMNS = {
    "region_id", "chromosome", "start", "end", "region_length", "fold",
    "graph_complexity", "model_score", "is_eligible", "eligibility_reason",
}
FINAL_REQUIRED_COLUMNS = [
    "region_id", "chromosome", "start", "end", "is_prioritized",
    "region_length", "gc_content", "mappability", "graph_complexity",
    "variant_density", "distance_to_gene",
]


def progress(message: str) -> None:
    print(f"[region-covariates {time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}", flush=True)


def open_text(path: Path) -> TextIO:
    return (
        gzip.open(path, "rt", encoding="utf-8", errors="replace")
        if path.suffix in {".gz", ".bgz"}
        else path.open("r", encoding="utf-8", errors="replace")
    )


def sha256sum(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coerce_boolean(values: pd.Series, label: str) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError(f"{label} must contain boolean values")
    return normalized.isin({"true", "1"})


def validate_regions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_SCORE_COLUMNS - set(frame)
    if missing:
        raise ValueError(f"Dense score table misses columns: {sorted(missing)}")
    result = frame.copy()
    result["region_id"] = result["region_id"].astype(str)
    if result["region_id"].duplicated().any():
        raise ValueError("Dense score table contains duplicate region_id values")
    result["chromosome"] = result["chromosome"].map(normalize_chrom)
    result = result.loc[result["chromosome"].isin(CANONICAL_CHROMOSOMES)].copy()
    for column in ("start", "end", "region_length"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    for column in ("graph_complexity", "model_score"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["is_eligible"] = coerce_boolean(result["is_eligible"], "is_eligible")
    if (result["start"] < 0).any() or (result["end"] <= result["start"]).any():
        raise ValueError("Dense score table contains invalid intervals")
    if not np.array_equal(
        result["region_length"].to_numpy(np.int64),
        (result["end"] - result["start"]).to_numpy(np.int64),
    ):
        raise ValueError("region_length does not equal end-start")
    for chrom, group in result.groupby("chromosome", sort=False):
        ordered = group.sort_values(["start", "end", "region_id"])
        starts = ordered["start"].to_numpy(np.int64)
        ends = ordered["end"].to_numpy(np.int64)
        if len(group) > 1 and np.any(starts[1:] < ends[:-1]):
            raise ValueError(f"Region intervals overlap on {chrom}")
    return result.reset_index(drop=True)


def interval_index(regions: pd.DataFrame) -> dict[str, dict[str, object]]:
    index: dict[str, dict[str, object]] = {}
    for chrom, group in regions.groupby("chromosome", sort=False):
        ordered = group.sort_values(["start", "end", "region_id"])
        index[str(chrom)] = {
            "row_indices": ordered.index.to_numpy(np.int64),
            "starts": ordered["start"].to_numpy(np.int64),
            "ends": ordered["end"].to_numpy(np.int64),
        }
    return index


def overlapping_interval_positions(
    starts: np.ndarray,
    ends: np.ndarray,
    start: int,
    end: int,
) -> range:
    """Return candidate interval positions for a non-overlapping index."""

    if end <= start or len(starts) == 0:
        return range(0)
    first = max(0, int(np.searchsorted(ends, start, side="right")))
    last = int(np.searchsorted(starts, end, side="left"))
    return range(first, last)


def fasta_covariates(
    regions: pd.DataFrame,
    fasta: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read compressed FASTA once and accumulate bases per interval.

    UCSC FASTA uses short wrapped lines.  Iterating over roughly sixty million
    individual sequence lines is unnecessarily slow, while the production
    server has ample RAM.  One decompressed byte buffer keeps this stage to a
    small number of vector-sized operations (about 3.2 GB for GRCh38).
    """

    index = interval_index(regions)
    gc = np.zeros(len(regions), dtype=np.int64)
    acgt = np.zeros(len(regions), dtype=np.int64)
    observed = np.zeros(len(regions), dtype=np.int64)
    if fasta.suffix in {".gz", ".bgz"}:
        with gzip.open(fasta, "rb") as handle:
            payload = handle.read()
    else:
        payload = fasta.read_bytes()
    cursor = 0
    while True:
        header_start = payload.find(b">", cursor)
        if header_start < 0:
            break
        header_end = payload.find(b"\n", header_start)
        if header_end < 0:
            break
        next_header = payload.find(b"\n>", header_end)
        record_end = len(payload) if next_header < 0 else next_header + 1
        chrom = normalize_chrom(
            payload[header_start + 1 : header_end].split(None, 1)[0].decode(
                "utf-8", errors="replace"
            )
        )
        if chrom in index:
            sequence = (
                payload[header_end + 1 : record_end]
                .replace(b"\n", b"")
                .replace(b"\r", b"")
                .upper()
            )
            item = index[chrom]
            for offset, row in enumerate(item["row_indices"]):
                start = int(item["starts"][offset])
                end = int(item["ends"][offset])
                piece = sequence[start:end]
                row = int(row)
                gc[row] = piece.count(b"G") + piece.count(b"C")
                acgt[row] = sum(piece.count(base) for base in (b"A", b"C", b"G", b"T"))
                observed[row] = len(piece)
        cursor = record_end
    return gc, acgt, observed


def mappability_covariates(
    regions: pd.DataFrame,
    bedgraph: Path,
    *,
    chunksize: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full-length mean mappability and explicitly reported coverage.

    Parsing is chunked and vectorized because the k=100 track is hundreds of
    compressed megabytes.  The spill loop is normally executed once; it also
    makes the aggregation exact if one bedGraph run crosses a tile boundary.
    """

    index = interval_index(regions)
    weighted = np.zeros(len(regions), dtype=np.float64)
    reported = np.zeros(len(regions), dtype=np.int64)
    reader = pd.read_csv(
        bedgraph,
        compression="infer",
        sep=r"\s+",
        comment="#",
        header=None,
        names=["chrom", "start", "end", "value"],
        usecols=[0, 1, 2, 3],
        chunksize=chunksize,
        dtype={"chrom": "string"},
    )
    for chunk_index, chunk in enumerate(reader, start=1):
        chunk["start"] = pd.to_numeric(chunk["start"], errors="coerce")
        chunk["end"] = pd.to_numeric(chunk["end"], errors="coerce")
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        # This also discards optional UCSC track/browser header records.
        chunk = chunk.dropna(subset=["start", "end", "value"])
        chunk["chrom"] = chunk["chrom"].map(normalize_chrom)
        for chrom, group in chunk.groupby("chrom", sort=False):
            if chrom not in index:
                continue
            starts = group["start"].to_numpy(np.int64)
            ends = group["end"].to_numpy(np.int64)
            values = group["value"].to_numpy(float)
            if np.any(starts < 0) or np.any(ends <= starts) or not np.isfinite(values).all():
                raise ValueError(f"Invalid bedGraph records on {chrom}")
            item = index[str(chrom)]
            region_starts = item["starts"]
            region_ends = item["ends"]
            offsets = np.searchsorted(region_starts, starts, side="right") - 1
            offsets = np.maximum(offsets, 0)
            active = offsets < len(region_starts)
            while np.any(active):
                selected = np.flatnonzero(active)
                current = offsets[selected]
                overlap = np.maximum(
                    0,
                    np.minimum(ends[selected], region_ends[current])
                    - np.maximum(starts[selected], region_starts[current]),
                )
                positive = overlap > 0
                if np.any(positive):
                    rows = item["row_indices"][current[positive]]
                    np.add.at(weighted, rows, overlap[positive] * values[selected[positive]])
                    np.add.at(reported, rows, overlap[positive])
                spills = ends[selected] > region_ends[current]
                active[selected] = False
                if np.any(spills):
                    spill_selected = selected[spills]
                    offsets[spill_selected] += 1
                    active[spill_selected] = offsets[spill_selected] < len(region_starts)
        if chunk_index % 10 == 0:
            progress(f"mappability chunks processed: {chunk_index:,}")
    length = regions["region_length"].to_numpy(np.float64)
    return weighted / length, reported


def load_gene_intervals(gtf: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    genes: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with open_text(gtf) as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 9:
                raise ValueError(f"Malformed GTF record at line {line_number}")
            if fields[2] != "gene":
                continue
            chrom = normalize_chrom(fields[0])
            if chrom not in CANONICAL_CHROMOSOMES:
                continue
            start = int(fields[3]) - 1
            end = int(fields[4])
            if start < 0 or end <= start:
                raise ValueError(f"Invalid GTF gene interval at line {line_number}")
            genes[chrom].append((start, end))
            if line_number % 5_000_000 == 0:
                progress(f"GTF lines processed: {line_number:,}")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, values in genes.items():
        values.sort()
        starts = np.asarray([value[0] for value in values], dtype=np.int64)
        ends = np.asarray([value[1] for value in values], dtype=np.int64)
        result[chrom] = starts, np.maximum.accumulate(ends)
    return result


def distance_to_nearest_gene(
    regions: pd.DataFrame,
    genes: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    distances = np.full(len(regions), np.nan, dtype=float)
    for row in regions.itertuples():
        chrom = str(row.chromosome)
        if chrom not in genes:
            continue
        starts, prefix_max_end = genes[chrom]
        index = int(np.searchsorted(starts, int(row.end), side="left"))
        left_end = int(prefix_max_end[index - 1]) if index else None
        if left_end is not None and left_end > int(row.start):
            distance = 0
        else:
            left_distance = (
                int(row.start) - left_end if left_end is not None else math.inf
            )
            right_distance = (
                int(starts[index]) - int(row.end) if index < len(starts) else math.inf
            )
            distance = min(left_distance, right_distance)
        distances[int(row.Index)] = float(distance)
    return distances


def count_variant_anchors(
    regions: pd.DataFrame,
    variants: Path,
    *,
    chunksize: int,
) -> tuple[np.ndarray, int, int]:
    """Count each normalized variant once by its zero-based POS anchor."""

    index = interval_index(regions)
    counts = np.zeros(len(regions), dtype=np.int64)
    rows_processed = 0
    assigned = 0
    for chunk_index, chunk in enumerate(pd.read_csv(
        variants,
        compression="infer",
        usecols=["chrom", "pos"],
        chunksize=chunksize,
    ), start=1):
        rows_processed += len(chunk)
        chunk["chrom"] = chunk["chrom"].map(normalize_chrom)
        chunk["pos0"] = pd.to_numeric(chunk["pos"], errors="coerce") - 1
        chunk = chunk.loc[chunk["pos0"].notna()].copy()
        for chrom, group in chunk.groupby("chrom", sort=False):
            if chrom not in index:
                continue
            item = index[str(chrom)]
            positions = group["pos0"].to_numpy(np.int64)
            offsets = np.searchsorted(item["starts"], positions, side="right") - 1
            valid_offset = offsets >= 0
            candidate_offsets = offsets[valid_offset]
            candidate_positions = positions[valid_offset]
            in_interval = candidate_positions < item["ends"][candidate_offsets]
            selected_offsets = candidate_offsets[in_interval]
            if len(selected_offsets):
                region_rows = item["row_indices"][selected_offsets]
                np.add.at(counts, region_rows, 1)
                assigned += int(len(region_rows))
        if chunk_index % 10 == 0:
            progress(f"variant rows processed: {rows_processed:,}")
    return counts, rows_processed, assigned


def assign_priorities(
    regions: pd.DataFrame,
    *,
    priority_fraction: float,
    controls_per_case: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mark top-scoring eligible regions independently within chromosome."""

    if not 0 < priority_fraction < 1:
        raise ValueError("priority_fraction must lie strictly between 0 and 1")
    work = regions.copy()
    matching_columns = [
        "region_length", "gc_content", "mappability", "graph_complexity",
        "variant_density", "distance_to_gene", "model_score",
    ]
    finite = np.isfinite(work[matching_columns].to_numpy(float)).all(axis=1)
    eligible = work["is_eligible"].astype(bool) & finite
    reasons = np.where(
        work["is_eligible"].astype(bool), "eligible", work["eligibility_reason"].astype(str)
    ).astype(object)
    reasons[work["is_eligible"].astype(bool) & ~finite] = "missing_or_nonfinite_covariate"
    work["priority_eligibility_reason"] = reasons
    work["is_prioritized"] = False
    work["priority_rank_within_chromosome"] = np.nan
    work["priority_fraction_within_chromosome"] = np.nan
    chromosome_exclusions = []
    for chrom, group in work.loc[eligible].groupby("chromosome", sort=True):
        ordered = group.sort_values(
            ["model_score", "region_id"], ascending=[False, True], kind="mergesort"
        )
        n_regions = len(ordered)
        if n_regions < controls_per_case + 1:
            work.loc[ordered.index, "priority_eligibility_reason"] = (
                "insufficient_same_chromosome_region_universe"
            )
            chromosome_exclusions.append(
                {
                    "chromosome": chrom,
                    "eligible_regions": n_regions,
                    "reason": "fewer_regions_than_one_case_plus_requested_controls",
                }
            )
            continue
        n_priority = max(1, int(math.floor(n_regions * priority_fraction)))
        n_priority = min(n_priority, n_regions // (controls_per_case + 1))
        ranks = np.arange(1, n_regions + 1)
        work.loc[ordered.index, "priority_rank_within_chromosome"] = ranks
        work.loc[ordered.index, "priority_fraction_within_chromosome"] = (
            ranks / n_regions
        )
        work.loc[ordered.index[:n_priority], "is_prioritized"] = True
    retained = work.loc[
        work["priority_eligibility_reason"].eq("eligible")
    ].copy()
    if retained.empty or not retained["is_prioritized"].any():
        raise ValueError("No prioritized regions remain after eligibility checks")
    if not (~retained["is_prioritized"]).any():
        raise ValueError("No non-prioritized controls remain after eligibility checks")
    excluded = work.loc[~work.index.isin(retained.index)].copy()
    return retained, excluded, pd.DataFrame(chromosome_exclusions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-scores", type=Path, required=True)
    parser.add_argument(
        "--dense-audit",
        type=Path,
        help="Defaults to audit.json beside --dense-scores.",
    )
    parser.add_argument("--reference-fasta", type=Path, required=True)
    parser.add_argument("--mappability-bedgraph", type=Path, required=True)
    parser.add_argument("--gene-annotation", type=Path, required=True)
    parser.add_argument("--normalized-variants", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--priority-fraction", type=float, default=0.10)
    parser.add_argument("--controls-per-case", type=int, default=5)
    parser.add_argument("--mappability-chunksize", type=int, default=1_000_000)
    parser.add_argument("--variant-chunksize", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.out_dir}")
    if args.controls_per_case < 1:
        parser.error("--controls-per-case must be positive")
    if args.variant_chunksize < 1:
        parser.error("--variant-chunksize must be positive")
    if args.mappability_chunksize < 1:
        parser.error("--mappability-chunksize must be positive")
    for path in (
        args.dense_scores,
        args.reference_fasta,
        args.mappability_bedgraph,
        args.gene_annotation,
        args.normalized_variants,
    ):
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    dense_audit_path = args.dense_audit or args.dense_scores.parent / "audit.json"
    if not dense_audit_path.is_file():
        parser.error(f"Missing dense-score audit: {dense_audit_path}")
    dense_audit = json.loads(dense_audit_path.read_text())
    if dense_audit.get("status") != "complete":
        parser.error("Dense-score audit is not complete")
    if dense_audit.get("closure") != "strict":
        parser.error("Only strict-context dense scores are admissible")
    if dense_audit.get("downstream_signal_access") != (
        "none; QTL, GWAS, cCRE, and gene annotations are not read"
    ):
        parser.error("Dense-score audit does not prove signal-blind scoring")
    expected_score_hash = dense_audit.get("output_sha256", {}).get(
        args.dense_scores.name
    )
    if expected_score_hash and sha256sum(args.dense_scores) != expected_score_hash:
        parser.error("Dense score checksum differs from its audit")

    started = time.monotonic()
    progress("validating dense scores and their signal-blind audit")
    regions = validate_regions(pd.read_csv(args.dense_scores))
    progress(f"computing GC for {len(regions)} regions from reference FASTA")
    gc, acgt, observed = fasta_covariates(regions, args.reference_fasta)
    expected = regions["region_length"].to_numpy(np.int64)
    if not np.array_equal(observed, expected):
        bad = regions.loc[observed != expected, "region_id"].astype(str).tolist()
        raise ValueError(
            "Reference FASTA does not cover every region exactly; examples: "
            + ", ".join(bad[:10])
        )
    if np.any(acgt <= 0):
        bad = regions.loc[acgt <= 0, "region_id"].astype(str).tolist()
        raise ValueError(f"Regions have no canonical reference bases: {bad[:10]}")
    regions["gc_content"] = gc / acgt
    progress("computing full-length mean k=100 mappability (omitted bases are zero)")
    mappability, reported = mappability_covariates(
        regions,
        args.mappability_bedgraph,
        chunksize=args.mappability_chunksize,
    )
    if np.any(reported > expected):
        bad = regions.loc[reported > expected, "region_id"].astype(str).tolist()
        raise ValueError(
            "Mappability bedGraph reports overlapping coverage for regions: "
            + ", ".join(bad[:10])
        )
    if np.any((mappability < 0) | (mappability > 1)):
        bad = regions.loc[
            (mappability < 0) | (mappability > 1), "region_id"
        ].astype(str).tolist()
        raise ValueError(f"Mappability means fall outside [0,1]: {bad[:10]}")
    regions["mappability"] = mappability
    regions["mappability_reported_fraction"] = reported / expected
    progress("loading GENCODE genes and computing nearest-gene distance")
    genes = load_gene_intervals(args.gene_annotation)
    regions["distance_to_gene"] = distance_to_nearest_gene(regions, genes)
    progress("counting HPRC normalized variant anchors in chunks")
    variant_counts, variant_rows, variants_assigned = count_variant_anchors(
        regions,
        args.normalized_variants,
        chunksize=args.variant_chunksize,
    )
    regions["variant_count"] = variant_counts
    regions["variant_density"] = variant_counts / expected * 1_000_000.0
    progress("applying within-chromosome prespecified priority rule")
    retained, priority_exclusions, chromosome_exclusions = assign_priorities(
        regions,
        priority_fraction=args.priority_fraction,
        controls_per_case=args.controls_per_case,
    )
    order = {chrom: index for index, chrom in enumerate(CANONICAL_CHROMOSOMES)}
    retained["_chrom_order"] = retained["chromosome"].map(order)
    retained = retained.sort_values(
        ["_chrom_order", "start", "end", "region_id"]
    ).drop(columns="_chrom_order")
    output_columns = FINAL_REQUIRED_COLUMNS + [
        column
        for column in retained.columns
        if column not in FINAL_REQUIRED_COLUMNS
    ]

    args.out_dir.mkdir(parents=True, exist_ok=False)
    progress(f"writing region universe to {args.out_dir}")
    retained[output_columns].to_csv(args.out_dir / "regions.csv", index=False)
    priority_exclusions.to_csv(args.out_dir / "region_exclusions.csv", index=False)
    chromosome_exclusions.to_csv(
        args.out_dir / "chromosome_priority_exclusions.csv", index=False
    )
    covariate_summary = retained[
        [
            "gc_content", "mappability", "graph_complexity", "variant_density",
            "distance_to_gene", "model_score",
        ]
    ].describe().transpose()
    covariate_summary.to_csv(args.out_dir / "covariate_summary.csv")
    output_paths = [
        args.out_dir / "regions.csv",
        args.out_dir / "region_exclusions.csv",
        args.out_dir / "chromosome_priority_exclusions.csv",
        args.out_dir / "covariate_summary.csv",
    ]
    progress("hashing multi-gigabyte inputs and output tables for the final audit")
    audit = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "complete",
        "inputs": {
            "dense_scores": str(args.dense_scores.resolve()),
            "dense_audit": str(dense_audit_path.resolve()),
            "reference_fasta": str(args.reference_fasta.resolve()),
            "mappability_bedgraph": str(args.mappability_bedgraph.resolve()),
            "gene_annotation": str(args.gene_annotation.resolve()),
            "normalized_variants": str(args.normalized_variants.resolve()),
        },
        "input_sha256": {
            "dense_scores": sha256sum(args.dense_scores),
            "dense_audit": sha256sum(dense_audit_path),
            "reference_fasta": sha256sum(args.reference_fasta),
            "mappability_bedgraph": sha256sum(args.mappability_bedgraph),
            "gene_annotation": sha256sum(args.gene_annotation),
            "normalized_variants": sha256sum(args.normalized_variants),
        },
        "input_regions": int(len(regions)),
        "retained_regions": int(len(retained)),
        "excluded_regions": int(len(priority_exclusions)),
        "prioritized_regions": int(retained["is_prioritized"].sum()),
        "control_regions": int((~retained["is_prioritized"]).sum()),
        "chromosome_counts": {
            str(key): int(value)
            for key, value in retained["chromosome"].value_counts().sort_index().items()
        },
        "prioritized_chromosome_counts": {
            str(key): int(value)
            for key, value in retained.loc[
                retained["is_prioritized"], "chromosome"
            ].value_counts().sort_index().items()
        },
        "priority_rule": (
            f"top floor({args.priority_fraction:g} * eligible regions), at least one, "
            "ranked within chromosome by descending model_score with region_id tie-break"
        ),
        "priority_fraction": args.priority_fraction,
        "controls_per_case_reserved": args.controls_per_case,
        "gc_definition": "(G+C)/(A+C+G+T) from the GRCh38 reference",
        "mappability_definition": (
            "mean Umap/Bismap k=100 multi-read mappability over full region length; "
            "bedGraph-omitted bases are zero"
        ),
        "graph_complexity_definition": "2 * strict native-tile link count / segment count",
        "variant_density_definition": (
            "HPRC R2 normalized variant POS anchors per megabase"
        ),
        "distance_to_gene_definition": (
            "minimum base-pair distance to a GENCODE gene interval; zero on overlap"
        ),
        "variant_rows_processed": variant_rows,
        "variant_anchors_assigned_to_regions": variants_assigned,
        "gene_counts": {chrom: int(len(values[0])) for chrom, values in genes.items()},
        "mappability_zero_policy": "unreported Umap bases contribute zero, not missing",
        "external_signal_access": "none; no QTL or GWAS signal file is accepted by this program",
        "interpretation": (
            "prespecified model-prioritized case/control universe for matched enrichment; "
            "not causality, colocalization, or fine-mapping"
        ),
        "power_warning": (
            "native 5-Mb scoring yields a modest number of cases; chr8-only QTL "
            "and very dense all-trait signals may be underpowered or overlap-saturated"
        ),
        "output_sha256": {
            path.name: sha256sum(path) for path in output_paths
        },
        "wall_seconds": time.monotonic() - started,
    }
    audit_path = args.out_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    with (args.out_dir / "SHA256SUMS").open("w") as handle:
        for path in [*output_paths, audit_path]:
            handle.write(f"{sha256sum(path)}  {path.name}\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
