#!/usr/bin/env python3
"""Map normalized phased-SV records to GRCh38 reference graph nodes.

The output is a variant-level table with both breakpoint node identifiers and a
unique node table compatible with ``prepare_ccre_feature_cache.py``.  Mapping
is coordinate based and audited; no variant label is copied into the node
feature cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.splits import normalize_chrom


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_chromosome(value: object) -> str:
    result = normalize_chrom(str(value))
    if not result.startswith("chr") and result in {
        *(str(index) for index in range(1, 23)), "X", "Y", "M", "MT"
    }:
        result = f"chr{result}"
    return result


def reference_nodes(full_segments: Path, reference_prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(
        full_segments,
        compression="infer",
        usecols=["id", "SN", "SO", "LN"],
    ).rename(columns={"id": "segid"})
    frame = frame.loc[frame["SN"].astype(str).str.startswith(reference_prefix)].copy()
    frame["chrom"] = frame["SN"].map(canonical_chromosome)
    frame["SO"] = pd.to_numeric(frame["SO"], errors="coerce")
    frame["LN"] = pd.to_numeric(frame["LN"], errors="coerce")
    frame = frame.dropna(subset=["SO", "LN"])
    frame["SO"] = frame["SO"].astype(np.int64)
    frame["LN"] = frame["LN"].astype(np.int64)
    frame["end0"] = frame["SO"] + frame["LN"]
    frame = frame.loc[frame["LN"] > 0].sort_values(["chrom", "SO", "segid"])
    if frame.empty:
        raise ValueError(f"No reference nodes matched prefix {reference_prefix!r}")
    return frame.reset_index(drop=True)


def map_coordinate(
    coordinate0: int,
    starts: np.ndarray,
    ends: np.ndarray,
    segids: np.ndarray,
    *,
    maximum_nearest_distance: int,
) -> tuple[int | None, int | None, str]:
    insertion = int(np.searchsorted(starts, coordinate0, side="right") - 1)
    if insertion >= 0 and coordinate0 < int(ends[insertion]):
        return int(segids[insertion]), 0, "overlap"
    candidates: list[tuple[int, int]] = []
    if insertion >= 0:
        candidates.append((max(0, coordinate0 - int(ends[insertion]) + 1), insertion))
    following = insertion + 1
    if following < len(starts):
        candidates.append((max(0, int(starts[following]) - coordinate0), following))
    if not candidates:
        return None, None, "outside_reference"
    distance, index = min(candidates)
    if distance > maximum_nearest_distance:
        return None, distance, "nearest_too_far"
    return int(segids[index]), int(distance), "nearest"


def prepare(
    *,
    normalized_variants: Path,
    full_segments: Path,
    out_dir: Path,
    reference_prefix: str,
    svtypes: set[str],
    minimum_length: int,
    maximum_nearest_distance: int,
) -> dict[str, object]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    variants = pd.read_csv(normalized_variants, compression="infer")
    required = {"chrom", "pos", "end", "variant_id", "svtype", "svlen", "length_bin", "allele_frequency", "af_bin", "filter"}
    missing = required - set(variants)
    if missing:
        raise ValueError(f"Normalized variant table misses columns: {sorted(missing)}")
    variants["chrom"] = variants["chrom"].map(canonical_chromosome)
    variants["svtype"] = variants["svtype"].astype(str).str.upper()
    variants["svlen"] = pd.to_numeric(variants["svlen"], errors="coerce")
    input_count = len(variants)
    variants = variants.loc[
        variants["svtype"].isin(svtypes)
        & variants["svlen"].ge(minimum_length)
        & variants["filter"].astype(str).isin({"PASS", "."})
    ].copy()
    variants["start0"] = pd.to_numeric(variants["pos"], errors="coerce") - 1
    variants["end0"] = pd.to_numeric(variants["end"], errors="coerce") - 1
    variants = variants.dropna(subset=["start0", "end0"])
    variants[["start0", "end0"]] = variants[["start0", "end0"]].astype(np.int64)

    nodes = reference_nodes(full_segments, reference_prefix)
    indices = {
        chrom: (
            group["SO"].to_numpy(np.int64),
            group["end0"].to_numpy(np.int64),
            group["segid"].to_numpy(np.int64),
        )
        for chrom, group in nodes.groupby("chrom", sort=False)
    }
    rows: list[dict[str, object]] = []
    reason_counts: dict[str, int] = {}
    for row in variants.itertuples(index=False):
        if row.chrom not in indices:
            reason_counts["chromosome_absent"] = reason_counts.get("chromosome_absent", 0) + 1
            continue
        starts, ends, segids = indices[row.chrom]
        start_segid, start_distance, start_rule = map_coordinate(
            int(row.start0), starts, ends, segids,
            maximum_nearest_distance=maximum_nearest_distance,
        )
        end_segid, end_distance, end_rule = map_coordinate(
            int(row.end0), starts, ends, segids,
            maximum_nearest_distance=maximum_nearest_distance,
        )
        if start_segid is None or end_segid is None:
            reason = f"start_{start_rule}" if start_segid is None else f"end_{end_rule}"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            continue
        payload = row._asdict()
        payload.update(
            {
                "start_segid": start_segid,
                "end_segid": end_segid,
                "start_mapping_distance": start_distance,
                "end_mapping_distance": end_distance,
                "start_mapping_rule": start_rule,
                "end_mapping_rule": end_rule,
                "binary_svtype_label": int(str(row.svtype) == "INS"),
            }
        )
        rows.append(payload)
    examples = pd.DataFrame(rows)
    if examples.empty:
        raise RuntimeError("No SV records mapped to reference nodes")
    examples = examples.sort_values(["chrom", "start0", "end0", "variant_id"]).reset_index(drop=True)
    examples.insert(0, "example_id", np.arange(len(examples), dtype=np.int64))
    examples.to_csv(out_dir / "sv_breakpoint_examples.csv.gz", index=False, compression="gzip")

    needed_segids = set(examples["start_segid"].astype(int)) | set(examples["end_segid"].astype(int))
    feature_nodes = nodes.loc[nodes["segid"].isin(needed_segids), ["segid", "SN", "chrom", "SO", "LN"]].copy()
    feature_nodes["ccre_label"] = 0  # schema compatibility only; never used as an SV target
    feature_nodes["orient"] = 0
    feature_nodes.to_csv(out_dir / "feature_nodes.csv.gz", index=False, compression="gzip")

    per_stratum = (
        examples.groupby(["chrom", "svtype", "length_bin", "af_bin"], dropna=False)
        .size()
        .rename("n_examples")
        .reset_index()
    )
    per_stratum.to_csv(out_dir / "sv_example_strata.csv", index=False)
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "complete",
        "normalized_variants": str(normalized_variants.resolve()),
        "normalized_variants_sha256": sha256_file(normalized_variants),
        "full_segments": str(full_segments.resolve()),
        "full_segments_sha256": sha256_file(full_segments),
        "reference_prefix": reference_prefix,
        "requested_svtypes": sorted(svtypes),
        "minimum_length": minimum_length,
        "maximum_nearest_distance": maximum_nearest_distance,
        "input_variant_count": input_count,
        "eligible_variant_count": int(len(variants)),
        "mapped_example_count": int(len(examples)),
        "mapped_fraction_of_eligible": float(len(examples) / len(variants)) if len(variants) else 0.0,
        "unique_feature_nodes": int(len(feature_nodes)),
        "svtype_counts": {str(key): int(value) for key, value in examples["svtype"].value_counts().sort_index().items()},
        "chromosome_counts": {str(key): int(value) for key, value in examples["chrom"].value_counts().sort_index().items()},
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "label_definition": "INS=1, DEL=0; other requested classes require a multiclass probe",
        "leakage_note": "feature_nodes contains no SV label; targets remain variant-level",
        "wall_seconds": time.monotonic() - started,
    }
    (out_dir / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-variants", type=Path, required=True)
    parser.add_argument("--full-segments", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-prefix", default="GRCh38#0")
    parser.add_argument("--svtypes", nargs="+", default=["INS", "DEL"])
    parser.add_argument("--minimum-length", type=int, default=50)
    parser.add_argument("--maximum-nearest-distance", type=int, default=10_000)
    args = parser.parse_args()
    payload = prepare(
        normalized_variants=args.normalized_variants,
        full_segments=args.full_segments,
        out_dir=args.out_dir,
        reference_prefix=args.reference_prefix,
        svtypes={value.upper() for value in args.svtypes},
        minimum_length=args.minimum_length,
        maximum_nearest_distance=args.maximum_nearest_distance,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
