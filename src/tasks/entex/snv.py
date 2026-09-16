"""Stream the accessible EN-TEx SNV source; retain supplied assay-specific calls."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tasks.entex.prepare import fingerprint, validate_as

ASSAYS = {"ctcf": "TF-ChIP-seq_CTCF", "h3k27ac": "HM-ChIP-seq_H3K27ac"}


def clean_measurements(frame: pd.DataFrame, assay: str) -> tuple[pd.DataFrame, int]:
    validate_as(frame, snv=True)
    if not frame.assay.eq(ASSAYS[assay]).all():
        raise ValueError("Assay mismatch")
    keys = ["chr", "ref_start", "experiment_accession"]
    unique = frame.drop_duplicates()
    if unique.duplicated(keys).any():
        raise ValueError("Conflicting records for one SNV/experiment")
    excluded = len(frame) - len(unique)
    unique = unique.rename(
        columns={
            "chr": "chrom",
            "ref_start": "start",
            "ref_end": "end",
            "imbalance_significance": "label",
        }
    ).copy()
    unique["locus_id"] = unique.chrom + ":" + unique.start.astype(str)
    unique["measurement_id"] = unique.locus_id + ":" + unique.experiment_accession
    if unique.label.nunique() != 2:
        raise ValueError("SNV task requires both supplied AS classes")
    unique["task"] = "p2"
    unique["subtask"] = assay
    return unique.sort_values(["chrom", "start", "experiment_accession"]).reset_index(
        drop=True
    ), excluded


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/entex/v1/p2"))
    ap.add_argument("--chunksize", type=int, default=200000)
    args = ap.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit = dict(
        version=1,
        source=fingerprint(args.source),
        raw_rows=0,
        definition="Per accessible SNV/experiment supplied imbalance_significance; no absence-derived negatives",
        coordinates="GRCh38 zero-based half-open",
        requested_assays=ASSAYS,
    )
    counts = Counter()
    writers = {}
    try:
        for frame in pd.read_csv(args.source, sep="\t", chunksize=args.chunksize):
            validate_as(frame, snv=True)
            audit["raw_rows"] += len(frame)
            counts.update(frame.assay)
            for key, assay in ASSAYS.items():
                selected = frame.loc[frame.assay.eq(assay)]
                if selected.empty:
                    continue
                table = pa.Table.from_pandas(selected, preserve_index=False)
                if key not in writers:
                    writers[key] = pq.ParquetWriter(
                        args.out_dir / f"{key}_source.parquet",
                        table.schema,
                        compression="zstd",
                    )
                writers[key].write_table(table)
    finally:
        for writer in writers.values():
            writer.close()
    audit["assay_counts"] = dict(counts)
    audit["tasks"] = {}
    for key in ASSAYS:
        if key not in writers:
            raise ValueError(f"Requested assay missing: {key}")
        # Only the selected assay's compact cache is loaded, never the full TSV.
        frame, duplicates = clean_measurements(
            pd.read_parquet(args.out_dir / f"{key}_source.parquet"), key
        )
        frame.to_parquet(
            args.out_dir / f"{key}_measurements.parquet",
            index=False,
            compression="zstd",
        )
        loci = frame[["locus_id", "chrom", "start", "end"]].drop_duplicates()
        if loci.locus_id.duplicated().any():
            raise ValueError("Inconsistent locus coordinates")
        loci.to_parquet(args.out_dir / f"{key}_loci.parquet", index=False)
        audit["tasks"][key] = dict(
            rows=len(frame),
            unique_loci=len(loci),
            positive=int(frame.label.sum()),
            negative=int(frame.label.eq(0).sum()),
            prevalence=float(frame.label.mean()),
            identical_duplicates_excluded=duplicates,
            chromosomes=frame.chrom.value_counts().to_dict(),
            tissues=frame.tissue.value_counts().to_dict(),
            donors=frame.donor.value_counts().to_dict(),
        )
    audit["status"] = "complete"
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
