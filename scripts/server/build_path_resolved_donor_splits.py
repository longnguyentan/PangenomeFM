#!/usr/bin/env python3
"""Build deterministic donor-held-out splits from GBZ path metadata.

This creates the split and path-record manifests required before path-resolved
examples are materialized.  It does not claim to remove a donor's contribution
from an already aggregated graph; that requires graph rematerialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.server.audit_path_sample_overlap import REFERENCE_SAMPLES, sha256_file


PRIMARY_CHROMOSOMES = {
    *(f"chr{number}" for number in range(1, 23)),
    "chrX",
    "chrY",
    "chrM",
}


def annotate_reference_chromosome_blocks(frame: pd.DataFrame) -> pd.Series:
    """Record the primary reference chromosome governing each metadata row.

    ``vg paths -M`` emits HPRC haplotype paths with assembly-contig LOCUS
    values inside blocks introduced by REFERENCE rows.  The block label must
    be retained before those reference rows are removed; it cannot generally
    be reconstructed from an assembly-contig accession afterwards.
    """

    current: str | None = None
    blocks: list[str | None] = []
    for sense, locus in zip(frame["SENSE"], frame["LOCUS"], strict=True):
        if str(sense) == "REFERENCE":
            candidate = str(locus)
            current = candidate if candidate in PRIMARY_CHROMOSOMES else None
        blocks.append(current)
    return pd.Series(blocks, index=frame.index, dtype="string")


def deterministic_split(
    samples: list[str],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, str]:
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("Invalid train/validation fractions")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below one")
    ordered = sorted(
        samples,
        key=lambda sample: hashlib.sha256(f"{seed}:{sample}".encode()).hexdigest(),
    )
    n = len(ordered)
    n_train = int(round(n * train_fraction))
    n_validation = int(round(n * validation_fraction))
    # Keep every split nonempty whenever at least three donors are available.
    if n >= 3:
        n_train = min(max(n_train, 1), n - 2)
        n_validation = min(max(n_validation, 1), n - n_train - 1)
    assignments = {}
    for index, sample in enumerate(ordered):
        assignments[sample] = (
            "train"
            if index < n_train
            else "validation"
            if index < n_train + n_validation
            else "test"
        )
    return assignments


def build(
    *,
    path_metadata: Path,
    dataset: str,
    out_dir: Path,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    exclude_samples: set[str],
) -> dict[str, object]:
    separator = "\t" if path_metadata.name.endswith(".tsv") or path_metadata.name.endswith(".tsv.gz") else ","
    frame = pd.read_csv(path_metadata, sep=separator, compression="infer")
    if {"#NAME", "SENSE", "SAMPLE", "HAPLOTYPE", "LOCUS"}.issubset(frame):
        frame["chromosome"] = annotate_reference_chromosome_blocks(frame)
        frame = frame.rename(
            columns={
                "#NAME": "path_name",
                "SENSE": "sense",
                "SAMPLE": "sample",
                "HAPLOTYPE": "haplotype",
                "LOCUS": "locus",
            }
        )
    elif {"path_name", "sample", "haplotype", "contig"}.issubset(frame):
        frame = frame.rename(columns={"contig": "locus"})
        frame["sense"] = "HAPLOTYPE"
        frame["chromosome"] = frame["locus"].where(
            frame["locus"].astype(str).isin(PRIMARY_CHROMOSOMES)
        )
    else:
        raise ValueError(f"Unsupported path metadata columns: {frame.columns.tolist()}")
    frame["sample"] = frame["sample"].astype(str)
    frame = frame[
        ~frame["sample"].isin(REFERENCE_SAMPLES | exclude_samples)
        & frame["sense"].astype(str).eq("HAPLOTYPE")
    ].copy()
    samples = sorted(frame["sample"].unique())
    assignments = deterministic_split(
        samples,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    frame["dataset"] = dataset
    frame["donor_split"] = frame["sample"].map(assignments)
    selected_columns = [
        "dataset",
        "donor_split",
        "sample",
        "haplotype",
        "locus",
        "chromosome",
        "path_name",
    ]
    for optional in ("PHASE_BLOCK", "SUBRANGE"):
        if optional in frame:
            selected_columns.append(optional)
    path_records = frame[selected_columns].drop_duplicates()
    donor_rows = []
    for sample, group in path_records.groupby("sample", sort=True):
        donor_rows.append(
            {
                "dataset": dataset,
                "sample": sample,
                "split": assignments[sample],
                "haplotypes": group["haplotype"].nunique(),
                "path_records": len(group),
                "loci": group["locus"].nunique(),
            }
        )
    donors = pd.DataFrame(donor_rows)
    overlap = donors.groupby("sample")["split"].nunique()
    if (overlap > 1).any():
        raise AssertionError("A donor was assigned to multiple splits")

    out_dir.mkdir(parents=True, exist_ok=True)
    donors.to_csv(out_dir / "donor_splits.csv", index=False)
    path_records.to_csv(out_dir / "path_records.csv.gz", index=False, compression="gzip")
    split_summary = (
        donors.groupby("split")
        .agg(donors=("sample", "nunique"), phased_haplotypes=("haplotypes", "sum"), path_records=("path_records", "sum"))
        .reset_index()
    )
    split_summary.to_csv(out_dir / "split_summary.csv", index=False)
    result = {
        "schema_version": 1,
        "status": "manifest_complete_examples_not_materialized",
        "dataset": dataset,
        "path_metadata": str(path_metadata.resolve()),
        "path_metadata_sha256": sha256_file(path_metadata),
        "seed": seed,
        "requested_fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": 1 - train_fraction - validation_fraction,
        },
        "excluded_samples": sorted(exclude_samples),
        "donors": int(donors["sample"].nunique()),
        "phased_haplotypes": int(donors["haplotypes"].sum()),
        "path_records": int(len(path_records)),
        "chromosome_path_record_counts": {
            str(key): int(value)
            for key, value in path_records["chromosome"]
            .fillna("unassigned")
            .value_counts()
            .sort_index()
            .items()
        },
        "unassigned_chromosome_path_records": int(
            (~path_records["chromosome"].isin(PRIMARY_CHROMOSOMES)).sum()
        ),
        "split_summary": split_summary.to_dict("records"),
        "leakage_check": "each sample occurs in exactly one split",
        "blocking_note": (
            "The manifest defines donor-held-out path records but the current aggregate GFA tables do not retain "
            "path-specific node/edge membership. Materialize examples from GBZ paths or rebuild cohort-filtered graphs."
        ),
    }
    (out_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-metadata", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--exclude-samples", nargs="*", default=[])
    args = parser.parse_args()
    result = build(
        path_metadata=args.path_metadata,
        dataset=args.dataset,
        out_dir=args.out_dir,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        exclude_samples=set(args.exclude_samples),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
