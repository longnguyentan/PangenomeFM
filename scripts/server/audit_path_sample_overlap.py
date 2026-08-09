#!/usr/bin/env python3
"""Audit exact donor/haplotype overlap from versioned GBZ path metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import pandas as pd


REFERENCE_SAMPLES = {"CHM13", "GRCh38", "GRCh37", "_gbwt_ref", "_gbwt_generic"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected DATASET=PATH, received {value!r}")
    label, raw_path = value.split("=", 1)
    if not label:
        raise ValueError("Dataset label cannot be empty")
    return label, Path(raw_path)


def read_path_metadata(path: Path) -> pd.DataFrame:
    separator = "\t" if path.name.endswith(".tsv") or path.name.endswith(".tsv.gz") else ","
    frame = pd.read_csv(path, sep=separator, compression="infer")
    if {"SAMPLE", "HAPLOTYPE", "LOCUS"}.issubset(frame):
        selected = frame.rename(
            columns={"SAMPLE": "sample", "HAPLOTYPE": "haplotype", "LOCUS": "contig"}
        )
    elif {"sample", "haplotype", "contig"}.issubset(frame):
        selected = frame
    else:
        raise ValueError(
            f"Unsupported path metadata schema in {path}; columns={frame.columns.tolist()}"
        )
    selected = selected[["sample", "haplotype", "contig"]].copy()
    selected["sample"] = selected["sample"].astype(str)
    selected["haplotype"] = selected["haplotype"].astype(str)
    selected["contig"] = selected["contig"].astype(str)
    selected = selected[~selected["sample"].isin(REFERENCE_SAMPLES)]
    return selected.drop_duplicates().reset_index(drop=True)


def audit(inputs: dict[str, Path], out_dir: Path) -> dict[str, object]:
    if len(inputs) < 2:
        raise ValueError("At least two path metadata inputs are required")
    path_tables = {label: read_path_metadata(path) for label, path in inputs.items()}
    sample_sets = {label: set(frame["sample"]) for label, frame in path_tables.items()}
    all_samples = sorted(set().union(*sample_sets.values()))
    membership = pd.DataFrame(
        {
            "sample": all_samples,
            **{label: [sample in samples for sample in all_samples] for label, samples in sample_sets.items()},
        }
    )
    pair_rows: list[dict[str, object]] = []
    shared_rows: list[dict[str, str]] = []
    for left, right in combinations(sorted(inputs), 2):
        shared = sorted(sample_sets[left] & sample_sets[right])
        pair_rows.append(
            {
                "dataset_a": left,
                "dataset_b": right,
                "donors_a": len(sample_sets[left]),
                "donors_b": len(sample_sets[right]),
                "shared_donors": len(shared),
                "fraction_a_shared": len(shared) / len(sample_sets[left]),
                "fraction_b_shared": len(shared) / len(sample_sets[right]),
                "shared_sample_ids": ";".join(shared),
            }
        )
        shared_rows.extend(
            {"dataset_a": left, "dataset_b": right, "sample": sample} for sample in shared
        )
    summaries = []
    for label, frame in path_tables.items():
        summaries.append(
            {
                "dataset": label,
                "path_metadata_records_after_reference_exclusion": len(frame),
                "donors": frame["sample"].nunique(),
                "phased_haplotypes": frame[["sample", "haplotype"]].drop_duplicates().shape[0],
                "distinct_path_loci_or_contigs": frame["contig"].nunique(),
                "source": str(inputs[label].resolve()),
                "source_sha256": sha256_file(inputs[label]),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    membership.to_csv(out_dir / "sample_membership.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out_dir / "pairwise_overlap.csv", index=False)
    pd.DataFrame(shared_rows, columns=["dataset_a", "dataset_b", "sample"]).to_csv(
        out_dir / "shared_samples.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(out_dir / "dataset_path_summary.csv", index=False)
    result = {
        "schema_version": 1,
        "status": "complete",
        "reference_samples_excluded": sorted(REFERENCE_SAMPLES),
        "datasets": summaries,
        "pairwise_overlaps": pair_rows,
        "interpretation": (
            "A cohort-transfer result is not independent-donor transfer when shared_donors > 0. "
            "Removing GBWT paths alone does not prove removal of donor-contributed aggregate graph topology."
        ),
    }
    (out_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="DATASET=PATH")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = dict(parse_labeled_path(value) for value in args.inputs)
    result = audit(inputs, args.out_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
