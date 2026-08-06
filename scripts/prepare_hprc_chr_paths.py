#!/usr/bin/env python3
"""Resolve exact donor chromosome paths from an HPRC GBZ path metadata table.

The HPRC metadata is emitted in reference-chromosome blocks.  A donor assembly
can use chromosome accessions, unlocalized WGS contigs, or a mixture of both,
so accession-number order is not a valid chromosome mapping.  We therefore
inherit the chromosome label from the surrounding REFERENCE records and select
all donor paths in the requested block.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PRIMARY_CHROMOSOMES = {
    *(f"chr{number}" for number in range(1, 23)),
    "chrX",
    "chrY",
    "chrM",
}


def annotate_reference_chromosome_blocks(metadata: pd.DataFrame) -> pd.Series:
    """Return the primary reference chromosome governing each metadata row."""

    current: str | None = None
    blocks: list[str | None] = []
    for row in metadata.itertuples(index=False):
        if str(row.SENSE) == "REFERENCE":
            locus = str(row.LOCUS)
            current = locus if locus in PRIMARY_CHROMOSOMES else None
        blocks.append(current)
    return pd.Series(blocks, index=metadata.index, dtype="string")


def prepare_paths(
    *,
    metadata_path: str | Path,
    cohort_path: str | Path,
    out_dir: str | Path,
    chromosome: str,
) -> dict[str, object]:
    metadata = pd.read_csv(metadata_path, sep="\t", dtype=str)
    required = {"#NAME", "SENSE", "SAMPLE", "HAPLOTYPE", "LOCUS"}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    if chromosome not in PRIMARY_CHROMOSOMES:
        raise ValueError(f"Unsupported primary chromosome: {chromosome}")
    metadata = metadata.copy()
    metadata["_chromosome_block"] = annotate_reference_chromosome_blocks(metadata)
    chromosome_rows = metadata[metadata["_chromosome_block"].eq(chromosome)]
    if chromosome_rows.empty:
        raise ValueError(
            f"No REFERENCE-delimited metadata block was found for {chromosome}."
        )
    cohort = pd.read_csv(cohort_path, sep="\t", dtype=str)
    donors = sorted(cohort["donor_id"].dropna().unique())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for donor in donors:
        donor_rows = chromosome_rows[
            chromosome_rows["SAMPLE"].eq(donor)
            & chromosome_rows["SENSE"].eq("HAPLOTYPE")
        ]
        if donor_rows.empty:
            raise ValueError(
                f"Donor {donor} has no HAPLOTYPE paths in the {chromosome} block."
            )
        selected_names: list[str] = []
        for haplotype in ("1", "2"):
            hap_rows = donor_rows[donor_rows["HAPLOTYPE"].eq(haplotype)]
            if hap_rows.empty:
                raise ValueError(
                    f"Donor {donor} lacks haplotype {haplotype} in the "
                    f"{chromosome} block."
                )
            paths = sorted(hap_rows["#NAME"].dropna().unique())
            loci = sorted(hap_rows["LOCUS"].dropna().unique())
            if not paths:
                raise ValueError(
                    f"No paths found for {donor} haplotype {haplotype} in "
                    f"{chromosome}."
                )
            selected_names.extend(paths)
            rows.append(
                {
                    "donor_id": donor,
                    "chromosome": chromosome,
                    "haplotype": haplotype,
                    "locus": ";".join(loci),
                    "n_loci": len(loci),
                    "resolution_method": "reference_chromosome_block",
                    "n_path_fragments": len(paths),
                    "first_path": paths[0],
                    "last_path": paths[-1],
                }
            )
        (out_dir / f"{donor}.{chromosome}.paths.txt").write_text(
            "\n".join(selected_names) + "\n", encoding="utf-8"
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / f"{chromosome}.path_manifest.tsv", sep="\t", index=False)
    summary = {
        "chromosome": chromosome,
        "n_donors": len(donors),
        "n_haplotypes": len(rows),
        "n_path_fragments": int(manifest["n_path_fragments"].sum()),
        "resolution_methods": manifest["resolution_method"].value_counts().to_dict(),
        "metadata_path": str(metadata_path),
        "cohort_path": str(cohort_path),
    }
    (out_dir / f"{chromosome}.path_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chromosome", default="chr8")
    args = parser.parse_args()
    summary = prepare_paths(
        metadata_path=args.metadata,
        cohort_path=args.cohort,
        out_dir=args.out_dir,
        chromosome=args.chromosome,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
