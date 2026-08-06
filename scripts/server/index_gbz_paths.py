#!/usr/bin/env python3
"""Create and validate compact vg path-metadata tables from downloaded GBZs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path


REQUIRED_COLUMNS = {"#NAME", "SENSE", "SAMPLE", "HAPLOTYPE", "LOCUS"}
REFERENCE_SAMPLES = {"CHM13", "GRCh38", "GRCh37"}


def summarize(path: Path) -> dict[str, object]:
    samples: set[str] = set()
    sample_haplotypes: set[tuple[str, str]] = set()
    loci: set[str] = set()
    senses: dict[str, int] = {}
    rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"vg metadata output is missing columns: {sorted(missing)}")
        for row in reader:
            rows += 1
            sample = row["SAMPLE"]
            haplotype = row["HAPLOTYPE"]
            loci.add(row["LOCUS"])
            sense = row["SENSE"]
            senses[sense] = senses.get(sense, 0) + 1
            if sample not in REFERENCE_SAMPLES:
                samples.add(sample)
                sample_haplotypes.add((sample, haplotype))
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "path_metadata": str(path),
        "path_records": rows,
        "samples_excluding_references": len(samples),
        "sample_haplotypes_excluding_references": len(sample_haplotypes),
        "loci": len(loci),
        "sense_counts": senses,
        "status": "verified",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gbz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gbz = args.gbz.expanduser().resolve()
    output = args.output.expanduser().resolve()
    command = ["vg", "paths", "-x", str(gbz), "-M"]
    if args.dry_run:
        print(" ".join(command) + f" > {output}")
        return 0
    if not gbz.is_file():
        raise FileNotFoundError(gbz)
    if shutil.which("vg") is None:
        raise RuntimeError("vg is required; run scripts/server/bootstrap_server.sh")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or args.force:
        temporary = output.with_name(output.name + ".tmp")
        temporary.unlink(missing_ok=True)
        with temporary.open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                command, stdout=handle, stderr=subprocess.PIPE, text=True, check=False
            )
        if process.returncode != 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"vg paths failed ({process.returncode}) for {gbz}: {process.stderr.strip()}"
            )
        temporary.replace(output)

    summary = summarize(output)
    summary["gbz"] = str(gbz)
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary_summary.replace(summary_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
