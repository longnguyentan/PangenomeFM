"""Check assay support on every fixed chromosome partition before new fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tasks.entex.snv import ASSAYS


def partition_counts(counts: dict, folds: list[dict], minimum: int) -> pd.DataFrame:
    """Validate the complete measured population, never select using model scores."""
    known = [chrom for fold in folds for chrom in fold["test"]]
    if len(known) != len(set(known)) or not set(counts).issubset(known):
        raise ValueError("Duplicate test chromosomes or loci outside manuscript folds")
    rows = []
    for fold in folds:
        test, val = set(fold["test"]), set(fold["validation"])
        if test & val or not (test | val).issubset(known):
            raise ValueError("Invalid chromosome partitions")
        for partition, chroms in [("train", set(known) - test - val), ("validation", val), ("test", test)]:
            row = dict(fold=fold["name"], partition=partition)
            for label in ("positive", "negative"):
                row[label] = sum(counts.get(chrom, {}).get(label, 0) for chrom in chroms)
                if row[label] < minimum:
                    raise ValueError(f"{fold['name']}/{partition} has only {row[label]} {label}; requires {minimum}")
            row["prevalence"] = row["positive"] / (row["positive"] + row["negative"])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/entex_extension_20260927.json"))
    args = parser.parse_args()
    plan = json.loads(args.config.read_text())
    base = json.loads(Path(plan["base_config"]).read_text())
    manuscript = json.loads(Path(base["manuscript_config"]).read_text())
    audit = json.loads((args.data_dir / "audit.json").read_text())
    if (audit.get("status") != "complete" or audit["source"]["sha256"] != plan["source_sha256"]
            or audit["requested_assays"] != {name: ASSAYS[name] for name in plan["assays"]}):
        raise ValueError("Prepared assay panel/source differs from the fixed plan")
    frames = []
    for assay in plan["assays"]:
        frame = partition_counts(audit["tasks"][assay]["chromosome_class_counts"],
                                 manuscript["rotating_chromosome_folds"], plan["minimum_per_class_per_partition"])
        frame.insert(0, "assay", assay)
        frames.append(frame)
    pd.concat(frames).to_csv(args.data_dir / "partition_counts.csv", index=False)
    (args.data_dir / "panel_readiness.json").write_text(json.dumps({
        "status": "complete", "scope": "source and partition class support; mapping and feature gates remain in their existing runners",
        "config": str(args.config), "assays": plan["assays"],
        "source_sha256": audit["source"]["sha256"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
