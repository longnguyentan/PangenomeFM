"""Audit exact cCRE accession coverage before any enhancer registry intersection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from tasks.entex.prepare import fingerprint, CHROMS


def read_registry(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", header=None)
    if frame.shape[1] == 11:
        # ENCODE V2 BED9+: name is the cCRE accession; column 10 is its class.
        frame = frame[[0, 1, 2, 3, 9]]
    elif frame.shape[1] == 6:
        # SCREEN download: column 4 is rDHS (EH38D), column 5 is cCRE (EH38E).
        frame = frame[[0, 1, 2, 4, 5]]
    else:
        raise ValueError(f"Unsupported registry schema with {frame.shape[1]} columns")
    frame.columns = ["chrom", "start", "end", "ccre_id", "ccre_class"]
    if frame.isna().any().any() or frame.ccre_id.duplicated().any():
        raise ValueError("Missing registry values or duplicate cCRE accession")
    if not frame.ccre_id.str.startswith("EH38E").all():
        raise ValueError("Expected GRCh38 EH38E cCRE accession column")
    if (
        not set(frame.chrom).issubset(CHROMS)
        or ((frame.start < 0) | (frame.end <= frame.start)).any()
    ):
        raise ValueError("Invalid registry coordinates")
    return frame


def audit_registry(
    registry: Path, cache_root: Path, out: Path, source_url: str
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    frame = read_registry(registry)
    ids = set(frame.ccre_id)
    summaries = []
    tissues = []
    for state in ["active", "repressed"]:
        total = found = 0
        unique = set()
        for batch in pq.ParquetFile(
            cache_root / f"{state}.combined_set.txt.zip.parquet"
        ).iter_batches(batch_size=100000):
            d = batch.to_pandas()
            d["matched"] = d.ccre_id.isin(ids)
            total += len(d)
            found += int(d.matched.sum())
            unique.update(d.ccre_id)
            tissues.append(
                d.groupby("tissue")
                .matched.agg(rows="size", matched_rows="sum")
                .reset_index()
                .assign(state=state)
            )
        summaries.append(
            dict(
                state=state,
                rows=total,
                matched_rows=found,
                row_coverage=found / total,
                unique_ids=len(unique),
                matched_unique_ids=len(unique & ids),
            )
        )
    tissue = (
        pd.concat(tissues)
        .groupby(["state", "tissue"], as_index=False)[["rows", "matched_rows"]]
        .sum()
    )
    tissue["row_coverage"] = tissue.matched_rows / tissue.rows
    tissue.to_csv(out / "coverage_by_tissue.csv", index=False)
    audit = dict(
        registry=fingerprint(registry),
        source_url=source_url,
        registry_rows=len(frame),
        coverage=summaries,
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, default=Path("data/entex/v1"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source-url", required=True)
    a = ap.parse_args()
    print(
        json.dumps(
            audit_registry(a.registry, a.cache_root, a.out_dir, a.source_url), indent=2
        )
    )


if __name__ == "__main__":
    main()
