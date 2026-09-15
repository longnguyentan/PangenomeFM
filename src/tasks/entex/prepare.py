"""Stream EN-TEx inputs into audited caches; never infer negatives from absence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

VERSION = 1
CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}


def fingerprint(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return dict(
        path=str(path.resolve()),
        bytes=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns,
        sha256=h.hexdigest(),
    )


def validate_as(frame: pd.DataFrame, snv: bool = False) -> None:
    start, end = ("ref_start", "ref_end") if snv else ("start", "end")
    required = [
        "chr",
        start,
        end,
        "experiment_accession",
        "donor",
        "tissue",
        "assay",
        "p_betabinom",
        "imbalance_significance",
    ]
    counts = ["cA", "cC", "cG", "cT"] if snv else ["hap1_count", "hap2_count"]
    required += counts + (
        ["ref_allele", "hap1_allele", "hap2_allele"] if snv else ["region_id"]
    )
    if set(required) - set(frame):
        raise ValueError(f"Missing columns: {set(required) - set(frame)}")
    if frame[required].isna().any().any():
        raise ValueError(
            "Missing required EN-TEx values; cannot assign an informative label"
        )
    if not set(frame.chr).issubset(CHROMS):
        raise ValueError(f"Unexpected chromosomes: {set(frame.chr) - CHROMS}")
    if (
        (frame[start] < 0)
        | (frame[end] <= frame[start])
        | (frame[start] % 1 != 0)
        | (frame[end] % 1 != 0)
    ).any():
        raise ValueError("Malformed zero-based half-open coordinates")
    if snv and (frame[end] - frame[start] != 1).any():
        raise ValueError("SNV intervals must be one base")
    if not frame.imbalance_significance.isin([0, 1]).all():
        raise ValueError("Unexpected AS call; expected supplied binary significance")
    if not frame.p_betabinom.between(0, 1).all():
        raise ValueError("Invalid p-values")
    if (frame[counts] < 0).any().any() or (frame[counts] % 1 != 0).any().any():
        raise ValueError("Invalid read counts")
    if (frame[counts].sum(axis=1) == 0).any():
        raise ValueError("Unmeasurable zero-read record in accessible call table")


def inspect_source(path: Path, out: Path, chunksize: int) -> dict:
    """Scan every row and every archive member, retaining a compact Parquet cache."""
    out.mkdir(parents=True, exist_ok=True)
    audit = dict(
        schema_version=VERSION,
        source=fingerprint(path),
        raw_row_count=0,
        missing_values={},
        value_counts={},
        examples=[],
        delimiter="tab",
    )
    is_zip = zipfile.is_zipfile(path)
    audit["compression"] = "zip" if is_zip else "none"
    counters: dict[str, Counter] = {}
    missing = Counter()
    loci: set = set()
    writer = None
    archive = zipfile.ZipFile(path) if is_zip else None
    temporary = out / (path.name + ".partial.parquet")
    try:
        if archive:
            audit["archive_members"] = [
                dict(
                    name=i.filename, bytes=i.file_size, compressed_bytes=i.compress_size
                )
                for i in archive.infolist()
            ]
            bad = archive.testzip()
            if bad:
                raise ValueError(f"Corrupt ZIP member: {bad}")
            members = [
                i.filename
                for i in archive.infolist()
                if not i.is_dir() and not i.filename.startswith("__MACOSX/")
            ]
        else:
            members = [None]
        for member in members:
            source = archive.open(member) if archive else path
            reader = pd.read_csv(
                source,
                sep="\t",
                chunksize=chunksize,
                **(
                    {"header": None, "names": ["ccre_id", "state", "tissue"]}
                    if archive
                    else {}
                ),
            )
            for frame in reader:
                if not archive:
                    validate_as(frame, snv="ref_start" in frame)
                    coord = (
                        ["chr", "ref_start", "ref_end"]
                        if "ref_start" in frame
                        else ["chr", "start", "end"]
                    )
                    loci.update(frame[coord].itertuples(index=False, name=None))
                else:
                    if frame.isna().any().any():
                        raise ValueError("Missing archive annotation")
                    loci.update(frame.ccre_id)
                audit["raw_row_count"] += len(frame)
                audit["columns"] = list(frame)
                if not audit["examples"]:
                    audit["examples"] = frame.head(3).to_dict("records")
                missing.update(frame.isna().sum().to_dict())
                for key in [
                    "chr",
                    "donor",
                    "tissue",
                    "assay",
                    "state",
                    "imbalance_significance",
                ]:
                    if key in frame:
                        counters.setdefault(key, Counter()).update(
                            frame[key].astype(str)
                        )
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary, table.schema, compression="zstd"
                    )
                writer.write_table(table)
        if writer:
            writer.close()
            writer = None
        temporary.replace(out / (path.name + ".parquet"))
    finally:
        if writer:
            writer.close()
        if archive:
            archive.close()
    audit["missing_values"] = dict(missing)
    audit["value_counts"] = {k: dict(v) for k, v in counters.items()}
    audit["unique_genomic_loci" if not archive else "unique_ccre_ids"] = len(loci)
    audit["coordinate_convention"] = (
        "GRCh38 zero-based half-open; EN-TEx paper methods"
        if not archive
        else "no coordinates; registry accession join required"
    )
    (out / (path.name + ".qc.json")).write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def aggregate_ccre(
    measurements: pd.DataFrame, *, require_both_classes: bool = True
) -> pd.DataFrame:
    """Union provided calls across distinct informative experiments per interval."""
    validate_as(measurements)
    keys = ["chr", "start", "end"]
    identity = keys + ["experiment_accession"]
    # Conflicting duplicate records cannot be resolved by counting or silently taking max.
    exact = measurements.drop_duplicates()
    if exact.duplicated(identity).any():
        raise ValueError("Conflicting duplicate cCRE/experiment records")
    grouped = exact.groupby(keys, sort=True, observed=True)
    result = (
        grouped.agg(
            label=("imbalance_significance", "max"),
            n_informative_experiments=("experiment_accession", "nunique"),
            n_as_experiments=("imbalance_significance", "sum"),
            donor_count=("donor", "nunique"),
            tissue_count=("tissue", "nunique"),
            assay_count=("assay", "nunique"),
            region_ids=("region_id", lambda x: "|".join(sorted(set(x)))),
        )
        .reset_index()
        .rename(columns={"chr": "chrom"})
    )
    result["ccre_id"] = result.region_ids.str.split("_").str[0]
    result["locus_id"] = (
        result.chrom + ":" + result.start.astype(str) + "-" + result.end.astype(str)
    )
    if require_both_classes and result.label.nunique() != 2:
        raise ValueError("Prepared cCRE task does not contain both classes")
    return result


def prepare_ccre(cache: Path, out: Path) -> dict:
    """Read the compressed cache by chromosome to bound aggregation memory."""
    parquet = pq.ParquetFile(cache)
    chroms = set()
    for batch in parquet.iter_batches(columns=["chr"]):
        chroms.update(batch.column(0).to_pylist())
    parts = []
    # Arrow row-group statistics skip non-overlapping chromosome groups where possible.
    for chrom in sorted(chroms):
        frame = pd.read_parquet(cache, filters=[("chr", "=", chrom)])
        # Some small chromosomes may legitimately have only one class: validate globally below.
        parts.append(aggregate_ccre(frame, require_both_classes=False))
    loci = pd.concat(parts, ignore_index=True)
    if loci.label.nunique() != 2:
        raise ValueError("Prepared cCRE task does not contain both classes")
    loci.to_parquet(out / "p0_loci.parquet", index=False)
    audit = dict(
        schema_version=VERSION,
        source_cache=fingerprint(cache),
        raw_row_count=parquet.metadata.num_rows,
        unique_genomic_loci=len(loci),
        positive_count=int(loci.label.sum()),
        negative_count=int((loci.label == 0).sum()),
        positive_prevalence=float(loci.label.mean()),
        chromosome_distribution=loci.chrom.value_counts().to_dict(),
        excluded_count=0,
        definition="Any supplied significant call among accessible experiments",
        mapping_status="pending exact manuscript HPRC R2 segment resource",
        feature_coverage={"C": None, "S": None, "T": None},
    )
    (out / "p0_qc.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/entex/v1"))
    ap.add_argument("--chunksize", type=int, default=100_000)
    ap.add_argument("--inspect-all", action="store_true")
    args = ap.parse_args()
    names = ["cCREs_default_AS.tsv"]
    if args.inspect_all:
        names += [
            "hetSNVs_high-confidence_AS.tsv",
            "active.combined_set.txt.zip",
            "repressed.combined_set.txt.zip",
        ]
    for name in names:
        cache = args.out_dir / (name + ".parquet")
        audit_path = args.out_dir / (name + ".qc.json")
        current = fingerprint(args.data_dir / name)
        if not (
            cache.exists()
            and audit_path.exists()
            and json.loads(audit_path.read_text()).get("source") == current
        ):
            audit = inspect_source(args.data_dir / name, args.out_dir, args.chunksize)
            print(name, audit["raw_row_count"], flush=True)
    print(
        json.dumps(
            prepare_ccre(args.out_dir / "cCREs_default_AS.tsv.parquet", args.out_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
