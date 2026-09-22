"""Prepare clonal HG008 INS/DEL records without changing the manuscript target."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.server.audit_sv_truth import (
    LENGTH_BINS,
    infer_end_and_length,
    infer_svtype,
    label_bin,
    parse_info,
)
from tasks.entex.prepare import fingerprint


def verify_md5(path: Path, checksums: Path) -> None:
    entries = [
        line.split()
        for line in checksums.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    expected = dict(entries).get(path.name)  # NIST release uses filename then hash.
    if expected is None or hashlib.md5(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"NIST release checksum mismatch: {path.name}")


def in_regions(bed: pd.DataFrame, chrom: str, position0: int) -> bool:
    return bool(
        ((bed.chrom == chrom) & (bed.start <= position0) & (position0 < bed.end)).any()
    )


def normalize(vcf: Path, bed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, rejected = [], []
    counts = Counter()
    with gzip.open(vcf, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 8:
                raise ValueError("Malformed VCF row")
            chrom, pos, identifier, ref, alt, _, filt, raw_info = fields[:8]
            pos = int(pos)
            if pos < 1 or not chrom.startswith("chr"):
                raise ValueError("Expected one-based GRCh38 chr-prefixed coordinates")
            info = parse_info(raw_info)
            kind = infer_svtype(ref, alt, info)
            end, length = infer_end_and_length(pos, ref, alt, kind, info)
            counts[kind] += 1
            reason = None
            if filt != "PASS":
                reason = "not_PASS"
            elif kind not in {"INS", "DEL"}:
                reason = "not_INS_or_DEL"
            elif info.get("SUBCLONAL") != "n":
                reason = "subclonal_or_unspecified"
            elif length is None or length < 50:
                reason = "length_below_50_or_missing"
            # For INS, the biological coordinate is the VCF anchor. The old
            # representation's inferred pseudo-end is retained only for features.
            elif not in_regions(bed, chrom, pos - 1) or (
                kind == "DEL" and not in_regions(bed, chrom, end - 1)
            ):
                reason = "biological_endpoint_outside_clonal_BED"
            if reason:
                rejected.append(dict(variant_id=identifier, svtype=kind, reason=reason))
                continue
            rows.append(
                dict(
                    chrom=chrom,
                    pos=pos,
                    end=end,
                    variant_id=identifier,
                    svtype=kind,
                    svlen=length,
                    filter=filt,
                    length_bin=label_bin(length, LENGTH_BINS, "len"),
                    allele_frequency=float("nan"),
                    af_bin="missing",
                    event_id=info.get("EVENT", identifier),
                    complex=info.get("COMPLEX", "."),
                    subclonal=info["SUBCLONAL"],
                    end_inferred="END" not in info,
                    biological_start0=pos - 1,
                    biological_end0=(end - 1 if kind == "DEL" else pos - 1),
                )
            )
    frame = pd.DataFrame(rows)
    if frame.empty or set(frame.svtype) != {"INS", "DEL"}:
        raise ValueError("Eligible HG008 records must include both INS and DEL")
    if frame.variant_id.duplicated().any():
        raise ValueError("Duplicate HG008 variant IDs")
    excluded = pd.DataFrame(rejected)
    audit = dict(
        raw_rows=sum(counts.values()),
        raw_svtypes=dict(counts),
        eligible_rows=len(frame),
        svtypes=frame.svtype.value_counts().to_dict(),
        chromosomes=frame.chrom.value_counts().to_dict(),
        exclusion_counts=excluded.reason.value_counts().to_dict()
        if len(excluded)
        else {},
        positive_prevalence=float(frame.svtype.eq("INS").mean()),
        unique_events=int(frame.event_id.nunique()),
        inferred_end_rows=int(frame.end_inferred.sum()),
        endpoint_convention="Original HGSVC normalizer: POS-1 and END-1, or POS+max(REF length, SVLEN)-2 when END absent",
        target="INS=1 versus DEL=0, length>=50; no genome-background negatives",
    )
    return frame, excluded, audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    vcf = (
        args.source_dir / "GRCh38_HG008-T-V0.5_somatic-stvar_PASS.draftbenchmark.vcf.gz"
    )
    bedpath = (
        args.source_dir / "GRCh38_HG008-T-V0.5_somatic-stvar-clonal.draftbenchmark.bed"
    )
    for path in [vcf, bedpath]:
        verify_md5(path, args.source_dir / "checksums.md5")
    bed = pd.read_csv(bedpath, sep="\t", header=None, names=["chrom", "start", "end"])
    if (bed.start < 0).any() or (bed.end <= bed.start).any():
        raise ValueError("Malformed clonal BED")
    frame, excluded, audit = normalize(vcf, bed)
    audit["inputs"] = {p.name: fingerprint(p) for p in [vcf, bedpath]}
    audit["processing_version"] = 1
    args.out_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.out_dir / "normalized_variants.csv.gz", index=False)
    excluded.to_csv(args.out_dir / "exclusions.csv", index=False)
    (args.out_dir / "qc.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
