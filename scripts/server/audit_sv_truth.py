#!/usr/bin/env python3
"""Audit and normalize a versioned phased-SV VCF without loading it into RAM.

The audit intentionally distinguishes release provenance from biological truth:
it reports phasing, missingness, FILTER state, inferred class/length/frequency,
and writes a normalized table that later graph/SV experiments can join against.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, TextIO


LENGTH_BINS = [0, 50, 100, 500, 1_000, 10_000, 100_000, 1_000_000]
AF_BINS = [0.0, 0.001, 0.01, 0.05, 0.5, 1.0000001]


def open_text(path: Path) -> TextIO:
    if path.suffix in {".gz", ".bgz"}:
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def parse_info(text: str) -> dict[str, str | bool]:
    values: dict[str, str | bool] = {}
    if text == ".":
        return values
    for item in text.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            values[key] = value
        elif item:
            values[item] = True
    return values


def first_number(value: object) -> float | None:
    if value in {None, "", "."}:
        return None
    for item in str(value).split(","):
        try:
            result = float(item)
        except ValueError:
            continue
        if math.isfinite(result):
            return result
    return None


def infer_svtype(ref: str, alt: str, info: dict[str, object]) -> str:
    if info.get("SVTYPE") not in {None, "", "."}:
        return str(info["SVTYPE"]).upper()
    symbolic = alt.strip("<>").split(":", 1)[0]
    if alt.startswith("<") and alt.endswith(">"):
        return symbolic.upper()
    delta = len(alt.split(",", 1)[0]) - len(ref)
    if delta >= 50:
        return "INS"
    if delta <= -50:
        return "DEL"
    return "OTHER"


def infer_end_and_length(
    pos: int, ref: str, alt: str, svtype: str, info: dict[str, object]
) -> tuple[int, int | None]:
    raw_length = first_number(info.get("SVLEN"))
    raw_end = first_number(info.get("END"))
    if raw_length is not None:
        length = abs(int(raw_length))
    elif raw_end is not None:
        length = abs(int(raw_end) - pos)
    elif svtype in {"INS", "DEL", "DUP", "INV", "CNV"}:
        length = abs(len(alt.split(",", 1)[0]) - len(ref))
    else:
        length = None
    end = int(raw_end) if raw_end is not None else pos + max(len(ref), length or 1) - 1
    return end, length


def label_bin(value: float | int | None, edges: list[float | int], prefix: str) -> str:
    if value is None or not math.isfinite(float(value)):
        return "missing"
    for lower, upper in zip(edges[:-1], edges[1:]):
        if lower <= value < upper:
            return f"{prefix}[{lower:g},{upper:g})"
    return f"{prefix}>={edges[-1]:g}"


def parse_genotypes(
    format_text: str,
    sample_values: list[str],
) -> tuple[list[dict[str, int]], int, int, int]:
    keys = format_text.split(":") if format_text not in {"", "."} else []
    gt_index = keys.index("GT") if "GT" in keys else None
    per_sample: list[dict[str, int]] = []
    ac = 0
    an = 0
    phased_called = 0
    for value in sample_values:
        stats = {
            "records": 1,
            "called": 0,
            "phased": 0,
            "alt_carrier": 0,
            "heterozygous": 0,
            "homozygous_alt": 0,
            "alt_alleles": 0,
            "called_alleles": 0,
        }
        if gt_index is None:
            per_sample.append(stats)
            continue
        fields = value.split(":")
        gt = fields[gt_index] if gt_index < len(fields) else "."
        separator = "|" if "|" in gt else "/"
        alleles = gt.split(separator)
        called = [item for item in alleles if item not in {"", "."}]
        if not called:
            per_sample.append(stats)
            continue
        alt = sum(int(item) > 0 for item in called if item.isdigit())
        stats["called"] = 1
        stats["called_alleles"] = len(called)
        stats["alt_alleles"] = alt
        stats["alt_carrier"] = int(alt > 0)
        stats["heterozygous"] = int(0 < alt < len(called))
        stats["homozygous_alt"] = int(alt == len(called) and alt > 0)
        if separator == "|" and len(called) == len(alleles):
            stats["phased"] = 1
            phased_called += 1
        ac += alt
        an += len(called)
        per_sample.append(stats)
    return per_sample, ac, an, phased_called


def sha256sum(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_vcf(
    vcf: Path,
    out_dir: Path,
    *,
    dataset: str,
    release: str,
    write_variants: bool = True,
    compute_sha256: bool = True,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_names: list[str] = []
    sample_totals: list[Counter] = []
    header_lines = 0
    malformed = 0
    variants = 0
    pass_variants = 0
    phased_records = 0
    missing_length = 0
    missing_frequency = 0
    class_counts: Counter = Counter()
    chromosome_counts: Counter = Counter()
    filter_counts: Counter = Counter()
    stratified: Counter = Counter()

    normalized_path = out_dir / "normalized_variants.csv.gz"
    normalized_handle = gzip.open(normalized_path, "wt", newline="") if write_variants else None
    fieldnames = [
        "dataset", "release", "chrom", "pos", "end", "variant_id", "ref_length",
        "alt_length", "svtype", "svlen", "length_bin", "allele_frequency", "af_bin",
        "filter", "n_called_samples", "n_phased_samples", "n_alt_carriers",
        "alt_allele_count", "called_allele_count",
    ]
    writer = csv.DictWriter(normalized_handle, fieldnames=fieldnames) if normalized_handle else None
    if writer:
        writer.writeheader()

    try:
        with open_text(vcf) as handle:
            for line in handle:
                if line.startswith("##"):
                    header_lines += 1
                    continue
                if line.startswith("#CHROM"):
                    columns = line.rstrip("\n").split("\t")
                    sample_names = columns[9:]
                    sample_totals = [Counter() for _ in sample_names]
                    continue
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 8:
                    malformed += 1
                    continue
                try:
                    chrom, pos_text, variant_id, ref, alt, _, filter_value, info_text = fields[:8]
                    pos = int(pos_text)
                except (TypeError, ValueError):
                    malformed += 1
                    continue
                info = parse_info(info_text)
                svtype = infer_svtype(ref, alt, info)
                end, svlen = infer_end_and_length(pos, ref, alt, svtype, info)
                per_sample, ac, an, n_phased = parse_genotypes(
                    fields[8] if len(fields) > 8 else ".",
                    fields[9:] if len(fields) > 9 else [],
                )
                for total, current in zip(sample_totals, per_sample):
                    total.update(current)
                info_af = first_number(info.get("AF"))
                af = info_af if info_af is not None else (ac / an if an else None)
                if svlen is None:
                    missing_length += 1
                if af is None:
                    missing_frequency += 1
                length_bin = label_bin(svlen, LENGTH_BINS, "bp")
                af_bin = label_bin(af, AF_BINS, "af")
                variants += 1
                pass_variants += int(filter_value in {"PASS", "."})
                phased_records += int(n_phased > 0)
                class_counts[svtype] += 1
                chromosome_counts[chrom] += 1
                filter_counts[filter_value] += 1
                stratified[(chrom, svtype, length_bin, af_bin)] += 1
                if writer:
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "release": release,
                            "chrom": chrom,
                            "pos": pos,
                            "end": end,
                            "variant_id": variant_id,
                            "ref_length": len(ref),
                            "alt_length": len(alt.split(",", 1)[0]),
                            "svtype": svtype,
                            "svlen": "" if svlen is None else svlen,
                            "length_bin": length_bin,
                            "allele_frequency": "" if af is None else f"{af:.8g}",
                            "af_bin": af_bin,
                            "filter": filter_value,
                            "n_called_samples": sum(row["called"] for row in per_sample),
                            "n_phased_samples": n_phased,
                            "n_alt_carriers": sum(row["alt_carrier"] for row in per_sample),
                            "alt_allele_count": ac,
                            "called_allele_count": an,
                        }
                    )
    finally:
        if normalized_handle:
            normalized_handle.close()

    with (out_dir / "variant_strata.csv").open("w", newline="") as handle:
        writer2 = csv.writer(handle)
        writer2.writerow(["chrom", "svtype", "length_bin", "af_bin", "n_variants"])
        for key, count in sorted(stratified.items()):
            writer2.writerow([*key, count])

    with (out_dir / "sample_phasing_summary.csv").open("w", newline="") as handle:
        writer3 = csv.writer(handle)
        writer3.writerow(
            [
                "sample", "records", "called_records", "called_fraction", "phased_records",
                "phased_fraction_among_called", "alt_carrier_records", "heterozygous_records",
                "homozygous_alt_records", "alt_alleles", "called_alleles",
            ]
        )
        for sample, total in zip(sample_names, sample_totals):
            called = total["called"]
            records = total["records"]
            writer3.writerow(
                [
                    sample,
                    records,
                    called,
                    called / records if records else "",
                    total["phased"],
                    total["phased"] / called if called else "",
                    total["alt_carrier"],
                    total["heterozygous"],
                    total["homozygous_alt"],
                    total["alt_alleles"],
                    total["called_alleles"],
                ]
            )

    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": dataset,
        "release": release,
        "vcf": str(vcf.resolve()),
        "vcf_bytes": vcf.stat().st_size,
        "vcf_sha256": sha256sum(vcf) if compute_sha256 else None,
        "header_lines": header_lines,
        "sample_count": len(sample_names),
        "samples": sample_names,
        "variant_count": variants,
        "pass_or_unfiltered_variant_count": pass_variants,
        "records_with_any_phased_sample": phased_records,
        "records_missing_length": missing_length,
        "records_missing_frequency": missing_frequency,
        "malformed_record_count": malformed,
        "svtype_counts": dict(sorted(class_counts.items())),
        "chromosome_counts": dict(sorted(chromosome_counts.items())),
        "filter_counts": dict(sorted(filter_counts.items())),
        "normalized_variants": str(normalized_path) if write_variants else None,
        "success_criteria": {
            "has_variants": variants > 0,
            "has_samples": len(sample_names) > 0,
            "has_phased_genotypes": phased_records > 0,
            "no_malformed_records": malformed == 0,
        },
    }
    payload["status"] = (
        "validated_phased_sv_candidate"
        if all(payload["success_criteria"].values())
        else "audit_warning"
    )
    (out_dir / "sv_truth_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--no-variant-table", action="store_true")
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()
    if not args.vcf.is_file():
        parser.error(f"VCF does not exist: {args.vcf}")
    payload = audit_vcf(
        args.vcf,
        args.out_dir,
        dataset=args.dataset,
        release=args.release,
        write_variants=not args.no_variant_table,
        compute_sha256=not args.skip_sha256,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "validated_phased_sv_candidate" else 1


if __name__ == "__main__":
    raise SystemExit(main())
