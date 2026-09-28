"""Prepare explicit INS/DEL/INV labels from existing HGSVC3 BED+6 annotations."""
from __future__ import annotations

import argparse
import hashlib
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tasks.entex.prepare import fingerprint

CLASSES = {"DEL": 0, "INS": 1, "INV": 2}
LENGTH_EDGES = [50, 100, 500, 1000, 5000, 10000, 50000, 100000, 1000000, np.inf]


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Verify this release's BED coordinates and ID contract; never coerce INV to DEL."""
    required = ["ID", "#CHROM", "POS", "END", "SVTYPE", "SVLEN"]
    if set(required)-set(frame) or frame[required].isna().any().any():
        raise ValueError("Missing SV-Pop BED+6 fields")
    data = frame[required].copy()
    if data.ID.duplicated().any() or not set(data.SVTYPE) <= set(CLASSES):
        raise ValueError("Duplicate event ID or unsupported SV class")
    for field in ["POS", "END", "SVLEN"]:
        value = pd.to_numeric(data[field], errors="raise")
        if not np.isfinite(value).all() or (value % 1 != 0).any():
            raise ValueError("Noninteger SV coordinates or length")
        data[field] = value.astype(np.int64)
    span = data.END-data.POS
    if ((data.POS < 0).any() or (data.SVLEN < 50).any()
            or not span.where(data.SVTYPE.ne("INS"), 1).eq(data.SVLEN.where(data.SVTYPE.ne("INS"), 1)).all()
            or not span.loc[data.SVTYPE.eq("INS")].eq(1).all()):
        raise ValueError("Inconsistent insertion point or affected interval")
    encoded = data.ID.str.extract(r"^(.*)-(\d+)-(INS|DEL|INV)-(\d+)(?:\.\d+)?(?:-.*)?$")
    if (encoded.isna().any().any() or not encoded[0].eq(data["#CHROM"]).all()
            or not encoded[2].eq(data.SVTYPE).all()
            or not encoded[3].astype(int).eq(data.SVLEN).all()):
        raise ValueError("Event ID disagrees with BED coordinate/type/length contract")
    # IDs are provenance identifiers, not authoritative coordinates. This
    # release includes 181 INS/DEL IDs retaining POS rather than POS+1; their
    # BED starts agree with the padded-allele VCF POS. Keep and audit them.
    data["id_position_minus_bed_start"] = encoded[1].astype(int)-data.POS
    if not data.id_position_minus_bed_start.isin([0, 1]).all():
        raise ValueError("Unrecognized ID coordinate discrepancy; inspect the release")
    data = data.rename(columns={"ID": "variant_id", "#CHROM": "chrom", "POS": "start", "END": "event_end",
                                "SVTYPE": "svtype", "SVLEN": "svlen"})
    # One reference anchor for every class avoids exposing class-specific
    # endpoint conventions or a span directly through C/S/T feature assembly.
    data["end"] = data.start+1
    data["locus_id"] = data.chrom+":"+data.start.astype(str)
    data["label"] = data.svtype.map(CLASSES).astype(np.int8)
    data["length_bin"] = pd.cut(data.svlen, bins=LENGTH_EDGES, right=False, labels=False).astype(int)
    return data.sort_values(["chrom", "start", "variant_id"]).reset_index(drop=True)


def matched_subset(data: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal class counts in fixed chromosome/length bins, without replacement."""
    records, selected = [], []
    for (chrom, length_bin), group in data.groupby(["chrom", "length_bin"], sort=True):
        counts = group.svtype.value_counts().reindex(CLASSES, fill_value=0)
        n = int(counts.min())
        records.append(dict(chrom=chrom, length_bin=int(length_bin), n_per_class=n,
                            **{f"available_{c}": int(counts[c]) for c in CLASSES}))
        for kind in CLASSES:
            part = group.loc[group.svtype.eq(kind)].copy()
            part["selection_hash"] = part.variant_id.map(lambda v: hashlib.sha256(f"{seed}:{v}".encode()).hexdigest())
            selected.append(part.sort_values(["selection_hash", "variant_id"]).head(n))
    if not selected or not sum(len(p) for p in selected):
        raise ValueError("No three-class common support in predefined length/chromosome bins")
    return pd.concat(selected).sort_values(["chrom", "start", "variant_id"]).reset_index(drop=True), pd.DataFrame(records)


def validate_padded_vcf(events: pd.DataFrame, path: Path) -> dict:
    """Independently verify first-affected-base anchors against existing alleles."""
    expected = events.loc[events.svtype.isin(['INS', 'DEL'])].set_index('variant_id')
    seen = set()
    with gzip.open(path, 'rt') as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            fields = line.rstrip().split('\t', 8)
            chrom, pos, variant, ref, alt = fields[:5]
            if variant not in expected.index or variant in seen:
                raise ValueError('VCF event universe differs or duplicates IDs')
            row = expected.loc[variant]
            delta = len(alt)-len(ref)
            if (chrom != row.chrom or int(pos) != row.start or ref[0] != alt[0]
                    or abs(delta) != row.svlen or ('INS' if delta > 0 else 'DEL') != row.svtype
                    or (len(ref) != 1 if row.svtype == 'INS' else len(alt) != 1)):
                raise ValueError(f'Padded VCF allele/coordinate mismatch: {variant}')
            seen.add(variant)
    if seen != set(expected.index):
        raise ValueError('VCF omits annotation events')
    return dict(status='pass', n_events=len(seen), source_vcf=fingerprint(path),
        contract='BED start0 equals the numeric 1-based VCF anchor position because the first affected base follows the common padding base; table POS is authoritative, ID positions are provenance only.')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--matching-seed", type=int, default=20260927)
    args = ap.parse_args()
    frames = [pd.read_csv(p, sep="\t") for p in args.annotations]
    data = normalize(pd.concat(frames, ignore_index=True))
    cfg = json.loads(Path("configs/entex_v1.json").read_text())
    folds = json.loads(Path(cfg["manuscript_config"]).read_text())["rotating_chromosome_folds"]
    chromosomes = {c for f in folds for c in f["test"]}
    eligible = data.loc[data.chrom.isin(chromosomes)].copy()
    excluded = data.loc[~data.chrom.isin(chromosomes)].assign(exclusion_reason="chromosome_not_in_manuscript_folds")
    matched, strata = matched_subset(eligible, args.matching_seed)
    support = []
    for cohort, examples in [("natural", eligible), ("length_matched", matched)]:
        for fold in folds:
            for split in ["train", "validation", "test"]:
                chrs = (chromosomes-set(fold["test"])-set(fold["validation"])) if split == "train" else set(fold[split])
                count = examples.loc[examples.chrom.isin(chrs)].svtype.value_counts().reindex(CLASSES, fill_value=0)
                for kind in CLASSES:
                    support.append(dict(cohort=cohort, fold=fold["name"], split=split, svtype=kind, n=int(count[kind])))
    support = pd.DataFrame(support)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in [("all_source_events", data), ("natural_events", eligible), ("length_matched_events", matched), ("excluded_events", excluded)]:
        frame.to_parquet(args.out_dir/(name+".parquet"), index=False)
    eligible[["locus_id", "chrom", "start", "end"]].drop_duplicates().to_parquet(args.out_dir/"loci.parquet", index=False)
    matched[["locus_id", "chrom", "start", "end"]].drop_duplicates().to_parquet(args.out_dir/"length_matched_loci.parquet", index=False)
    strata.to_csv(args.out_dir/"matching_strata.csv", index=False)
    support.to_csv(args.out_dir/"fold_support.csv", index=False)
    cohorts = {}
    for name, frame in [("natural", eligible), ("length_matched", matched)]:
        mixed = frame.groupby("locus_id").label.nunique().gt(1)
        cohorts[name] = dict(n=len(frame), n_loci=frame.locus_id.nunique(), classes=frame.svtype.value_counts().to_dict(),
            class_prevalence=frame.svtype.value_counts(normalize=True).to_dict(), mixed_label_anchors=int(mixed.sum()))
    audit = dict(status="prepared", sources=[fingerprint(p) for p in args.annotations], raw_rows=len(data),
        cohorts=cohorts, id_position_minus_bed_start=data.id_position_minus_bed_start.value_counts().to_dict(), n_excluded=len(excluded), exclusion_chromosomes=excluded.chrom.value_counts().to_dict(),
        matching_seed=args.matching_seed, length_bins=[*LENGTH_EDGES[:-1], "infinity"],
        class_definition=CLASSES, representation="One first-affected-base reference anchor for every class; SV length is a separate explicit control only.",
        matched_interpretation="Exploratory equal-class chromosome/length-bin common-support task; prevalence is not natural. No performance-based subset selection.",
        mapping="Pending existing interval mapper; no nearest fallback", no_model_fitted=True,
        coordinate_reference="https://github.com/EichlerLab/svpop#sv-pop-pipeline")
    (args.out_dir/"qc.json").write_text(json.dumps(audit, indent=2)+'\n')
    if (support.n < 10).any():
        raise ValueError("Fewer than ten examples of a class in a declared split; inspect support without changing bins")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
