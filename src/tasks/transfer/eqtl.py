"""Prepare matched GTEx high/low posterior loci; never call absent variants negative."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
import tarfile

import numpy as np
import pandas as pd

from tasks.entex.prepare import fingerprint


def gene_tss(gtf: Path) -> pd.DataFrame:
    rows = []
    with gtf.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9:
                raise ValueError("Malformed GTF")
            chrom, _, kind, start, end, _, strand, _, attributes = fields
            if kind != "gene":
                continue
            match = re.search(r'gene_id "([^";]+)"', attributes)
            if match is None or strand not in {"+", "-"}:
                raise ValueError("Missing gene ID or strand")
            rows.append(
                (match[1], chrom, int(start) - 1 if strand == "+" else int(end) - 1)
            )
    result = pd.DataFrame(rows, columns=["phenotype_id", "gene_chrom", "tss0"])
    if result.phenotype_id.duplicated().any():
        raise ValueError("Duplicate GTF gene identifiers")
    return result


def prepare_tissue(
    raw: pd.DataFrame, genes: pd.DataFrame, config: dict, tissue: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = {"phenotype_id", "variant_id", "pip", "af", "cs_id"}
    if not required.issubset(raw) or raw[list(required)].isna().any().any():
        raise ValueError("GTEx schema/missing required values")
    if not raw.pip.between(0, 1).all() or not raw.af.between(0, 1).all():
        raise ValueError("Invalid PIP or allele frequency")
    x = raw.copy()
    parts = x.variant_id.str.extract(
        r"^(chr(?:[1-9]|1[0-9]|2[0-2]|X|Y))_([1-9][0-9]*)_([ACGT]+)_([ACGT]+)_b38$"
    )
    if parts.isna().any().any():
        raise ValueError("Malformed or non-GRCh38 variant ID")
    x[["chrom", "pos", "ref", "alt"]] = parts
    x["start"] = x.pos.astype(np.int64) - 1
    x["end"] = x.start + 1
    x["locus_id"] = x.chrom + ":" + x.pos
    audit = dict(
        raw_rows=len(x), raw_unique_loci=int(x.locus_id.nunique()), tissue=tissue
    )
    x = x.loc[x.ref.str.len().eq(1) & x.alt.str.len().eq(1) & x.ref.ne(x.alt)].copy()
    audit["excluded_non_snv"] = len(raw) - len(x)
    # Collapse repeated credible-set membership before eligibility. Max PIP means
    # a variant strongly supported for any released gene is never a low control.
    maximum = x.groupby("locus_id").pip.transform("max")
    minimum = x.groupby("locus_id").pip.transform("min")
    conflicting = maximum.ge(config["positive_min_pip"]) & minimum.le(
        config["negative_max_pip"]
    )
    multiallelic = x.groupby("locus_id").variant_id.transform("nunique").gt(1)
    audit["excluded_conflicting_or_multiallelic_rows"] = int(
        (conflicting | multiallelic).sum()
    )
    x = x.loc[~(conflicting | multiallelic)].copy()
    x = x.sort_values("pip", ascending=False).drop_duplicates(
        ["phenotype_id", "variant_id"]
    )
    maximum = x.groupby("locus_id").pip.transform("max")
    positive = x.pip.ge(config["positive_min_pip"])
    negative = maximum.le(config["negative_max_pip"])
    audit["excluded_intermediate_pip_rows"] = int((~(positive | negative)).sum())
    x["label"] = positive.astype(int)
    x = x.loc[positive | negative].copy()
    x = x.merge(genes, on="phenotype_id", how="left", validate="many_to_one")
    if x.tss0.isna().any() or not x.chrom.eq(x.gene_chrom).all():
        raise ValueError(
            "Version-matched cis gene/TSS annotation missing or chromosome mismatch"
        )
    x["maf"] = np.minimum(x.af, 1 - x.af)
    x["distance_to_tss"] = (x.start - x.tss0).abs().astype(np.int64)
    x["maf_bin"] = pd.cut(x.maf, config["maf_bins"], include_lowest=True, labels=False)
    x["distance_bin"] = pd.cut(
        x.distance_to_tss, config["distance_bins_bp"], include_lowest=True, labels=False
    )
    audit["excluded_outside_matching_bins"] = int(
        x[["maf_bin", "distance_bin"]].isna().any(axis=1).sum()
    )
    x = x.dropna(subset=["maf_bin", "distance_bin"])
    audit["eligible_positive_rows"] = int(x.label.sum())
    audit["eligible_negative_rows"] = int(x.label.eq(0).sum())
    keys = ["phenotype_id", "chrom", "maf_bin", "distance_bin"]
    rng = np.random.default_rng(config["matching_seed"])
    x = x.sort_values(["phenotype_id", "variant_id"]).reset_index(drop=True)
    x["match_rank"] = rng.random(len(x))
    used, selected, pairs = set(), [], []
    for key, group in x.groupby(keys, sort=True):
        p = group.loc[group.label.eq(1)].sort_values("match_rank")
        n = group.loc[group.label.eq(0)].sort_values("match_rank")
        available = [i for i in n.index if x.at[i, "locus_id"] not in used]
        for i in p.index:
            if x.at[i, "locus_id"] in used:
                continue
            available = [j for j in available if x.at[j, "locus_id"] not in used]
            if not available:
                break
            j = available.pop(0)
            pair = len(pairs)
            used.update([x.at[i, "locus_id"], x.at[j, "locus_id"]])
            selected.extend([i, j])
            x.loc[[i, j], "match_pair"] = pair
            pairs.append(
                dict(
                    zip(keys, key),
                    match_pair=pair,
                    positive_locus=x.at[i, "locus_id"],
                    negative_locus=x.at[j, "locus_id"],
                )
            )
    if not selected:
        raise ValueError("No high/low posterior pairs satisfy prespecified matching")
    result = x.loc[selected].sort_values(["chrom", "start"]).reset_index(drop=True)
    result["task"], result["subtask"] = "eqtl", tissue
    result["match_pair"] = result.match_pair.astype(int)
    if result.locus_id.duplicated().any() or result.label.nunique() != 2:
        raise ValueError("Invalid matched locus universe")
    audit.update(
        selected_loci=len(result),
        positive_count=int(result.label.sum()),
        negative_count=int(result.label.eq(0).sum()),
        positive_prevalence=float(result.label.mean()),
        genes=int(result.phenotype_id.nunique()),
        chromosome_counts=result.chrom.value_counts().to_dict(),
        unmatched_eligible_rows=len(x) - len(result),
        definition=config,
        label_warning="Low-PIP released credible-set controls, not confirmed noncausal or all-tested negatives; LD unmatched",
    )
    return result, pd.DataFrame(pairs), audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--gtf", type=Path, required=True)
    ap.add_argument(
        "--config", type=Path, default=Path("configs/downstream_transfer_v1.json")
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())["eqtl"]
    genes = gene_tss(args.gtf)
    inputs = {p.name: fingerprint(p) for p in [args.archive, args.gtf, args.config]}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.archive) as archive:
        for tissue in cfg["tissues"]:
            member = f"{tissue}.v10.eQTLs.SuSiE_summary.parquet"
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Missing archive member: {member}")
            raw = pd.read_parquet(io.BytesIO(source.read()))
            result, pairs, audit = prepare_tissue(raw, genes, cfg, tissue)
            out = args.out_dir / tissue
            out.mkdir()
            result.to_parquet(out / "loci.parquet", index=False)
            pairs.to_parquet(out / "matching.parquet", index=False)
            audit.update(inputs=inputs, processing_version=1, archive_member=member)
            (out / "qc.json").write_text(json.dumps(audit, indent=2) + "\n")
            print(tissue, len(result), "loci", audit["genes"], "genes", flush=True)


if __name__ == "__main__":
    main()
