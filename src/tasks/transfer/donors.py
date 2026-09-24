"""Donor/haplotype-stratified chromosome-held-out SV scores with alias-aware overlap."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.server.run_ccre_frozen_probe_fold import binary_metrics
from scripts.server.run_ccre_frozen_probe_matrix import build_jobs
from tasks.entex.analyze import BASE, FULL, estimate
from tasks.entex.prepare import fingerprint


def overlap_table(
    samples: list[str], membership: pd.DataFrame, aliases: dict
) -> pd.DataFrame:
    def canonical(sample):
        return aliases.get(sample, sample)
    if len({canonical(s) for s in samples}) != len(samples):
        raise ValueError(
            "VCF contains duplicate biological donors under verified aliases"
        )
    hprc = {canonical(s) for s in membership.loc[membership.hprc_r2, "sample"]}
    return pd.DataFrame(
        [
            dict(
                sample=s,
                canonical_sample=canonical(s),
                in_hprc_pretraining_cohort=canonical(s) in hprc,
            )
            for s in samples
        ]
    )


def parse_gt(text: str) -> tuple[int, int]:
    """Return phased per-haplotype ALT presence; missing/unphased remain unknown."""
    alleles = text.replace("/", "|").split("|")
    if len(alleles) > 2 or any(x not in {".", "0", "1"} for x in alleles):
        raise ValueError("Expected biallelic haploid/diploid GT")
    if "/" in text:
        return -1, -1
    values = [-1 if x == "." else int(x) for x in alleles]
    return tuple((values + [-1, -1])[:2])


def genotype_cache(
    vcf: Path, examples: pd.DataFrame, out: Path
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    identity = dict(vcf=fingerprint(vcf), variant_ids=examples.variant_id.tolist())
    if out.exists():
        with np.load(out, allow_pickle=False) as stored:
            audit = json.loads(str(stored["audit"].item()))
            if audit["identity"] != identity:
                raise ValueError("Genotype cache provenance mismatch")
            return (
                stored["carriers"].copy(),
                stored["haplotypes"].copy(),
                stored["samples"].tolist(),
                audit,
            )
    lookup = {value: i for i, value in enumerate(examples.variant_id)}
    if len(lookup) != len(examples):
        raise ValueError("Duplicate normalized variant IDs")
    seen, samples, count = set(), [], 0
    unphased = 0
    with gzip.open(vcf, "rt") as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                samples = line.rstrip().split("\t")[9:]
                if len(samples) != len(set(samples)):
                    raise ValueError("Duplicate VCF samples")
                carriers = np.zeros((len(examples), len(samples)), dtype=bool)
                haplotypes = np.full(
                    (len(examples), len(samples), 2), -1, dtype=np.int8
                )
            elif line.startswith("#"):
                continue
            else:
                count += 1
                fields = line.rstrip().split("\t")
                if fields[2] not in lookup:
                    continue
                if not samples or fields[2] in seen or len(fields) != 9 + len(samples):
                    raise ValueError("VCF genotype schema or duplicate record")
                i = lookup[fields[2]]
                if (
                    fields[0] != examples.iloc[i].chrom
                    or int(fields[1]) != examples.iloc[i].pos
                ):
                    raise ValueError("VCF ID/coordinate mismatch")
                seen.add(fields[2])
                gt_index = fields[8].split(":").index("GT")
                for j, text in enumerate(fields[9:]):
                    gt = text.split(":")[gt_index]
                    unphased += int("/" in gt)
                    carriers[i, j] = "1" in gt.replace("/", "|").split("|")
                    haplotypes[i, j] = parse_gt(gt)
    if len(seen) != len(examples):
        raise ValueError("Original examples missing from genotype VCF")
    audit = dict(
        identity=identity,
        raw_rows=count,
        matched_variants=len(seen),
        samples=len(samples),
        unphased_genotypes=unphased,
        missing_haplotype_genotypes=int((haplotypes < 0).sum()),
        positive_carrier_records=int(carriers.sum()),
        missing_policy="Unknown alleles never count as reference or alternate; unphased ALT carriers enter donor-only analysis",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        carriers=carriers,
        haplotypes=haplotypes,
        samples=np.array(samples),
        audit=json.dumps(audit),
    )
    return carriers, haplotypes, samples, audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ["vcf", "examples", "membership", "probe-root", "out-dir", "cache"]:
        ap.add_argument("--" + key, type=Path, required=True)
    args = ap.parse_args()
    examples = pd.read_csv(args.examples).reset_index(drop=True)
    carriers, haplotypes, samples, audit = genotype_cache(
        args.vcf, examples, args.cache
    )
    aliases = json.loads(Path("configs/verified_sample_aliases.json").read_text())
    donors = overlap_table(samples, pd.read_csv(args.membership), aliases["aliases"])
    by_id = pd.Series(np.arange(len(examples)), index=examples.example_id)
    rows, seen = [], set()
    for path in sorted(args.probe_root.glob("fold_*/seed_*/*/test_predictions.csv.gz")):
        p = pd.read_csv(path)
        p["feature_set"] = p.feature_set.str.removesuffix("_pair")
        p = p.loc[p.feature_set.isin([BASE, FULL])]
        key = tuple(p[["fold", "seed", "closure"]].drop_duplicates().iloc[0])
        if key in seen or p.duplicated(["example_id", "feature_set"]).any():
            raise ValueError("Duplicate run or prediction")
        seen.add(key)
        if not p.groupby("example_id").feature_set.nunique().eq(2).all():
            raise ValueError("Unpaired prediction universe")
        for feature, frame in p.groupby("feature_set"):
            index = frame.example_id.map(by_id).to_numpy(int)
            if not np.array_equal(
                frame.y_true, examples.iloc[index].binary_svtype_label
            ):
                raise ValueError("Genotype/prediction label mismatch")
            for j, donor in donors.iterrows():
                for unit, selected in [
                    ("donor", carriers[index, j]),
                    ("haplotype_1", haplotypes[index, j, 0] == 1),
                    ("haplotype_2", haplotypes[index, j, 1] == 1),
                ]:
                    group = frame.loc[selected]
                    if group.empty:
                        continue
                    rows.append(
                        dict(
                            fold=key[0],
                            seed=key[1],
                            context=key[2],
                            feature_set=feature,
                            sample=donor["sample"],
                            canonical_sample=donor.canonical_sample,
                            in_hprc=donor.in_hprc_pretraining_cohort,
                            unit=unit,
                            **binary_metrics(
                                group.y_true.to_numpy(),
                                group.p_calibrated.to_numpy(),
                                float(group.threshold.iloc[0]),
                            ),
                        )
                    )
        print(key, "complete", flush=True)
    jobs = build_jobs(
        json.loads(Path("configs/server_full_multicohort_20260806.json").read_text())
    )
    if seen != {(j.fold, j.seed, j.closure) for j in jobs}:
        raise ValueError("Incomplete fold/seed/context matrix")
    metrics = pd.DataFrame(rows)
    keys = ["sample", "canonical_sample", "in_hprc", "unit", "fold", "seed", "context"]
    wide = metrics.pivot(index=keys, columns="feature_set", values="auprc")
    gains = (wide[FULL] - wide[BASE]).rename("gain").reset_index()
    summary = []
    for (sample, unit, context, in_hprc), group in gains.groupby(
        ["sample", "unit", "context", "in_hprc"]
    ):
        summary.append(
            dict(
                sample=sample,
                unit=unit,
                context=context,
                in_hprc=in_hprc,
                **estimate(group, "gain", 10000, 20260924),
            )
        )
    macro = (
        gains.loc[~gains.in_hprc & gains.unit.eq("donor")]
        .groupby(["context", "fold", "seed"])
        .gain.mean()
        .reset_index()
    )
    macro_summary = [
        dict(context=c, **estimate(g, "gain", 10000, 20260924))
        for c, g in macro.groupby("context")
    ]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(args.out_dir / "per_run.csv", index=False)
    pd.DataFrame(summary).to_csv(
        args.out_dir / "per_donor_haplotype_gains.csv", index=False
    )
    pd.DataFrame(macro_summary).to_csv(
        args.out_dir / "nonoverlap_donor_macro_gains.csv", index=False
    )
    donors.to_csv(args.out_dir / "donor_overlap.csv", index=False)
    audit.pop("identity")
    audit.update(
        status="complete",
        vcf=fingerprint(args.vcf),
        examples=fingerprint(args.examples),
        aliases=aliases,
        overlap_donors=donors.loc[donors.in_hprc_pretraining_cohort, "sample"].tolist(),
        encoder_updated=False,
        probes_refitted=False,
        interpretation="Donor/haplotype stratification of chromosome-held-out predictions; downstream HGSVC probe training included these donors on other chromosomes. NOT donor-held-out probe validation or personal embeddings.",
        shared_variants="Donor outcomes are correlated through shared variants; macro uncertainty resamples chromosome folds and seeds, never treats donors as independent replicates",
    )
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
