"""Prepare measured COSIGT locus quality; absent outcomes remain unmeasured."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tasks.entex.prepare import fingerprint


def prepare_quality(outcomes: pd.DataFrame, regions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Keep one observed donor/locus, with continuous targets and explicit support.

    This is conditional quality among publisher-evaluable haplotypes. It is not
    variant concordance, callability, or an unseen-donor PangenomeFM benchmark.
    """
    required = ["sample", "region", "gene_name", "QV_1_pred", "QV_2_pred", "QV_sum_pred",
                "QV_1_best", "QV_2_best", "QV_sum_best", "avg_error_rate_pred"]
    if set(required) - set(outcomes) or set(["chrom", "start", "end", "gene_name"]) - set(regions):
        raise ValueError("Missing published outcome or region columns")
    if outcomes.empty or regions.empty or outcomes[required].isna().any().any():
        raise ValueError("Empty or missing measured outcomes")
    loci = regions.copy()
    coords = loci[["start", "end"]].to_numpy(float)
    if (not np.isfinite(coords).all() or (coords != np.floor(coords)).any()
            or (coords[:, 0] < 0).any() or (coords[:, 1] <= coords[:, 0]).any()
            or not loci.chrom.str.fullmatch(r"chr(?:[1-9]|1[0-9]|2[0-2]|X|Y)").all()):
        raise ValueError("Malformed GRCh38 BED coordinates")
    loci[["start", "end"]] = loci[["start", "end"]].astype(np.int64)
    loci["locus_id"] = loci.chrom + "_" + loci.start.astype(str) + "_" + loci.end.astype(str)
    if loci.locus_id.duplicated().any() or loci.gene_name.isna().any():
        raise ValueError("Duplicate or missing region identity")
    frame = outcomes.drop_duplicates().copy()
    if frame.duplicated(["sample", "region"]).any():
        raise ValueError("Conflicting measurements for one sample and region")
    expected_gene = frame.region.map(loci.set_index("locus_id").gene_name)
    if expected_gene.isna().any() or not expected_gene.eq(frame.gene_name).all():
        raise ValueError("Outcome region/gene does not match supplied BED")
    numeric = required[3:]
    values = frame[numeric].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (frame.QV_sum_best <= 0).any():
        raise ValueError("Invalid quality/error measurements")
    if (frame.avg_error_rate_pred > 1).any():
        raise ValueError("Invalid error rate")
    for kind in ["pred", "best"]:
        if not np.allclose(frame[f"QV_sum_{kind}"], frame[f"QV_1_{kind}"] + frame[f"QV_2_{kind}"], rtol=0, atol=1e-6):
            raise ValueError("Haplotype QV sum does not replay")
    # Published rounded QVs can exceed the best-match sum by <0.001.
    # Retain, count, and never silently clip these values.
    excess = frame.QV_sum_pred - frame.QV_sum_best
    if (excess > 0.001).any():
        raise ValueError("Prediction exceeds best-available QV beyond published rounding")
    frame["qv_fraction"] = frame.QV_sum_pred / frame.QV_sum_best
    frame["qv_mean_pred"] = frame.QV_sum_pred / 2
    frame["locus_id"] = frame.region
    frame = frame.merge(loci[["locus_id", "chrom", "start", "end"]], on="locus_id", validate="many_to_one")
    summary = frame.groupby("locus_id").agg(
        n_observed_donors=("sample", "nunique"),
        mean_qv_fraction=("qv_fraction", "mean"),
        median_qv_fraction=("qv_fraction", "median"),
        mean_qv_pred=("qv_mean_pred", "mean"),
        mean_error_rate_pred=("avg_error_rate_pred", "mean"),
    )
    loci = loci.merge(summary, on="locus_id", how="left", validate="one_to_one")
    loci["n_observed_donors"] = loci.n_observed_donors.fillna(0).astype(int)
    universe = pd.MultiIndex.from_product([sorted(frame["sample"].unique()), loci.locus_id], names=["sample", "locus_id"])
    observed = pd.MultiIndex.from_frame(frame[["sample", "locus_id"]])
    unobserved = universe.difference(observed).to_frame(index=False)
    unobserved["status"] = "no_published_evaluable_outcome; not a negative"
    qc = dict(raw_rows=len(outcomes), exact_duplicate_rows_removed=len(outcomes)-len(frame),
        measured_sample_locus_pairs=len(frame), n_samples=frame["sample"].nunique(), n_loci=len(loci),
        loci_with_outcomes=int(loci.n_observed_donors.gt(0).sum()),
        cartesian_sample_locus_pairs=len(universe), unobserved_pairs=len(unobserved),
        observed_pair_fraction=len(frame)/len(universe), missing_required_values=0,
        qv_fraction_above_one=int((frame.qv_fraction > 1).sum()),
        max_qv_sum_excess=float(excess.max()), rounding_clipped=False,
        min_observed_donors=int(loci.n_observed_donors.min()), max_observed_donors=int(loci.n_observed_donors.max()),
        chromosomes=loci.chrom.value_counts().sort_index().to_dict(),
        estimand="continuous measured locus quality conditional on published evaluable donors",
        missingness="Cartesian support is descriptive; unobserved pairs are not asserted attempted, callable, failed, or negative",
        no_performance_results=True)
    return frame, loci, unobserved, qc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outcomes", type=Path, required=True)
    ap.add_argument("--regions", type=Path, required=True)
    ap.add_argument("--source-receipt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    receipt = json.loads(args.source_receipt.read_text())
    expected = {Path(x["path"]).name: x["sha256"] for x in receipt["files"]}
    for path in [args.outcomes, args.regions]:
        if expected.get(path.name) != fingerprint(path)["sha256"]:
            raise ValueError("Source differs from pinned download receipt")
    frame, loci, unobserved, qc = prepare_quality(pd.read_csv(args.outcomes, sep="\t"),
        pd.read_csv(args.regions, sep="\t", header=None, names=["chrom", "start", "end", "gene_name"]))
    qc.update(source_receipt=receipt, implementation=fingerprint(Path(__file__)),
              outcomes=fingerprint(args.outcomes), regions=fingerprint(args.regions))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(args.out_dir / "measured_outcomes.parquet", index=False)
    loci.to_parquet(args.out_dir / "loci.parquet", index=False)
    loci.to_csv(args.out_dir / "loci.csv", index=False)
    unobserved.to_parquet(args.out_dir / "unobserved_pairs.parquet", index=False)
    (args.out_dir / "qc.json").write_text(json.dumps(qc, indent=2) + "\n")


if __name__ == "__main__":
    main()
