"""Prespecified P0 sensitivity datasets and exact exposure matching."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from tasks.entex.prepare import aggregate_ccre, fingerprint


def match_exposure(loci: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select equal classes within chromosome × exact measurement count.

    Stable hash ranking makes selection independent of row order, probe seed,
    graph features and predictions. Apply after joint feature coverage at fitting.
    """
    keys = ["chrom", "n_informative_experiments"]
    if loci.locus_id.duplicated().any() or not loci.label.isin([0, 1]).all():
        raise ValueError("Matching requires unique binary-labeled loci")
    if loci[keys].isna().any().any() or (loci.n_informative_experiments < 1).any():
        raise ValueError("Missing or nonpositive exposure")
    selected = []
    rows = []
    for (chrom, exposure), group in loci.groupby(keys, sort=True):
        counts = group.label.value_counts()
        n = min(int(counts.get(0, 0)), int(counts.get(1, 0)))
        rows.append(
            dict(
                chrom=chrom,
                n_informative_experiments=int(exposure),
                available_negative=int(counts.get(0, 0)),
                available_positive=int(counts.get(1, 0)),
                selected_per_class=n,
            )
        )
        for label in [0, 1]:
            candidates = group.loc[group.label.eq(label)].copy()
            candidates["_rank"] = candidates.locus_id.map(
                lambda locus: hashlib.sha256(f"{seed}:{locus}".encode()).hexdigest()
            )
            selected.append(
                candidates.sort_values(["_rank", "locus_id"])
                .head(n)
                .drop(columns="_rank")
            )
    matched = (
        pd.concat(selected).sort_values("locus_id").reset_index(drop=True)
        if selected
        else loci.iloc[:0].copy()
    )
    if matched.empty:
        raise ValueError("No overlapping exposure support between classes")
    return matched, pd.DataFrame(rows)


def prepare(cache: Path, primary: Path, out: Path, config: dict) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite sensitivity preparation: {out}")
    out.mkdir(parents=True, exist_ok=True)
    settings = config["sensitivities"]
    # Save definitions before examining assay-specific outcomes.
    (out / "definitions.json").write_text(json.dumps(settings, indent=2) + "\n")
    chroms = set()
    for batch in pq.ParquetFile(cache).iter_batches(columns=["chr"]):
        chroms.update(batch.column(0).to_pylist())
    rows = []
    source = fingerprint(cache)
    for name, assay in settings["assays"].items():
        parts = []
        raw_rows = 0
        for chrom in sorted(chroms):
            measurements = pd.read_parquet(
                cache, filters=[("chr", "=", chrom), ("assay", "=", assay)]
            )
            raw_rows += len(measurements)
            if not measurements.empty:
                parts.append(aggregate_ccre(measurements, require_both_classes=False))
        if not parts:
            raise ValueError(f"No informative measurements for exact assay {assay}")
        loci = pd.concat(parts, ignore_index=True)
        if loci.label.nunique() != 2:
            raise ValueError(f"{name} lacks both classes")
        loci["sensitivity"] = name
        loci.to_parquet(out / f"{name}_loci.parquet", index=False)
        audit = dict(
            sensitivity=name,
            assay=assay,
            source=source,
            raw_rows=raw_rows,
            n_loci=len(loci),
            positive_count=int(loci.label.sum()),
            negative_count=int(loci.label.eq(0).sum()),
            prevalence=float(loci.label.mean()),
            definition="Union supplied AS calls after filtering exact assay; negatives measured in that assay",
            chromosome_counts=loci.chrom.value_counts().to_dict(),
            mapping_status="not_run",
            feature_coverage=None,
            biological_fitting="not_run",
        )
        (out / f"{name}_qc.json").write_text(json.dumps(audit, indent=2) + "\n")
        rows.append(
            {k: v for k, v in audit.items() if k not in ["source", "chromosome_counts"]}
        )
    primary_loci = pd.read_parquet(primary)
    matched, balance = match_exposure(primary_loci, settings["matching_seed"])
    matched.to_parquet(out / "exposure_matched_preview.parquet", index=False)
    balance.to_csv(out / "exposure_balance_preview.csv", index=False)
    audit = dict(
        sensitivity="exposure_matched",
        source=fingerprint(primary),
        n_loci=len(matched),
        positive_count=int(matched.label.sum()),
        negative_count=int(matched.label.eq(0).sum()),
        prevalence=float(matched.label.mean()),
        n_excluded=len(primary_loci) - len(matched),
        matching_seed=settings["matching_seed"],
        keys=settings["matching_keys"],
        status="pre-feature QC preview; rematch joint feature universe separately for each run",
        biological_fitting="not_run",
    )
    (out / "exposure_matched_qc.json").write_text(json.dumps(audit, indent=2) + "\n")
    rows.append({k: v for k, v in audit.items() if k not in ["source", "keys"]})
    pd.DataFrame(rows).to_csv(out / "preparation_summary.csv", index=False)
    return {"datasets": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache", type=Path, default=Path("data/entex/v1/cCREs_default_AS.tsv.parquet")
    )
    ap.add_argument(
        "--primary", type=Path, default=Path("data/entex/v1/p0_loci.parquet")
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/entex/v1/sensitivities"))
    ap.add_argument("--config", type=Path, default=Path("configs/entex_v1.json"))
    args = ap.parse_args()
    print(
        json.dumps(
            prepare(
                args.cache,
                args.primary,
                args.out_dir,
                json.loads(args.config.read_text()),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
