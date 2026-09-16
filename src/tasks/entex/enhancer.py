"""Prepare tissue-specific active versus explicitly repressed V2 dELS tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

from tasks.entex.prepare import fingerprint
from tasks.entex.registry import read_registry


def prepare_tissue(
    active: pd.DataFrame, repressed: pd.DataFrame, registry: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    pieces = []
    qc = {}
    for name, frame, label in [("active", active, 1), ("repressed", repressed, 0)]:
        qc[name + "_raw_rows"] = len(frame)
        if not frame.state.str.startswith(name + ".").all():
            raise ValueError("Archive state disagrees with source file")
        merged = frame.merge(
            registry, on="ccre_id", how="left", validate="many_to_one", indicator=True
        )
        qc[name + "_unmatched_rows"] = int(merged["_merge"].eq("left_only").sum())
        if qc[name + "_unmatched_rows"]:
            raise ValueError("Registry coverage must be complete before P1 preparation")
        dels = merged.ccre_class.str.split(",").str[0].eq("dELS")
        distal = merged.state.str.split(".").str[1].eq("distal")
        qc[name + "_excluded_non_dels"] = int((~dels).sum())
        qc[name + "_excluded_non_distal_dels"] = int((dels & ~distal).sum())
        selected = merged.loc[dels & distal].drop(columns="_merge").copy()
        selected["label"] = label
        pieces.append(selected)
    frame = pd.concat(pieces, ignore_index=True)
    frame["locus_id"] = (
        frame.chrom + ":" + frame.start.astype(str) + "-" + frame.end.astype(str)
    )
    conflict = frame.groupby("locus_id").label.nunique().gt(1)
    qc["conflicting_loci_excluded"] = int(conflict.sum())
    frame = frame.loc[~frame.locus_id.isin(conflict[conflict].index)].copy()
    frame = frame.groupby(
        ["locus_id", "chrom", "start", "end", "tissue", "label"], as_index=False
    ).agg(
        ccre_id=("ccre_id", lambda x: "|".join(sorted(set(x)))),
        source_states=("state", lambda x: "|".join(sorted(set(x)))),
        ccre_class=("ccre_class", "first"),
    )
    qc.update(
        n_loci=len(frame),
        positive_count=int(frame.label.sum()),
        negative_count=int(frame.label.eq(0).sum()),
    )
    return frame, qc


def select_tissues(counts: pd.DataFrame, number: int, minimum_class: int) -> list[str]:
    eligible = counts.loc[
        (counts.positive_count >= minimum_class)
        & (counts.negative_count >= minimum_class)
    ].copy()
    eligible["smaller_class"] = eligible[["positive_count", "negative_count"]].min(
        axis=1
    )
    return (
        eligible.sort_values(
            ["smaller_class", "n_loci", "tissue"], ascending=[False, False, True]
        )
        .head(number)
        .tissue.tolist()
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--registry",
        type=Path,
        default=Path("data/encode/legacy_v2/ENCFF924IMH.bed.gz"),
    )
    ap.add_argument("--cache-root", type=Path, default=Path("data/entex/v1"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/entex/v1/p1"))
    ap.add_argument("--config", type=Path, default=Path("configs/entex_v1.json"))
    args = ap.parse_args()
    config = json.loads(args.config.read_text())["p1"]
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)
    if fingerprint(args.registry)["sha256"] != config["registry_sha256"]:
        raise ValueError("P1 registry checksum mismatch")
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "definitions.json").write_text(json.dumps(config, indent=2) + "\n")
    registry = read_registry(args.registry)
    tissues = set()
    for state in ["active", "repressed"]:
        for batch in pq.ParquetFile(
            args.cache_root / f"{state}.combined_set.txt.zip.parquet"
        ).iter_batches(columns=["tissue"]):
            tissues.update(batch.column(0).to_pylist())
    counts = []
    for tissue in sorted(tissues):
        frames = [
            pd.read_parquet(
                args.cache_root / f"{state}.combined_set.txt.zip.parquet",
                filters=[("tissue", "=", tissue)],
            )
            for state in ["active", "repressed"]
        ]
        loci, qc = prepare_tissue(*frames, registry)
        qc["tissue"] = tissue
        counts.append(qc)
        loci["task"] = "p1"
        loci["subtask"] = tissue
        loci.to_parquet(args.out_dir / f"{tissue}_loci.parquet", index=False)
    counts = pd.DataFrame(counts)
    chosen = select_tissues(counts, config["n_tissues"], config["minimum_per_class"])
    if len(chosen) < 3:
        raise ValueError("Fewer than three sufficiently powered tissues")
    counts["selected"] = counts.tissue.isin(chosen)
    counts["positive_prevalence"] = counts.positive_count / counts.n_loci
    counts.to_csv(args.out_dir / "tissue_counts.csv", index=False)
    audit = dict(
        task="p1",
        registry=fingerprint(args.registry),
        sources={
            state: fingerprint(
                args.cache_root / f"{state}.combined_set.txt.zip.parquet"
            )
            for state in ["active", "repressed"]
        },
        definitions=config,
        selected_tissues=chosen,
        biological_fitting="not_run",
    )
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(counts.loc[counts.selected].to_string(index=False))


if __name__ == "__main__":
    main()
