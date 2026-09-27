#!/usr/bin/env python3
"""Check the released HGSVC3 GRCh38 inversion labels and exact endpoint coverage."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.server.prepare_sv_breakpoint_examples import reference_nodes, map_coordinate
from tasks.entex.prepare import fingerprint


def inversion_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ID", "#CHROM", "POS", "END", "SVTYPE", "SVLEN"}
    if required - set(frame) or frame[list(required)].isna().any().any():
        raise ValueError("Missing inversion annotation fields")
    if frame.ID.duplicated().any() or not frame.SVTYPE.eq("INV").all():
        raise ValueError("Expected unique inversion events")
    frame = frame.copy()
    for key in ["POS", "END", "SVLEN"]:
        values = pd.to_numeric(frame[key], errors="raise")
        if not np.isfinite(values).all() or (values % 1 != 0).any():
            raise ValueError("Noninteger inversion coordinates")
        frame[key] = values.astype(np.int64)
    if (frame.POS < 0).any() or not (frame.END - frame.POS).eq(frame.SVLEN).all() or (frame.SVLEN < 50).any():
        raise ValueError("Inconsistent BED interval/length")
    # HGSVC's annotation table is the SV-Pop BED+6 format. IDs encode the
    # corresponding 1-based first affected base; check this release convention.
    encoded = frame.ID.str.extract(r"^(chr[^-]+)-(\d+)-INV-(\d+)(?:-.*)?$")
    if (encoded.isna().any().any() or not encoded[0].eq(frame["#CHROM"]).all()
            or not encoded[1].astype(int).eq(frame.POS + 1).all()
            or not encoded[2].astype(int).eq(frame.SVLEN).all()):
        raise ValueError("Inversion IDs disagree with the BED coordinate contract")
    return frame.rename(columns={"#CHROM": "chrom", "POS": "start0", "END": "end0", "ID": "variant_id"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotation", type=Path, required=True)
    ap.add_argument("--full-segments", type=Path, required=True)
    ap.add_argument("--sequence-cache", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    graph = fingerprint(args.full_segments)
    config = json.loads(Path("configs/entex_v1.json").read_text())
    if graph["sha256"] != config["full_segments_sha256"]:
        raise ValueError("Not the manuscript graph")
    data = inversion_intervals(pd.read_csv(args.annotation, sep="\t"))
    nodes = reference_nodes(args.full_segments, "GRCh38#0")
    index = {chrom: (g.SO.to_numpy(), g.end0.to_numpy(), g.segid.to_numpy()) for chrom, g in nodes.groupby("chrom")}
    with np.load(args.sequence_cache, allow_pickle=False) as cache:
        available = set(cache["segid"].astype(int))
    for key, coordinate in [("start", data.start0), ("last_base", data.end0 - 1)]:
        mapped = [map_coordinate(int(pos), *index[chrom], maximum_nearest_distance=0)[0]
                  if chrom in index else None for pos, chrom in zip(coordinate, data.chrom)]
        data[key + "_segid"] = pd.array(mapped, dtype="Int64")
    data["both_mapped"] = data[["start_segid", "last_base_segid"]].notna().all(axis=1)
    data["both_sequence_covered"] = data.start_segid.isin(available) & data.last_base_segid.isin(available)
    folds = json.loads(Path(config["manuscript_config"]).read_text())["rotating_chromosome_folds"]
    rows = []
    for fold in folds:
        for split in ["train", "validation", "test"]:
            chrs = set(fold[split]) if split != "train" else set().union(*(set(f["test"]) for f in folds)) - set(fold["test"]) - set(fold["validation"])
            part = data.loc[data.chrom.isin(chrs)]
            rows.append(dict(fold=fold["name"], split=split, n_inv=len(part),
                             n_mapped=int(part.both_mapped.sum()), n_sequence_covered=int(part.both_sequence_covered.sum())))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    data.to_parquet(args.out_dir / "inversion_events.parquet", index=False)
    pd.DataFrame(rows).to_csv(args.out_dir / "fold_support.csv", index=False)
    data.groupby("chrom").agg(n=("variant_id", "size"), mapped=("both_mapped", "sum"),
                             sequence_covered=("both_sequence_covered", "sum")).to_csv(args.out_dir / "chromosome_counts.csv")
    (args.out_dir / "audit.json").write_text(json.dumps(dict(status="label_mapping_audit_complete",
        annotation=fingerprint(args.annotation), graph=graph, sequence_cache=fingerprint(args.sequence_cache),
        n_events=len(data), n_mapped=int(data.both_mapped.sum()), n_sequence_covered=int(data.both_sequence_covered.sum()),
        coordinate_contract="SV-Pop BED+6 [POS,END); map first and last affected bases, with no nearest fallback",
        coordinate_source="https://github.com/EichlerLab/svpop#sv-pop-pipeline",
        no_model_fitted=True, not_binary_insdel="INV must never be relabeled as DEL by the historical binary probe",
        remaining=["Independently defined multiclass probe and disjoint event labels",
                   "Event-overlap handling and length-matched controls", "Frozen checkpoint coverage; inversion class support limits precision"]
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
