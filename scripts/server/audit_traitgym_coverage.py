#!/usr/bin/env python3
"""Audit the official TraitGym examples before any graph-restricted benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.server.prepare_sv_breakpoint_examples import canonical_chromosome, reference_nodes
from tasks.entex.mapping import map_loci
from tasks.entex.prepare import fingerprint


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"chrom", "pos", "ref", "alt", "label", "match_group"}
    if required - set(frame) or frame[list(required)].isna().any().any():
        raise ValueError("Missing official TraitGym fields")
    frame = frame.copy()
    positions = pd.to_numeric(frame.pos, errors="raise")
    if not np.isfinite(positions).all() or (positions < 1).any() or (positions % 1 != 0).any():
        raise ValueError("TraitGym positions must be positive 1-based integers")
    if not frame.label.isin([True, False, 0, 1]).all() or frame.label.nunique() != 2:
        raise ValueError("Require both supplied binary labels")
    if any(not frame[c].str.fullmatch("[ACGT]").all() for c in ["ref", "alt"]):
        raise ValueError("This audit supports the official single-base substitution datasets")
    if frame.duplicated(["chrom", "pos", "ref", "alt"]).any():
        raise ValueError("Duplicate variant identities")
    frame["chrom"] = frame.chrom.map(canonical_chromosome)
    if not frame.chrom.str.fullmatch(r"chr(?:[1-9]|1[0-9]|2[0-2]|X|Y)").all():
        raise ValueError("Unexpected chromosome naming")
    frame["start"] = positions.astype(np.int64) - 1
    frame["end"] = positions.astype(np.int64)
    frame["locus_id"] = frame.chrom + ":" + frame.end.astype(str)
    frame["variant_id"] = frame.locus_id + ":" + frame.ref + ":" + frame.alt
    frame["label"] = frame.label.astype(np.int8)
    return frame


def audit(frame: pd.DataFrame, nodes: pd.DataFrame, caches: dict[str, set[int]]) -> tuple:
    loci = frame[["locus_id", "chrom", "start", "end"]].drop_duplicates()
    overlaps = map_loci(loci, nodes)
    mapped = overlaps.groupby("locus_id").size()
    frame = frame.copy()
    frame["n_segments"] = frame.locus_id.map(mapped).fillna(0).astype(int)
    frame["graph_mapped"] = frame.n_segments > 0
    for name, ids in caches.items():
        complete = overlaps.assign(covered=overlaps.segid.isin(ids)).groupby("locus_id").covered.all()
        frame[name + "_covered"] = frame.locus_id.map(complete).eq(True)
    flags = ["graph_mapped", *[name + "_covered" for name in caches]]
    rows = []
    for group, subset in [("all", frame), *[(f"label={k}", v) for k, v in frame.groupby("label")],
                          *[(f"chrom={k}", v) for k, v in frame.groupby("chrom")]]:
        rows.append(dict(group=group, n=len(subset), positives=int(subset.label.sum()),
                         prevalence=float(subset.label.mean()), unique_loci=subset.locus_id.nunique(),
                         **{flag: float(subset[flag].mean()) for flag in flags}))
    return frame, overlaps, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", action="append", required=True, help="NAME=official parquet")
    ap.add_argument("--feature-cache", action="append", default=[], help="NAME=NPZ containing segid")
    ap.add_argument("--full-segments", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    graph = fingerprint(args.full_segments)
    expected = json.loads(Path("configs/entex_v1.json").read_text())["full_segments_sha256"]
    if graph["sha256"] != expected:
        raise ValueError("Not the exact manuscript graph")
    nodes = reference_nodes(args.full_segments, "GRCh38#0")
    caches, cache_sources = {}, {}
    for spec in args.feature_cache:
        name, path = spec.split("=", 1)
        with np.load(path, allow_pickle=False) as values:
            ids = values["segid"]
            if len(np.unique(ids)) != len(ids):
                raise ValueError("Duplicate cache identities")
            caches[name] = set(ids.astype(int))
        cache_sources[name] = fingerprint(Path(path))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    receipts = []
    for spec in args.dataset:
        name, path = spec.split("=", 1)
        if not name.replace("_", "").isalnum():
            raise ValueError("Use a simple dataset name")
        frame, overlaps, coverage = audit(normalize(pd.read_parquet(path)), nodes, caches)
        frame.to_parquet(args.out_dir / (name + "_examples.parquet"), index=False)
        overlaps.to_parquet(args.out_dir / (name + "_overlaps.parquet"), index=False)
        coverage.to_csv(args.out_dir / (name + "_coverage.csv"), index=False)
        receipts.append(dict(dataset=name, source=fingerprint(Path(path)), n=len(frame),
                             unique_loci=frame.locus_id.nunique(), multi_segment_rows=int((frame.n_segments > 1).sum()),
                             mixed_label_loci=int((frame.groupby("locus_id").label.nunique() > 1).sum()),
                             graph_coverage=float(frame.graph_mapped.mean()),
                             official_match_groups=frame.match_group.nunique()))
    (args.out_dir / "audit.json").write_text(json.dumps(dict(
        status="coverage_audit_complete", graph=graph, caches=cache_sources, datasets=receipts,
        coordinate_contract="Official TraitGym pos is 1-based; map [pos-1,pos) without nearest fallback",
        no_model_fitted=True, original_rows_retained=True,
        interpretation="Cache membership is potential input coverage, not proof of complete checkpoint/window embedding coverage",
        next_gate="Preserve official matched controls and chromosome protocol; audit class-dependent loss before fitting",
        representation_limit="Static segment T is a locus prior and does not distinguish ref versus alt alleles"
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
