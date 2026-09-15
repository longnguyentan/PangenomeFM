"""Locus-preserving adapter over the manuscript cCRE overlap implementation."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tasks.ccre.encoding import _accumulate_segment_class_overlaps
from scripts.server.prepare_sv_breakpoint_examples import reference_nodes
from tasks.entex.prepare import fingerprint


def map_loci(loci: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    if loci.locus_id.duplicated().any():
        raise ValueError("Map unique loci, not repeated tissue/donor measurements")
    if nodes.segid.duplicated().any():
        raise ValueError("Duplicate reference segment IDs")
    rows = []
    for chrom, intervals in loci.groupby("chrom", sort=True):
        segments = nodes.loc[nodes.chrom == chrom].sort_values("SO")
        # The existing accumulator accepts arbitrary class keys. Using locus IDs
        # retains identity before the original mapper's dominant-class collapse.
        intervals = intervals.rename(columns={"locus_id": "cCRE_class"}).sort_values(
            "start"
        )
        overlaps = _accumulate_segment_class_overlaps(intervals, segments)
        for segid, values in overlaps.items():
            rows.extend((locus, segid, bp) for locus, bp in values.items())
    return pd.DataFrame(rows, columns=["locus_id", "segid", "overlap_bp"])


def aggregate_features(
    loci: pd.DataFrame,
    overlaps: pd.DataFrame,
    segids: np.ndarray,
    values: np.ndarray,
    method: str,
) -> np.ndarray:
    if method not in {"mean", "length_weighted"}:
        raise ValueError("Unknown pooling method")
    if len(set(segids)) != len(segids):
        raise ValueError("Duplicate feature IDs")
    positions = pd.Series(np.arange(len(segids)), index=segids)
    edges = overlaps.loc[overlaps.locus_id.isin(loci.locus_id)].copy()
    indices = edges.segid.map(positions)
    if indices.isna().any():
        raise ValueError("Missing segment features; partial-locus pooling is forbidden")
    target = pd.Series(np.arange(len(loci)), index=loci.locus_id)
    row = edges.locus_id.map(target).to_numpy(int)
    weights = (
        edges.overlap_bp.to_numpy(float)
        if method == "length_weighted"
        else np.ones(len(edges))
    )
    total = np.bincount(row, weights=weights, minlength=len(loci))
    if (total == 0).any():
        raise ValueError("Unmapped locus in feature universe")
    result = np.zeros((len(loci), values.shape[1]), dtype=np.float64)
    np.add.at(result, row, values[indices.to_numpy(int)] * weights[:, None])
    result /= total[:, None]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite aggregated features")
    return result.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loci", type=Path, required=True)
    ap.add_argument("--full-segments", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--minimum-mapping", type=float, default=0.95)
    args = ap.parse_args()
    config = json.loads(Path("configs/entex_v1.json").read_text())
    if fingerprint(args.full_segments)["sha256"] != config["full_segments_sha256"]:
        raise ValueError("Graph differs from exact manuscript HPRC R2 resource")
    loci = pd.read_parquet(args.loci)
    nodes = reference_nodes(args.full_segments, "GRCh38#0")
    overlaps = map_loci(loci, nodes)
    counts = overlaps.groupby("locus_id").size().reindex(loci.locus_id, fill_value=0)
    audit = dict(
        source_loci=fingerprint(args.loci),
        source_graph=fingerprint(args.full_segments),
        total_loci=len(loci),
        mapped_loci=int((counts > 0).sum()),
        unmapped_loci=int((counts == 0).sum()),
        fraction_mapped=float((counts > 0).mean()),
        one_segment=int((counts == 1).sum()),
        multi_segment=int((counts > 1).sum()),
        multi_segment_fraction=float((counts > 1).mean()),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlaps.to_parquet(args.out_dir / "overlaps.parquet", index=False)
    loci.loc[~loci.locus_id.isin(overlaps.locus_id)].to_parquet(
        args.out_dir / "unmapped.parquet", index=False
    )
    # This is a target inventory for the existing feature-cache builder, not biological segment labels.
    targets = nodes.loc[nodes.segid.isin(overlaps.segid)].copy()
    targets["ccre_label"] = 0
    targets.to_csv(args.out_dir / "feature_targets.csv.gz", index=False)
    (args.out_dir / "mapping_qc.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["fraction_mapped"] < args.minimum_mapping:
        raise ValueError(
            f"Mapping below gate; inspect {args.out_dir / 'mapping_qc.json'}"
        )


if __name__ == "__main__":
    main()
