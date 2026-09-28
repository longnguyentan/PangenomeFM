#!/usr/bin/env python3
"""Audit whether each TraitGym base survives the original NT raw-node sampling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tasks.entex.prepare import fingerprint


def raw_base_retained(offset: np.ndarray, length: np.ndarray, maximum_bases: int) -> np.ndarray:
    """Mirror balanced_sequence's prefix/N/suffix policy at exact base offsets."""
    offset, length = np.asarray(offset), np.asarray(length)
    if (maximum_bases < 3 or offset.shape != length.shape or not np.isfinite(offset).all()
            or not np.isfinite(length).all() or (offset % 1 != 0).any() or (length % 1 != 0).any()
            or (offset < 0).any() or (offset >= length).any()):
        raise ValueError("Invalid base offset or segment length")
    left = (maximum_bases - 1) // 2
    right = maximum_bases - 1 - left
    return (length <= maximum_bases) | (offset < left) | (offset >= length - right)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapping-dir", type=Path, required=True)
    ap.add_argument("--full-segments", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(Path("configs/entex_v1.json").read_text())
    graph = fingerprint(args.full_segments)
    mapping = json.loads((args.mapping_dir / "audit.json").read_text())
    if graph["sha256"] != cfg["full_segments_sha256"] or mapping["graph"]["sha256"] != graph["sha256"]:
        raise ValueError("Graph is not the exact mapped manuscript resource")
    nodes = pd.read_csv(args.full_segments, usecols=["id", "SO", "LN"]).rename(columns={"id": "segid"})
    rows, summaries, sources = [], [], []
    for name in ["complex_traits", "mendelian_traits"]:
        a, b = [args.mapping_dir / (name + suffix) for suffix in ["_examples.parquet", "_overlaps.parquet"]]
        examples, overlaps = pd.read_parquet(a), pd.read_parquet(b)
        if overlaps.locus_id.duplicated().any():
            raise ValueError("Require one containing segment per locus")
        f = examples.merge(overlaps, on="locus_id", how="left", validate="many_to_one").merge(
            nodes, on="segid", how="left", validate="many_to_one")
        if f[["segid", "SO", "LN"]].isna().any().any():
            raise ValueError("Unmapped variant or missing node metadata")
        f["offset"] = f.start - f.SO
        f["variant_base_in_raw_nt_input"] = raw_base_retained(f.offset, f.LN, cfg["nt_max_bases"])
        f["dataset"] = name
        rows.append(f[["dataset", "variant_id", "chrom", "label", "segid", "LN", "offset", "variant_base_in_raw_nt_input"]])
        for group, part in [("all", f), *[(f"label={label}", g) for label, g in f.groupby("label")]]:
            summaries.append(dict(dataset=name, group=group, n=len(part),
                raw_base_retained=int(part.variant_base_in_raw_nt_input.sum()),
                raw_base_excluded=int((~part.variant_base_in_raw_nt_input).sum()),
                raw_retention_fraction=float(part.variant_base_in_raw_nt_input.mean()),
                median_segment_length=float(part.LN.median())))
        sources.extend([fingerprint(a), fingerprint(b)])
    args.out_dir.mkdir(parents=True, exist_ok=False)
    pd.concat(rows).to_parquet(args.out_dir / "variant_visibility.parquet", index=False)
    pd.DataFrame(summaries).to_csv(args.out_dir / "visibility.csv", index=False)
    (args.out_dir / "audit.json").write_text(json.dumps(dict(status="complete", graph=graph, sources=sources,
        implementation=fingerprint(Path(__file__)), maximum_raw_bases=cfg["nt_max_bases"],
        definition="Existing balanced prefix+N+suffix raw-node sampling, before tokenization",
        limitation="Retention is an upper bound on tokenizer-visible bases; tokenizer truncation can remove further bases. Even retained SNPs use reference-only segment embeddings.",
        predictions_used=False, no_model_fitted=True), indent=2) + "\n")


if __name__ == "__main__":
    main()
