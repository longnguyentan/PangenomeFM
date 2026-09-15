#!/usr/bin/env python3
"""Inventory exact EN-TEx prerequisites and audit local enhancer registry coverage."""

from pathlib import Path
import argparse
import json
import pandas as pd
import pyarrow.parquet as pq
from tasks.entex.analyze import complexity_labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("results/entex/v1/qc"))
    ap.add_argument("--data-root", type=Path, default=Path("server_workspace/data"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    resources = {
        "full_segments": args.data_root / "processed/hprc_r2_sv/full_segments.csv.gz",
        "full_links": args.data_root / "processed/hprc_r2_sv/full_links.csv.gz",
        "manuscript_C_K": args.data_root
        / "processed/hprc_r2_ccre_screen_v4_features.npz",
        "manuscript_NT": Path(
            "server_workspace/results/frozen_sequence_fm_cache_20260815/hprc_target_union_sequence_fm.npz"
        ),
    }
    pd.DataFrame(
        [dict(resource=k, path=str(v), exists=v.exists()) for k, v in resources.items()]
    ).to_csv(args.out_dir / "resource_inventory.csv", index=False)
    registry = pd.read_csv(
        "data/encode/GRCh38-human-cCREs.bed",
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "ccre", "rdhs", "class"],
    )
    ids = set(registry.ccre) | set(registry.rdhs)
    rows = []
    for state in ["active", "repressed"]:
        path = Path(f"data/entex/v1/{state}.combined_set.txt.zip.parquet")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=100000):
            d = batch.to_pandas()
            d["id_found"] = d.ccre_id.isin(ids)
            for tissue, g in d.groupby("tissue"):
                rows.append(
                    dict(
                        state=state,
                        tissue=tissue,
                        rows=len(g),
                        id_found=int(g.id_found.sum()),
                    )
                )
    counts = (
        pd.DataFrame(rows)
        .groupby(["state", "tissue"], as_index=False)[["rows", "id_found"]]
        .sum()
    )
    counts["id_coverage"] = counts.id_found / counts.rows
    counts.to_csv(
        args.out_dir / "enhancer_registry_coverage_by_tissue.csv", index=False
    )
    config = json.loads(Path("configs/entex_v1.json").read_text())
    loci = complexity_labels(
        pd.read_parquet("data/entex/v1/p0_loci.parquet"), Path(config["complexity"])
    )
    counts = loci.groupby("complexity").label.agg(
        n="size", positive_count="sum", positive_prevalence="mean"
    )
    counts.to_csv(args.out_dir / "p0_complexity_counts.csv")
    (args.out_dir / "status.json").write_text(
        json.dumps(
            dict(
                p0_preparation="complete",
                p0_mapping="blocked_missing_exact_graph",
                p0_full_experiment="not_run",
                p0b_gain="not_run",
                p1="not_started_pending_P0",
                p2="not_started_pending_P0",
                p3="not_started",
                p4="not_started",
                enhancer_warning="Local SCREEN v4 ID coverage is incomplete; do not silently drop unmatched IDs or equate distal with dELS",
            ),
            indent=2,
        )
        + "\n"
    )
    print(counts.to_string())


if __name__ == "__main__":
    main()
