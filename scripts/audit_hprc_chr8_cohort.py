#!/usr/bin/env python3
"""Validate exact-pair donor artifacts before cohort assembly and evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_complete_marker(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        raise FileNotFoundError(f"Missing extraction marker: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def _audit_embeddings(
    path: Path, expected_ids: pd.Index, expected_dim: int
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing embeddings: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"ids", "embeddings"}:
            raise ValueError(f"Unexpected arrays in {path}: {payload.files}")
        ids = pd.Index(payload["ids"].astype(str))
        embeddings = payload["embeddings"]
        if embeddings.shape != (len(expected_ids), expected_dim):
            raise ValueError(
                f"{path} has shape {embeddings.shape}; expected "
                f"({len(expected_ids)}, {expected_dim})."
            )
        if ids.has_duplicates:
            raise ValueError(f"{path} contains duplicate embedding IDs.")
        if set(ids) != set(expected_ids):
            missing = len(set(expected_ids) - set(ids))
            extra = len(set(ids) - set(expected_ids))
            raise ValueError(
                f"{path} ID mismatch: {missing} missing and {extra} extra."
            )
        if not np.isfinite(embeddings).all():
            raise ValueError(f"{path} contains non-finite embeddings.")
        return {
            "n_embeddings": int(len(ids)),
            "embedding_dim": int(embeddings.shape[1]),
            "all_finite": True,
            "ids_exact": True,
        }


def _anchor_methods(pairs: pd.DataFrame, source: Path) -> tuple[pd.Series, str]:
    if "anchor_method" in pairs:
        return pairs["anchor_method"].astype(str), "reported"
    required = {"shared_nodes", "shared_node_bp"}
    if not required.issubset(pairs):
        raise ValueError(
            f"{source} lacks anchor_method and the shared-node evidence "
            "needed for legacy-schema inference."
        )
    valid = (
        pairs["shared_nodes"].fillna(0).astype(float).ge(2)
        & pairs["shared_node_bp"].fillna(0).astype(float).ge(500)
    )
    if not valid.all():
        raise ValueError(
            f"{source} lacks anchor_method and contains rows below the "
            "shared-node anchor thresholds."
        )
    return (
        pd.Series("shared_graph_nodes", index=pairs.index),
        "inferred_from_shared_nodes_ge_2_and_shared_bp_ge_500",
    )


def audit_cohort(
    *,
    cohort_path: str | Path,
    path_manifest: str | Path,
    path_batch_dir: str | Path,
    chunks_root: str | Path,
    result_root: str | Path,
    require_shared_node_anchors: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cohort = pd.read_csv(cohort_path, sep="\t")
    if cohort["donor_id"].duplicated().any():
        raise ValueError("Cohort metadata contains duplicate donor IDs.")
    manifest = pd.read_csv(path_manifest, sep="\t")
    donors = cohort["donor_id"].astype(str).tolist()
    if set(manifest["donor_id"].astype(str)) != set(donors):
        raise ValueError("Path manifest donors do not exactly match cohort metadata.")

    path_batch_dir = Path(path_batch_dir)
    chunks_root = Path(chunks_root)
    result_root = Path(result_root)
    rows: list[dict[str, Any]] = []
    for donor in donors:
        donor_manifest = manifest[manifest["donor_id"].astype(str) == donor]
        path_list = path_batch_dir / f"{donor}.chr8.paths.txt"
        paths = [
            line.strip()
            for line in path_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected_fragments = int(donor_manifest["n_path_fragments"].sum())
        if len(paths) != expected_fragments or len(set(paths)) != len(paths):
            raise ValueError(
                f"{donor} path list has {len(paths)} unique/total entries; "
                f"manifest expects {expected_fragments}."
            )
        chunks_dir = chunks_root / f"{donor}.chr8.chunks2m"
        chunks = sorted(chunks_dir.glob("chunk_*.gfa.gz"))
        marker = _read_complete_marker(chunks_dir / ".complete")
        if int(marker.get("chunks", "-1")) != len(chunks):
            raise ValueError(f"{donor} extraction marker/chunk count mismatch.")
        if int(marker.get("paths", "-1")) != len(paths):
            raise ValueError(f"{donor} extraction marker/path count mismatch.")

        donor_out = result_root / "donors" / donor
        pair_path = donor_out / "paired_windows.csv.gz"
        if not pair_path.is_file():
            raise FileNotFoundError(f"Missing paired windows: {pair_path}")
        pairs = pd.read_csv(pair_path, compression="infer")
        if len(pairs) == 0:
            raise ValueError(f"{donor} has zero paired windows.")
        if set(pairs["donor_id"].astype(str)) != {donor}:
            raise ValueError(f"{donor} pair table contains another donor.")
        expected_ids = pd.Index(
            pd.concat(
                [pairs["h1_embedding_id"], pairs["h2_embedding_id"]],
                ignore_index=True,
            ).astype(str)
        )
        if expected_ids.has_duplicates:
            raise ValueError(f"{donor} pair table contains duplicate embedding IDs.")
        anchor_methods, anchor_provenance = _anchor_methods(pairs, pair_path)
        anchor_counts = {
            str(key): int(value)
            for key, value in anchor_methods.value_counts().items()
        }
        if require_shared_node_anchors and set(anchor_counts) != {
            "shared_graph_nodes"
        }:
            raise ValueError(
                f"{donor} has unexpected anchor methods: {anchor_counts}."
            )

        graph = _audit_embeddings(
            donor_out / "graphgenomefm_embeddings.npz", expected_ids, 48
        )
        sequence = _audit_embeddings(
            donor_out / "nucleotide_transformer_embeddings.npz", expected_ids, 512
        )
        dataset_summary_path = donor_out / "dataset_summary.json"
        dataset_summary = json.loads(
            dataset_summary_path.read_text(encoding="utf-8")
        )
        rows.append(
            {
                "donor_id": donor,
                "population": cohort.loc[
                    cohort["donor_id"].astype(str) == donor, "population"
                ].iloc[0],
                "super_population": cohort.loc[
                    cohort["donor_id"].astype(str) == donor,
                    "super_population",
                ].iloc[0],
                "n_path_fragments": len(paths),
                "n_chunks": len(chunks),
                "n_labeled_windows": int(
                    dataset_summary["n_labeled_haplotype_windows"]
                ),
                "n_pairs": int(len(pairs)),
                "anchor_methods": json.dumps(anchor_counts, sort_keys=True),
                "anchor_method_provenance": anchor_provenance,
                "n_graph_embeddings": graph["n_embeddings"],
                "graph_embedding_dim": graph["embedding_dim"],
                "n_sequence_embeddings": sequence["n_embeddings"],
                "sequence_embedding_dim": sequence["embedding_dim"],
                "ids_exact": True,
                "all_embeddings_finite": True,
            }
        )

    audit = pd.DataFrame(rows)
    anchor_method_counts: Counter[str] = Counter()
    for donor_counts in audit["anchor_methods"].map(json.loads):
        anchor_method_counts.update(donor_counts)
    summary = {
        "status": "passed",
        "n_donors": int(len(audit)),
        "n_path_fragments": int(audit["n_path_fragments"].sum()),
        "n_chunks": int(audit["n_chunks"].sum()),
        "n_labeled_windows": int(audit["n_labeled_windows"].sum()),
        "n_pairs": int(audit["n_pairs"].sum()),
        "n_embeddings_per_model": int(2 * audit["n_pairs"].sum()),
        "all_embedding_ids_exact": bool(audit["ids_exact"].all()),
        "all_embeddings_finite": bool(audit["all_embeddings_finite"].all()),
        "anchor_method_counts": {
            str(key): int(value)
            for key, value in sorted(anchor_method_counts.items())
        },
    }
    return audit, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--path-manifest", required=True)
    parser.add_argument("--path-batch-dir", required=True)
    parser.add_argument("--chunks-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--out-tsv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--require-shared-node-anchors", action="store_true")
    args = parser.parse_args()
    audit, summary = audit_cohort(
        cohort_path=args.cohort,
        path_manifest=args.path_manifest,
        path_batch_dir=args.path_batch_dir,
        chunks_root=args.chunks_root,
        result_root=args.result_root,
        require_shared_node_anchors=args.require_shared_node_anchors,
    )
    out_tsv = Path(args.out_tsv)
    out_json = Path(args.out_json)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_tsv, sep="\t", index=False)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
