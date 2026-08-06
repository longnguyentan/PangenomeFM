#!/usr/bin/env python3
"""Build label-blind embedding controls and an H1/H2-swapped analysis table.

The controls preserve the embedding archive schema used by the downstream
held-out-donor model:

* ``random_moment_matched`` replaces every vector with independent Gaussian
  noise having the pretrained archive's per-dimension mean and variance.
* ``shuffled_within_donor`` preserves the exact empirical vector distribution
  for each donor while breaking its window assignment.
* ``shuffled_within_local_block`` permutes vectors within donor, haplotype,
  contig, and fixed-width coordinate blocks.  This is the most conservative
  control because it preserves broad local genomic context.

The swapped table reverses H1/H2 identities, signed delta features, and the
target while leaving symmetric context features unchanged.  It is deliberately
compact and contains only columns consumed by the hierarchical experiment and
the coverage audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEQUENCE_DELTAS = [
    "delta_gc_fraction",
    "delta_cpg_density",
    "delta_base_entropy",
]
ANNOTATION_DELTAS = [
    "delta_repeat_fraction",
    "delta_hmmflagger_fraction",
    "delta_hmmflagger_dup_fraction",
    "delta_hmmflagger_col_fraction",
    "delta_hmmflagger_err_fraction",
    "delta_hmmflagger_nnn_fraction",
    "delta_copy_number_proxy",
    "delta_graph_mappability_proxy",
]
SYMMETRIC_CONTEXT = ["sv_context_score", "sv_length_delta_bp", "node_jaccard"]


def _parse_embedding_id(value: str) -> tuple[str, str, str, int]:
    track, start = str(value).rsplit(":", 1)
    donor, haplotype, contig = track.split("#", 2)
    return donor, haplotype, contig, int(start)


def _save_archive(
    path: Path,
    ids: np.ndarray,
    embeddings: np.ndarray,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.asarray(ids, dtype=str),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _permutation_within_groups(
    groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    order = np.arange(len(groups))
    frame = pd.DataFrame({"group": groups, "row": order})
    for _, rows in frame.groupby("group", sort=False)["row"]:
        positions = rows.to_numpy()
        order[positions] = rng.permutation(positions)
    return order


def build_controls(
    *,
    pairs_path: str | Path,
    graph_embeddings_path: str | Path,
    out_dir: str | Path,
    seed: int = 42,
    block_bp: int = 1_000_000,
) -> dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = np.load(graph_embeddings_path)
    ids = payload["ids"].astype(str)
    embeddings = payload["embeddings"].astype(np.float32)
    parsed = [_parse_embedding_id(value) for value in ids]
    donors = np.asarray([row[0] for row in parsed], dtype=str)
    local_blocks = np.asarray(
        [
            f"{donor}#{haplotype}#{contig}:{start // block_bp}"
            for donor, haplotype, contig, start in parsed
        ],
        dtype=str,
    )
    outputs: dict[str, str] = {}

    rng = np.random.default_rng(seed)
    means = embeddings.mean(axis=0)
    standard_deviations = embeddings.std(axis=0)
    random_embeddings = rng.normal(
        loc=means,
        scale=np.maximum(standard_deviations, 1e-6),
        size=embeddings.shape,
    ).astype(np.float32)
    random_path = out_dir / "graph_random_moment_matched.npz"
    _save_archive(
        random_path,
        ids,
        random_embeddings,
        {
            "control": "random_moment_matched",
            "source": str(graph_embeddings_path),
            "seed": seed,
            "n_embeddings": int(len(ids)),
            "embedding_dim": int(embeddings.shape[1]),
            "label_independent": True,
        },
    )
    outputs["random_moment_matched"] = str(random_path)

    donor_order = _permutation_within_groups(donors, np.random.default_rng(seed + 1))
    donor_path = out_dir / "graph_shuffled_within_donor.npz"
    _save_archive(
        donor_path,
        ids,
        embeddings[donor_order],
        {
            "control": "shuffled_within_donor",
            "source": str(graph_embeddings_path),
            "seed": seed + 1,
            "n_embeddings": int(len(ids)),
            "embedding_dim": int(embeddings.shape[1]),
            "label_independent": True,
        },
    )
    outputs["shuffled_within_donor"] = str(donor_path)

    block_order = _permutation_within_groups(
        local_blocks, np.random.default_rng(seed + 2)
    )
    block_path = out_dir / "graph_shuffled_within_local_block.npz"
    _save_archive(
        block_path,
        ids,
        embeddings[block_order],
        {
            "control": "shuffled_within_local_block",
            "source": str(graph_embeddings_path),
            "seed": seed + 2,
            "block_bp": block_bp,
            "n_blocks": int(len(set(local_blocks))),
            "n_singleton_blocks": int(
                pd.Series(local_blocks).value_counts().eq(1).sum()
            ),
            "n_embeddings": int(len(ids)),
            "embedding_dim": int(embeddings.shape[1]),
            "label_independent": True,
        },
    )
    outputs["shuffled_within_local_block"] = str(block_path)

    header = pd.read_csv(pairs_path, compression="infer", nrows=0).columns
    kmer_deltas = [column for column in header if column.startswith("delta_kmer3_")]
    paired_columns = [
        "embedding_id",
        "contig",
        "window_start",
        "methylation",
        "cpgs",
        "depth",
    ]
    requested = {
        "donor_id",
        "methylation_delta",
        *SEQUENCE_DELTAS,
        *kmer_deltas,
        *ANNOTATION_DELTAS,
        *SYMMETRIC_CONTEXT,
        *(f"{side}_{suffix}" for side in ("h1", "h2") for suffix in paired_columns),
    }
    columns = [column for column in header if column in requested]
    pairs = pd.read_csv(
        pairs_path,
        compression="infer",
        low_memory=False,
        usecols=columns,
    )
    for suffix in paired_columns:
        left = f"h1_{suffix}"
        right = f"h2_{suffix}"
        if left in pairs and right in pairs:
            pairs[left], pairs[right] = pairs[right].copy(), pairs[left].copy()
    signed_columns = [
        "methylation_delta",
        *SEQUENCE_DELTAS,
        *kmer_deltas,
        *ANNOTATION_DELTAS,
    ]
    for column in signed_columns:
        if column in pairs:
            pairs[column] = -pairs[column]
    swapped_path = out_dir / "paired_windows_h1_h2_swapped.csv.gz"
    pairs.to_csv(swapped_path, index=False, compression="gzip")
    outputs["h1_h2_swapped_pairs"] = str(swapped_path)

    summary = {
        "task": "haplotype_negative_controls",
        "source_pairs": str(pairs_path),
        "source_graph_embeddings": str(graph_embeddings_path),
        "seed": seed,
        "block_bp": block_bp,
        "n_embeddings": int(len(ids)),
        "embedding_dim": int(embeddings.shape[1]),
        "n_donors": int(len(set(donors))),
        "outputs": outputs,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--graph-embeddings", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-bp", type=int, default=1_000_000)
    args = parser.parse_args()
    result = build_controls(
        pairs_path=args.pairs,
        graph_embeddings_path=args.graph_embeddings,
        out_dir=args.out_dir,
        seed=args.seed,
        block_bp=args.block_bp,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
