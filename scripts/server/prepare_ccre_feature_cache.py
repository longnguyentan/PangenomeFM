#!/usr/bin/env python3
"""Prepare leakage-audited cCRE baseline features once for all GPU probes.

The HPRC segment table is several gigabytes.  Streaming nucleotide features in
every seed/fold job would waste substantial server time, so this command writes
one versioned feature cache.  It never overwrites an existing cache unless
``--force`` is supplied and records input checksums in a JSON sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tasks.ccre.aligned_baselines import _feature_matrix


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_cache(
    *,
    full_segments: Path,
    full_links: Path,
    node_labels: Path,
    output: Path,
    force: bool,
) -> dict[str, object]:
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing cache: {output}")
    labels = pd.read_csv(node_labels, compression="infer")
    if labels["segid"].duplicated().any():
        raise ValueError("node_labels contains duplicate segid values")
    required = {"segid", "SO", "LN", "chrom", "ccre_label"}
    missing = required - set(labels)
    if missing:
        raise ValueError(f"node_labels is missing columns: {sorted(missing)}")

    started = time.monotonic()
    structural, structural_names = _feature_matrix(
        feature_set="structural",
        feature_policy="leakage_safe",
        labels=labels,
        full_segments=full_segments,
        full_links=full_links,
    )
    sequence, sequence_names = _feature_matrix(
        feature_set="sequence_kmer",
        feature_policy="leakage_safe",
        labels=labels,
        full_segments=full_segments,
        full_links=full_links,
    )
    structural_index = {name: index for index, name in enumerate(structural_names)}
    coordinate_names = [name for name in ("log1p_SO", "log1p_LN", "orient") if name in structural_index]
    graph_names = ["log1p_degree"]
    coordinate = structural[:, [structural_index[name] for name in coordinate_names]]
    graph = structural[:, [structural_index[name] for name in graph_names]]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        segid=labels["segid"].to_numpy(np.int64),
        chrom=labels["chrom"].astype(str).to_numpy(),
        ccre_label=labels["ccre_label"].to_numpy(np.int64),
        coordinate=coordinate.astype(np.float32),
        graph=graph.astype(np.float32),
        structural=structural.astype(np.float32),
        sequence_kmer=sequence.astype(np.float32),
        coordinate_names=np.asarray(coordinate_names),
        graph_names=np.asarray(graph_names),
        structural_names=np.asarray(structural_names),
        sequence_kmer_names=np.asarray(sequence_names),
    )
    temporary.replace(output)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "inputs": {
            "full_segments": str(full_segments.resolve()),
            "full_links": str(full_links.resolve()),
            "node_labels": str(node_labels.resolve()),
            "full_segments_sha256": sha256_file(full_segments),
            "full_links_sha256": sha256_file(full_links),
            "node_labels_sha256": sha256_file(node_labels),
        },
        "n_nodes": int(len(labels)),
        "feature_dimensions": {
            "coordinate": int(coordinate.shape[1]),
            "graph": int(graph.shape[1]),
            "structural": int(structural.shape[1]),
            "sequence_kmer": int(sequence.shape[1]),
        },
        "feature_access": {
            "coordinate": "reference coordinate, node length, and constant orientation; no graph adjacency or nucleotides",
            "graph": "global node degree only; no coordinates or nucleotides",
            "structural": "coordinate, node length, degree, and orientation; SR and reference-identity fields excluded",
            "sequence_kmer": "length-normalized mono/di/tri-nucleotide composition from at most 2,048 balanced terminal bases",
        },
        "wall_seconds": time.monotonic() - started,
    }
    audit_path = output.with_suffix(output.suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-segments", type=Path, required=True)
    parser.add_argument("--full-links", type=Path, required=True)
    parser.add_argument("--node-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    audit = prepare_cache(
        full_segments=args.full_segments,
        full_links=args.full_links,
        node_labels=args.node_labels,
        output=args.output,
        force=args.force,
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
