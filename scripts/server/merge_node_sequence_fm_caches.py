#!/usr/bin/env python3
"""Merge audited frozen node sequence-model shards into one exact cache."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.server.prepare_node_sequence_fm_cache import requested_segids, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, nargs="+", required=True)
    parser.add_argument("--target-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_path = Path(f"{args.output}.audit.json")
    if args.output.exists() or audit_path.exists():
        raise FileExistsError("Refusing to overwrite an existing merged cache or audit")
    started = time.monotonic()
    target_union, target_inputs = requested_segids(args.target_cache)
    arrays: list[np.ndarray] = []
    segid_arrays: list[np.ndarray] = []
    audits: list[dict[str, object]] = []
    for shard in args.shard:
        sidecar = Path(f"{shard}.audit.json")
        audit = json.loads(sidecar.read_text(encoding="utf-8"))
        if audit.get("status") != "complete":
            raise ValueError(f"Shard audit is not complete: {sidecar}")
        if sha256_file(shard) != audit.get("output_sha256"):
            raise ValueError(f"Shard checksum differs from audit: {shard}")
        with np.load(shard, allow_pickle=False) as cache:
            segid_arrays.append(cache["segid"].astype(np.int64))
            arrays.append(cache["embeddings"].astype(np.float32))
        audits.append(audit)
    model_identity = {
        (audit.get("model_name"), audit.get("resolved_revision")) for audit in audits
    }
    dimensions = {int(array.shape[1]) for array in arrays}
    if len(model_identity) != 1 or len(dimensions) != 1:
        raise ValueError("Sequence-model shard identities or dimensions differ")
    segids = np.concatenate(segid_arrays)
    embeddings = np.concatenate(arrays, axis=0)
    if len(np.unique(segids)) != len(segids):
        raise ValueError("Sequence-model shards overlap in segid")
    observed = set(int(value) for value in segids)
    missing = target_union - observed
    extra = observed - target_union
    if missing or extra:
        raise ValueError(
            f"Merged cache differs from target union: missing={len(missing)}, extra={len(extra)}"
        )
    order = np.argsort(segids, kind="stable")
    segids = segids[order]
    embeddings = embeddings[order]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, segid=segids, embeddings=embeddings)
    temporary.replace(args.output)
    model_name, resolved_revision = next(iter(model_identity))
    audit = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "target_caches": target_inputs,
        "target_union_nodes": int(len(target_union)),
        "embedded_nodes": int(len(segids)),
        "coverage_fraction": 1.0,
        "embedding_dimension": int(embeddings.shape[1]),
        "model_name": model_name,
        "resolved_revision": resolved_revision,
        "source_shards": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "audit_sha256": sha256_file(Path(f"{path}.audit.json")),
            }
            for path in args.shard
        ],
        "fine_tuned": False,
        "model_parameters_frozen": True,
        "downstream_label_access": "none",
        "target_selection_note": "target caches contribute segid arrays only; no label or split array is read",
        "scope": "complete_target_union",
        "wall_seconds": time.monotonic() - started,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
