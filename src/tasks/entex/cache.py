"""Reuse frozen topology extraction across label-only P0 sensitivities."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from evaluation.modality_factorial import load_frozen_node_embedding_cache
from tasks.entex.prepare import fingerprint


def cached_topology(
    path: Path,
    requested: set[int],
    identity: dict,
    extract: Callable,
) -> tuple[dict[int, np.ndarray], dict]:
    """Cache the existing extractor's output, never labels or fitted probe state.

    A superset of requested targets can serve assay subsets. Targets which were
    requested but not embedded stay missing; the ordinary coverage gate applies.
    Identity hashes bind the checkpoint, graph and manifest to the cache.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        sidecar = Path(str(path) + ".audit.json")
        if path.exists() or sidecar.exists():
            segids, embeddings, audit = load_frozen_node_embedding_cache(path)
            if audit.get("identity") != identity:
                raise ValueError(
                    "Topology cache checkpoint/graph/manifest identity mismatch"
                )
            if audit.get("output_sha256") != fingerprint(path)["sha256"]:
                raise ValueError("Topology cache checksum mismatch")
            with np.load(path, allow_pickle=False) as stored:
                cached_requested = set(stored["requested_segids"].tolist())
            if not requested.issubset(cached_requested):
                raise ValueError(
                    "Topology cache did not request all required targets; use a new cache root"
                )
            return {
                int(s): e for s, e in zip(segids, embeddings) if int(s) in requested
            }, {
                **audit["extraction"],
                "cache_status": "reused",
                "cache_path": str(path),
            }
        frozen, _, extraction = extract()
        if not frozen:
            raise ValueError("No topology embeddings extracted")
        segids = np.array(sorted(frozen), dtype=np.int64)
        temporary = path.with_name(path.name + f".tmp-{os.getpid()}.npz")
        np.savez_compressed(
            temporary,
            segid=segids,
            embeddings=np.stack([frozen[int(s)] for s in segids]),
            requested_segids=np.array(sorted(requested), dtype=np.int64),
        )
        audit = dict(
            status="complete",
            schema_version=1,
            downstream_label_access="none",
            model_parameters_frozen=True,
            identity=identity,
            extraction=extraction,
            output_sha256=fingerprint(temporary)["sha256"],
        )
        temporary.replace(path)
        temp_audit = sidecar.with_name(sidecar.name + ".tmp")
        temp_audit.write_text(json.dumps(audit, indent=2) + "\n")
        temp_audit.replace(sidecar)
        return frozen, {
            **extraction,
            "cache_status": "created",
            "cache_path": str(path),
        }
