#!/usr/bin/env python3
"""Fill only missing canonical graph segments using the audited manuscript NT policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.server.merge_node_sequence_fm_caches import sequence_contract
from scripts.server.prepare_node_sequence_fm_cache import iter_segment_rows, segment_id, sha256_file


def graph_targets(full_segments: Path, cached_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(np.unique(cached_ids)) != len(cached_ids):
        raise ValueError("Duplicate identifiers in existing sequence cache")
    cached = set(map(int, cached_ids))
    ids, missing = [], []
    for index, row in enumerate(iter_segment_rows(full_segments)):
        segid = segment_id(row, index)
        if segid != index:
            raise ValueError("Graph name-derived IDs differ from canonical row indices")
        ids.append(segid)
        if segid not in cached:
            if not row["seq"] or row["seq"] == "*":
                raise ValueError(f"Missing-cache segment {segid} has no sequence")
            missing.append(segid)
    if not ids or cached - set(ids):
        raise ValueError("Empty graph or existing cache IDs outside canonical graph")
    return np.asarray(ids, dtype=np.int64), np.asarray(missing, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-segments", type=Path, required=True)
    parser.add_argument("--existing-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Use a new cache-completion directory")
    contract = sequence_contract(args.existing_cache)
    if sha256_file(args.full_segments) != contract["full_segments_sha256"]:
        raise ValueError("Sequence-cache source differs from canonical graph")
    with np.load(args.existing_cache, allow_pickle=False) as cache:
        ids, missing = graph_targets(args.full_segments, cache["segid"])
    args.out_dir.mkdir(parents=True)
    np.savez(args.out_dir / "all_targets.npz", segid=ids)
    np.savez(args.out_dir / "missing_targets.npz", segid=missing)
    partial, complete = args.out_dir / "missing_nt.npz", args.out_dir / "whole_graph_nt.npz"
    commands = []
    if len(missing):
        commands.append([sys.executable, "scripts/server/prepare_node_sequence_fm_cache.py",
                         "--full-segments", str(args.full_segments), "--target-cache", str(args.out_dir / "missing_targets.npz"),
                         "--output", str(partial), "--model-name", contract["model_name"],
                         "--revision", contract["resolved_revision"], "--max-length", str(contract["maximum_token_length"]),
                         "--max-bases", str(contract["maximum_raw_bases"]), "--batch-size", str(args.batch_size),
                         "--device", args.device, "--trust-remote-code"])
    commands.append([sys.executable, "scripts/server/merge_node_sequence_fm_caches.py", "--shard",
                     str(args.existing_cache), *([str(partial)] if len(missing) else []),
                     "--target-cache", str(args.out_dir / "all_targets.npz"), "--output", str(complete)])
    receipt = dict(status="planned", graph_segments=len(ids), missing_segments=len(missing),
                   existing_coverage=float((len(ids) - len(missing)) / len(ids)),
                   contract=contract, commands=commands, existing_cache_modified=False,
                   model_downloads_allowed=False, biological_labels_used=False)
    status_path = args.out_dir / "status.json"

    def save() -> None:
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(receipt, indent=2) + "\n")
        temp.replace(status_path)

    save()
    if args.execute:
        env = dict(os.environ, PYTHONPATH="src:.", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        try:
            for index, command in enumerate(commands):
                receipt.update(status="running", active_command=index)
                save()
                with (args.out_dir / f"command_{index}.log").open("w") as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            receipt.update(status="complete", output=str(complete))
        except Exception as error:
            receipt.update(status="failed", error=str(error))
            raise
        finally:
            save()


if __name__ == "__main__":
    main()
