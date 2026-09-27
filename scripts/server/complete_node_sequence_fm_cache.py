#!/usr/bin/env python3
"""Fill only missing canonical graph segments using the audited manuscript NT policy."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from graph.slicing import build_global_index, map_links_to_segids
from scripts.server.merge_node_sequence_fm_caches import sequence_contract
from scripts.server.prepare_node_sequence_fm_cache import iter_segment_rows, segment_id, sha256_file


def graph_targets(full_segments: Path, cached_ids: np.ndarray,
                  required_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if len(np.unique(cached_ids)) != len(cached_ids):
        raise ValueError("Duplicate identifiers in existing sequence cache")
    cached = set(map(int, cached_ids))
    required = None if required_ids is None else set(map(int, required_ids)) | cached
    ids, missing = [], []
    for index, row in enumerate(iter_segment_rows(full_segments)):
        segid = segment_id(row, index)
        if segid != index:
            raise ValueError("Graph name-derived IDs differ from canonical row indices")
        ids.append(segid)
        if segid not in cached and (required is None or segid in required):
            if not row["seq"] or row["seq"] == "*":
                raise ValueError(f"Missing-cache segment {segid} has no sequence")
            missing.append(segid)
    if not ids or cached - set(ids):
        raise ValueError("Empty graph or existing cache IDs outside canonical graph")
    if required is not None and required - set(ids):
        raise ValueError("Requested IDs outside canonical graph")
    return np.asarray(ids if required is None else sorted(required), dtype=np.int64), np.asarray(missing, dtype=np.int64)


def benchmark_targets(full_segments: Path, manifest: Path, context: str) -> tuple[np.ndarray, dict]:
    """Use native unmasked link endpoints, including every chromosome and alternative node."""
    names = pd.read_csv(full_segments, usecols=["name"], dtype={"name": "string"})
    seg_index, _ = build_global_index(names)
    if len(seg_index) != len(names):
        raise ValueError("Duplicate graph names would change row-derived sequence-cache IDs")
    rows = pd.read_csv(manifest)
    rows = rows.loc[rows.closure.eq(context)]
    if rows.empty:
        raise ValueError("No benchmark windows for requested context")
    wanted = set()
    for row in rows.itertuples(index=False):
        links = pd.read_csv(row.links_path, usecols=["from_seg", "to_seg"])
        source, target = map_links_to_segids(links, seg_index)
        wanted.update(source.tolist())
        wanted.update(target.tolist())
    return np.asarray(sorted(wanted), dtype=np.int64), dict(
        graph_segments=len(seg_index), benchmark_segments=len(wanted), benchmark_windows=len(rows),
        manifest=str(manifest.resolve()), manifest_sha256=sha256_file(manifest), context=context,
        target_selection="existing-cache union native unmasked benchmark link endpoints; no labels")


def completion_commands(args, contract: dict, n_missing: int) -> tuple[list[list[str]], Path]:
    complete = args.out_dir / ("benchmark_nt.npz" if args.manifest else "whole_graph_nt.npz")
    n_shards = len(args.shard_gpus) if args.shard_gpus else 1
    commands, partials = [], []
    if n_missing:
        for index in range(n_shards):
            partial = args.out_dir / (f"missing_nt_{index}.npz" if n_shards > 1 else "missing_nt.npz")
            partials.append(str(partial))
            commands.append([sys.executable, "scripts/server/prepare_node_sequence_fm_cache.py",
                             "--full-segments", str(args.full_segments), "--target-cache", str(args.out_dir / "missing_targets.npz"),
                             "--output", str(partial), "--model-name", contract["model_name"],
                             "--revision", contract["resolved_revision"], "--max-length", str(contract["maximum_token_length"]),
                             "--max-bases", str(contract["maximum_raw_bases"]), "--batch-size", str(args.batch_size),
                             "--device", args.device, "--trust-remote-code", "--num-shards", str(n_shards),
                             "--shard-index", str(index)])
    commands.append([sys.executable, "scripts/server/merge_node_sequence_fm_caches.py", "--shard",
                     str(args.existing_cache), *partials,
                     "--target-cache", str(args.out_dir / "all_targets.npz"), "--output", str(complete)])
    return commands, complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-segments", type=Path, required=True)
    parser.add_argument("--existing-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--manifest", type=Path,
                        help="Complete only the native benchmark-node union, retaining all existing entries")
    parser.add_argument("--context", choices=["strict", "1hop"], default="1hop")
    parser.add_argument("--maximum-new-segments", type=int, default=10000,
                        help="Refuse unexpectedly broad inference before loading the model")
    parser.add_argument("--shard-gpus", type=int, nargs="+",
                        help="Explicit distinct GPUs for concurrent native cache shards")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.shard_gpus and (len(set(args.shard_gpus)) != len(args.shard_gpus) or args.device != "cuda"):
        raise ValueError("Sharding requires distinct GPUs and --device cuda")
    if args.out_dir.exists():
        raise FileExistsError("Use a new cache-completion directory")
    contract = sequence_contract(args.existing_cache)
    if sha256_file(args.full_segments) != contract["full_segments_sha256"]:
        raise ValueError("Sequence-cache source differs from canonical graph")
    required, scope = benchmark_targets(args.full_segments, args.manifest, args.context) if args.manifest else (None, {})
    with np.load(args.existing_cache, allow_pickle=False) as cache:
        ids, missing = graph_targets(args.full_segments, cache["segid"], required)
    args.out_dir.mkdir(parents=True)
    np.savez(args.out_dir / "all_targets.npz", segid=ids)
    np.savez(args.out_dir / "missing_targets.npz", segid=missing)
    commands, complete = completion_commands(args, contract, len(missing))
    receipt = dict(status="planned", graph_segments=scope.get("graph_segments", len(ids)),
                   target_segments=len(ids), missing_segments=len(missing), target_scope=scope,
                   existing_coverage=float((len(ids) - len(missing)) / len(ids)),
                   contract=contract, commands=commands, existing_cache_modified=False,
                   model_downloads_allowed=False, biological_labels_used=False, shard_gpus=args.shard_gpus)
    status_path = args.out_dir / "status.json"

    def save() -> None:
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(receipt, indent=2) + "\n")
        temp.replace(status_path)

    save()
    if len(missing) > args.maximum_new_segments:
        receipt.update(status="stopped_scope_guard", error=f"{len(missing)} new segments exceeds {args.maximum_new_segments}")
        save()
        raise ValueError(receipt["error"])
    if args.execute:
        env = dict(os.environ, PYTHONPATH="src:.", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")

        def execute(index: int, command: list[str], gpu: int | None = None) -> None:
            child_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu)) if gpu is not None else env
            with (args.out_dir / f"command_{index}.log").open("w") as log:
                subprocess.run(command, env=child_env, stdout=log, stderr=subprocess.STDOUT, check=True)

        try:
            if args.shard_gpus and len(missing):
                receipt.update(status="running", active_commands=list(range(len(commands) - 1)), phase="embedding_shards")
                save()
                with ThreadPoolExecutor(max_workers=len(args.shard_gpus)) as pool:
                    futures = [pool.submit(execute, index, command, gpu)
                               for index, (command, gpu) in enumerate(zip(commands[:-1], args.shard_gpus))]
                    for future in futures:
                        future.result()
                receipt.update(phase="merging", active_commands=[len(commands) - 1])
                save()
                execute(len(commands) - 1, commands[-1])
            else:
                for index, command in enumerate(commands):
                    receipt.update(status="running", active_command=index)
                    save()
                    execute(index, command)
            receipt.update(status="complete", output=str(complete))
        except BaseException as error:
            receipt.update(status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                           error=f"{type(error).__name__}: {error}")
            raise
        finally:
            save()


if __name__ == "__main__":
    main()
