#!/usr/bin/env python3
"""Create a label-blind frozen sequence-model embedding cache for graph nodes.

Target caches contribute node identifiers only.  No downstream label, split,
or outcome column is read.  The resulting ``segid``/``embeddings`` NPZ is the
common adapter consumed by the cCRE and SV exact-universe probe runners.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import numpy as np
import torch


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def configure_csv_field_size_limit() -> int:
    """Use the largest CSV field supported by the current Python build."""

    limit = sys.maxsize
    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit //= 10
    raise RuntimeError("Unable to configure a usable CSV field-size limit")


def iter_segment_rows(path: Path) -> Iterator[dict[str, str]]:
    """Yield segment rows without Python's 128-KiB default CSV-field ceiling."""

    configure_csv_field_size_limit()
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        required_columns = {"name", "seq"}
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"full_segments needs columns {sorted(required_columns)}; "
                f"got {reader.fieldnames}"
            )
        yield from reader


def requested_segids(paths: list[Path]) -> tuple[set[int], list[dict[str, object]]]:
    requested: set[int] = set()
    inputs: list[dict[str, object]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as cache:
            if "segid" not in cache:
                raise KeyError(f"Target cache has no segid array: {path}")
            segids = cache["segid"].astype(np.int64)
        before = len(requested)
        requested.update(int(value) for value in segids)
        inputs.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "segids": int(len(segids)),
                "unique_segids": int(len(np.unique(segids))),
                "new_union_segids": int(len(requested) - before),
            }
        )
    return requested, inputs


def non_special_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_ids: list[int],
) -> torch.Tensor:
    mask = attention_mask.bool()
    for special_id in special_ids:
        mask &= input_ids.ne(int(special_id))
    return mask


def segment_id(row: dict[str, str], fallback: int) -> int:
    name = str(row["name"])
    if name.startswith("s") and name[1:].isdigit():
        return int(name[1:]) - 1
    return fallback


def balanced_sequence(sequence: str, max_bases: int) -> tuple[str, bool]:
    """Retain both node ends when a raw sequence exceeds the base cap."""

    if len(sequence) <= max_bases:
        return sequence, False
    left = (max_bases - 1) // 2
    right = max_bases - 1 - left
    return sequence[:left] + "N" + sequence[-right:], True


def prepare_cache(
    *,
    full_segments: Path,
    target_caches: list[Path],
    output: Path,
    model_name: str,
    revision: str | None,
    batch_size: int,
    device: str,
    max_length: int,
    max_bases: int,
    trust_remote_code: bool,
    allow_unpinned_model: bool,
    max_nodes: int | None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> dict[str, object]:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    started = time.monotonic()
    if output.exists() or Path(f"{output}.audit.json").exists():
        raise FileExistsError("Refusing to overwrite an existing cache or audit")
    if revision is None and not allow_unpinned_model:
        raise ValueError(
            "--revision is required for a scientific run; use "
            "--allow-unpinned-model only for a disposable pilot"
        )
    requested_union, target_inputs = requested_segids(target_caches)
    if not requested_union:
        raise ValueError("Target caches contain no node identifiers")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Shard index must satisfy 0 <= shard_index < num_shards")
    requested = {
        segid for segid in requested_union if segid % num_shards == shard_index
    }
    if not requested:
        raise ValueError("Selected shard contains no target nodes")
    if max_nodes is not None:
        if max_nodes < 1:
            raise ValueError("--max-nodes must be positive")
        requested = set(
            sorted(
                requested,
                key=lambda value: hashlib.blake2b(
                    str(value).encode("ascii"), digest_size=8
                ).digest(),
            )[:max_nodes]
        )

    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
    ).to(resolved_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output_segids: list[int] = []
    embedding_batches: list[np.ndarray] = []
    batch_segids: list[int] = []
    batch_sequences: list[str] = []
    sequence_less_requested = 0
    raw_sequences_sampled = 0

    def embed_batch() -> None:
        if not batch_segids:
            return
        encoded = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
        with torch.inference_mode():
            result = model(**encoded, output_hidden_states=True)
            hidden = result.hidden_states[-1]
            mask = non_special_mask(
                encoded["input_ids"],
                encoded["attention_mask"],
                list(tokenizer.all_special_ids),
            )
            pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(
                1, keepdim=True
            ).clamp_min(1)
        values = pooled.detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Sequence model emitted non-finite embeddings")
        output_segids.extend(batch_segids)
        embedding_batches.append(values)
        batch_segids.clear()
        batch_sequences.clear()
        completed = len(output_segids)
        if completed == batch_size or completed % max(batch_size * 25, 1) == 0:
            print(
                f"[sequence-fm] embedded {completed:,}/{len(requested):,} requested nodes",
                flush=True,
            )

    remaining = set(requested)
    fallback = 0
    for row in iter_segment_rows(full_segments):
        segid = segment_id(row, fallback)
        fallback += 1
        if segid not in remaining:
            continue
        sequence = str(row["seq"]).upper()
        if not sequence or sequence == "*":
            sequence_less_requested += 1
            remaining.remove(segid)
            continue
        sequence, sampled = balanced_sequence(sequence, max_bases)
        raw_sequences_sampled += int(sampled)
        batch_segids.append(segid)
        batch_sequences.append(sequence)
        remaining.remove(segid)
        if len(batch_segids) >= batch_size:
            embed_batch()
        if not remaining:
            break
    embed_batch()
    if remaining:
        raise KeyError(
            f"{len(remaining)} requested segment IDs were absent; examples={sorted(remaining)[:10]}"
        )
    if sequence_less_requested:
        raise ValueError(
            f"{sequence_less_requested} requested nodes have no sequence; an exact sequence-FM comparison is impossible"
        )
    embeddings = (
        np.concatenate(embedding_batches, axis=0)
        if embedding_batches
        else np.empty((0, 0), dtype=np.float32)
    )
    segids = np.asarray(output_segids, dtype=np.int64)
    order = np.argsort(segids, kind="stable")
    segids = segids[order]
    embeddings = embeddings[order]
    if len(segids) != len(requested) or len(np.unique(segids)) != len(segids):
        raise ValueError("Output cache does not contain every requested node exactly once")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, segid=segids, embeddings=embeddings)
    temporary.replace(output)
    resolved_revision = (
        getattr(model.config, "_commit_hash", None)
        or revision
        or "unpinned-pilot"
    )
    audit = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "full_segments": str(full_segments.resolve()),
        "full_segments_sha256": sha256_file(full_segments),
        "target_caches": target_inputs,
        "target_union_nodes": int(len(requested_union)),
        "num_shards": int(num_shards),
        "shard_index": int(shard_index),
        "requested_nodes": int(len(requested)),
        "embedded_nodes": int(len(segids)),
        "coverage_fraction": float(len(segids) / len(requested)),
        "embedding_dimension": int(embeddings.shape[1]),
        "model_name": model_name,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "pooling": "mean final hidden state over non-special, non-padding tokens",
        "maximum_token_length": int(max_length),
        "maximum_raw_bases": int(max_bases),
        "raw_sequences_balanced_sampled": int(raw_sequences_sampled),
        "raw_sequence_sampling": "long nodes retain balanced prefix and suffix separated by N before tokenizer truncation",
        "truncation_policy": "tokenizer truncation at maximum_token_length; complete raw sequences remain in the source table",
        "device": resolved_device,
        "batch_size": int(batch_size),
        "fine_tuned": False,
        "model_parameters_frozen": True,
        "downstream_label_access": "none",
        "target_selection_note": "target caches contribute segid arrays only; no label or split array is read",
        "scope": (
            "pilot"
            if max_nodes is not None
            else "complete_target_union"
            if num_shards == 1
            else "complete_shard"
        ),
        "wall_seconds": time.monotonic() - started,
    }
    Path(f"{output}.audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-segments", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    )
    parser.add_argument("--revision")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument("--max-bases", type=int, default=6000)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-unpinned-model", action="store_true")
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 2 or args.max_bases < 3:
        parser.error(
            "--batch-size must be positive, --max-length at least 2, and --max-bases at least 3"
        )
    audit = prepare_cache(
        full_segments=args.full_segments,
        target_caches=args.target_cache,
        output=args.output,
        model_name=args.model_name,
        revision=args.revision,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
        max_bases=args.max_bases,
        trust_remote_code=args.trust_remote_code,
        allow_unpinned_model=args.allow_unpinned_model,
        max_nodes=args.max_nodes,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
