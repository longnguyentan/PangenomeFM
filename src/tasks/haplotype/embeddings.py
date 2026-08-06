"""Generate frozen sequence-FM and GraphGenome-FM embeddings on identical pairs."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd
import torch

from analysis.network import build_adjacency, find_connected_components
from graph.neg_sampling import compute_oriented_degrees
from models.dual_stream_gat import DualStreamPangenomeGAT
from models.gat import build_node_features


def _open_text(path: Path) -> TextIO:
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open()


def _steps(value: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    marker = ""
    token = []
    for char in value:
        if char in "><":
            if marker and token:
                result.append(("".join(token), "+" if marker == ">" else "-"))
            marker, token = char, []
        else:
            token.append(char)
    if marker and token:
        result.append(("".join(token), "+" if marker == ">" else "-"))
    return result


def _p_steps(value: str) -> list[tuple[str, str]]:
    """Parse the comma-separated oriented segment list used by GFA P records."""
    result: list[tuple[str, str]] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token[-1] not in "+-":
            raise ValueError(f"Malformed GFA P step: {token}")
        result.append((token[:-1], token[-1]))
    return result


_P_SLICE_SUFFIX = re.compile(r"\[(\d+)-(\d+)\]$")


def _target_p_walk(
    fields: list[str],
    target_sample: str,
    target_path_names: set[str] | None = None,
) -> dict[str, Any] | None:
    """Recover the exact selected GBZ path emitted by ``vg chunk`` as a P line.

    Context paths are emitted with a ``[start-end]`` suffix.  The selected path
    is the unique unsuffixed P record and its fourth ``#`` component is the
    chromosome-path offset.  Downstream pair IDs intentionally use the stable
    sample/haplotype/contig triple and whole-contig coordinates.
    """
    if len(fields) < 3 or fields[0] != "P":
        return None
    name = fields[1]
    match = _P_SLICE_SUFFIX.search(name)
    slice_start = int(match.group(1)) if match else 0
    source_name = name[: match.start()] if match else name
    if target_path_names is not None and source_name not in target_path_names:
        return None
    # Without an explicit path manifest, only an unsuffixed selected path is
    # unambiguous; bracketed P records may merely be context paths.
    if target_path_names is None and match:
        return None
    parts = source_name.split("#")
    if len(parts) != 4 or parts[0] != target_sample:
        return None
    try:
        start = int(parts[3]) + slice_start
    except ValueError:
        return None
    return {
        "track": "#".join(parts[:3]),
        "start": start,
        "steps": _p_steps(fields[2]),
    }


def _segment_length(fields: list[str]) -> int:
    if len(fields) < 3 or fields[0] != "S":
        raise ValueError("Expected a GFA S record.")
    if fields[2] != "*":
        return len(fields[2])
    for tag in fields[3:]:
        if tag.startswith("LN:i:"):
            return int(tag[5:])
    raise ValueError(f"Sequence-less segment {fields[1]!r} has no LN tag.")


def _save_embeddings(
    path: Path, ids: list[str], embeddings: np.ndarray, metadata: dict[str, Any]
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


def _natural_key(path: Path) -> list[str | int]:
    return [
        int(token) if token.isdigit() else token
        for token in re.split(r"(\d+)", path.name)
    ]


def _manifest_target_for_chunk(
    path: Path, target_path_names: set[str] | None
) -> set[str] | None:
    if target_path_names is None:
        return None
    matched = [name for name in target_path_names if name in path.name]
    if not matched:
        raise ValueError(f"Cannot map chunk filename to path manifest: {path.name}")
    longest = max(map(len, matched))
    winners = {name for name in matched if len(name) == longest}
    if len(winners) != 1:
        raise ValueError(f"Ambiguous target path in chunk filename: {path.name}")
    return winners


def _walk_base_matches_active_target(
    source_name: str, active_target_paths: set[str] | None
) -> bool:
    if active_target_paths is None or source_name in active_target_paths:
        return True
    # GFA W records omit a zero phase-block suffix present in metadata and
    # chunk filenames. Nonzero phase blocks are P records; same-contig W
    # records in those chunks are context.
    return any(
        target.endswith("#0")
        and target.rsplit("#", 1)[0] == source_name
        for target in active_target_paths
    )


def _uncovered_length(
    covered: dict[str, list[tuple[int, int]]],
    key: str,
    left: int,
    right: int,
) -> int:
    """Return and record uncovered bases in [left, right)."""
    if right <= left:
        return 0
    pieces = [(left, right)]
    for old_left, old_right in covered.get(key, []):
        next_pieces: list[tuple[int, int]] = []
        for piece_left, piece_right in pieces:
            if old_right <= piece_left or old_left >= piece_right:
                next_pieces.append((piece_left, piece_right))
                continue
            if piece_left < old_left:
                next_pieces.append((piece_left, old_left))
            if old_right < piece_right:
                next_pieces.append((old_right, piece_right))
        pieces = next_pieces
        if not pieces:
            return 0
    intervals = covered.setdefault(key, [])
    intervals.extend(pieces)
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for piece_left, piece_right in intervals:
        if not merged or piece_left > merged[-1][1]:
            merged.append((piece_left, piece_right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], piece_right))
    covered[key] = merged
    return sum(piece_right - piece_left for piece_left, piece_right in pieces)


def _non_special_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_ids: list[int],
) -> torch.Tensor:
    mask = attention_mask.bool()
    for special_id in special_ids:
        mask &= input_ids.ne(int(special_id))
    return mask


def generate_sequence_embeddings(
    *,
    pairs_path: str | Path,
    out_path: str | Path,
    model_name: str = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    batch_size: int = 16,
    device: str = "cpu",
    max_length: int = 1000,
) -> dict[str, Any]:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    pairs = pd.read_csv(pairs_path, compression="infer")
    sequence_by_id: dict[str, str] = {}
    for side in ("h1", "h2"):
        for embedding_id, sequence in zip(
            pairs[f"{side}_embedding_id"], pairs[f"{side}_sequence"]
        ):
            sequence = str(sequence).upper()
            prior = sequence_by_id.setdefault(str(embedding_id), sequence)
            if prior != sequence:
                raise ValueError(f"Conflicting sequences for {embedding_id}.")
    ids = sorted(sequence_by_id)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True
    ).to(device)
    model.eval()
    batches: list[np.ndarray] = []
    n_batches = (len(ids) + batch_size - 1) // batch_size
    with torch.inference_mode():
        for batch_index, start in enumerate(
            range(0, len(ids), batch_size), start=1
        ):
            if batch_index == 1 or batch_index % 25 == 0:
                print(
                    f"[sequence-embed] batch {batch_index}/{n_batches}",
                    flush=True,
                )
            batch_ids = ids[start : start + batch_size]
            encoded = tokenizer(
                [sequence_by_id[key] for key in batch_ids],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=True)
            hidden = output.hidden_states[-1]
            # The NT v2 remote tokenizer currently returns a special-token
            # mask one position longer than input_ids.  Derive the mask from
            # actual IDs instead; attention_mask still handles padding.
            mask = _non_special_mask(
                encoded["input_ids"],
                encoded["attention_mask"],
                tokenizer.all_special_ids,
            )
            pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(
                1, keepdim=True
            ).clamp_min(1)
            batches.append(pooled.detach().cpu().numpy().astype(np.float32))
    matrix = np.concatenate(batches) if batches else np.empty((0, 0), np.float32)
    out_path = Path(out_path)
    metadata = {
        "embedding_type": "frozen_sequence_foundation_model",
        "model": model_name,
        "pooling": "mean final hidden state over non-special tokens",
        "n_embeddings": len(ids),
        "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "pairs_path": str(pairs_path),
        "max_length_tokens": int(max_length),
        "fine_tuned": False,
    }
    _save_embeddings(out_path, ids, matrix, metadata)
    return metadata


def _load_chunk(
    path: Path,
    target_sample: str,
    target_path_names: set[str] | None = None,
) -> dict[str, Any]:
    active_target_paths = _manifest_target_for_chunk(path, target_path_names)
    target_walks: list[dict[str, Any]] = []
    reference_nodes: set[str] = set()
    target_nodes: set[str] = set()
    with _open_text(path) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields and fields[0] == "W" and len(fields) >= 7:
                walk_steps = _steps(fields[6])
                if fields[1] == "GRCh38":
                    reference_nodes.update(node for node, _ in walk_steps)
                source_name = f"{fields[1]}#{fields[2]}#{fields[3]}"
                if fields[1] == target_sample and (
                    _walk_base_matches_active_target(
                        source_name, active_target_paths
                    )
                ):
                    target_nodes.update(node for node, _ in walk_steps)
                    target_walks.append(
                        {
                            "track": f"{fields[1]}#{fields[2]}#{fields[3]}",
                            "start": int(fields[4]),
                            "steps": walk_steps,
                        }
                    )
            elif fields and fields[0] == "P":
                walk = _target_p_walk(
                    fields,
                    target_sample,
                    target_path_names=active_target_paths,
                )
                if walk is not None:
                    target_nodes.update(node for node, _ in walk["steps"])
                    target_walks.append(walk)
    # vg has already materialized a one-context-step chunk.  Retain the exact
    # target path and its incident graph edges, but not the dense web of edges
    # among context paths.  This is an explicit bounded-transfer adaptation:
    # it preserves the selected path's one-step graph neighborhood while
    # avoiding a chromosome-scale context graph that is not comparable to the
    # checkpoint's bounded training slices.
    selected_nodes = set(target_nodes)
    links: set[tuple[str, str, str, str]] = set()
    with _open_text(path) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if (
                fields
                and fields[0] == "L"
                and len(fields) >= 5
                and (fields[1] in target_nodes or fields[3] in target_nodes)
            ):
                selected_nodes.update((fields[1], fields[3]))
                links.add((fields[1], fields[2], fields[3], fields[4]))

    lengths: dict[str, int] = {}
    with _open_text(path) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if not fields:
                continue
            if (
                fields[0] == "S"
                and len(fields) >= 3
                and fields[1] in selected_nodes
            ):
                lengths.setdefault(fields[1], _segment_length(fields))
    return {
        "lengths": lengths,
        "links": sorted(links),
        "target_walks": target_walks,
        "reference_nodes": reference_nodes & selected_nodes,
    }


def _encode_chunk(
    chunk: dict[str, Any],
    model: DualStreamPangenomeGAT,
    device: str,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, list[tuple[int, int, str, str]]]]:
    segment_names = sorted(chunk["lengths"])
    segment_index = {name: idx for idx, name in enumerate(segment_names)}

    def oid(node: str, orientation: str) -> int:
        return 2 * segment_index[node] + int(orientation == "-")

    u = np.asarray(
        [oid(left, left_o) for left, left_o, _, _ in chunk["links"]],
        dtype=np.int64,
    )
    v = np.asarray(
        [oid(right, right_o) for _, _, right, right_o in chunk["links"]],
        dtype=np.int64,
    )
    target_oids = [
        oid(node, orientation)
        for walk in chunk["target_walks"]
        for node, orientation in walk["steps"]
    ]
    nodes = np.unique(
        np.concatenate([u, v, np.asarray(target_oids, dtype=np.int64)])
    )
    oid_to_dense = {int(value): idx for idx, value in enumerate(nodes)}
    src = np.asarray([oid_to_dense[int(value)] for value in u], dtype=np.int64)
    dst = np.asarray([oid_to_dense[int(value)] for value in v], dtype=np.int64)
    degrees = compute_oriented_degrees(u, v, nodes)
    adjacency = build_adjacency(u, v, nodes)
    components, _ = find_connected_components(adjacency)
    so: dict[int, int] = {}
    target_spans: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
    for walk in chunk["target_walks"]:
        cursor = int(walk["start"])
        for node, orientation in walk["steps"]:
            end = cursor + int(chunk["lengths"][node])
            target_spans[walk["track"]].append((cursor, end, node, orientation))
            for bit in (0, 1):
                so.setdefault(2 * segment_index[node] + bit, cursor)
            cursor = end
    lengths = {
        2 * idx + bit: int(chunk["lengths"][name])
        for name, idx in segment_index.items()
        for bit in (0, 1)
    }
    reference = {
        2 * idx + bit: int(name in chunk["reference_nodes"])
        for name, idx in segment_index.items()
        for bit in (0, 1)
    }
    zeros = {int(value): 0 for value in nodes}
    features = build_node_features(
        nodes, so, lengths, zeros, reference, degrees, components
    )
    with torch.inference_mode():
        encoded = model.encode_nodes(
            torch.tensor(features, dtype=torch.float32, device=device),
            torch.tensor([so.get(int(value), 0) for value in nodes], device=device),
            torch.tensor(src, dtype=torch.long, device=device),
            torch.tensor(dst, dtype=torch.long, device=device),
            torch.ones(len(nodes), dtype=torch.float32, device=device),
            orient=torch.tensor(nodes % 2, dtype=torch.int8, device=device),
        )
    by_oriented_node = {
        (segment_names[int(value) // 2], "-" if int(value) % 2 else "+"):
        encoded[index].detach().cpu().numpy()
        for index, value in enumerate(nodes)
    }
    return by_oriented_node, target_spans


def _load_frozen_graph_model(
    checkpoint_path: str | Path, device: str
) -> tuple[DualStreamPangenomeGAT, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    model = DualStreamPangenomeGAT(
        in_dim=int(checkpoint["in_dim"]),
        hidden_dim=int(args["hidden_dim"]),
        n_heads=int(args["n_heads"]),
        n_layers=int(args["n_layers"]),
        dropout=float(args["dropout"]),
        edge_mlp_dim=int(args["hidden_dim"]) * 2,
        edge_feat_dim=int(checkpoint.get("edge_feat_dim", 0)),
        use_rope=not bool(args.get("no_rope", False)),
        use_fusion_gate=not bool(args.get("no_fusion_gate", False)),
        window_k=int(args.get("adaptive_window_base", 64)),
        use_multiscale_rope=bool(args.get("multiscale_rope", False)),
        n_rope_scales=int(args.get("n_rope_scales", 3)),
        use_orientation=bool(args.get("orientation_rope", False)),
        pop_embed_dim=int(args.get("pop_embed_dim", 0))
        if args.get("pop_cond")
        else 0,
        stream_mode=str(checkpoint.get("stream_mode", args.get("stream_mode", "full"))),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def generate_graph_embeddings(
    *,
    pairs_path: str | Path,
    chunks_dir: str | Path,
    checkpoint_path: str | Path,
    out_path: str | Path,
    path_list: str | Path | None = None,
    device: str = "cpu",
    window_bp: int = 10_000,
) -> dict[str, Any]:
    pairs = pd.read_csv(pairs_path, compression="infer")
    required = set(pairs["h1_embedding_id"]) | set(pairs["h2_embedding_id"])
    donor = str(pairs["donor_id"].iloc[0])
    target_path_names = None
    if path_list is not None:
        target_path_names = {
            line.strip()
            for line in Path(path_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    model, checkpoint = _load_frozen_graph_model(checkpoint_path, device)
    sums: dict[str, np.ndarray] = {}
    weights: dict[str, int] = defaultdict(int)
    covered: dict[str, list[tuple[int, int]]] = {}
    chunks = sorted(Path(chunks_dir).glob("chunk*.gfa.gz"), key=_natural_key)
    for chunk_index, chunk_path in enumerate(chunks, start=1):
        print(
            f"[graph-embed] {donor} chunk {chunk_index}/{len(chunks)}: "
            f"{chunk_path.name}",
            flush=True,
        )
        chunk = _load_chunk(
            chunk_path, donor, target_path_names=target_path_names
        )
        if not chunk["target_walks"]:
            raise ValueError(
                f"No exact {donor} W walk or unsuffixed P path in {chunk_path}."
            )
        node_embeddings, target_spans = _encode_chunk(chunk, model, device)
        for track, spans in target_spans.items():
            for left, right, node, orientation in spans:
                first = left // window_bp
                last = max(first, (right - 1) // window_bp)
                for window in range(first, last + 1):
                    start, end = window * window_bp, (window + 1) * window_bp
                    embedding_id = f"{track}:{start}"
                    if embedding_id not in required:
                        continue
                    overlap = _uncovered_length(
                        covered,
                        embedding_id,
                        max(left, start),
                        min(right, end),
                    )
                    if not overlap:
                        continue
                    value = node_embeddings[(node, orientation)].astype(np.float64)
                    if embedding_id not in sums:
                        sums[embedding_id] = np.zeros_like(value)
                    sums[embedding_id] += value * overlap
                    weights[embedding_id] += overlap
        del chunk, node_embeddings, target_spans
        gc.collect()
    missing = sorted(required - set(sums))
    if missing:
        raise ValueError(
            f"Graph embeddings are missing {len(missing)} required windows; "
            f"examples: {missing[:5]}"
        )
    ids = sorted(required)
    matrix = np.stack([sums[key] / weights[key] for key in ids]).astype(np.float32)
    metadata = {
        "embedding_type": "frozen_graphgenome_fm",
        "checkpoint": str(checkpoint_path),
        "checkpoint_best_val_auc": float(checkpoint["best_val_auc"]),
        "checkpoint_closure": checkpoint["closure"],
        "checkpoint_stream_mode": checkpoint["stream_mode"],
        "checkpoint_test_chromosomes": checkpoint["args"].get("test_chrs", []),
        "pooling": "path-bp-weighted mean of frozen node embeddings",
        "feature_adaptation": (
            "GBZ chunks omit original segment tags; length, path offset, incident "
            "one-step graph degree/component, orientation, and GRCh38 path "
            "membership are rebuilt using the original seven-feature schema. "
            "Inference is bounded to the selected path plus incident edges."
        ),
        "n_chunks": len(chunks),
        "path_list": str(path_list) if path_list is not None else None,
        "n_embeddings": len(ids),
        "embedding_dim": int(matrix.shape[1]),
        "fine_tuned": False,
    }
    _save_embeddings(Path(out_path), ids, matrix, metadata)
    return metadata


def merge_embedding_files(
    *, input_paths: list[str | Path], out_path: str | Path
) -> dict[str, Any]:
    values: dict[str, np.ndarray] = {}
    dimensions: set[int] = set()
    for path in input_paths:
        payload = np.load(path)
        for key, embedding in zip(payload["ids"], payload["embeddings"]):
            key = str(key)
            embedding = np.asarray(embedding, dtype=np.float32)
            dimensions.add(int(embedding.shape[0]))
            prior = values.setdefault(key, embedding)
            if not np.allclose(prior, embedding, atol=1e-6):
                raise ValueError(f"Conflicting embedding values for {key}.")
    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimensions differ: {sorted(dimensions)}")
    ids = sorted(values)
    matrix = np.stack([values[key] for key in ids])
    metadata = {
        "embedding_type": "merged_frozen_embeddings",
        "input_paths": [str(path) for path in input_paths],
        "n_embeddings": len(ids),
        "embedding_dim": next(iter(dimensions)),
    }
    _save_embeddings(Path(out_path), ids, matrix, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seq = subparsers.add_parser("sequence")
    seq.add_argument("--pairs", required=True)
    seq.add_argument("--out", required=True)
    seq.add_argument(
        "--model",
        default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    )
    seq.add_argument("--batch-size", type=int, default=16)
    seq.add_argument("--device", default="cpu")
    graph = subparsers.add_parser("graph")
    graph.add_argument("--pairs", required=True)
    graph.add_argument("--chunks-dir", required=True)
    graph.add_argument("--checkpoint", required=True)
    graph.add_argument("--out", required=True)
    graph.add_argument("--path-list")
    graph.add_argument("--device", default="cpu")
    merge = subparsers.add_parser("merge")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "sequence":
        result = generate_sequence_embeddings(
            pairs_path=args.pairs,
            out_path=args.out,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.command == "graph":
        result = generate_graph_embeddings(
            pairs_path=args.pairs,
            chunks_dir=args.chunks_dir,
            checkpoint_path=args.checkpoint,
            out_path=args.out,
            path_list=args.path_list,
            device=args.device,
        )
    else:
        result = merge_embedding_files(
            input_paths=args.inputs,
            out_path=args.out,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
