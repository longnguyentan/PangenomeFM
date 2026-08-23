"""Pilot trainer for the joint sequence/topology/path PangenomeFM v2 model.

The trainer deliberately uses a compact, full-batch NPZ contract.  It is a
correctness and objective-interaction pilot, not the chromosome-scale loader.
Once a pilot passes, the same model can be connected to the existing server
slice/path loaders without changing objective semantics.

Required NPZ arrays
-------------------
``topology_embeddings`` [nodes, topology_dim] and either
``sequence_tokens`` [nodes, length] or ``sequence_features`` [nodes, dim].
``coordinates`` [nodes, coordinate_dim] is required unless coordinate_dim=0.

At least one objective group must be present.  Each group can include a split
array with values 0=train, 1=validation, 2=test:

* edge: edge_u, edge_v, edge_labels, edge_split
* masked tokens: masked_token_labels, node_split
* path order: path_nodes, path_pair_left/right, path_order_labels,
  path_order_split
* branch choice: branch_sources, branch_candidates, branch_labels, branch_split
* cross graph: cross_graph_left_nodes/right_nodes, cross_graph_split

Use ``--synthetic-smoke`` for an end-to-end server preflight without external
data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from models.multimodal_pangenome import (
    MASK_TOKEN,
    ObjectiveWeights,
    PangenomeFoundationModelV2,
)


SPLIT_NAMES = {0: "train", 1: "validation", 2: "test"}


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_arrays(seed: int = 20260823) -> dict[str, np.ndarray]:
    """Create a deterministic batch exercising every supported objective."""

    rng = np.random.default_rng(seed)
    n_nodes, seq_len, topology_dim = 36, 24, 12
    tokens = rng.integers(1, 6, size=(n_nodes, seq_len), dtype=np.int64)
    tokens[:, -2:] = 0
    labels = np.full_like(tokens, -100)
    for node in range(n_nodes):
        position = node % (seq_len - 2)
        labels[node, position] = tokens[node, position]
        tokens[node, position] = MASK_TOKEN

    edge_u = np.arange(24, dtype=np.int64) % n_nodes
    edge_v = (edge_u + rng.integers(1, 8, size=len(edge_u))) % n_nodes
    edge_labels = ((edge_u + edge_v) % 3 == 0).astype(np.float32)
    edge_split = np.repeat(np.arange(3, dtype=np.int64), 8)

    path_nodes = np.full((9, 6), -1, dtype=np.int64)
    for path in range(len(path_nodes)):
        path_nodes[path, :5] = (np.arange(5) + 3 * path) % n_nodes
    path_pair_left = np.arange(6, dtype=np.int64)
    path_pair_right = (path_pair_left + 1) % len(path_nodes)
    path_order_labels = (path_pair_left % 2 == 0).astype(np.float32)

    branch_sources = np.arange(9, dtype=np.int64)
    branch_candidates = np.stack(
        [(branch_sources + offset + 1) % n_nodes for offset in range(3)], axis=1
    )
    branch_labels = branch_sources % 3
    cross_left = np.arange(9, dtype=np.int64)
    cross_right = (cross_left + 9) % n_nodes
    group_split = np.repeat(np.arange(3, dtype=np.int64), 3)

    return {
        "topology_embeddings": rng.normal(size=(n_nodes, topology_dim)).astype(np.float32),
        "sequence_tokens": tokens,
        "coordinates": rng.normal(size=(n_nodes, 4)).astype(np.float32),
        "population_ids": (np.arange(n_nodes) % 4).astype(np.int64),
        "haplotype_ids": (np.arange(n_nodes) % 3).astype(np.int64),
        "node_split": (np.arange(n_nodes) % 3).astype(np.int64),
        "masked_token_labels": labels,
        "edge_u": edge_u,
        "edge_v": edge_v,
        "edge_labels": edge_labels,
        "edge_split": edge_split,
        "path_nodes": path_nodes,
        "path_pair_left": path_pair_left,
        "path_pair_right": path_pair_right,
        "path_order_labels": path_order_labels,
        "path_order_split": np.repeat(np.arange(3, dtype=np.int64), 2),
        "branch_sources": branch_sources,
        "branch_candidates": branch_candidates,
        "branch_labels": branch_labels.astype(np.int64),
        "branch_split": group_split,
        "cross_graph_left_nodes": cross_left,
        "cross_graph_right_nodes": cross_right,
        "cross_graph_split": group_split,
    }


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as packed:
        return {name: packed[name] for name in packed.files}


def validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    if "topology_embeddings" not in arrays:
        raise ValueError("NPZ is missing topology_embeddings")
    topology = arrays["topology_embeddings"]
    if topology.ndim != 2 or not np.isfinite(topology).all():
        raise ValueError("topology_embeddings must be finite [nodes, features]")
    n_nodes = len(topology)
    if ("sequence_tokens" in arrays) == ("sequence_features" in arrays):
        raise ValueError("Provide exactly one of sequence_tokens or sequence_features")
    for name in ["sequence_tokens", "sequence_features", "coordinates", "population_ids", "haplotype_ids"]:
        if name in arrays and len(arrays[name]) != n_nodes:
            raise ValueError(f"{name} has a different node count")

    groups = [
        {"edge_u", "edge_v", "edge_labels"},
        {"masked_token_labels"},
        {"path_nodes", "path_pair_left", "path_pair_right", "path_order_labels"},
        {"branch_sources", "branch_candidates", "branch_labels"},
        {"cross_graph_left_nodes", "cross_graph_right_nodes"},
    ]
    if not any(group.issubset(arrays) for group in groups):
        raise ValueError("NPZ has no complete supported objective group")
    for split_name in ["node_split", "edge_split", "path_order_split", "branch_split", "cross_graph_split"]:
        if split_name in arrays and not np.isin(arrays[split_name], [0, 1, 2]).all():
            raise ValueError(f"{split_name} must contain only 0, 1, and 2")


def tensorize(arrays: Mapping[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    integer_names = {
        "sequence_tokens", "population_ids", "haplotype_ids", "node_split",
        "edge_u", "edge_v", "edge_split", "path_nodes", "path_pair_left",
        "path_pair_right", "path_order_split", "branch_sources",
        "branch_candidates", "branch_labels", "branch_split",
        "cross_graph_left_nodes", "cross_graph_right_nodes", "cross_graph_split",
        "masked_token_labels",
    }
    result: dict[str, torch.Tensor] = {}
    for name, values in arrays.items():
        dtype = torch.long if name in integer_names else torch.float32
        result[name] = torch.as_tensor(values, dtype=dtype, device=device)
    return result


def split_batch(batch: Mapping[str, torch.Tensor], split: int) -> dict[str, torch.Tensor]:
    """Filter objective labels while retaining shared node-level inputs."""

    result = {
        name: value
        for name, value in batch.items()
        if name in {
            "topology_embeddings", "sequence_tokens", "sequence_features",
            "coordinates", "population_ids", "haplotype_ids", "path_nodes",
        }
    }
    if {"edge_u", "edge_v", "edge_labels"}.issubset(batch):
        mask = batch.get("edge_split", torch.zeros_like(batch["edge_u"])).eq(split)
        if mask.any():
            for name in ["edge_u", "edge_v", "edge_labels"]:
                result[name] = batch[name][mask]
    if "masked_token_labels" in batch:
        labels = batch["masked_token_labels"].clone()
        node_split = batch.get("node_split", torch.zeros(len(labels), dtype=torch.long, device=labels.device))
        labels[node_split.ne(split)] = -100
        if labels.ne(-100).any():
            result["masked_token_labels"] = labels
    if {"path_pair_left", "path_pair_right", "path_order_labels"}.issubset(batch):
        mask = batch.get("path_order_split", torch.zeros_like(batch["path_pair_left"])).eq(split)
        if mask.any():
            for name in ["path_pair_left", "path_pair_right", "path_order_labels"]:
                result[name] = batch[name][mask]
    if {"branch_sources", "branch_candidates", "branch_labels"}.issubset(batch):
        mask = batch.get("branch_split", torch.zeros_like(batch["branch_sources"])).eq(split)
        if mask.any():
            for name in ["branch_sources", "branch_candidates", "branch_labels"]:
                result[name] = batch[name][mask]
    if {"cross_graph_left_nodes", "cross_graph_right_nodes"}.issubset(batch):
        default = torch.zeros_like(batch["cross_graph_left_nodes"])
        mask = batch.get("cross_graph_split", default).eq(split)
        if mask.sum() >= 2:
            for name in ["cross_graph_left_nodes", "cross_graph_right_nodes"]:
                result[name] = batch[name][mask]
    return result


def edge_metrics(model: PangenomeFoundationModelV2, batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
    if not {"edge_u", "edge_v", "edge_labels"}.issubset(batch):
        return {}
    with torch.no_grad():
        encoded = model.encode_nodes(
            topology_embeddings=batch["topology_embeddings"],
            sequence_tokens=batch.get("sequence_tokens"),
            sequence_features=batch.get("sequence_features"),
            coordinates=batch.get("coordinates"),
            population_ids=batch.get("population_ids"),
            haplotype_ids=batch.get("haplotype_ids"),
        )["node_embeddings"]
        scores = torch.sigmoid(model.edge_logits(encoded, batch["edge_u"], batch["edge_v"]))
    labels = batch["edge_labels"].detach().cpu().numpy()
    predictions = scores.detach().cpu().numpy()
    metrics: dict[str, float] = {}
    if len(np.unique(labels)) == 2:
        metrics["edge_auroc"] = float(roc_auc_score(labels, predictions))
        metrics["edge_auprc"] = float(average_precision_score(labels, predictions))
    return metrics


def _loss_scalars(outputs: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {
        name: float(value.detach().cpu())
        for name, value in outputs.items()
        if name not in {"node_embeddings", "modality_weights"}
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    started = time.monotonic()
    device = torch.device(args.device)
    if args.synthetic_smoke:
        arrays = synthetic_arrays(args.seed)
        input_sha256 = None
        input_label = "deterministic_synthetic_smoke"
    else:
        if args.input_npz is None:
            raise ValueError("--input-npz is required unless --synthetic-smoke is set")
        arrays = load_arrays(args.input_npz)
        input_sha256 = sha256_file(args.input_npz)
        input_label = str(args.input_npz.resolve())
    validate_arrays(arrays)
    batch = tensorize(arrays, device)

    sequence_feature_dim = arrays["sequence_features"].shape[1] if "sequence_features" in arrays else 0
    coordinate_dim = arrays["coordinates"].shape[1] if "coordinates" in arrays else 0
    n_populations = int(arrays["population_ids"].max()) + 1 if "population_ids" in arrays else 0
    n_haplotype_states = int(arrays["haplotype_ids"].max()) + 1 if "haplotype_ids" in arrays else 0
    model = PangenomeFoundationModelV2(
        topology_dim=arrays["topology_embeddings"].shape[1],
        hidden_dim=args.hidden_dim,
        coordinate_dim=coordinate_dim,
        sequence_feature_dim=sequence_feature_dim,
        n_heads=args.n_heads,
        sequence_layers=args.sequence_layers,
        path_layers=args.path_layers,
        dropout=args.dropout,
        max_sequence_length=args.max_sequence_length,
        max_path_length=args.max_path_length,
        n_populations=n_populations,
        n_haplotype_states=n_haplotype_states,
        modality_dropout=args.modality_dropout,
    ).to(device)
    weights = ObjectiveWeights(
        edge=args.edge_weight,
        masked_token=args.masked_token_weight,
        path_order=args.path_order_weight,
        branch_choice=args.branch_choice_weight,
        cross_graph=args.cross_graph_weight,
    )
    train = split_batch(batch, 0)
    validation = split_batch(batch, 1)
    test = split_batch(batch, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_outputs = model.compute_objectives(train, weights=weights)
        train_outputs["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_outputs = model.compute_objectives(validation, weights=weights)
        validation_loss = float(validation_outputs["total"].cpu())
        history.append({
            "epoch": epoch + 1,
            **{f"train_{key}": value for key, value in _loss_scalars(train_outputs).items()},
            **{f"validation_{key}": value for key, value in _loss_scalars(validation_outputs).items()},
        })
        if validation_loss < best_loss - args.min_delta:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_outputs = model.compute_objectives(test, weights=weights)
    test_losses = _loss_scalars(test_outputs)
    metrics = edge_metrics(model, test)
    modality_weights = test_outputs["modality_weights"].mean(dim=0).detach().cpu().tolist()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model_class": "PangenomeFoundationModelV2",
            "model_kwargs": {
                "topology_dim": arrays["topology_embeddings"].shape[1],
                "hidden_dim": args.hidden_dim,
                "coordinate_dim": coordinate_dim,
                "sequence_feature_dim": sequence_feature_dim,
                "n_heads": args.n_heads,
                "sequence_layers": args.sequence_layers,
                "path_layers": args.path_layers,
                "dropout": args.dropout,
                "max_sequence_length": args.max_sequence_length,
                "max_path_length": args.max_path_length,
                "n_populations": n_populations,
                "n_haplotype_states": n_haplotype_states,
                "modality_dropout": args.modality_dropout,
            },
            "objective_weights": weights.as_dict(),
            "input_sha256": input_sha256,
            "seed": args.seed,
            "best_epoch": best_epoch,
        },
        checkpoint,
    )
    (args.out_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary = {
        "status": "complete",
        "scope": "multimodal_objective_pilot",
        "promotable_to_primary_result": False,
        "input": input_label,
        "input_sha256": input_sha256,
        "device": str(device),
        "seed": args.seed,
        "nodes": int(len(arrays["topology_embeddings"])),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "objective_weights": weights.as_dict(),
        "test_losses": test_losses,
        "test_metrics": metrics,
        "mean_modality_weights": modality_weights,
        "runtime_seconds": time.monotonic() - started,
        "peak_rss_platform_units": int(peak_rss),
        "checkpoint": str(checkpoint.resolve()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-npz", type=Path)
    source.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--sequence-layers", type=int, default=2)
    parser.add_argument("--path-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--modality-dropout", type=float, default=0.1)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--max-path-length", type=int, default=1024)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--masked-token-weight", type=float, default=1.0)
    parser.add_argument("--path-order-weight", type=float, default=0.25)
    parser.add_argument("--branch-choice-weight", type=float, default=0.5)
    parser.add_argument("--cross-graph-weight", type=float, default=0.1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if args.gradient_clip <= 0:
        raise ValueError("gradient_clip must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

