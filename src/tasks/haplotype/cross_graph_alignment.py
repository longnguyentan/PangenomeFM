"""Build sequence-anchored HPRC/HGSVC loci and train cross-graph InfoNCE alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from analysis.network import build_adjacency, find_connected_components
from graph.neg_sampling import compute_oriented_degrees
from models.gat import build_node_features
from models.hierarchical_path import CrossGraphContrastiveLoss
from tasks.haplotype.embeddings import _load_frozen_graph_model

CHROMOSOME = re.compile(r"(chr(?:[0-9]+|X|Y))")


def _chromosome(value: str) -> str:
    match = CHROMOSOME.search(str(value))
    return match.group(1) if match else ""


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _sampled_kmers(sequence: str, k: int, stride: int) -> Iterator[tuple[int, str]]:
    sequence = sequence.upper()
    for offset in range(0, max(0, len(sequence) - k + 1), stride):
        kmer = sequence[offset : offset + k]
        if set(kmer) <= {"A", "C", "G", "T"}:
            yield offset, min(kmer, _reverse_complement(kmer))


def _digest(kmer: str) -> bytes:
    return hashlib.blake2b(kmer.encode("ascii"), digest_size=8).digest()


def _entropy(sequence: str) -> float:
    values = np.asarray([sequence.count(base) for base in "ACGT"], float)
    values = values[values > 0]
    if not len(values):
        return 0.0
    probabilities = values / values.sum()
    return float(-(probabilities * np.log2(probabilities)).sum() / 2.0)


def _reference_chunks(
    path: str | Path, chromosomes: set[str], chunksize: int = 100_000
) -> Iterator[pd.DataFrame]:
    columns = ["name", "seq", "LN", "SN", "SO", "SR"]
    for chunk in pd.read_csv(
        path, usecols=columns, compression="infer", chunksize=chunksize
    ):
        chunk["chromosome"] = chunk["SN"].map(_chromosome)
        selected = chunk[chunk["SR"].eq(0) & chunk["chromosome"].isin(chromosomes)]
        if not selected.empty:
            yield selected


def _selected_degrees(
    links_path: str | Path, selected: set[str]
) -> dict[str, int]:
    degrees = {name: 0 for name in selected}
    for chunk in pd.read_csv(
        links_path,
        usecols=["from_seg", "to_seg"],
        compression="infer",
        chunksize=250_000,
    ):
        for column in ("from_seg", "to_seg"):
            counts = chunk.loc[chunk[column].isin(selected), column].value_counts()
            for name, count in counts.items():
                degrees[str(name)] += int(count)
    return degrees


def build_same_locus_anchors(
    *,
    hprc_segments: str | Path,
    hprc_links: str | Path,
    hgsvc_segments: str | Path,
    hgsvc_links: str | Path,
    out_path: str | Path,
    chromosomes: tuple[str, ...] = ("chr19", "chr21", "chr22", "chrY"),
    k: int = 31,
    stride: int = 128,
    max_per_chromosome: int = 2_000,
) -> dict[str, Any]:
    chromosome_set = set(chromosomes)
    hgsvc_index: dict[bytes, tuple[str, dict[str, Any]] | None] = {}
    for chunk in _reference_chunks(hgsvc_segments, chromosome_set):
        for row in chunk.itertuples(index=False):
            sequence = str(row.seq).upper()
            for offset, kmer in _sampled_kmers(sequence, k, stride):
                key = _digest(kmer)
                value = (
                    kmer,
                    {
                        "hgsvc_name": str(row.name),
                        "hgsvc_chromosome": str(row.chromosome),
                        "hgsvc_so": int(row.SO),
                        "hgsvc_offset": offset,
                        "hgsvc_ln": int(row.LN),
                        "hgsvc_gc": (sequence.count("G") + sequence.count("C"))
                        / max(len(sequence), 1),
                        "hgsvc_entropy": _entropy(sequence),
                    },
                )
                if key in hgsvc_index:
                    hgsvc_index[key] = None
                else:
                    hgsvc_index[key] = value

    matches: list[dict[str, Any]] = []
    for chunk in _reference_chunks(hprc_segments, chromosome_set):
        for row in chunk.itertuples(index=False):
            sequence = str(row.seq).upper()
            for offset, kmer in _sampled_kmers(sequence, k, stride):
                candidate = hgsvc_index.get(_digest(kmer))
                if candidate is None or candidate[0] != kmer:
                    continue
                right = candidate[1]
                if right["hgsvc_chromosome"] != row.chromosome:
                    continue
                matches.append(
                    {
                        **right,
                        "hprc_name": str(row.name),
                        "hprc_chromosome": str(row.chromosome),
                        "hprc_so": int(row.SO),
                        "hprc_offset": offset,
                        "hprc_ln": int(row.LN),
                        "hprc_gc": (
                            sequence.count("G") + sequence.count("C")
                        )
                        / max(len(sequence), 1),
                        "hprc_entropy": _entropy(sequence),
                        "anchor_hash": _digest(kmer).hex(),
                    }
                )
    anchors = pd.DataFrame(matches)
    if anchors.empty:
        raise ValueError("No exact unique cross-graph sequence anchors were found.")
    anchors = anchors.drop_duplicates(["hprc_name", "hgsvc_name"])
    sampled = []
    for _, frame in anchors.groupby("hprc_chromosome"):
        frame = frame.sort_values(["hprc_so", "hgsvc_so"])
        if len(frame) > max_per_chromosome:
            indices = np.linspace(0, len(frame) - 1, max_per_chromosome).astype(int)
            frame = frame.iloc[indices]
        sampled.append(frame)
    anchors = pd.concat(sampled, ignore_index=True)
    hprc_degree = _selected_degrees(hprc_links, set(anchors["hprc_name"]))
    hgsvc_degree = _selected_degrees(hgsvc_links, set(anchors["hgsvc_name"]))
    chrom_max_hprc = anchors.groupby("hprc_chromosome")["hprc_so"].transform("max")
    chrom_max_hgsvc = anchors.groupby("hgsvc_chromosome")["hgsvc_so"].transform("max")
    anchors["hprc_degree"] = anchors["hprc_name"].map(hprc_degree).fillna(0)
    anchors["hgsvc_degree"] = anchors["hgsvc_name"].map(hgsvc_degree).fillna(0)
    for prefix, chrom_max in (
        ("hprc", chrom_max_hprc),
        ("hgsvc", chrom_max_hgsvc),
    ):
        relative_coordinate = (
            anchors[f"{prefix}_so"] + anchors[f"{prefix}_offset"]
        ) / chrom_max.clip(lower=1)
        anchors[f"{prefix}_f0_log_degree"] = np.log1p(
            anchors[f"{prefix}_degree"]
        )
        anchors[f"{prefix}_f1_branching"] = anchors[f"{prefix}_degree"].gt(2).astype(float)
        anchors[f"{prefix}_f2_relative_coordinate"] = relative_coordinate
        anchors[f"{prefix}_f3_sin_coordinate"] = np.sin(
            2 * np.pi * relative_coordinate
        )
        anchors[f"{prefix}_f4_cos_coordinate"] = np.cos(
            2 * np.pi * relative_coordinate
        )
        anchors[f"{prefix}_f5_sin_coordinate_2"] = np.sin(
            4 * np.pi * relative_coordinate
        )
        anchors[f"{prefix}_f6_cos_coordinate_2"] = np.cos(
            4 * np.pi * relative_coordinate
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(out_path, index=False, compression="infer")
    summary = {
        "task": "same_locus_cross_graph_anchors",
        "anchor_definition": (
            f"unique exact canonical {k}-mer shared by reference paths on the "
            "same named chromosome"
        ),
        "n_anchors": len(anchors),
        "anchors_per_chromosome": anchors["hprc_chromosome"].value_counts().to_dict(),
        "chromosomes": list(chromosomes),
        "k": k,
        "stride": stride,
        "max_per_chromosome": max_per_chromosome,
        "output": str(out_path),
    }
    out_path.with_suffix("").with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _encode_anchor_graph(
    *,
    anchor_names: set[str],
    segments_path: str | Path,
    links_path: str | Path,
    checkpoint_path: str | Path,
    device: str,
) -> dict[str, np.ndarray]:
    """Encode anchor nodes with their incident one-hop graph neighborhoods."""

    incident_parts: list[pd.DataFrame] = []
    selected_names = set(anchor_names)
    link_columns = ["from_seg", "from_orient", "to_seg", "to_orient"]
    for chunk in pd.read_csv(
        links_path,
        usecols=link_columns,
        compression="infer",
        chunksize=250_000,
    ):
        selected = chunk[
            chunk["from_seg"].isin(anchor_names)
            | chunk["to_seg"].isin(anchor_names)
        ]
        if selected.empty:
            continue
        incident_parts.append(selected.copy())
        selected_names.update(selected["from_seg"].astype(str))
        selected_names.update(selected["to_seg"].astype(str))
    if not incident_parts:
        raise ValueError("No incident links were found for the requested anchors.")
    links = pd.concat(incident_parts, ignore_index=True).drop_duplicates()

    segment_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        segments_path,
        usecols=["name", "LN", "SO", "SR"],
        compression="infer",
        chunksize=100_000,
    ):
        selected = chunk[chunk["name"].isin(selected_names)]
        if not selected.empty:
            segment_parts.append(selected.copy())
    segments = (
        pd.concat(segment_parts, ignore_index=True)
        .drop_duplicates("name")
        .set_index("name")
    )
    available_names = set(segments.index.astype(str))
    missing_anchors = sorted(anchor_names - available_names)
    if missing_anchors:
        raise ValueError(
            f"Segment table lacks {len(missing_anchors)} anchor nodes."
        )
    # Some serialized link tables retain endpoints removed during graph
    # canonicalization.  They cannot contribute features, so exclude those
    # dangling records while requiring every anchor itself to be present.
    links = links[
        links["from_seg"].isin(available_names)
        & links["to_seg"].isin(available_names)
    ].copy()
    names = sorted(available_names)
    name_to_index = {name: index for index, name in enumerate(names)}

    def oriented_id(name: str, orientation: str) -> int:
        return 2 * name_to_index[str(name)] + int(str(orientation) == "-")

    u = np.asarray(
        [
            oriented_id(row.from_seg, row.from_orient)
            for row in links.itertuples(index=False)
        ],
        dtype=np.int64,
    )
    v = np.asarray(
        [
            oriented_id(row.to_seg, row.to_orient)
            for row in links.itertuples(index=False)
        ],
        dtype=np.int64,
    )
    anchor_orientations = np.asarray(
        [
            2 * name_to_index[name] + bit
            for name in sorted(anchor_names)
            for bit in (0, 1)
        ],
        dtype=np.int64,
    )
    nodes = np.unique(np.concatenate([u, v, anchor_orientations]))
    dense = {int(value): index for index, value in enumerate(nodes)}
    src = np.asarray([dense[int(value)] for value in u], dtype=np.int64)
    dst = np.asarray([dense[int(value)] for value in v], dtype=np.int64)
    degrees = compute_oriented_degrees(u, v, nodes)
    adjacency = build_adjacency(u, v, nodes)
    components, _ = find_connected_components(adjacency)
    so: dict[int, int] = {}
    lengths: dict[int, int] = {}
    reference: dict[int, int] = {}
    for name, index in name_to_index.items():
        row = segments.loc[name]
        coordinate = int(row["SO"]) if pd.notna(row["SO"]) else 0
        length = int(row["LN"])
        is_reference = int(row["SR"]) == 0
        for bit in (0, 1):
            oid = 2 * index + bit
            so[oid] = coordinate
            lengths[oid] = length
            reference[oid] = int(is_reference)
    zeros = {int(value): 0 for value in nodes}
    features = build_node_features(
        nodes,
        so,
        lengths,
        zeros,
        reference,
        degrees,
        components,
    )
    model, _ = _load_frozen_graph_model(checkpoint_path, device)
    with torch.inference_mode():
        encoded = model.encode_nodes(
            torch.tensor(features, dtype=torch.float32, device=device),
            torch.tensor(
                [so.get(int(value), 0) for value in nodes],
                dtype=torch.long,
                device=device,
            ),
            torch.tensor(src, dtype=torch.long, device=device),
            torch.tensor(dst, dtype=torch.long, device=device),
            torch.ones(len(nodes), dtype=torch.float32, device=device),
            orient=torch.tensor(nodes % 2, dtype=torch.int8, device=device),
        ).detach().cpu().numpy()
    output: dict[str, np.ndarray] = {}
    for name in anchor_names:
        plus = encoded[dense[2 * name_to_index[name]]]
        minus = encoded[dense[2 * name_to_index[name] + 1]]
        output[name] = (plus + minus) / 2.0
    return output


def add_frozen_graph_embeddings(
    *,
    anchors_path: str | Path,
    hprc_segments: str | Path,
    hprc_links: str | Path,
    hgsvc_segments: str | Path,
    hgsvc_links: str | Path,
    checkpoint_path: str | Path,
    out_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Add actual frozen GraphGenome-FM vectors to same-locus anchors."""

    anchors = pd.read_csv(anchors_path, compression="infer")
    hprc = _encode_anchor_graph(
        anchor_names=set(anchors["hprc_name"].astype(str)),
        segments_path=hprc_segments,
        links_path=hprc_links,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    hgsvc = _encode_anchor_graph(
        anchor_names=set(anchors["hgsvc_name"].astype(str)),
        segments_path=hgsvc_segments,
        links_path=hgsvc_links,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    hprc_matrix = np.stack([hprc[str(name)] for name in anchors["hprc_name"]])
    hgsvc_matrix = np.stack([hgsvc[str(name)] for name in anchors["hgsvc_name"]])
    for index in range(hprc_matrix.shape[1]):
        anchors[f"hprc_emb_{index:03d}"] = hprc_matrix[:, index]
        anchors[f"hgsvc_emb_{index:03d}"] = hgsvc_matrix[:, index]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(out_path, index=False, compression="infer")
    summary = {
        "task": "frozen_graphgenomefm_cross_graph_anchor_embeddings",
        "anchors_path": str(anchors_path),
        "checkpoint": str(checkpoint_path),
        "n_anchors": int(len(anchors)),
        "embedding_dim": int(hprc_matrix.shape[1]),
        "orientation_pooling": "mean of plus and minus oriented node embeddings",
        "graph_context": "anchor nodes plus incident one-hop edges",
        "output": str(out_path),
    }
    out_path.with_suffix("").with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return values

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


class _Projector(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Linear(input_dim, 64, bias=False),
            nn.Tanh(),
            nn.Linear(64, input_dim, bias=False),
        )
        nn.init.zeros_(self.residual[-1].weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(values + self.residual(values), dim=1)


def _retrieval(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    similarity = left @ right.T
    ranking = torch.argsort(similarity, dim=1, descending=True)
    target = torch.arange(len(left), device=left.device).unsqueeze(1)
    ranks = (ranking == target).nonzero()[:, 1] + 1
    reverse = torch.argsort(similarity.T, dim=1, descending=True)
    reverse_ranks = (reverse == target).nonzero()[:, 1] + 1
    both = torch.cat([ranks, reverse_ranks]).float()
    return {
        "recall_at_1": float((both <= 1).float().mean()),
        "recall_at_5": float((both <= 5).float().mean()),
        "median_rank": float(both.median()),
    }


def train_cross_graph_alignment(
    *,
    anchors_path: str | Path,
    out_dir: str | Path,
    val_chromosome: str = "chr22",
    test_chromosome: str = "chrY",
    adversarial_weights: tuple[float, ...] = (0.0, 0.001, 0.01, 0.05, 0.1),
    epochs: int = 100,
    batch_size: int = 256,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    anchors = pd.read_csv(anchors_path, compression="infer")
    embedding_columns_present = any(
        column.startswith("hprc_emb_") for column in anchors
    )
    left_prefix = "hprc_emb_" if embedding_columns_present else "hprc_f"
    right_prefix = "hgsvc_emb_" if embedding_columns_present else "hgsvc_f"
    left_columns = sorted(
        column for column in anchors if column.startswith(left_prefix)
    )
    right_columns = sorted(
        column for column in anchors if column.startswith(right_prefix)
    )
    if len(left_columns) != len(right_columns):
        raise ValueError("Cross-graph feature dimensions differ.")
    chromosome = anchors["hprc_chromosome"].astype(str).to_numpy()
    train_idx = np.flatnonzero(
        (chromosome != val_chromosome) & (chromosome != test_chromosome)
    )
    val_idx = np.flatnonzero(chromosome == val_chromosome)
    test_idx = np.flatnonzero(chromosome == test_chromosome)
    if min(len(train_idx), len(val_idx), len(test_idx)) < 2:
        raise ValueError("Train, validation, and test chromosomes need >=2 anchors.")
    left = anchors[left_columns].to_numpy(np.float32)
    right = anchors[right_columns].to_numpy(np.float32)
    left_mean, left_std = left[train_idx].mean(0), left[train_idx].std(0)
    right_mean, right_std = right[train_idx].mean(0), right[train_idx].std(0)
    left_std[left_std < 1e-6] = 1
    right_std[right_std < 1e-6] = 1
    left = (left - left_mean) / left_std
    right = (right - right_mean) / right_std
    left_tensor = torch.tensor(left, device=device)
    right_tensor = torch.tensor(right, device=device)
    loss_function = CrossGraphContrastiveLoss(temperature=0.1)
    sweep_rows = []
    states = {}
    with torch.inference_mode():
        raw_left = F.normalize(left_tensor, dim=1)
        raw_right = F.normalize(right_tensor, dim=1)
        raw_validation = _retrieval(raw_left[val_idx], raw_right[val_idx])
        raw_test = _retrieval(raw_left[test_idx], raw_right[test_idx])

    for weight in adversarial_weights:
        torch.manual_seed(seed)
        left_model = _Projector(left.shape[1]).to(device)
        right_model = _Projector(right.shape[1]).to(device)
        discriminator = nn.Sequential(
            nn.Linear(left.shape[1], 32), nn.Tanh(), nn.Linear(32, 2)
        ).to(device)
        optimizer = torch.optim.AdamW(
            [*left_model.parameters(), *right_model.parameters(), *discriminator.parameters()],
            lr=2e-3,
            weight_decay=1e-4,
        )
        rng = np.random.default_rng(seed)
        best_state = {
            "left": {
                key: value.detach().cpu().clone()
                for key, value in left_model.state_dict().items()
            },
            "right": {
                key: value.detach().cpu().clone()
                for key, value in right_model.state_dict().items()
            },
            "discriminator": {
                key: value.detach().cpu().clone()
                for key, value in discriminator.state_dict().items()
            },
        }
        best_validation = raw_validation
        for epoch in range(epochs):
            shuffled = rng.permutation(train_idx)
            for start in range(0, len(shuffled), batch_size):
                batch = shuffled[start : start + batch_size]
                if len(batch) < 2:
                    continue
                batch_tensor = torch.tensor(batch, dtype=torch.long, device=device)
                projected_left = left_model(left_tensor[batch_tensor])
                projected_right = right_model(right_tensor[batch_tensor])
                contrastive = loss_function(projected_left, projected_right)
                joined = torch.cat([projected_left, projected_right])
                labels = torch.cat(
                    [
                        torch.zeros(len(batch), dtype=torch.long, device=device),
                        torch.ones(len(batch), dtype=torch.long, device=device),
                    ]
                )
                reversed_values = _GradientReverse.apply(joined, float(weight))
                domain_loss = F.cross_entropy(discriminator(reversed_values), labels)
                loss = contrastive + domain_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 5 == 0:
                left_model.eval()
                right_model.eval()
                with torch.inference_mode():
                    current_validation = _retrieval(
                        left_model(left_tensor[val_idx]),
                        right_model(right_tensor[val_idx]),
                    )
                current_key = (
                    current_validation["recall_at_1"],
                    current_validation["recall_at_5"],
                    -current_validation["median_rank"],
                )
                best_key = (
                    best_validation["recall_at_1"],
                    best_validation["recall_at_5"],
                    -best_validation["median_rank"],
                )
                if current_key > best_key:
                    best_validation = current_validation
                    best_state = {
                        "left": {
                            key: value.detach().cpu().clone()
                            for key, value in left_model.state_dict().items()
                        },
                        "right": {
                            key: value.detach().cpu().clone()
                            for key, value in right_model.state_dict().items()
                        },
                        "discriminator": {
                            key: value.detach().cpu().clone()
                            for key, value in discriminator.state_dict().items()
                        },
                    }
                left_model.train()
                right_model.train()
        left_model.load_state_dict(best_state["left"])
        right_model.load_state_dict(best_state["right"])
        discriminator.load_state_dict(best_state["discriminator"])
        left_model.eval()
        right_model.eval()
        discriminator.eval()
        with torch.inference_mode():
            projected_left = left_model(left_tensor)
            projected_right = right_model(right_tensor)
            val_metrics = _retrieval(
                projected_left[val_idx], projected_right[val_idx]
            )
            test_metrics = _retrieval(
                projected_left[test_idx], projected_right[test_idx]
            )
            joined = torch.cat([projected_left[val_idx], projected_right[val_idx]])
            labels = torch.cat(
                [
                    torch.zeros(len(val_idx), dtype=torch.long, device=device),
                    torch.ones(len(val_idx), dtype=torch.long, device=device),
                ]
            )
            domain_accuracy = float(
                (discriminator(joined).argmax(1) == labels).float().mean()
            )
        row = {
            "adversarial_weight": weight,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
            "val_domain_accuracy": domain_accuracy,
        }
        sweep_rows.append(row)
        states[weight] = {
            "left": left_model.state_dict(),
            "right": right_model.state_dict(),
            "discriminator": discriminator.state_dict(),
        }
    sweep = pd.DataFrame(sweep_rows)
    best_row = sweep.sort_values(
        ["val_recall_at_1", "val_recall_at_5", "adversarial_weight"],
        ascending=[False, False, True],
    ).iloc[0]
    best_weight = float(best_row["adversarial_weight"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out_dir / "dann_validation_sweep.csv", index=False)
    torch.save(
        {
            **states[best_weight],
            "adversarial_weight": best_weight,
            "left_columns": left_columns,
            "right_columns": right_columns,
            "normalization": {
                "left_mean": left_mean,
                "left_std": left_std,
                "right_mean": right_mean,
                "right_std": right_std,
            },
        },
        out_dir / "best_alignment.pt",
    )
    summary = {
        "task": "cross_graph_infonce_then_dann_sweep",
        "n_anchors": len(anchors),
        "feature_source": (
            "frozen_graphgenomefm_embeddings"
            if embedding_columns_present
            else "handcrafted_structural_coordinate_features"
        ),
        "train_chromosomes": sorted(set(chromosome[train_idx])),
        "validation_chromosome": val_chromosome,
        "test_chromosome": test_chromosome,
        "selection_rule": "maximize validation recall@1, then recall@5; lower adversarial weight breaks ties",
        "unaligned_validation": raw_validation,
        "unaligned_locked_test": raw_test,
        "best_adversarial_weight": best_weight,
        "best_validation": {
            key: float(value)
            for key, value in best_row.items()
            if key.startswith("val_")
        },
        "locked_test": {
            key.removeprefix("test_"): float(value)
            for key, value in best_row.items()
            if key.startswith("test_")
        },
        "sweep": sweep_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--hprc-segments", required=True)
    build.add_argument("--hprc-links", required=True)
    build.add_argument("--hgsvc-segments", required=True)
    build.add_argument("--hgsvc-links", required=True)
    build.add_argument("--out", required=True)
    align = subparsers.add_parser("align")
    align.add_argument("--anchors", required=True)
    align.add_argument("--out-dir", required=True)
    align.add_argument("--epochs", type=int, default=100)
    align.add_argument("--device", default="cpu")
    embed = subparsers.add_parser("embed")
    embed.add_argument("--anchors", required=True)
    embed.add_argument("--hprc-segments", required=True)
    embed.add_argument("--hprc-links", required=True)
    embed.add_argument("--hgsvc-segments", required=True)
    embed.add_argument("--hgsvc-links", required=True)
    embed.add_argument("--checkpoint", required=True)
    embed.add_argument("--out", required=True)
    embed.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "build":
        result = build_same_locus_anchors(
            hprc_segments=args.hprc_segments,
            hprc_links=args.hprc_links,
            hgsvc_segments=args.hgsvc_segments,
            hgsvc_links=args.hgsvc_links,
            out_path=args.out,
        )
    elif args.command == "align":
        result = train_cross_graph_alignment(
            anchors_path=args.anchors,
            out_dir=args.out_dir,
            epochs=args.epochs,
            device=args.device,
        )
    else:
        result = add_frozen_graph_embeddings(
            anchors_path=args.anchors,
            hprc_segments=args.hprc_segments,
            hprc_links=args.hprc_links,
            hgsvc_segments=args.hgsvc_segments,
            hgsvc_links=args.hgsvc_links,
            checkpoint_path=args.checkpoint,
            out_path=args.out,
            device=args.device,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
