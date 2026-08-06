"""Quantitative latent-space audit for trained GraphGenome-FM checkpoints.

The unit of analysis is a node occurrence in a benchmark slice. Keeping slice
occurrences separate preserves an unambiguous local graph for topology and
shortest-path calculations, while still allowing joint HPRC/HGSVC projections.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/graphgenomefm-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/graphgenomefm-xdg-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/graphgenomefm-numba")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.external import _build_model_from_checkpoint, _namespace_from_checkpoint
from graph.features import build_oid_metadata_from_segments
from graph.slicing import build_global_index
from training.pretrain import (
    _compute_adaptive_window_k,
    load_slice,
    mask_positive_query_edges,
    tensorize_slice,
)


def _select_rows(manifest: Path, closure: str, max_slices: int) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    if closure != "all":
        df = df[df["closure"].astype(str) == closure]
    if len(df) <= max_slices:
        return df.reset_index(drop=True)
    selected = np.linspace(0, len(df) - 1, max_slices).round().astype(int)
    return df.iloc[np.unique(selected)].reset_index(drop=True)


def _load_raw_slices(
    dataset_name: str,
    manifest: Path,
    full_segments: Path,
    args: argparse.Namespace,
    closure: str,
    max_slices: int,
) -> list[dict[str, Any]]:
    print(f"[latent] loading {dataset_name}: {manifest}")
    rows = _select_rows(manifest, closure, max_slices)
    segment_frames = [
        pd.read_csv(
            row["segments_path"],
            compression="infer",
            usecols=["id", "name", "LN", "SN", "SO", "SR"],
        )
        for _, row in rows.iterrows()
    ]
    selected_segments = pd.concat(segment_frames, ignore_index=True).drop_duplicates("name")
    names = selected_segments["name"].astype(str)
    standard_numbering = (
        names.str.match(r"^s\d+$").all()
        and (
            selected_segments["id"].astype(str).to_numpy()
            == selected_segments["name"].astype(str).to_numpy()
        ).all()
    )
    if standard_numbering:
        max_segid = int(names.str[1:].astype(int).max()) - 1
        seg_index = pd.Index([f"s{i}" for i in range(1, max_segid + 2)], dtype="string")
        metadata: dict[str, dict[int, Any]] = {
            "oid_to_sn": {},
            "oid_to_so": {},
            "oid_to_ln": {},
            "oid_to_sr": {},
            "oid_to_is_grch38": {},
        }
        for row in selected_segments.itertuples(index=False):
            segid = int(str(row.name)[1:]) - 1
            is_reference = int(
                int(row.SR) == 0
                or str(row.SN).startswith(("GRCh38", "CHM13", "id=CHM13"))
            )
            for bit in (0, 1):
                oid = segid * 2 + bit
                metadata["oid_to_sn"][oid] = str(row.SN)
                metadata["oid_to_so"][oid] = int(row.SO)
                metadata["oid_to_ln"][oid] = int(row.LN)
                metadata["oid_to_sr"][oid] = int(row.SR)
                metadata["oid_to_is_grch38"][oid] = is_reference
    else:
        # Targeted HGSVC tables renumber rows while retaining original s<number>
        # names, so reconstruct that smaller global index from the full table.
        full_metadata = pd.read_csv(
            full_segments,
            compression="infer",
            usecols=["id", "name", "LN", "SN", "SO", "SR"],
        )
        seg_index, _ = build_global_index(full_metadata)
        metadata = build_oid_metadata_from_segments(full_metadata, seg_index)
    slices: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        raw = load_slice(row, seg_index, metadata, selected_segments, args)
        if raw is not None:
            raw["dataset"] = dataset_name
            slices.append(raw)
    del selected_segments, metadata
    print(f"[latent] loaded {len(slices)} evaluable {dataset_name} slices")
    return slices


def _gate_hooks(model):
    captured: list[np.ndarray] = []
    handles = []
    if model.fusion_modules is None:
        return captured, handles
    for module in model.fusion_modules:
        gate = getattr(module, "gate", None)
        if gate is not None:
            handles.append(
                gate.register_forward_hook(
                    lambda _module, _inputs, output: captured.append(
                        output.detach().cpu().numpy()
                    )
                )
            )
    return captured, handles


@torch.no_grad()
def _encode(
    raw_slices: list[dict[str, Any]],
    checkpoint: Path,
    seed: int,
    device_name: str,
    mask_queries: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    device = torch.device(device_name)
    ckpt = torch.load(checkpoint, map_location=device)
    eval_args = _namespace_from_checkpoint(ckpt, seed=seed)
    model, _ = _build_model_from_checkpoint(ckpt, eval_args, device)
    all_h: list[np.ndarray] = []
    all_raw: list[np.ndarray] = []
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, int]] = []
    global_offset = 0

    for slice_idx, raw in enumerate(raw_slices):
        sd = tensorize_slice(raw, device, eval_args)
        if eval_args.adaptive_window:
            window_k = _compute_adaptive_window_k(
                sd["branching_frac"],
                eval_args.adaptive_window_base,
                eval_args.adaptive_window_alpha,
            )
            for layer in model.linear_layers:
                layer.window_k = window_k

        src, dst, edge_attr = sd["src"], sd["dst"], sd["edge_attr"]
        if mask_queries:
            all_idx = torch.arange(len(sd["labels"]), dtype=torch.long, device=device)
            src, dst, edge_attr = mask_positive_query_edges(
                src,
                dst,
                edge_attr,
                sd["q_u"],
                sd["q_v"],
                sd["labels"],
                all_idx,
            )

        captured, handles = _gate_hooks(model)
        h = model.encode_nodes(
            sd["X"],
            sd["so"],
            src,
            dst,
            sd["temps"],
            edge_attr,
            sd["orient"],
            sd["pop_ids"],
        )
        for handle in handles:
            handle.remove()
        gate = (
            np.mean(np.stack([value.mean(axis=1) for value in captured]), axis=0)
            if captured
            else np.full(len(raw["nodes"]), np.nan)
        )
        h_np = h.detach().cpu().numpy().astype(np.float32)
        all_h.append(h_np)
        all_raw.append(raw["node_feats"].astype(np.float32))

        degree = np.bincount(
            np.concatenate([raw["src"], raw["dst"]]),
            minlength=len(raw["nodes"]),
        )
        chrom = str(raw["target_sn"]).split("#")[-1].split("|")[-1]
        for local_idx, oid in enumerate(raw["nodes"]):
            node_rows.append(
                {
                    "global_id": global_offset + local_idx,
                    "dataset": raw["dataset"],
                    "slice_idx": slice_idx,
                    "slice": raw["name"],
                    "target_sn": raw["target_sn"],
                    "chrom": chrom,
                    "oid": int(oid),
                    "segid": int(oid) // 2,
                    "orientation": int(oid) % 2,
                    "SO": int(raw["so_arr"][local_idx]),
                    "degree": int(degree[local_idx]),
                    "is_reference": float(raw["node_feats"][local_idx, 3]),
                    "branching": int(degree[local_idx] > 2),
                    "fusion_gate_coordinate_weight": float(gate[local_idx]),
                }
            )
        for u, v in zip(raw["src"], raw["dst"]):
            edge_rows.append(
                {
                    "slice_idx": slice_idx,
                    "source": global_offset + int(u),
                    "target": global_offset + int(v),
                }
            )
        global_offset += len(raw["nodes"])

    return (
        np.concatenate(all_h),
        np.concatenate(all_raw),
        pd.DataFrame(node_rows),
        pd.DataFrame(edge_rows),
    )


def _subsample(
    H: np.ndarray,
    raw: np.ndarray,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    max_nodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    if len(nodes) <= max_nodes:
        nodes = nodes.copy()
        nodes["occurrence_id"] = nodes["global_id"]
        return H, raw, nodes, edges
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(nodes), max_nodes, replace=False))
    old_to_new = {int(old): new for new, old in enumerate(keep)}
    edge_keep = edges["source"].isin(old_to_new) & edges["target"].isin(old_to_new)
    edges = edges.loc[edge_keep].copy()
    edges["source"] = edges["source"].map(old_to_new)
    edges["target"] = edges["target"].map(old_to_new)
    nodes = nodes.iloc[keep].reset_index(drop=True)
    nodes["occurrence_id"] = nodes["global_id"]
    nodes["global_id"] = np.arange(len(nodes))
    return H[keep], raw[keep], nodes, edges.reset_index(drop=True)


def _knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    n_neighbors = min(k + 1, len(X))
    return NearestNeighbors(n_neighbors=n_neighbors).fit(X).kneighbors(return_distance=False)[:, 1:]


def _knn_overlap(high: np.ndarray, low: np.ndarray, k: int) -> float:
    high_nn = _knn_indices(high, k)
    low_nn = _knn_indices(low, k)
    return float(
        np.mean(
            [
                len(set(a.tolist()).intersection(b.tolist())) / max(len(a), 1)
                for a, b in zip(high_nn, low_nn)
            ]
        )
    )


def _edge_neighbor_recall(X: np.ndarray, edges: pd.DataFrame, k: int) -> float:
    if edges.empty:
        return float("nan")
    knn = _knn_indices(X, k)
    graph_neighbors: list[set[int]] = [set() for _ in range(len(X))]
    for u, v in edges[["source", "target"]].itertuples(index=False):
        graph_neighbors[int(u)].add(int(v))
        graph_neighbors[int(v)].add(int(u))
    recalls = []
    for i, neighbors in enumerate(graph_neighbors):
        if neighbors:
            recalls.append(len(neighbors.intersection(knn[i].tolist())) / len(neighbors))
    return float(np.mean(recalls)) if recalls else float("nan")


def _distance_correlation(
    H: np.ndarray,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    seed: int,
) -> tuple[float, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for slice_idx, grp in nodes.groupby("slice_idx"):
        global_ids = grp.index.to_numpy()
        if len(global_ids) < 3:
            continue
        local_map = {int(global_id): i for i, global_id in enumerate(global_ids)}
        graph = nx.Graph()
        graph.add_nodes_from(range(len(global_ids)))
        slice_edges = edges[edges["slice_idx"] == slice_idx]
        graph.add_edges_from(
            (local_map[int(u)], local_map[int(v)])
            for u, v in slice_edges[["source", "target"]].itertuples(index=False)
            if int(u) in local_map and int(v) in local_map
        )
        sources = rng.choice(len(global_ids), min(25, len(global_ids)), replace=False)
        for source in sources:
            lengths = nx.single_source_shortest_path_length(graph, int(source), cutoff=8)
            targets = [target for target, dist in lengths.items() if target != source and dist > 0]
            if not targets:
                continue
            for target in rng.choice(targets, min(8, len(targets)), replace=False):
                rows.append(
                    {
                        "slice_idx": int(slice_idx),
                        "graph_distance": int(lengths[int(target)]),
                        "embedding_distance": float(
                            np.linalg.norm(H[global_ids[int(source)]] - H[global_ids[int(target)]])
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    if len(frame) < 3:
        return float("nan"), frame
    rho = spearmanr(frame["graph_distance"], frame["embedding_distance"]).statistic
    return float(rho), frame


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    denom = np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro")
    return float(cross / denom) if denom > 0 else float("nan")


def _safe_silhouette(X: np.ndarray, labels: np.ndarray, seed: int) -> float:
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or np.min(counts) < 2:
        return float("nan")
    if len(X) > 5000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), 5000, replace=False)
        X, labels = X[idx], labels[idx]
    return float(silhouette_score(X, labels))


def _plot_projection(
    projection: np.ndarray,
    nodes: pd.DataFrame,
    method: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("dataset", "Dataset", "tab10"),
        ("chrom", "Chromosome", "tab20"),
        ("degree", "log1p(degree)", "viridis"),
    ]
    for ax, (column, title, cmap) in zip(axes, panels):
        if column == "degree":
            values = np.log1p(nodes[column].to_numpy(float))
            scatter = ax.scatter(
                projection[:, 0],
                projection[:, 1],
                c=values,
                cmap=cmap,
                s=4,
                alpha=0.55,
                linewidths=0,
            )
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        else:
            categories = pd.Categorical(nodes[column])
            ax.scatter(
                projection[:, 0],
                projection[:, 1],
                c=categories.codes,
                cmap=cmap,
                s=4,
                alpha=0.55,
                linewidths=0,
            )
            labels = list(categories.categories)
            if len(labels) <= 12:
                handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        color=plt.get_cmap(cmap)(i / max(len(labels) - 1, 1)),
                        label=str(label),
                        markersize=5,
                    )
                    for i, label in enumerate(labels)
                ]
                ax.legend(handles=handles, fontsize=6, frameon=False, loc="best")
        ax.set_title(title)
        ax.set_xlabel(f"{method}-1")
        ax.set_ylabel(f"{method}-2")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs=3,
        action="append",
        metavar=("NAME", "MANIFEST", "FULL_SEGMENTS"),
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        required=True,
        help="First checkpoint supplies the paper projection; extras quantify stability.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--closure", choices=["strict", "1hop", "all"], default="1hop")
    parser.add_argument("--max-slices-per-dataset", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--keep-query-edges", action="store_true")
    parser.add_argument("--skip-phate", action="store_true")
    parser.add_argument("--skip-tsne", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    first_ckpt = Path(args.checkpoint[0][1])
    train_ckpt = torch.load(first_ckpt, map_location="cpu")
    load_args = _namespace_from_checkpoint(train_ckpt, seed=args.seed)

    raw_slices: list[dict[str, Any]] = []
    for name, manifest, full_segments in args.dataset:
        raw_slices.extend(
            _load_raw_slices(
                name,
                Path(manifest),
                Path(full_segments),
                load_args,
                args.closure,
                args.max_slices_per_dataset,
            )
        )
    if not raw_slices:
        raise RuntimeError("No slices were loaded.")

    checkpoint_embeddings: dict[str, np.ndarray] = {}
    H, raw, nodes, edges = _encode(
        raw_slices,
        first_ckpt,
        args.seed,
        args.device,
        not args.keep_query_edges,
    )
    checkpoint_embeddings[args.checkpoint[0][0]] = H
    H, raw, nodes, edges = _subsample(H, raw, nodes, edges, args.max_nodes, args.seed)
    kept_ids = nodes["occurrence_id"].to_numpy(int)

    for label, checkpoint in args.checkpoint[1:]:
        H_other, _, _, _ = _encode(
            raw_slices,
            Path(checkpoint),
            args.seed,
            args.device,
            not args.keep_query_edges,
        )
        checkpoint_embeddings[label] = H_other[kept_ids]
    checkpoint_embeddings[args.checkpoint[0][0]] = H

    scaled = StandardScaler().fit_transform(H)
    pca = PCA(n_components=2, random_state=args.seed).fit_transform(scaled)
    projections: dict[str, np.ndarray] = {"pca": pca}
    try:
        import umap

        projections["umap"] = umap.UMAP(
            n_neighbors=args.neighbors,
            min_dist=0.1,
            metric="euclidean",
            random_state=args.seed,
        ).fit_transform(scaled)
    except Exception as exc:
        print(f"[latent] UMAP unavailable ({exc}); skipped UMAP")
    if not args.skip_tsne:
        projections["tsne"] = TSNE(
            n_components=2,
            perplexity=min(30, max(5, (len(H) - 1) // 3)),
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
        ).fit_transform(scaled)
    if not args.skip_phate:
        try:
            import phate

            projections["phate"] = phate.PHATE(
                n_components=2,
                knn=args.neighbors,
                random_state=args.seed,
                verbose=False,
            ).fit_transform(scaled)
        except Exception as exc:
            print(f"[latent] PHATE unavailable ({exc}); skipped PHATE")

    metrics: list[dict[str, Any]] = []
    for method, projection in projections.items():
        nodes[f"{method}_1"] = projection[:, 0]
        nodes[f"{method}_2"] = projection[:, 1]
        metrics.extend(
            [
                {
                    "analysis": "projection",
                    "method": method,
                    "metric": "trustworthiness",
                    "value": float(
                        trustworthiness(
                            scaled,
                            projection,
                            n_neighbors=min(args.neighbors, len(H) // 2 - 1),
                        )
                    ),
                },
                {
                    "analysis": "projection",
                    "method": method,
                    "metric": "knn_preservation",
                    "value": _knn_overlap(scaled, projection, args.neighbors),
                },
            ]
        )
        _plot_projection(projection, nodes, method.upper(), out_dir / f"{method}_latent.png")

    coordinate = raw[:, :2]
    metrics.extend(
        [
            {
                "analysis": "topology",
                "method": "learned_embedding",
                "metric": f"edge_neighbor_recall_at_{args.neighbors}",
                "value": _edge_neighbor_recall(scaled, edges, args.neighbors),
            },
            {
                "analysis": "topology",
                "method": "coordinate_features",
                "metric": f"edge_neighbor_recall_at_{args.neighbors}",
                "value": _edge_neighbor_recall(coordinate, edges, args.neighbors),
            },
            {
                "analysis": "separation",
                "method": "learned_embedding",
                "metric": "silhouette_dataset",
                "value": _safe_silhouette(scaled, nodes["dataset"].to_numpy(), args.seed),
            },
            {
                "analysis": "separation",
                "method": "learned_embedding",
                "metric": "silhouette_chromosome",
                "value": _safe_silhouette(scaled, nodes["chrom"].to_numpy(), args.seed),
            },
        ]
    )

    rho, distance_frame = _distance_correlation(scaled, nodes, edges, args.seed)
    metrics.append(
        {
            "analysis": "topology",
            "method": "learned_embedding",
            "metric": "spearman_graph_vs_embedding_distance",
            "value": rho,
        }
    )
    if not distance_frame.empty:
        distance_frame.to_csv(out_dir / "graph_embedding_distances.csv", index=False)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(
            distance_frame["graph_distance"],
            distance_frame["embedding_distance"],
            s=8,
            alpha=0.35,
        )
        ax.set_xlabel("Shortest-path distance")
        ax.set_ylabel("Embedding distance")
        ax.set_title(f"Graph vs embedding distance (Spearman ρ={rho:.3f})")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / "graph_vs_embedding_distance.png", dpi=200)
        plt.close(fig)

    labels = list(checkpoint_embeddings)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            left = checkpoint_embeddings[labels[i]]
            right = checkpoint_embeddings[labels[j]]
            disparity = procrustes(left, right)[2]
            metrics.extend(
                [
                    {
                        "analysis": "stability",
                        "method": f"{labels[i]}_vs_{labels[j]}",
                        "metric": "linear_cka",
                        "value": _linear_cka(left, right),
                    },
                    {
                        "analysis": "stability",
                        "method": f"{labels[i]}_vs_{labels[j]}",
                        "metric": "procrustes_disparity",
                        "value": float(disparity),
                    },
                ]
            )

    try:
        from ripser import ripser

        rng = np.random.default_rng(args.seed)
        persistence_idx = (
            rng.choice(len(H), 1000, replace=False) if len(H) > 1000 else np.arange(len(H))
        )
        diagrams = ripser(scaled[persistence_idx], maxdim=1)["dgms"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for dim, diagram in enumerate(diagrams[:2]):
            finite = diagram[np.isfinite(diagram[:, 1])]
            axes[dim].scatter(finite[:, 0], finite[:, 1], s=10, alpha=0.6)
            if len(finite):
                limit = float(finite.max())
                axes[dim].plot([0, limit], [0, limit], "--", color="grey", linewidth=1)
            axes[dim].set_title(f"Persistence H{dim}")
            axes[dim].set_xlabel("Birth")
            axes[dim].set_ylabel("Death")
        fig.tight_layout()
        fig.savefig(out_dir / "persistence_diagrams.png", dpi=200)
        plt.close(fig)
        for dim, diagram in enumerate(diagrams[:2]):
            finite = diagram[np.isfinite(diagram[:, 1])]
            metrics.append(
                {
                    "analysis": "topology",
                    "method": "learned_embedding",
                    "metric": f"persistence_h{dim}_total",
                    "value": float(np.sum(finite[:, 1] - finite[:, 0])),
                }
            )
    except Exception as exc:
        print(f"[latent] persistence unavailable ({exc}); skipped persistence diagrams")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    nodes.to_csv(out_dir / "nodes.csv.gz", index=False, compression="gzip")
    edges.to_csv(out_dir / "edges.csv.gz", index=False, compression="gzip")
    np.savez_compressed(out_dir / "embeddings.npz", embedding=H, raw_features=raw)

    gate_summary = (
        nodes.groupby("branching")["fusion_gate_coordinate_weight"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    gate_summary.to_csv(out_dir / "fusion_gate_by_branching.csv", index=False)
    summary = {
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "n_slices": int(len(raw_slices)),
        "datasets": nodes["dataset"].value_counts().to_dict(),
        "chromosomes": nodes["chrom"].value_counts().to_dict(),
        "closure": args.closure,
        "query_edges_masked": not args.keep_query_edges,
        "checkpoints": {label: path for label, path in args.checkpoint},
        "metrics": metrics,
        "gate_by_branching": gate_summary.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics_df.to_string(index=False))
    print(f"[latent] outputs: {out_dir}")


if __name__ == "__main__":
    main()
