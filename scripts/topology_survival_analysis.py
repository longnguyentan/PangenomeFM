#!/usr/bin/env python
"""Compute topology-survival metrics for masked link-prediction candidates.

The script reconstructs local benchmark slices, removes each positive query edge
from the observed local topology, and records how much alternate neighborhood
context remains around the candidate endpoints.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from graph.features import build_oid_metadata_from_segments
from graph.io import read_segments_csv
from graph.slicing import build_global_index
from training.pretrain import load_slice


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    return root / path


def _adjacency(src: np.ndarray, dst: np.ndarray) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {}
    for u, v in zip(src.tolist(), dst.tolist()):
        u_i = int(u)
        v_i = int(v)
        adj.setdefault(u_i, set()).add(v_i)
        adj.setdefault(v_i, set()).add(u_i)
    return adj


def _masked_neighbors(adj: dict[int, set[int]], u: int, v: int, label: int) -> tuple[set[int], set[int]]:
    nu = set(adj.get(u, set()))
    nv = set(adj.get(v, set()))
    if label == 1:
        nu.discard(v)
        nv.discard(u)
    return nu, nv


def _shortest_path(
    adj: dict[int, set[int]],
    u: int,
    v: int,
    *,
    label: int,
    cutoff: int,
) -> int | None:
    if u == v:
        return 0
    seen = {u}
    queue: deque[tuple[int, int]] = deque([(u, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= cutoff:
            continue
        for nbr in adj.get(node, set()):
            if label == 1 and ((node == u and nbr == v) or (node == v and nbr == u)):
                continue
            if nbr == v:
                return depth + 1
            if nbr not in seen:
                seen.add(nbr)
                queue.append((nbr, depth + 1))
    return None


def _safe_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {"n": int(len(y)), "positive_fraction": float(np.mean(y)) if len(y) else float("nan")}
    if len(y) and len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def _bin_numeric(values: pd.Series, n_bins: int) -> pd.Series:
    if values.nunique(dropna=True) <= 1:
        return pd.Series(["all"] * len(values), index=values.index)
    try:
        return pd.qcut(values, q=n_bins, duplicates="drop").astype(str)
    except ValueError:
        return pd.cut(values, bins=n_bins, duplicates="drop").astype(str)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--full-segments", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shortest-path-cutoff", type=int, default=12)
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--filter-column", default=None)
    ap.add_argument("--filter-values", nargs="+", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions, compression="infer")
    needed = {"slice", "u_local", "v_local", "y_true", "p_edge"}
    missing = needed - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")
    if args.filter_column:
        if args.filter_column not in predictions.columns:
            raise ValueError(f"Predictions do not contain filter column {args.filter_column!r}")
        allowed = set(args.filter_values or [])
        predictions = predictions[predictions[args.filter_column].astype(str).isin(allowed)].reset_index(drop=True)
        if predictions.empty:
            raise ValueError(
                f"No prediction rows where {args.filter_column} is in {sorted(allowed)}"
            )

    segments = read_segments_csv(args.full_segments)
    seg_index, _ = build_global_index(segments)
    md = build_oid_metadata_from_segments(segments, seg_index)

    manifest = pd.read_csv(args.manifest)
    manifest_by_name = {str(row["name"]): row for _, row in manifest.iterrows()}
    loader_args = SimpleNamespace(seed=args.seed, pop_cond=False, use_edge_features=False)

    rows: list[dict[str, object]] = []
    for slice_name, pred_sub in predictions.groupby("slice", sort=False):
        if slice_name not in manifest_by_name:
            continue
        sd = load_slice(manifest_by_name[slice_name], seg_index, md, segments, loader_args)
        if sd is None:
            continue
        adj = _adjacency(sd["src"], sd["dst"])
        degrees = np.array([len(adj.get(i, set())) for i in range(len(sd["nodes"]))])
        branching_nodes = set(np.where(degrees > 2)[0].astype(int).tolist())
        for rec in pred_sub.itertuples(index=False):
            u = int(rec.u_local)
            v = int(rec.v_local)
            label = int(round(float(rec.y_true)))
            nu, nv = _masked_neighbors(adj, u, v, label)
            common = nu & nv
            union = nu | nv
            sp = _shortest_path(
                adj,
                u,
                v,
                label=label,
                cutoff=args.shortest_path_cutoff,
            )
            local_nodes = {u, v} | nu | nv
            branch_local = len(local_nodes & branching_nodes)
            rows.append(
                {
                    "slice": slice_name,
                    "target_sn": getattr(rec, "target_sn", sd["target_sn"]),
                    "closure": getattr(rec, "closure", sd["closure"]),
                    "u_local": u,
                    "v_local": v,
                    "y_true": float(rec.y_true),
                    "p_edge": float(rec.p_edge),
                    "connected_after_mask": sp is not None,
                    "shortest_path_after_mask": sp if sp is not None else np.nan,
                    "shared_neighbors_after_mask": int(len(common)),
                    "two_hop_paths_after_mask": int(len(common)),
                    "endpoint_degree_u_after_mask": int(len(nu)),
                    "endpoint_degree_v_after_mask": int(len(nv)),
                    "endpoint_degree_sum_after_mask": int(len(nu) + len(nv)),
                    "endpoint_degree_product_after_mask": int(len(nu) * len(nv)),
                    "jaccard_after_mask": float(len(common) / len(union)) if union else 0.0,
                    "local_branching_nodes": int(branch_local),
                    "local_branching_fraction": float(branch_local / max(len(local_nodes), 1)),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "candidate_topology_survival.csv.gz", index=False, compression="gzip")

    summary: dict[str, object] = {
        "predictions": str(args.predictions),
        "manifest": str(args.manifest),
        "n_candidates": int(len(out)),
    }
    if not out.empty:
        positives = out[out["y_true"] > 0.5]
        summary["n_positive_candidates"] = int(len(positives))
        if len(positives):
            summary["positive_connected_after_mask_fraction"] = float(positives["connected_after_mask"].mean())
            summary["positive_mean_shared_neighbors"] = float(positives["shared_neighbors_after_mask"].mean())
            summary["positive_mean_two_hop_paths"] = float(positives["two_hop_paths_after_mask"].mean())
            summary["positive_mean_endpoint_degree_sum"] = float(positives["endpoint_degree_sum_after_mask"].mean())
        summary["overall_metrics"] = _safe_metrics(
            out["y_true"].to_numpy(np.float32),
            out["p_edge"].to_numpy(np.float32),
        )

        bin_rows: list[dict[str, object]] = []
        for feature in [
            "connected_after_mask",
            "shortest_path_after_mask",
            "shared_neighbors_after_mask",
            "endpoint_degree_sum_after_mask",
            "local_branching_fraction",
        ]:
            tmp = out.copy()
            if tmp[feature].dtype == bool:
                tmp["_bin"] = tmp[feature].astype(str)
            else:
                finite = tmp[feature].replace([np.inf, -np.inf], np.nan)
                tmp["_bin"] = _bin_numeric(finite.fillna(finite.max() + 1 if finite.notna().any() else 0), args.bins)
            for bin_name, sub in tmp.groupby("_bin", sort=False):
                m = _safe_metrics(
                    sub["y_true"].to_numpy(np.float32),
                    sub["p_edge"].to_numpy(np.float32),
                )
                bin_rows.append(
                    {
                        "feature": feature,
                        "bin": str(bin_name),
                        **m,
                        "mean_p_edge_positive": float(sub.loc[sub["y_true"] > 0.5, "p_edge"].mean())
                        if (sub["y_true"] > 0.5).any()
                        else float("nan"),
                    }
                )
        pd.DataFrame(bin_rows).to_csv(out_dir / "metrics_by_topology_bin.csv", index=False)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
