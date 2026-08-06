"""Topology-only heuristic baselines for pangenome link prediction.

The evaluator scores candidate edges from an existing benchmark manifest with
standard non-parametric graph link-prediction heuristics. Positive query edges
are masked by default before scoring, so the direct edge itself is not counted as
observed topology.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from graph.neg_sampling import oriented_ids_from_links
from graph.slicing import build_global_index
from utils.versioning import resolve_run_dir


def _resolve_path(value: str | Path, manifest_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return manifest_dir / path


def _split_indices(n: int, split: str, seed: int) -> np.ndarray:
    if split == "all":
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.2)
    n_val = int(n * 0.1)
    if split == "test":
        return idx[:n_test]
    if split == "val":
        return idx[n_test : n_test + n_val]
    if split == "train":
        return idx[n_test + n_val :]
    raise ValueError(f"Unknown split={split!r}")


def _build_undirected_adjacency(edges: Iterable[Tuple[int, int]]) -> Dict[int, set[int]]:
    adj: Dict[int, set[int]] = {}
    for u, v in edges:
        u_i = int(u)
        v_i = int(v)
        adj.setdefault(u_i, set()).add(v_i)
        adj.setdefault(v_i, set()).add(u_i)
    return adj


def _candidate_neighbors(
    adj: Dict[int, set[int]],
    u: int,
    v: int,
    remove_direct: bool,
) -> Tuple[set[int], set[int]]:
    nu = adj.get(u, set())
    nv = adj.get(v, set())
    if remove_direct and v in nu:
        nu = set(nu)
        nu.discard(v)
    if remove_direct and u in nv:
        nv = set(nv)
        nv.discard(u)
    return nu, nv


def _masked_degree(adj: Dict[int, set[int]], node: int, other: int, remove_direct: bool) -> int:
    deg = len(adj.get(node, set()))
    if remove_direct and other in adj.get(node, set()):
        deg -= 1
    return max(deg, 0)


def _shortest_path_score(
    adj: Dict[int, set[int]],
    u: int,
    v: int,
    remove_direct: bool,
    cutoff: int,
) -> float:
    if u == v:
        return 1.0
    seen = {u}
    queue: deque[Tuple[int, int]] = deque([(u, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= cutoff:
            continue
        for nbr in adj.get(node, set()):
            if remove_direct and ((node == u and nbr == v) or (node == v and nbr == u)):
                continue
            if nbr == v:
                return 1.0 / float(depth + 2)
            if nbr not in seen:
                seen.add(nbr)
                queue.append((nbr, depth + 1))
    return 0.0


def _score_candidates(
    adj: Dict[int, set[int]],
    edge_df: pd.DataFrame,
    split_idx: np.ndarray,
    *,
    mask_query_edges: bool,
    shortest_path_cutoff: int,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for i in split_idx.tolist():
        rec = edge_df.iloc[int(i)]
        u = int(rec["u_oid"])
        v = int(rec["v_oid"])
        label = int(rec["label"])
        remove_direct = bool(mask_query_edges and label == 1)

        nu, nv = _candidate_neighbors(adj, u, v, remove_direct)
        common = nu & nv
        union = nu | nv
        deg_u = _masked_degree(adj, u, v, remove_direct)
        deg_v = _masked_degree(adj, v, u, remove_direct)

        adamic = 0.0
        resource = 0.0
        for w in common:
            deg_w = len(adj.get(w, set()))
            if deg_w > 1:
                adamic += 1.0 / math.log(deg_w)
            if deg_w > 0:
                resource += 1.0 / deg_w

        rows.append(
            {
                "row_index": int(i),
                "u_oid": u,
                "v_oid": v,
                "label": label,
                "common_neighbors": float(len(common)),
                "jaccard": float(len(common) / len(union)) if union else 0.0,
                "adamic_adar": float(adamic),
                "resource_allocation": float(resource),
                "preferential_attachment": float(deg_u * deg_v),
                "degree_sum": float(deg_u + deg_v),
                "shortest_path": _shortest_path_score(
                    adj, u, v, remove_direct, shortest_path_cutoff
                ),
            }
        )
    return pd.DataFrame(rows)


def _safe_metrics(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    out = {
        "n_edges": int(len(y)),
        "positive_fraction": float(np.mean(y)) if len(y) else float("nan"),
    }
    if len(y) == 0 or len(np.unique(y)) < 2:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
        return out
    try:
        out["auroc"] = float(roc_auc_score(y, score))
    except ValueError:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y, score))
    except ValueError:
        out["auprc"] = float("nan")
    return out


def run_link_heuristics(
    *,
    manifest: str | Path,
    full_segments: str | Path,
    out_dir: str | Path,
    closures: Optional[List[str]],
    target_chrs: Optional[List[str]],
    split: str,
    seed: int,
    mask_query_edges: bool,
    shortest_path_cutoff: int,
    save_scores: bool,
) -> Dict:
    manifest = Path(manifest)
    manifest_dir = manifest.parent
    out_path = resolve_run_dir(Path(out_dir))

    segments = pd.read_csv(full_segments, usecols=["name"], compression="infer")
    seg_index, _ = build_global_index(segments)

    manifest_df = pd.read_csv(manifest)
    if closures:
        manifest_df = manifest_df[manifest_df["closure"].astype(str).isin(closures)]
    if target_chrs:
        target_set = set(target_chrs)
        manifest_df = manifest_df[manifest_df["target_sn"].astype(str).isin(target_set)]
    if manifest_df.empty:
        raise RuntimeError("No manifest rows matched the requested filters.")

    score_frames: List[pd.DataFrame] = []
    per_slice_rows: List[Dict] = []
    metric_names = [
        "common_neighbors",
        "jaccard",
        "adamic_adar",
        "resource_allocation",
        "preferential_attachment",
        "degree_sum",
        "shortest_path",
    ]

    for _, row in manifest_df.iterrows():
        links_path = _resolve_path(row["links_path"], manifest_dir)
        edge_path = _resolve_path(row["edge_pred_path"], manifest_dir)
        links = pd.read_csv(links_path, compression="infer")
        edge_df = pd.read_csv(edge_path, compression="infer")
        u, v = oriented_ids_from_links(links, seg_index)
        adj = _build_undirected_adjacency(zip(u.tolist(), v.tolist()))
        split_idx = _split_indices(len(edge_df), split, seed)
        scores = _score_candidates(
            adj,
            edge_df,
            split_idx,
            mask_query_edges=mask_query_edges,
            shortest_path_cutoff=shortest_path_cutoff,
        )
        scores.insert(0, "slice", row["name"])
        scores.insert(1, "target_sn", row["target_sn"])
        scores.insert(2, "closure", row["closure"])
        score_frames.append(scores)

        y = scores["label"].to_numpy(np.float32)
        for metric in metric_names:
            vals = scores[metric].to_numpy(np.float32)
            per_slice_rows.append(
                {
                    "slice": row["name"],
                    "target_sn": row["target_sn"],
                    "closure": row["closure"],
                    "heuristic": metric,
                    **_safe_metrics(y, vals),
                }
            )

    all_scores = pd.concat(score_frames, ignore_index=True)
    per_slice = pd.DataFrame(per_slice_rows)

    summary_rows: List[Dict] = []
    for (closure, heuristic), sub in per_slice.groupby(["closure", "heuristic"]):
        score_sub = all_scores[all_scores["closure"] == closure]
        pooled = _safe_metrics(
            score_sub["label"].to_numpy(np.float32),
            score_sub[heuristic].to_numpy(np.float32),
        )
        summary_rows.append(
            {
                "closure": closure,
                "heuristic": heuristic,
                "n_slices": int(sub["slice"].nunique()),
                "mean_auroc": float(sub["auroc"].mean()),
                "mean_auprc": float(sub["auprc"].mean()),
                "pooled_auroc": pooled["auroc"],
                "pooled_auprc": pooled["auprc"],
                "n_edges": pooled["n_edges"],
                "positive_fraction": pooled["positive_fraction"],
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["closure", "heuristic"])

    per_slice.to_csv(out_path / "per_slice_metrics.csv", index=False)
    summary.to_csv(out_path / "summary.csv", index=False)
    if save_scores:
        all_scores.to_csv(out_path / "candidate_scores.csv.gz", index=False, compression="gzip")

    payload = {
        "manifest": str(manifest),
        "full_segments": str(full_segments),
        "closures": closures or "all",
        "target_chrs": target_chrs or "all",
        "split": split,
        "seed": seed,
        "mask_query_edges": mask_query_edges,
        "shortest_path_cutoff": shortest_path_cutoff,
        "n_slices": int(manifest_df.shape[0]),
        "summary": summary.to_dict(orient="records"),
    }
    (out_path / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("[link-heuristics] summary:")
    print(summary.to_string(index=False))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate topology-only link heuristics.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--full_segments", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--closures", nargs="+", choices=["strict", "1hop"], default=None)
    ap.add_argument("--target_chrs", nargs="+", default=None)
    ap.add_argument("--split", choices=["all", "train", "val", "test"], default="test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_mask_query_edges", action="store_true")
    ap.add_argument("--shortest_path_cutoff", type=int, default=4)
    ap.add_argument("--save_scores", action="store_true")
    args = ap.parse_args()

    run_link_heuristics(
        manifest=args.manifest,
        full_segments=args.full_segments,
        out_dir=args.out_dir,
        closures=args.closures,
        target_chrs=args.target_chrs,
        split=args.split,
        seed=args.seed,
        mask_query_edges=not args.no_mask_query_edges,
        shortest_path_cutoff=args.shortest_path_cutoff,
        save_scores=args.save_scores,
    )


if __name__ == "__main__":
    main()
