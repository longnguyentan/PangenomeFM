#!/usr/bin/env python3
"""Audit topology shortcuts in masked-connection pretraining benchmarks.

For each benchmark slice (``*_segments.csv.gz`` with sibling ``_links``), this
script rebuilds candidates in two ways and scores them with label-free
heuristics under explicit masking protocols:

* ``v1_degree_matched``: the manuscript construction.  Negatives are matched
  on genomic separation and on unmasked endpoint degree
  (``neg_distance_matched_paired``); each positive query is hidden alone.
* ``v1_distance_only``: the same without degree matching.
* ``junction_repair``: span-grouped junction re-pairing
  (``graph.junction_repair``); all validation/test junctions are hidden, plus the current training group
  for training candidates. This matches native evaluation and the minimum
  training group mask, excluding stochastic drop-edge and multi-group packing.

This is a diagnostic, not a trained downstream baseline. v1 single-query
masking is not a replay of batched encoder training. Heuristic direction is not known a priori (Platt scaling in the manuscript
baselines can flip it), so the report gives both the raw AUROC and the
direction-free AUPRC ``max(AP(s), AP(-s))``.  A shortcut-free benchmark should
show heuristic AUROC near 0.5 for degree-based scores and for the degree
deficit.  Labels define which positive edges are masked. The direction-free maximum
uses evaluation labels and is an optimistic diagnostic upper envelope, never
a validated predictor score. Raw positive and negative directions are saved.

Example::

    PYTHONPATH=src:. python scripts/audit_masking_degree_shortcut.py \
        --slices 'data/hgsvc3*/benchmark_*/*_strict_segments.csv.gz' \
        --out-dir results/audits/masking_degree_shortcut
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from graph.junction_repair import build_junction_repair_candidates
from graph.neg_sampling import (
    build_pos_set,
    canonical_oriented_pair,
    canonicalize_oriented_pairs,
    compute_oriented_degrees,
    neg_distance_matched_paired,
    oriented_ids_from_links,
    slice_oriented_node_set,
)

SCORES = [
    "preferential_attachment",
    "degree_sum",
    "degree_deficit",
    "common_neighbors",
    "adamic_adar",
    "shortest_path",
    "abs_distance",
]


def _adjacency(pairs: Iterable[Tuple[int, int]]) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {}
    for a, b in pairs:
        adj.setdefault(int(a), set()).add(int(b))
        adj.setdefault(int(b), set()).add(int(a))
    return adj


def _shortest(adj: Dict[int, Set[int]], a: int, b: int, cutoff: int = 4) -> float:
    if a == b:
        return 1.0
    seen = {a}
    queue: deque = deque([(a, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= cutoff:
            continue
        for nbr in adj.get(node, ()):
            if nbr == b:
                return 1.0 / (depth + 2)
            if nbr not in seen:
                seen.add(nbr)
                queue.append((nbr, depth + 1))
    return 0.0


def _score(adj, full_deg, so, a, b) -> Dict[str, float]:
    na, nb = adj.get(a, set()), adj.get(b, set())
    da, db = len(na), len(nb)
    common = na & nb
    aa = sum(1.0 / math.log(len(adj.get(w, ()))) for w in common if len(adj.get(w, ())) > 1)
    return {
        "preferential_attachment": float(da * db),
        "degree_sum": float(da + db),
        "degree_deficit": float(full_deg.get(a, 0) - da + full_deg.get(b, 0) - db),
        "common_neighbors": float(len(common)),
        "adamic_adar": float(aa),
        "shortest_path": _shortest(adj, a, b),
        "abs_distance": float(abs(so.get(a, 0) - so.get(b, 0))),
    }


def _slice_inputs(path: str):
    seg = pd.read_csv(path)
    links = pd.read_csv(path.replace("_segments", "_links"))
    index = pd.Index(seg["name"].astype("string"))
    u, v = oriented_ids_from_links(links, index)
    ok = (u >= 0) & (v >= 0)
    u, v = u[ok], v[ok]
    so: Dict[int, int] = {}
    sn: Dict[int, str] = {}
    for i, row in enumerate(seg.itertuples(index=False)):
        offset = int(row.SO) if pd.notna(row.SO) else 0
        for bit in (0, 1):
            so[i * 2 + bit] = offset
            sn[i * 2 + bit] = str(row.SN)
    return u, v, so, sn


def _full_degree_undirected(u, v) -> Dict[int, int]:
    return {k: len(s) for k, s in _adjacency(zip(u.tolist(), v.tolist())).items()}


def audit_slice(path: str, seed: int, span_size: int, scope: str) -> List[dict]:
    u, v, so, sn = _slice_inputs(path)
    if len(u) < 20:
        return []
    rows: List[dict] = []
    all_pairs = list(zip(u.tolist(), v.tolist()))
    full_adj = _adjacency(all_pairs)
    full_deg = {k: len(s) for k, s in full_adj.items()}

    nodes = slice_oriented_node_set(u, v)
    pos = canonicalize_oriented_pairs(np.stack([u, v], axis=1))
    pos_set = build_pos_set(u, v)
    deg_map = compute_oriented_degrees(u, v, nodes)
    for construction, matched in (("v1_degree_matched", True), ("v1_distance_only", False)):
        rng = np.random.default_rng(seed)
        neg, pidx = neg_distance_matched_paired(
            nodes, pos, pos_set, sn, so, deg_map, rng,
            same_sn=True, tol_bp=1000, tol_frac=0.1, degree_matched=matched,
        )
        for label, pairs in ((1, pos[pidx]), (0, np.asarray(neg, dtype=np.int64).reshape(-1, 2))):
            for a, b in pairs.tolist():
                if label == 1:  # single-query masking, as in the v1 heuristic baselines
                    na = set(full_adj.get(a, set()))
                    na.discard(b)
                    nb = set(full_adj.get(b, set()))
                    nb.discard(a)
                    local = dict(full_adj)
                    local[a] = na
                    local[b] = nb
                else:
                    local = full_adj
                rows.append({"slice": path, "construction": construction, "label": label,
                             **_score(local, full_deg, so, a, b)})

    cand, _audit = build_junction_repair_candidates(
        u, v, so, oid_to_sn=sn, scope=scope, span_size=span_size, split_seed=seed, rng_seed=seed
    )
    heldout = cand[(cand["label"] == 1) & cand["split"].isin(["val", "test"])]
    heldout_hidden = {canonical_oriented_pair(a, b)
                      for a, b in heldout[["u_oid", "v_oid"]].itertuples(index=False)}
    for (split_name, _group_id), sub in cand.groupby(["split", "group_id"]):
        hidden = heldout_hidden.copy()
        if split_name == "train":
            hidden.update(canonical_oriented_pair(a, b)
                          for a, b in sub.loc[sub["label"] == 1, ["u_oid", "v_oid"]].itertuples(index=False))
        visible = [(a, b) for a, b in all_pairs if canonical_oriented_pair(a, b) not in hidden]
        adj = _adjacency(visible)
        for a, b, label in sub[["u_oid", "v_oid", "label"]].itertuples(index=False):
            rows.append({"slice": path, "construction": f"junction_repair_{scope}",
                         "split": split_name, "label": int(label),
                         **_score(adj, full_deg, so, int(a), int(b))})
    # Candidate quality is retained separately from prediction performance.
    rows.append({"slice": path, "construction": "candidate_audit", **_audit.as_dict()})
    return rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    out = []
    for construction, group in frame.groupby("construction"):
        y = group["label"].to_numpy()
        for name in SCORES:
            s = group[name].to_numpy(float)
            auroc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan")
            ap = max(average_precision_score(y, s), average_precision_score(y, -s))
            out.append({
                "construction": construction, "score": name, "n": len(y),
                "prevalence": float(y.mean()), "auroc_raw": float(auroc),
                "auroc_direction_free": float(max(auroc, 1 - auroc)),
                "auprc_direction_free": float(ap),
                "auprc_raw": float(average_precision_score(y, s)),
                "auprc_reversed": float(average_precision_score(y, -s)),
            })
        deficit = group["degree_deficit"].to_numpy(float) > 0
        out.append({
            "construction": construction, "score": "fraction_endpoints_with_deficit",
            "n": len(y), "prevalence": float(y.mean()),
            "auroc_raw": float(deficit[y == 1].mean()), "auroc_direction_free": float("nan"),
            "auprc_direction_free": float(deficit[y == 0].mean()),
        })
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slices", required=True, help="Glob for *_segments.csv.gz slice files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--span-size", type=int, default=16)
    ap.add_argument("--scope", choices=["all", "branching"], default="branching")
    ap.add_argument("--max-slices", type=int, default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(args.slices))
    if args.max_slices:
        files = files[: args.max_slices]
    if not files:
        raise SystemExit(f"No slices match {args.slices!r}")
    rows: List[dict] = []
    for path in files:
        rows.extend(audit_slice(path, args.seed, args.span_size, args.scope))
    frame = pd.DataFrame(rows)
    candidate_audit = frame[frame["construction"] == "candidate_audit"].dropna(axis=1, how="all")
    frame = frame[frame["construction"] != "candidate_audit"]
    if frame.empty:
        raise ValueError("No candidates scored")
    summary = summarize(frame)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "audit.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed audit {out}")
    summary.to_csv(out / "summary.csv", index=False)
    candidate_audit.to_csv(out / "candidate_audit.csv", index=False)
    by_split = [summarize(g).assign(split=name) for name, g in frame.groupby("split")]
    if by_split:
        pd.concat(by_split).to_csv(out / "by_split.csv", index=False)
    (out / "audit.json").write_text(json.dumps({
        "slices": len(files), "slices_scored": int(frame["slice"].nunique()),
        "seed": args.seed, "span_size": args.span_size, "scope": args.scope,
        "processing_version": 2,
        "v1_mask": "single query; not a full batched training replay",
        "v2_mask": "all held-out positives plus current training group; no drop-edge",
        "direction_free_scores": "evaluation-label-selected optimistic diagnostic upper envelope",
        "note": ("Fraction row: auroc_raw = share of positive candidates whose endpoints "
                 "have a visible-degree deficit; auprc_direction_free = same share for negatives."),
    }, indent=2))
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
