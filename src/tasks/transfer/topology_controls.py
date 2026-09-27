"""Label-free handcrafted topology controls for frozen downstream probes.

A reviewer's first question for any graph foundation model is whether the
learned embedding beats cheap graph statistics.  This module computes, for
every segment of a pangenome graph, a fixed vector of local topology features
from ``full_segments``/``full_links`` only (no labels, splits, or outcomes):

* bidirected in/out degree of the forward and reverse handle;
* whether the segment is a branching handle (bubble entry or exit);
* hop distance (undirected, capped) to the nearest branching segment;
* number of distinct segments within 1, 2 and 3 undirected hops;
* local clustering coefficient;
* fraction of branching segments within 2 hops;
* deletion-bypass flag: a predecessor links directly to a successor;
* log segment length and log total length within 2 hops;
* reference-rank fraction within 2 hops is intentionally *excluded* (reference
  identity is a known cCRE shortcut).

The output uses the same ``segid``/``embeddings`` NPZ + ``.audit.json`` format
as the frozen sequence-model cache, so probes can load it with
``evaluation.modality_factorial.load_frozen_node_embedding_cache``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

FEATURE_NAMES: List[str] = [
    "log1p_out_deg_fwd", "log1p_in_deg_fwd", "log1p_out_deg_rev", "log1p_in_deg_rev",
    "is_branching", "hops_to_branching_capped8",
    "log1p_n_within_1hop", "log1p_n_within_2hop", "log1p_n_within_3hop",
    "clustering", "branching_fraction_2hop", "deletion_bypass",
    "log1p_length", "log1p_total_length_2hop",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def compute_topology_features(
    n_segments: int,
    lengths: np.ndarray,
    from_id: np.ndarray,
    from_rev: np.ndarray,
    to_id: np.ndarray,
    to_rev: np.ndarray,
    target_segids: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(segids, features)`` for ``target_segids`` (default: all)."""

    u = from_id.astype(np.int64) * 2 + from_rev.astype(np.int64)
    v = to_id.astype(np.int64) * 2 + to_rev.astype(np.int64)
    # Bidirected degree of oriented handles; reverse-equivalent rows counted once.
    keys = np.minimum(u * (2 * n_segments + 2) + v, (v ^ 1) * (2 * n_segments + 2) + (u ^ 1))
    _, first = np.unique(keys, return_index=True)
    u, v = u[first], v[first]
    out_deg = np.bincount(np.concatenate([u, v ^ 1]), minlength=2 * n_segments)
    in_deg = np.bincount(np.concatenate([v, u ^ 1]), minlength=2 * n_segments)

    # Undirected segment-level adjacency.
    su, sv = u // 2, v // 2
    keep = su != sv
    a = np.concatenate([su[keep], sv[keep]])
    b = np.concatenate([sv[keep], su[keep]])
    order = np.lexsort((b, a))
    a, b = a[order], b[order]
    pair = np.unique(np.stack([a, b], 1), axis=0) if len(a) else np.empty((0, 2), np.int64)
    indptr = np.zeros(n_segments + 1, dtype=np.int64)
    np.add.at(indptr, pair[:, 0] + 1, 1)
    indptr = np.cumsum(indptr)
    nbrs = pair[:, 1]

    def neighbors(s: int) -> np.ndarray:
        return nbrs[indptr[s] : indptr[s + 1]]

    seg_out = np.maximum(out_deg[0::2], out_deg[1::2])
    seg_in = np.maximum(in_deg[0::2], in_deg[1::2])
    branching = (seg_out > 1) | (seg_in > 1)

    # A multi-source traversal computes the declared capped distance for all
    # segments. The separate three-hop neighborhood below cannot supply it.
    branch_distance = np.full(n_segments, 8, dtype=np.int8)
    branch_distance[branching] = 0
    queue = deque(np.flatnonzero(branching).tolist())
    while queue:
        x = queue.popleft()
        distance = int(branch_distance[x]) + 1
        if distance >= 8:
            continue
        for y in neighbors(x):
            if branch_distance[y] > distance:
                branch_distance[y] = distance
                queue.append(int(y))

    # Directed successor sets per forward handle for the bypass test.
    succ: Dict[int, set] = {}
    for x, y in zip(u.tolist(), v.tolist()):
        succ.setdefault(x, set()).add(y)
        succ.setdefault(y ^ 1, set()).add(x ^ 1)

    targets = np.arange(n_segments) if target_segids is None else np.asarray(sorted(set(int(t) for t in target_segids)), dtype=np.int64)
    feats = np.zeros((len(targets), len(FEATURE_NAMES)), dtype=np.float32)
    for row, s in enumerate(targets.tolist()):
        # BFS up to 3 hops.
        seen = {s: 0}
        queue = deque([s])
        while queue:
            x = queue.popleft()
            if seen[x] >= 3:
                continue
            for y in neighbors(x).tolist():
                if y not in seen:
                    seen[y] = seen[x] + 1
                    queue.append(y)
        by_hop = [sum(1 for d in seen.values() if 0 < d <= k) for k in (1, 2, 3)]
        within2 = [x for x, d in seen.items() if 0 < d <= 2]
        hops_branch = float(branch_distance[s])
        n1 = neighbors(s)
        if len(n1) > 1:
            n1set = set(n1.tolist())
            links = sum(len(n1set.intersection(neighbors(x).tolist())) for x in n1.tolist()) / 2.0
            clustering = links / (len(n1) * (len(n1) - 1) / 2.0)
        else:
            clustering = 0.0
        fwd = 2 * s
        bypass = 0.0
        preds = succ.get(fwd ^ 1, set())  # predecessors of fwd, as reverse successors
        for p_rev in preds:
            p = p_rev ^ 1
            if succ.get(fwd, set()) & succ.get(p, set()):
                bypass = 1.0
                break
        feats[row] = [
            np.log1p(out_deg[fwd]), np.log1p(in_deg[fwd]),
            np.log1p(out_deg[fwd + 1]), np.log1p(in_deg[fwd + 1]),
            float(branching[s]), hops_branch,
            np.log1p(by_hop[0]), np.log1p(by_hop[1]), np.log1p(by_hop[2]),
            clustering,
            (sum(1 for x in within2 if branching[x]) / len(within2)) if within2 else 0.0,
            bypass,
            np.log1p(lengths[s]),
            np.log1p(float(lengths[within2].sum()) if within2 else 0.0),
        ]
    return targets, feats


def build_cache(
    full_segments: Path, full_links: Path, out: Path, target_segids: Sequence[int] | None
) -> dict:
    from graph.slicing import build_global_index

    segments = pd.read_csv(full_segments, compression="infer", usecols=["name", "LN"])
    seg_index, _ = build_global_index(segments)
    lengths = np.zeros(len(seg_index), dtype=np.float64)
    lengths[seg_index.get_indexer(segments["name"].astype("string"))] = segments["LN"].fillna(0).to_numpy(float)
    links = pd.read_csv(full_links, compression="infer", usecols=["from_seg", "from_orient", "to_seg", "to_orient"])
    fi = seg_index.get_indexer(links["from_seg"].astype("string"))
    ti = seg_index.get_indexer(links["to_seg"].astype("string"))
    ok = (fi >= 0) & (ti >= 0)
    segids, feats = compute_topology_features(
        len(seg_index), lengths,
        fi[ok], (links["from_orient"].astype(str) == "-").to_numpy()[ok],
        ti[ok], (links["to_orient"].astype(str) == "-").to_numpy()[ok],
        target_segids,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, segid=segids, embeddings=feats, feature_names=np.asarray(FEATURE_NAMES))
    audit = {
        "status": "complete",
        "downstream_label_access": "none",
        "kind": "handcrafted_topology_control",
        "processing_version": 2,
        "feature_names": FEATURE_NAMES,
        "n_segments": int(len(segids)),
        "full_segments_sha256": _sha256(full_segments),
        "full_links_sha256": _sha256(full_links),
        "excluded_by_design": ["SR", "is_grch38", "reference-rank fractions", "coordinates"],
    }
    Path(f"{out}.audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a handcrafted topology-control NPZ cache.")
    ap.add_argument("--full-segments", type=Path, required=True)
    ap.add_argument("--full-links", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--target-cache", type=Path, nargs="*", default=[],
        help="Optional NPZ caches whose 'segid' arrays define the segments to featurize "
             "(e.g. the NT cache). Only identifiers are read.",
    )
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    targets = None
    if args.target_cache:
        ids: set = set()
        for path in args.target_cache:
            with np.load(path, allow_pickle=False) as cache:
                ids.update(int(x) for x in cache["segid"].tolist())
        targets = sorted(ids)
    print(json.dumps(build_cache(args.full_segments, args.full_links, args.out, targets), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
