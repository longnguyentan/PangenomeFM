"""Node-input policies for PangenomeFM pretraining (v2 options).

All helpers default to the v1 behaviour when the corresponding argument is
absent, so archived checkpoints and their namespaces keep working unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:  # torch is optional for NumPy-only callers and tests
    import torch
except ImportError:  # pragma: no cover
    torch = None

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_CACHE: Dict[str, Tuple[Dict[int, int], np.ndarray]] = {}


def objective_of(args: Optional[argparse.Namespace]) -> str:
    return str(getattr(args, "objective", "edge_masking") or "edge_masking")


def structure_source_of(args: Optional[argparse.Namespace]) -> str:
    return str(getattr(args, "node_structure_source", "unmasked") or "unmasked")


def extra_features_of(args: Optional[argparse.Namespace]) -> str:
    return str(getattr(args, "node_extra_features", "none") or "none")


def visible_structure_features(X, src, dst, deg_norm: float):
    """Replace the degree column with degree on the visible (masked) graph.

    Degree is stored-direction out+in degree, identical in definition to
    ``graph.neg_sampling.compute_oriented_degrees`` and normalised by the same
    slice-level constant, so with no masking the column is unchanged.
    """

    n = X.shape[0]
    deg = torch.bincount(src, minlength=n) + torch.bincount(dst, minlength=n)
    out = X.clone()
    if deg_norm > 0:
        out[:, 4] = torch.log1p(deg.to(X.dtype)) / float(deg_norm)
    else:
        out[:, 4] = 0.0
    return out


def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _kmer_block(nodes: np.ndarray, seg_sub: pd.DataFrame, seg_index) -> np.ndarray:
    from tasks.ccre.aligned_baselines import _sequence_kmer_features_from_sequences

    if "seq" not in seg_sub.columns:
        raise ValueError("--node_extra_features kmer requires a 'seq' column in slice segments")
    positions = seg_index.get_indexer(seg_sub["name"].astype("string"))
    seq_by_segid = {
        int(pos): str(seq) if isinstance(seq, str) else ""
        for pos, seq in zip(positions, seg_sub["seq"].tolist())
        if pos >= 0
    }
    seqs = []
    for oid in np.asarray(nodes, dtype=np.int64).tolist():
        seq = seq_by_segid.get(oid // 2, "")
        seqs.append(_reverse_complement(seq) if oid % 2 else seq)
    block, _names = _sequence_kmer_features_from_sequences(seqs)
    return block.astype(np.float32)


def _load_cache(path: str) -> Tuple[Dict[int, int], np.ndarray]:
    if path not in _CACHE:
        from evaluation.modality_factorial import load_frozen_node_embedding_cache

        segids, values, _audit = load_frozen_node_embedding_cache(Path(path))
        _CACHE[path] = ({int(s): i for i, s in enumerate(segids.tolist())}, values)
    return _CACHE[path]


def _cache_block(nodes: np.ndarray, args) -> Tuple[np.ndarray, float]:
    path = getattr(args, "node_feature_cache", None)
    if not path:
        raise ValueError("--node_extra_features cache requires --node_feature_cache")
    position, values = _load_cache(str(path))
    block = np.zeros((len(nodes), values.shape[1]), dtype=np.float32)
    hits = 0
    for row, oid in enumerate(np.asarray(nodes, dtype=np.int64).tolist()):
        index = position.get(oid // 2)
        if index is not None:
            block[row] = values[index]
            hits += 1
    coverage = hits / max(len(nodes), 1)
    minimum = float(getattr(args, "node_feature_min_coverage", 0.99))
    if coverage < minimum:
        raise ValueError(
            f"Node feature cache covers {coverage:.4f} of slice nodes, below {minimum:.4f}. "
            "Build a whole-graph cache; partial caches make coverage itself a feature."
        )
    return block, coverage


def append_extra_node_features(X: np.ndarray, nodes, seg_sub, seg_index, args) -> np.ndarray:
    mode = extra_features_of(args)
    if mode == "none":
        return X
    if mode == "kmer":
        return np.concatenate([X, _kmer_block(nodes, seg_sub, seg_index)], axis=1)
    if mode == "cache":
        block, _coverage = _cache_block(nodes, args)
        return np.concatenate([X, block], axis=1)
    raise ValueError(f"Unknown node_extra_features {mode!r}")


PAIR_GEOMETRY_DIM = 4


def pair_geometry(u: np.ndarray, v: np.ndarray, md: dict) -> np.ndarray:
    """Explicit candidate geometry for the connection scorer.

    Columns: signed log offset ``SO(v)-SO(u)``; signed log contiguity gap
    ``SO(v)-(SO(u)+LN(u))`` (0 for reference-contiguous segments); orientation
    agreement; same coordinate system (SN).  Offsets are only comparable within
    one SN, so the last flag lets the scorer gate the first two.
    """

    so, ln, sn = md["oid_to_so"], md["oid_to_ln"], md.get("oid_to_sn", {})
    out = np.zeros((len(u), PAIR_GEOMETRY_DIM), dtype=np.float32)
    for row, (a, b) in enumerate(zip(np.asarray(u).tolist(), np.asarray(v).tolist())):
        same = (a in so and b in so and a in sn and b in sn
                and sn[a] not in (None, "", "*", "nan") and sn[a] == sn[b])
        d = float(so.get(b, 0)) - float(so.get(a, 0))
        gap = d - float(ln.get(a, 0))
        out[row, 0] = np.sign(d) * np.log1p(abs(d)) / 16.0 if same else 0.0
        out[row, 1] = np.sign(gap) * np.log1p(abs(gap)) / 16.0 if same else 0.0
        out[row, 2] = float((a % 2) == (b % 2))
        out[row, 3] = float(same)
    return out


def pair_geom_rows(sd: dict, index):
    geom = sd.get("pair_geom")
    return None if geom is None else geom[index]
