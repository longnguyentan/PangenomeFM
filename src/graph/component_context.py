"""Complete attached alternative components without inventing bubbles or paths.

Membership uses undirected connectivity after removing GRCh38 reference nodes.
Edges retain their original GFA orientations. Only components incident to the
original reference core are added; added anchors do not trigger recursive growth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from graph.slicing import build_global_index, map_links_to_segids

REFERENCE_PREFIX = 'GRCh38#0#'
PRIMARY = {f'chr{i}' for i in range(1, 23)} | {'chrX', 'chrY'}


@dataclass
class ComponentIndex:
    names: pd.Index
    nodes: pd.DataFrame
    source: np.ndarray
    target: np.ndarray
    reference: np.ndarray
    labels: np.ndarray
    members: dict[int, np.ndarray]
    anchors: dict[int, np.ndarray]
    incident: dict[int, np.ndarray]


def build_component_index(segments: pd.DataFrame, links: pd.DataFrame) -> ComponentIndex:
    """Use the repository's canonical IDs and mapper; count isolated alternatives."""
    names, nodes = build_global_index(segments)
    if len(nodes) != len(segments):
        raise ValueError('Canonical segment names must be unique')
    if nodes[['name', 'SN', 'SO', 'LN']].isna().any().any():
        raise ValueError('Missing segment identity or coordinate metadata')
    for field in ['SO', 'LN']:
        values = pd.to_numeric(nodes[field], errors='raise').to_numpy()
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError('Segment coordinates must be finite integers')
    if (nodes.LN <= 0).any() or (nodes.SO < 0).any():
        raise ValueError('Malformed segment coordinates')
    reference = nodes.SN.astype(str).str.startswith(REFERENCE_PREFIX).to_numpy()
    if not reference.any() or reference.all():
        raise ValueError('Require both GRCh38 reference and alternative segments')
    for column in ['from_orient', 'to_orient']:
        if column in links and not links[column].isin(['+', '-']).all():
            raise ValueError('Invalid GFA link orientation')
    source, target = map_links_to_segids(links, names)
    alt_edges = ~reference[source] & ~reference[target]
    adjacency = coo_matrix((np.ones(int(alt_edges.sum()), dtype=np.uint8),
        (source[alt_edges], target[alt_edges])), shape=(len(nodes), len(nodes))).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    alts = np.flatnonzero(~reference)
    members = {int(label): group.segment_id.to_numpy(np.int64)
        for label, group in pd.DataFrame(dict(component=labels[alts], segment_id=alts)).groupby('component')}
    crossing = reference[source] != reference[target]
    a = np.where(reference[source[crossing]], target[crossing], source[crossing])
    r = np.where(reference[source[crossing]], source[crossing], target[crossing])
    boundary = pd.DataFrame(dict(component=labels[a], reference_id=r)).drop_duplicates()
    anchors = {int(label): np.sort(group.reference_id.to_numpy(np.int64))
        for label, group in boundary.groupby('component')}
    incident = {int(node): np.sort(group.component.to_numpy(np.int64))
        for node, group in boundary.groupby('reference_id')}
    return ComponentIndex(names, nodes, source, target, reference, labels, members, anchors, incident)


def complete_context(index: ComponentIndex, core: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical IDs and component IDs; reject ambiguous chromosome context."""
    core = np.asarray(core)
    if (core.ndim != 1 or core.dtype.kind not in 'iu' or not len(core)
            or len(np.unique(core)) != len(core) or (core < 0).any() or (core >= len(index.nodes)).any()):
        raise ValueError('Core must contain unique, nonempty canonical integer IDs')
    if not index.reference[core].all():
        raise ValueError('Original core must be GRCh38 reference segments')
    contigs = index.nodes.SN.iloc[core].unique()
    if len(contigs) != 1 or str(contigs[0]).removeprefix(REFERENCE_PREFIX) not in PRIMARY:
        raise ValueError('Core must belong to one primary reference chromosome')
    components = np.unique(np.concatenate([index.incident.get(int(i), np.empty(0, np.int64)) for i in core]))
    pieces = [core]
    for label in components:
        anchors = index.anchors[int(label)]
        if not index.nodes.SN.iloc[anchors].eq(contigs[0]).all():
            raise ValueError('Alternative component crosses reference chromosomes')
        pieces.extend([index.members[int(label)], anchors])
    return np.unique(np.concatenate(pieces)), components


def attention_storage(n_handles: int, *, heads: int = 4, scalar_bytes: int = 4,
                      window_k: int = 128) -> dict[str, int]:
    """Exact dense-score bytes and a sparse-score upper bound, NOT peak GPU RAM.

    Both omit softmax, gradients, activations, graph messages and optimizer state.
    The sparse estimate follows the current implementation's clamped edge grid;
    it is not an endorsement of its semantics or a prediction that a model fits.
    """
    if min(n_handles, heads, scalar_bytes, window_k) < 1:
        raise ValueError('Memory dimensions must be positive')
    dense = int(heads) * int(n_handles)**2 * int(scalar_bytes)
    sparse = int(heads) * int(n_handles) * (2*(int(window_k)//2)+1) * int(scalar_bytes)
    return dict(dense_score_bytes=dense, sparse_score_bytes_upper_bound=min(dense, sparse))


def fold_membership_audit(window_ids: list[np.ndarray], chromosomes: list[str],
                          folds: list[dict], n_segments: int) -> pd.DataFrame:
    """Check actual context IDs, not merely window labels, for every split pair."""
    if len(window_ids) != len(chromosomes) or not set(chromosomes).issubset(PRIMARY):
        raise ValueError('Aligned primary-chromosome window identities required')
    rows = []
    for fold in folds:
        val, test = set(fold['validation']), set(fold['test'])
        if val & test or not (val | test).issubset(PRIMARY):
            raise ValueError('Invalid chromosome split')
        masks = {part: np.zeros(n_segments, dtype=bool) for part in ['train', 'validation', 'test']}
        for ids, chrom in zip(window_ids, chromosomes):
            part = 'test' if chrom in test else 'validation' if chrom in val else 'train'
            masks[part][ids] = True
        overlaps = {f'{a}_{b}_overlap': int((masks[a] & masks[b]).sum())
            for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}
        if any(overlaps.values()):
            raise ValueError(f"Context segment leakage in {fold['name']}: {overlaps}")
        rows.append(dict(fold=fold['name'], **{f'{p}_segments': int(m.sum()) for p, m in masks.items()}, **overlaps))
    return pd.DataFrame(rows)
