"""Build and benchmark haplotype-specific methylation labels on graph walks."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Iterable, TextIO

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.gfa_parser import _parse_p_walk, _parse_w_walk, _split_path_name


_P_SLICE_SUFFIX = re.compile(r"\[(\d+)-(\d+)\]$")


def _open_text(path: str | Path) -> TextIO:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


_RC_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")
_CANONICAL_3MER = {
    kmer: min(kmer, kmer.translate(_RC_TABLE)[::-1])
    for bases in product("ACGT", repeat=3)
    for kmer in ["".join(bases)]
}


def _canonical_kmer_counts(sequence: str, k: int = 3) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    sequence = sequence.upper()
    for idx in range(max(0, len(sequence) - k + 1)):
        kmer = sequence[idx : idx + k]
        if k == 3:
            canonical = _CANONICAL_3MER.get(kmer)
        elif set(kmer) <= {"A", "C", "G", "T"}:
            canonical = min(kmer, kmer.translate(_RC_TABLE)[::-1])
        else:
            canonical = None
        if canonical is not None:
            counts[canonical] += 1
    return dict(counts)


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(_RC_TABLE)[::-1]


def _base_entropy(sequence: str) -> float:
    counts = np.asarray(
        [sequence.upper().count(base) for base in "ACGT"], dtype=float
    )
    counts = counts[counts > 0]
    if not len(counts):
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum() / 2.0)


def _as_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _manifest_target_for_chunk(
    path: Path, path_names: set[str] | None
) -> set[str] | None:
    if path_names is None:
        return None
    matched = [name for name in path_names if name in path.name]
    if not matched:
        raise ValueError(f"Cannot map chunk filename to path manifest: {path.name}")
    longest = max(map(len, matched))
    winners = {name for name in matched if len(name) == longest}
    if len(winners) != 1:
        raise ValueError(f"Ambiguous target path in chunk filename: {path.name}")
    return winners


def _parse_walk(fields: list[str]) -> dict[str, Any] | None:
    if fields[0] == "W" and len(fields) >= 7:
        source_path_name = f"{fields[1]}#{fields[2]}#{fields[3]}"
        return {
            "record_type": "W",
            "path_name": (
                f"{source_path_name}[{fields[4]}-{fields[5]}]"
            ),
            "source_path_name": source_path_name,
            "sample": fields[1],
            "haplotype": fields[2],
            "contig": fields[3],
            "path_start": int(fields[4]),
            "path_end": int(fields[5]),
            "steps": _parse_w_walk(fields[6]),
        }
    if fields[0] != "P" or len(fields) < 3:
        return None
    match = _P_SLICE_SUFFIX.search(fields[1])
    slice_start = int(match.group(1)) if match else 0
    source_path_name = (
        fields[1][: match.start()] if match is not None else fields[1]
    )
    sample, haplotype, contig = _split_path_name(source_path_name)
    phase_start = 0
    if "#" in contig:
        contig_parts = contig.split("#")
        try:
            phase_start = int(contig_parts[-1]) + slice_start
            contig = "#".join(contig_parts[:-1])
        except ValueError:
            pass
    return {
        "record_type": "P",
        "path_name": fields[1],
        "source_path_name": source_path_name,
        "sample": sample,
        "haplotype": haplotype,
        "contig": contig,
        "path_start": phase_start,
        "path_end": 0,
        "steps": _parse_p_walk(fields[2]),
    }


def _walk_matches_active_target(
    walk: dict[str, Any], active_path_names: set[str]
) -> bool:
    source_path_name = str(walk.get("source_path_name", walk["path_name"]))
    if source_path_name in active_path_names:
        return True
    if walk.get("record_type") != "W":
        return False
    # GFA W records omit the GBZ phase-block suffix carried by metadata path
    # names when that phase block is zero (for example,
    # "D#1#CONTIG#0"). Nonzero phase blocks are emitted as P records and a W
    # record with the same contig is context, not the extraction target.
    return any(
        target.endswith("#0")
        and target.rsplit("#", 1)[0] == source_path_name
        for target in active_path_names
    )


def _load_selected_gfa(
    paths: str | Path | Iterable[str | Path],
    *,
    sample_ids: set[str] | None = None,
    path_names: set[str] | None = None,
) -> tuple[dict[str, str], dict[str, int], list[dict[str, Any]]]:
    """Load path chunks while retaining sequence only for selected path nodes."""

    gfa_paths = _as_paths(paths)
    walks_by_name: dict[str, dict[str, Any]] = {}
    selected_nodes: set[str] = set()
    for path in gfa_paths:
        active_path_names = _manifest_target_for_chunk(path, path_names)
        with _open_text(path) as handle:
            for raw_line in handle:
                if not raw_line or raw_line.startswith("#"):
                    continue
                fields = raw_line.rstrip("\n").split("\t")
                walk = _parse_walk(fields)
                if walk is None:
                    continue
                if sample_ids is not None and str(walk["sample"]) not in sample_ids:
                    continue
                if (
                    active_path_names is not None
                    and not _walk_matches_active_target(
                        walk, active_path_names
                    )
                ):
                    continue
                existing = walks_by_name.get(str(walk["path_name"]))
                if existing is not None and existing["steps"] != walk["steps"]:
                    raise ValueError(
                        f"Conflicting definitions for path {walk['path_name']!r}."
                    )
                walks_by_name[str(walk["path_name"])] = walk
                selected_nodes.update(node for node, _ in walk["steps"])

    sequences: dict[str, str] = {}
    degree: dict[str, int] = defaultdict(int)
    seen_edges: set[tuple[str, str, str, str]] = set()
    for path in gfa_paths:
        with _open_text(path) as handle:
            for raw_line in handle:
                if not raw_line or raw_line.startswith("#"):
                    continue
                fields = raw_line.rstrip("\n").split("\t")
                if (
                    fields[0] == "S"
                    and len(fields) >= 3
                    and fields[1] in selected_nodes
                ):
                    prior = sequences.get(fields[1])
                    if prior is not None and prior != fields[2]:
                        raise ValueError(
                            f"Conflicting sequences for graph node {fields[1]!r}."
                        )
                    sequences[fields[1]] = fields[2]
                elif fields[0] == "L" and len(fields) >= 5:
                    source_selected = fields[1] in selected_nodes
                    target_selected = fields[3] in selected_nodes
                    if not source_selected and not target_selected:
                        continue
                    edge = (fields[1], fields[2], fields[3], fields[4])
                    if edge in seen_edges:
                        continue
                    seen_edges.add(edge)
                    if source_selected:
                        degree[fields[1]] += 1
                    if target_selected:
                        degree[fields[3]] += 1
    return sequences, dict(degree), list(walks_by_name.values())


def _walk_windows(
    *,
    sequences: dict[str, str],
    degree: dict[str, int],
    walks: Iterable[dict[str, Any]],
    window_bp: int,
) -> pd.DataFrame:
    accumulators: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    covered_until: dict[str, int] = {}
    ordered_walks = sorted(
        walks,
        key=lambda walk: (
            str(walk["sample"]),
            str(walk["haplotype"]),
            str(walk["contig"]),
            int(walk["path_start"]),
            str(walk["path_name"]),
        ),
    )
    for walk in ordered_walks:
        cursor = int(walk["path_start"])
        track_contig = f"{walk['sample']}#{walk['haplotype']}#{walk['contig']}"
        trim_before = covered_until.get(track_contig, cursor)
        path_node_counts = Counter(node for node, _ in walk["steps"])
        for node, orientation in walk["steps"]:
            if node not in sequences:
                raise KeyError(f"Walk {walk['path_name']} references missing node {node!r}.")
            sequence = sequences[node]
            node_length = len(sequence)
            node_start = cursor
            node_end = cursor + node_length
            effective_start = max(node_start, trim_before)
            if node_end <= effective_start:
                cursor = node_end
                continue
            first_bin = effective_start // window_bp
            last_bin = max(
                first_bin, (max(effective_start, node_end - 1)) // window_bp
            )
            gc_count = sequence.upper().count("G") + sequence.upper().count("C")
            cpg_count = sequence.upper().count("CG")
            kmers = _canonical_kmer_counts(sequence, 3)
            for bin_idx in range(first_bin, last_bin + 1):
                window_start = bin_idx * window_bp
                window_end = window_start + window_bp
                overlap_start = max(effective_start, window_start)
                overlap_end = min(node_end, window_end)
                overlap = max(0, overlap_end - overlap_start)
                if overlap <= 0:
                    continue
                key = (
                    str(walk["sample"]),
                    str(walk["haplotype"]),
                    str(walk["contig"]),
                    window_start,
                )
                acc = accumulators.setdefault(
                    key,
                    {
                        "sample": str(walk["sample"]),
                        "haplotype": str(walk["haplotype"]),
                        "contig": str(walk["contig"]),
                        "track_contig": track_contig,
                        "window_start": window_start,
                        "window_end": window_end,
                        "path_names": set(),
                        "nodes": set(),
                        "oriented_nodes": set(),
                        "path_bp": 0,
                        "node_length_weighted": 0.0,
                        "degree_weighted": 0.0,
                        "branching_bp": 0,
                        "reverse_bp": 0,
                        "gc_count_weighted": 0.0,
                        "cpg_count_weighted": 0.0,
                        "base_entropy_weighted": 0.0,
                        "node_multiplicity_weighted": 0.0,
                        "multicopy_bp": 0,
                        "sequence_parts": [],
                        "sequence_sample_bp": 0,
                        "kmer_weighted": defaultdict(float),
                    },
                )
                fraction = overlap / max(node_length, 1)
                acc["path_names"].add(str(walk["path_name"]))
                acc["nodes"].add(node)
                acc["oriented_nodes"].add(f"{node}{orientation}")
                acc["path_bp"] += overlap
                acc["node_length_weighted"] += node_length * overlap
                acc["degree_weighted"] += degree.get(node, 0) * overlap
                acc["branching_bp"] += int(degree.get(node, 0) > 2) * overlap
                acc["reverse_bp"] += int(orientation == "-") * overlap
                acc["gc_count_weighted"] += gc_count * fraction
                acc["cpg_count_weighted"] += cpg_count * fraction
                acc["base_entropy_weighted"] += _base_entropy(sequence) * overlap
                multiplicity = int(path_node_counts[node])
                acc["node_multiplicity_weighted"] += multiplicity * overlap
                acc["multicopy_bp"] += int(multiplicity > 1) * overlap
                if acc["sequence_sample_bp"] < 512:
                    local_start = overlap_start - node_start
                    local_end = overlap_end - node_start
                    oriented_sequence = (
                        sequence if orientation == "+" else _reverse_complement(sequence)
                    )
                    remaining = 512 - int(acc["sequence_sample_bp"])
                    fragment = oriented_sequence[local_start:local_end][:remaining]
                    acc["sequence_parts"].append(fragment)
                    acc["sequence_sample_bp"] += len(fragment)
                for kmer, count in kmers.items():
                    acc["kmer_weighted"][kmer] += count * fraction
            cursor = node_end
        covered_until[track_contig] = max(
            covered_until.get(track_contig, cursor), cursor
        )

    rows: list[dict[str, Any]] = []
    for acc in accumulators.values():
        path_bp = max(int(acc["path_bp"]), 1)
        row = {
            "sample": acc["sample"],
            "haplotype": acc["haplotype"],
            "contig": acc["contig"],
            "track_contig": acc["track_contig"],
            "window_start": acc["window_start"],
            "window_end": acc["window_end"],
            "path_names": ",".join(sorted(acc["path_names"])),
            "node_ids": ",".join(sorted(acc["nodes"])),
            "oriented_node_ids": ",".join(sorted(acc["oriented_nodes"])),
            "node_count": len(acc["nodes"]),
            "path_bp": int(acc["path_bp"]),
            "mean_node_length": float(acc["node_length_weighted"] / path_bp),
            "mean_degree": float(acc["degree_weighted"] / path_bp),
            "branching_fraction": float(acc["branching_bp"] / path_bp),
            "reverse_fraction": float(acc["reverse_bp"] / path_bp),
            "gc_fraction": float(acc["gc_count_weighted"] / path_bp),
            "cpg_density": float(acc["cpg_count_weighted"] / path_bp),
            "base_entropy": float(acc["base_entropy_weighted"] / path_bp),
            "mean_path_node_multiplicity": float(
                acc["node_multiplicity_weighted"] / path_bp
            ),
            "multicopy_fraction": float(acc["multicopy_bp"] / path_bp),
            "sequence_sample": "".join(acc["sequence_parts"]),
        }
        kmer_total = sum(acc["kmer_weighted"].values())
        for kmer, count in acc["kmer_weighted"].items():
            row[f"kmer3_{kmer}"] = float(count / max(kmer_total, 1.0))
        rows.append(row)
    return pd.DataFrame(rows)


def _methylation_windows(
    paths: Iterable[str | Path],
    *,
    allowed_contigs: set[str],
    window_bp: int,
) -> pd.DataFrame:
    aggregates: dict[tuple[str, int], list[float]] = {}

    def consume(line: str) -> None:
        if not line or line.startswith("#"):
            return
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 7 or fields[0] not in allowed_contigs:
            return
        start = int(fields[1])
        methylation = float(fields[4])
        depth = int(float(fields[6]))
        window_start = (start // window_bp) * window_bp
        values = aggregates.setdefault(
            (fields[0], window_start), [0.0, 0.0, 0.0]
        )
        values[0] += methylation * depth
        values[1] += depth
        values[2] += 1

    for path in paths:
        path = Path(path)
        index_path = Path(f"{path}.tbi")
        if path.suffix == ".gz" and index_path.exists():
            try:
                import pysam

                with pysam.TabixFile(str(path)) as tabix:
                    for contig in sorted(allowed_contigs & set(tabix.contigs)):
                        for line in tabix.fetch(contig):
                            consume(line)
                continue
            except (ImportError, OSError, ValueError):
                pass
        with _open_text(path) as handle:
            for line in handle:
                consume(line)
    return pd.DataFrame(
        [
            {
                "track_contig": contig,
                "window_start": start,
                "methylation": weighted / max(depth, 1.0),
                "methylation_depth": int(depth),
                "methylation_cpgs": int(cpgs),
            }
            for (contig, start), (weighted, depth, cpgs) in aggregates.items()
        ]
    )


def _node_sets(frame: pd.DataFrame) -> list[set[str]]:
    return [set(str(value).split(",")) - {""} for value in frame["node_ids"]]


def _spaced_canonical_kmers(
    sequence: str, k: int = 31, stride: int = 31
) -> set[str]:
    sequence = str(sequence).upper()
    result: set[str] = set()
    for start in range(0, max(0, len(sequence) - k + 1), stride):
        kmer = sequence[start : start + k]
        if set(kmer) <= {"A", "C", "G", "T"}:
            result.add(min(kmer, kmer.translate(_RC_TABLE)[::-1]))
    return result


def _reciprocal_anchor_pairs(
    windows: pd.DataFrame,
    *,
    node_lengths: dict[str, int],
    min_shared_bp: int,
    min_shared_nodes: int,
    min_shared_kmer_anchors: int = 2,
    force_sequence_fallback: bool = False,
    signal_column: str = "methylation",
    signal_name: str = "methylation",
    support_columns: Iterable[str] = ("methylation_cpgs", "methylation_depth"),
) -> pd.DataFrame:
    h1 = windows[windows["haplotype"].astype(str).eq("1")].reset_index(drop=True)
    h2 = windows[windows["haplotype"].astype(str).eq("2")].reset_index(drop=True)
    actual_h1_sets = _node_sets(h1)
    actual_h2_sets = _node_sets(h2)
    h1_sets = (
        [set() for _ in actual_h1_sets]
        if force_sequence_fallback
        else actual_h1_sets
    )
    h2_sets = (
        [set() for _ in actual_h2_sets]
        if force_sequence_fallback
        else actual_h2_sets
    )
    h2_by_node: dict[str, set[int]] = defaultdict(set)
    h1_by_node: dict[str, set[int]] = defaultdict(set)
    for idx, nodes in enumerate(h2_sets):
        for node in nodes:
            h2_by_node[node].add(idx)
    for idx, nodes in enumerate(h1_sets):
        for node in nodes:
            h1_by_node[node].add(idx)

    def best_matches(
        source_sets: list[set[str]], target_by_node: dict[str, set[int]]
    ) -> dict[int, tuple[int, int, int]]:
        best: dict[int, tuple[int, int, int]] = {}
        for source_idx, nodes in enumerate(source_sets):
            counts: dict[int, set[str]] = defaultdict(set)
            for node in nodes:
                for target_idx in target_by_node.get(node, set()):
                    counts[target_idx].add(node)
            if not counts:
                continue
            ranked = []
            for target_idx, shared in counts.items():
                shared_bp = sum(node_lengths.get(node, 0) for node in shared)
                ranked.append((shared_bp, len(shared), -target_idx, target_idx))
            shared_bp, shared_count, _, target_idx = max(ranked)
            best[source_idx] = (target_idx, shared_bp, shared_count)
        return best

    best_12 = best_matches(h1_sets, h2_by_node)
    best_21 = best_matches(h2_sets, h1_by_node)
    rows: list[dict[str, Any]] = []
    feature_columns = [
        "node_count",
        "path_bp",
        "mean_node_length",
        "mean_degree",
        "branching_fraction",
        "reverse_fraction",
        "gc_fraction",
        "cpg_density",
        "base_entropy",
        "mean_path_node_multiplicity",
        "multicopy_fraction",
        *sorted(column for column in windows if column.startswith("kmer3_")),
    ]
    def append_pair(
        h1_idx: int,
        h2_idx: int,
        *,
        shared_bp: int,
        shared_count: int,
        anchor_method: str,
        shared_kmer31: int = 0,
    ) -> None:
        left = h1.iloc[h1_idx]
        right = h2.iloc[h2_idx]
        union = actual_h1_sets[h1_idx] | actual_h2_sets[h2_idx]
        actual_shared_nodes = actual_h1_sets[h1_idx] & actual_h2_sets[h2_idx]
        signal_delta = float(left[signal_column] - right[signal_column])
        row: dict[str, Any] = {
            "donor_id": left["sample"],
            "h1_contig": left["contig"],
            "h2_contig": right["contig"],
            "h1_window_start": int(left["window_start"]),
            "h2_window_start": int(right["window_start"]),
            "h1_embedding_id": f"{left['track_contig']}:{int(left['window_start'])}",
            "h2_embedding_id": f"{right['track_contig']}:{int(right['window_start'])}",
            "h1_track_contig": str(left["track_contig"]),
            "h2_track_contig": str(right["track_contig"]),
            "h1_node_ids": str(left["node_ids"]),
            "h2_node_ids": str(right["node_ids"]),
            "h1_oriented_node_ids": str(left["oriented_node_ids"]),
            "h2_oriented_node_ids": str(right["oriented_node_ids"]),
            "h1_sequence": str(left.get("sequence_sample", "")),
            "h2_sequence": str(right.get("sequence_sample", "")),
            f"h1_{signal_name}": float(left[signal_column]),
            f"h2_{signal_name}": float(right[signal_column]),
            f"{signal_name}_delta": signal_delta,
            "ase_value": signal_delta,
            "anchor_method": anchor_method,
            "shared_nodes": int(len(actual_shared_nodes)),
            "shared_node_bp": int(
                sum(node_lengths.get(node, 0) for node in actual_shared_nodes)
            ),
            "shared_kmer31": int(shared_kmer31),
            "shared_anchor_bp": int(shared_bp),
            "node_jaccard": float(len(actual_shared_nodes) / max(len(union), 1)),
        }
        for support_column in support_columns:
            if support_column not in windows:
                continue
            output_name = support_column.removeprefix(f"{signal_name}_")
            row[f"h1_{output_name}"] = float(left[support_column])
            row[f"h2_{output_name}"] = float(right[support_column])
        for feature in feature_columns:
            left_value = float(left.get(feature, 0.0))
            right_value = float(right.get(feature, 0.0))
            row[f"h1_{feature}"] = left_value
            row[f"h2_{feature}"] = right_value
            row[f"delta_{feature}"] = left_value - right_value
            row[f"divergence_x_delta_{feature}"] = (
                1.0 - row["node_jaccard"]
            ) * (left_value - right_value)
        rows.append(row)

    for h1_idx, (h2_idx, shared_bp, shared_count) in best_12.items():
        if best_21.get(h2_idx, (-1, 0, 0))[0] != h1_idx:
            continue
        if shared_bp < min_shared_bp or shared_count < min_shared_nodes:
            continue
        append_pair(
            h1_idx,
            h2_idx,
            shared_bp=shared_bp,
            shared_count=shared_count,
            anchor_method="shared_graph_nodes",
        )

    # Some assembly paths are represented by disjoint graph-node identifiers
    # despite sequence homology.  Only when graph-node pairing yields no usable
    # pairs, fall back to reciprocal unique spaced 31-mer anchors.  K-mers that
    # occur in more than one window on either haplotype are excluded, limiting
    # repeat-driven cross-locus matches.
    if not rows:
        h1_kmers = [
            _spaced_canonical_kmers(value) for value in h1["sequence_sample"]
        ]
        h2_kmers = [
            _spaced_canonical_kmers(value) for value in h2["sequence_sample"]
        ]
        h1_occurrences: dict[str, set[int]] = defaultdict(set)
        h2_occurrences: dict[str, set[int]] = defaultdict(set)
        for index, kmers in enumerate(h1_kmers):
            for kmer in kmers:
                h1_occurrences[kmer].add(index)
        for index, kmers in enumerate(h2_kmers):
            for kmer in kmers:
                h2_occurrences[kmer].add(index)
        unique_anchor_pairs = {
            kmer: (
                next(iter(h1_occurrences[kmer])),
                next(iter(h2_occurrences[kmer])),
            )
            for kmer in h1_occurrences.keys() & h2_occurrences.keys()
            if len(h1_occurrences[kmer]) == 1 and len(h2_occurrences[kmer]) == 1
        }
        counts_12: dict[int, Counter[int]] = defaultdict(Counter)
        counts_21: dict[int, Counter[int]] = defaultdict(Counter)
        for left_index, right_index in unique_anchor_pairs.values():
            counts_12[left_index][right_index] += 1
            counts_21[right_index][left_index] += 1
        best_kmer_12 = {
            source: max(
                ((count, -target, target) for target, count in targets.items())
            )
            for source, targets in counts_12.items()
        }
        best_kmer_21 = {
            source: max(
                ((count, -target, target) for target, count in targets.items())
            )
            for source, targets in counts_21.items()
        }
        for h1_idx, (count, _, h2_idx) in best_kmer_12.items():
            reverse = best_kmer_21.get(h2_idx)
            if reverse is None or reverse[2] != h1_idx:
                continue
            if count < min_shared_kmer_anchors:
                continue
            append_pair(
                h1_idx,
                h2_idx,
                shared_bp=count * 31,
                shared_count=0,
                anchor_method="reciprocal_unique_spaced_31mer",
                shared_kmer31=count,
            )
    return pd.DataFrame(rows)


def repair_pairs_with_sequence_anchors(
    *,
    windows_path: str | Path,
    out_dir: str | Path,
    min_shared_kmer_anchors: int = 2,
) -> dict[str, Any]:
    """Re-pair an existing labeled-window table using unique 31-mer anchors."""
    windows = pd.read_csv(windows_path, compression="infer")
    pairs = _reciprocal_anchor_pairs(
        windows,
        node_lengths={},
        min_shared_bp=500,
        min_shared_nodes=2,
        min_shared_kmer_anchors=min_shared_kmer_anchors,
        force_sequence_fallback=True,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_dir / "paired_windows.csv.gz", index=False, compression="gzip")
    summary_path = out_dir / "dataset_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {"task": "haplotype_specific_methylation"}
    )
    summary.update(
        {
            "pairing": (
                "reciprocal unique spaced 31-mer fallback because shared graph "
                "nodes yielded no usable pairs"
            ),
            "anchor_method_counts": (
                pairs["anchor_method"].value_counts().to_dict()
                if "anchor_method" in pairs
                else {}
            ),
            "n_reciprocal_h1_h2_pairs": int(len(pairs)),
            "n_donors": (
                int(pairs["donor_id"].astype(str).nunique())
                if not pairs.empty
                else 0
            ),
            "sequence_fallback_min_shared_kmer_anchors": int(
                min_shared_kmer_anchors
            ),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_methylation_pairs(
    *,
    gfa_path: str | Path | Iterable[str | Path],
    methylation_paths: Iterable[str | Path],
    out_dir: str | Path,
    sample_ids: Iterable[str] | None = None,
    path_names: Iterable[str] | None = None,
    window_bp: int = 10_000,
    min_cpgs: int = 10,
    min_depth: int = 100,
    min_shared_bp: int = 500,
    min_shared_nodes: int = 2,
) -> dict[str, Any]:
    gfa_paths = _as_paths(gfa_path)
    selected_samples = set(sample_ids) if sample_ids is not None else None
    selected_path_names = set(path_names) if path_names is not None else None
    print(
        f"[methylation-pairs] loading {len(gfa_paths)} GFA chunks",
        flush=True,
    )
    sequences, degree, walks = _load_selected_gfa(
        gfa_paths,
        sample_ids=selected_samples,
        path_names=selected_path_names,
    )
    print(
        f"[methylation-pairs] selected {len(walks)} path pieces and "
        f"{len(sequences)} graph nodes",
        flush=True,
    )
    path_windows = _walk_windows(
        sequences=sequences,
        degree=degree,
        walks=walks,
        window_bp=window_bp,
    )
    print(
        f"[methylation-pairs] built {len(path_windows)} path windows; "
        "fetching indexed methylation",
        flush=True,
    )
    methylation = _methylation_windows(
        methylation_paths,
        allowed_contigs=set(path_windows["track_contig"].astype(str)),
        window_bp=window_bp,
    )
    windows = path_windows.merge(
        methylation, on=["track_contig", "window_start"], how="inner"
    )
    windows = windows[
        windows["methylation_cpgs"].ge(min_cpgs)
        & windows["methylation_depth"].ge(min_depth)
    ].reset_index(drop=True)
    pairs = _reciprocal_anchor_pairs(
        windows,
        node_lengths={node: len(sequence) for node, sequence in sequences.items()},
        min_shared_bp=min_shared_bp,
        min_shared_nodes=min_shared_nodes,
    )
    print(
        f"[methylation-pairs] retained {len(windows)} labeled windows and "
        f"{len(pairs)} reciprocal pairs",
        flush=True,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(out_dir / "haplotype_windows.csv.gz", index=False, compression="gzip")
    pairs.to_csv(out_dir / "paired_windows.csv.gz", index=False, compression="gzip")
    summary = {
        "task": "haplotype_specific_methylation",
        "pairing": (
            "reciprocal-best shared graph-node anchors; reciprocal unique "
            "spaced 31-mer fallback only when graph-node pairing yields zero"
        ),
        "anchor_method_counts": (
            pairs["anchor_method"].value_counts().to_dict()
            if "anchor_method" in pairs
            else {}
        ),
        "gfa_paths": [str(path) for path in gfa_paths],
        "sample_ids": sorted(selected_samples) if selected_samples else None,
        "path_names": (
            sorted(selected_path_names) if selected_path_names is not None else None
        ),
        "methylation_paths": [str(path) for path in methylation_paths],
        "window_bp": int(window_bp),
        "n_graph_nodes": int(len(sequences)),
        "n_walk_records": int(len(walks)),
        "n_labeled_haplotype_windows": int(len(windows)),
        "n_reciprocal_h1_h2_pairs": int(len(pairs)),
        "n_donors": int(pairs["donor_id"].nunique()) if not pairs.empty else 0,
        "thresholds": {
            "min_cpgs": int(min_cpgs),
            "min_depth": int(min_depth),
            "min_shared_bp": int(min_shared_bp),
            "min_shared_nodes": int(min_shared_nodes),
            "fallback_min_shared_kmer_anchors": 2,
        },
    }
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rho = (
        spearmanr(y, pred).statistic
        if len(np.unique(y)) > 1 and len(np.unique(pred)) > 1
        else float("nan")
    )
    return {
        "spearman": float(rho),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)) if len(y) >= 2 else float("nan"),
        "direction_accuracy": float(np.mean((y > 0) == (pred > 0))),
    }


def _finite_quantile(values: list[float], quantile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile)) if len(finite) else float("nan")


def benchmark_methylation_pairs(
    *,
    pairs_path: str | Path | Iterable[str | Path],
    out_dir: str | Path,
    block_bp: int = 1_000_000,
    n_splits: int = 5,
    split_mode: str = "genomic_block",
    target_column: str = "methylation_delta",
    ridge_alpha: float = 10.0,
    n_bootstrap: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    pair_paths = _as_paths(pairs_path)
    pair_frames = []
    for path in pair_paths:
        frame = pd.read_csv(path, compression="infer")
        frame["pair_source"] = str(path)
        pair_frames.append(frame)
    pairs = pd.concat(pair_frames, ignore_index=True)
    if len(pairs) < 20:
        raise ValueError("At least 20 paired windows are required for blocked evaluation.")
    if target_column not in pairs:
        raise ValueError(
            f"Target column {target_column!r} is absent from the paired table."
        )
    y = pairs[target_column].to_numpy(float)
    if split_mode == "genomic_block":
        groups = (
            pairs["donor_id"].astype(str)
            + ":"
            + pairs["h1_contig"].astype(str)
            + ":"
            + (pairs["h1_window_start"].astype(int) // block_bp).astype(str)
        ).to_numpy()
        split_description = (
            f"GroupKFold by donor and {block_bp}-bp H1 path blocks"
        )
        resampling_unit = f"donor + {block_bp}-bp H1 path block"
    elif split_mode == "donor":
        groups = pairs["donor_id"].astype(str).to_numpy()
        split_description = "leave-one-donor-out GroupKFold"
        resampling_unit = "donor"
    else:
        raise ValueError(
            "split_mode must be either 'genomic_block' or 'donor'; "
            f"received {split_mode!r}."
        )
    unique_groups = np.unique(groups)
    folds = min(n_splits, len(unique_groups))
    if folds < 2:
        raise ValueError(
            f"{split_mode} evaluation requires at least two independent groups."
        )
    if split_mode == "donor":
        folds = len(unique_groups)

    sequence_features = sorted(
        column
        for column in pairs
        if column.startswith("delta_kmer3_")
        or column in {"delta_gc_fraction", "delta_cpg_density"}
    )
    graph_features = [
        column
        for column in [
            "delta_node_count",
            "delta_path_bp",
            "delta_mean_node_length",
            "delta_mean_degree",
            "delta_branching_fraction",
            "delta_reverse_fraction",
            "divergence_x_delta_mean_degree",
            "divergence_x_delta_branching_fraction",
        ]
        if column in pairs
    ]
    feature_sets = {
        "sequence_only": sequence_features,
        "graph_only": graph_features,
        "sequence_graph": sequence_features + graph_features,
    }
    splitter = GroupKFold(n_splits=folds)
    prediction_rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(pairs, y, groups)):
        fold_predictions: dict[str, np.ndarray] = {}
        for model_name, columns in feature_sets.items():
            model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=ridge_alpha, fit_intercept=False),
            )
            model.fit(pairs.iloc[train_idx][columns].fillna(0.0), y[train_idx])
            fold_predictions[model_name] = model.predict(
                pairs.iloc[test_idx][columns].fillna(0.0)
            )

        sequence_model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=ridge_alpha, fit_intercept=False),
        )
        sequence_model.fit(
            pairs.iloc[train_idx][sequence_features].fillna(0.0), y[train_idx]
        )
        train_sequence = sequence_model.predict(
            pairs.iloc[train_idx][sequence_features].fillna(0.0)
        )
        residual_model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=ridge_alpha, fit_intercept=False),
        )
        residual_model.fit(
            pairs.iloc[train_idx][graph_features].fillna(0.0),
            y[train_idx] - train_sequence,
        )
        fold_predictions["sequence_plus_graph_residual"] = (
            sequence_model.predict(
                pairs.iloc[test_idx][sequence_features].fillna(0.0)
            )
            + residual_model.predict(
                pairs.iloc[test_idx][graph_features].fillna(0.0)
            )
        )
        fold_predictions["zero"] = np.zeros(len(test_idx))

        for local_idx, row_idx in enumerate(test_idx):
            for model_name, predictions in fold_predictions.items():
                prediction_rows.append(
                    {
                        "row_index": int(row_idx),
                        "fold": int(fold),
                        "model": model_name,
                        "y_true": float(y[row_idx]),
                        "y_pred": float(predictions[local_idx]),
                        "group": str(groups[row_idx]),
                        "donor_id": str(pairs.iloc[row_idx]["donor_id"]),
                        "pair_source": str(pairs.iloc[row_idx]["pair_source"]),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    metrics = [
        {
            "model": model_name,
            "n": int(len(sub)),
            **_regression_metrics(
                sub["y_true"].to_numpy(float), sub["y_pred"].to_numpy(float)
            ),
        }
        for model_name, sub in predictions.groupby("model", sort=False)
    ]
    per_group_metrics = [
        {
            "model": model_name,
            "group": str(group),
            "donor_id": ",".join(sorted(sub["donor_id"].astype(str).unique())),
            "n": int(len(sub)),
            **_regression_metrics(
                sub["y_true"].to_numpy(float), sub["y_pred"].to_numpy(float)
            ),
        }
        for (model_name, group), sub in predictions.groupby(
            ["model", "group"], sort=False
        )
    ]
    wide = (
        predictions.pivot(
            index="row_index", columns="model", values="y_pred"
        )
        .join(
            predictions.drop_duplicates("row_index")
            .set_index("row_index")[["y_true", "group"]]
        )
        .sort_index()
    )
    grouped_indices = {
        group: np.flatnonzero(wide["group"].astype(str).to_numpy() == group)
        for group in wide["group"].astype(str).unique()
    }
    rng = np.random.default_rng(seed)
    comparison_rows: list[dict[str, Any]] = []
    reference = "sequence_only"
    uncertainty_valid = not (
        split_mode == "donor" and len(unique_groups) < 5
    )
    for model_name in [
        name
        for name in feature_sets
        if name != reference
    ] + ["sequence_plus_graph_residual", "zero"]:
        point_model = _regression_metrics(
            wide["y_true"].to_numpy(float), wide[model_name].to_numpy(float)
        )
        point_reference = _regression_metrics(
            wide["y_true"].to_numpy(float), wide[reference].to_numpy(float)
        )
        draws = {metric: [] for metric in point_model}
        groups_to_sample = np.array(list(grouped_indices))
        if uncertainty_valid:
            for _ in range(n_bootstrap):
                sampled_groups = rng.choice(
                    groups_to_sample, size=len(groups_to_sample), replace=True
                )
                sampled_indices = np.concatenate(
                    [grouped_indices[group] for group in sampled_groups]
                )
                y_sample = wide["y_true"].to_numpy(float)[sampled_indices]
                model_metrics = _regression_metrics(
                    y_sample, wide[model_name].to_numpy(float)[sampled_indices]
                )
                reference_metrics = _regression_metrics(
                    y_sample, wide[reference].to_numpy(float)[sampled_indices]
                )
                for metric in draws:
                    draws[metric].append(
                        model_metrics[metric] - reference_metrics[metric]
                    )
        for metric, values in draws.items():
            comparison_rows.append(
                {
                    "model": model_name,
                    "reference": reference,
                    "metric": metric,
                    "estimate": point_model[metric] - point_reference[metric],
                    "ci95_low": _finite_quantile(values, 0.025),
                    "ci95_high": _finite_quantile(values, 0.975),
                    "n_bootstrap": int(n_bootstrap) if uncertainty_valid else 0,
                    "resampling_unit": resampling_unit,
                    "uncertainty_valid": uncertainty_valid,
                }
            )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        out_dir / "blocked_predictions.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(metrics).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(per_group_metrics).to_csv(
        out_dir / "per_group_metrics.csv", index=False
    )
    pd.DataFrame(comparison_rows).to_csv(
        out_dir / "paired_block_bootstrap.csv", index=False
    )
    summary = {
        "task": f"paired_haplotype_{target_column}",
        "target_column": target_column,
        "split": split_description,
        "split_mode": split_mode,
        "pair_sources": [str(path) for path in pair_paths],
        "n_pairs": int(len(pairs)),
        "n_donors": int(pairs["donor_id"].astype(str).nunique()),
        "n_groups": int(len(unique_groups)),
        "n_splits": int(folds),
        "grouped_uncertainty_valid": uncertainty_valid,
        "feature_sets": feature_sets,
        "metrics": metrics,
        "per_group_metrics": per_group_metrics,
        "paired_block_bootstrap": comparison_rows,
        "interpretation_limit": (
            "A genomic-block split within one donor is a pipeline feasibility "
            "result, not held-out-donor biological generalization."
            if split_mode == "genomic_block"
            else "With only two donors, leave-one-donor-out is a direct "
            "generalization test but donor-level uncertainty is not stable; "
            "additional donors are required for a paper-level estimate."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--gfa", nargs="+", required=True)
    build.add_argument("--methylation", nargs="+", required=True)
    build.add_argument("--out-dir", required=True)
    build.add_argument("--sample", nargs="+")
    build.add_argument(
        "--path-list",
        help="Optional newline-delimited exact GFA path names to retain.",
    )
    build.add_argument("--window-bp", type=int, default=10_000)
    build.add_argument("--min-cpgs", type=int, default=10)
    build.add_argument("--min-depth", type=int, default=100)
    build.add_argument("--min-shared-bp", type=int, default=500)
    build.add_argument("--min-shared-nodes", type=int, default=2)

    repair = subparsers.add_parser("pair-windows")
    repair.add_argument("--windows", required=True)
    repair.add_argument("--out-dir", required=True)
    repair.add_argument("--min-shared-kmers", type=int, default=2)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--pairs", nargs="+", required=True)
    benchmark.add_argument("--out-dir", required=True)
    benchmark.add_argument("--block-bp", type=int, default=1_000_000)
    benchmark.add_argument("--n-splits", type=int, default=5)
    benchmark.add_argument(
        "--split-mode",
        choices=["genomic_block", "donor"],
        default="genomic_block",
    )
    benchmark.add_argument("--target-column", default="methylation_delta")
    benchmark.add_argument("--ridge-alpha", type=float, default=10.0)
    benchmark.add_argument("--n-bootstrap", type=int, default=1_000)
    benchmark.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.command == "build":
        selected_path_names = None
        if args.path_list:
            selected_path_names = [
                line.strip()
                for line in Path(args.path_list).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        result = build_methylation_pairs(
            gfa_path=args.gfa,
            methylation_paths=args.methylation,
            out_dir=args.out_dir,
            sample_ids=args.sample,
            path_names=selected_path_names,
            window_bp=args.window_bp,
            min_cpgs=args.min_cpgs,
            min_depth=args.min_depth,
            min_shared_bp=args.min_shared_bp,
            min_shared_nodes=args.min_shared_nodes,
        )
    elif args.command == "pair-windows":
        result = repair_pairs_with_sequence_anchors(
            windows_path=args.windows,
            out_dir=args.out_dir,
            min_shared_kmer_anchors=args.min_shared_kmers,
        )
    else:
        result = benchmark_methylation_pairs(
            pairs_path=args.pairs,
            out_dir=args.out_dir,
            block_bp=args.block_bp,
            n_splits=args.n_splits,
            split_mode=args.split_mode,
            target_column=args.target_column,
            ridge_alpha=args.ridge_alpha,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
