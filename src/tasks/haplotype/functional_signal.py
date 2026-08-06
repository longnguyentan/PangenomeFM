"""Build paired H1-H2 labels from haplotype-coordinate BigWig tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from tasks.haplotype.methylation import (
    _as_paths,
    _load_selected_gfa,
    _reciprocal_anchor_pairs,
    _walk_windows,
)


def _bigwig_windows(
    path_windows: pd.DataFrame,
    tracks: Iterable[str | Path],
) -> pd.DataFrame:
    try:
        import pyBigWig
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "pyBigWig is required for haplotype functional-track ingestion."
        ) from exc

    handles = []
    try:
        for path in tracks:
            handle = pyBigWig.open(str(path))
            handles.append((str(path), handle, handle.chroms()))
        rows: list[dict[str, Any]] = []
        for row in path_windows.itertuples(index=False):
            contig = str(row.track_contig)
            start = int(row.window_start)
            end = int(row.window_end)
            total_signal = 0.0
            total_coverage = 0.0
            contributing_tracks = 0
            for _, handle, chromosomes in handles:
                if contig not in chromosomes or start >= chromosomes[contig]:
                    continue
                bounded_end = min(end, int(chromosomes[contig]))
                if bounded_end <= start:
                    continue
                mean = handle.stats(
                    contig, start, bounded_end, type="mean", exact=True
                )[0]
                coverage = handle.stats(
                    contig, start, bounded_end, type="coverage", exact=True
                )[0]
                if mean is None or coverage is None or coverage <= 0:
                    continue
                # Strand-specific minus tracks use negative values. Absolute
                # magnitude makes plus/minus coverage directly additive.
                total_signal += abs(float(mean)) * float(coverage)
                total_coverage += float(coverage)
                contributing_tracks += 1
            if contributing_tracks:
                rows.append(
                    {
                        "track_contig": contig,
                        "window_start": start,
                        "expression_signal": total_signal,
                        "expression_coverage": total_coverage,
                        "expression_tracks": contributing_tracks,
                    }
                )
        return pd.DataFrame(rows)
    finally:
        for _, handle, _ in handles:
            handle.close()


def build_bigwig_signal_pairs(
    *,
    gfa_path: str | Path | Iterable[str | Path],
    bigwig_paths: Iterable[str | Path],
    out_dir: str | Path,
    sample_ids: Iterable[str] | None = None,
    path_names: Iterable[str] | None = None,
    window_bp: int = 10_000,
    min_coverage: float = 0.001,
    min_shared_bp: int = 500,
    min_shared_nodes: int = 2,
) -> dict[str, Any]:
    gfa_paths = _as_paths(gfa_path)
    tracks = _as_paths(bigwig_paths)
    selected_samples = set(sample_ids) if sample_ids is not None else None
    selected_path_names = set(path_names) if path_names is not None else None
    sequences, degree, walks = _load_selected_gfa(
        gfa_paths,
        sample_ids=selected_samples,
        path_names=selected_path_names,
    )
    path_windows = _walk_windows(
        sequences=sequences,
        degree=degree,
        walks=walks,
        window_bp=window_bp,
    )
    signal = _bigwig_windows(path_windows, tracks)
    windows = path_windows.merge(
        signal, on=["track_contig", "window_start"], how="inner"
    )
    windows = windows[
        windows["expression_coverage"].ge(min_coverage)
    ].reset_index(drop=True)
    pairs = _reciprocal_anchor_pairs(
        windows,
        node_lengths={node: len(sequence) for node, sequence in sequences.items()},
        min_shared_bp=min_shared_bp,
        min_shared_nodes=min_shared_nodes,
        signal_column="expression_signal",
        signal_name="expression",
        support_columns=("expression_coverage", "expression_tracks"),
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(
        out_dir / "haplotype_windows.csv.gz", index=False, compression="gzip"
    )
    pairs.to_csv(
        out_dir / "paired_windows.csv.gz", index=False, compression="gzip"
    )
    summary = {
        "task": "haplotype_specific_expression_track",
        "signal_definition": (
            "sum across tracks of abs(non-missing mean signal) times covered "
            "window fraction"
        ),
        "pairing": "reciprocal-best shared graph-node anchors",
        "gfa_paths": [str(path) for path in gfa_paths],
        "bigwig_paths": [str(path) for path in tracks],
        "sample_ids": sorted(selected_samples) if selected_samples else None,
        "path_names": (
            sorted(selected_path_names) if selected_path_names is not None else None
        ),
        "window_bp": int(window_bp),
        "min_coverage": float(min_coverage),
        "n_graph_nodes": int(len(sequences)),
        "n_walk_records": int(len(walks)),
        "n_labeled_haplotype_windows": int(len(windows)),
        "n_reciprocal_h1_h2_pairs": int(len(pairs)),
        "n_donors": int(pairs["donor_id"].nunique()) if not pairs.empty else 0,
    }
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gfa", nargs="+", required=True)
    parser.add_argument("--bigwig", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample", nargs="+")
    parser.add_argument("--path-list")
    parser.add_argument("--window-bp", type=int, default=10_000)
    parser.add_argument("--min-coverage", type=float, default=0.001)
    parser.add_argument("--min-shared-bp", type=int, default=500)
    parser.add_argument("--min-shared-nodes", type=int, default=2)
    args = parser.parse_args()
    selected_path_names = None
    if args.path_list:
        selected_path_names = [
            line.strip()
            for line in Path(args.path_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    result = build_bigwig_signal_pairs(
        gfa_path=args.gfa,
        bigwig_paths=args.bigwig,
        out_dir=args.out_dir,
        sample_ids=args.sample,
        path_names=selected_path_names,
        window_bp=args.window_bp,
        min_coverage=args.min_coverage,
        min_shared_bp=args.min_shared_bp,
        min_shared_nodes=args.min_shared_nodes,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
