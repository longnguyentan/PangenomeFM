"""Add exact repeat/QC overlaps and graph-native SV/CN proxies to paired windows."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import pyBigWig
except ImportError:  # pragma: no cover - reported at runtime
    pyBigWig = None


def interval_overlap_fraction(
    intervals: Iterable[tuple[int, int]], start: int, end: int
) -> float:
    clipped = sorted(
        (max(start, left), min(end, right))
        for left, right in intervals
        if right > start and left < end
    )
    if not clipped or end <= start:
        return 0.0
    covered = 0
    cursor_start, cursor_end = clipped[0]
    for left, right in clipped[1:]:
        if left > cursor_end:
            covered += cursor_end - cursor_start
            cursor_start, cursor_end = left, right
        else:
            cursor_end = max(cursor_end, right)
    covered += cursor_end - cursor_start
    return float(covered / (end - start))


def _load_bed(path: Path) -> dict[str, list[tuple[int, int]]]:
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                intervals[fields[0]].append((int(fields[1]), int(fields[2])))
    return {chrom: sorted(values) for chrom, values in intervals.items()}


def _load_labeled_bed(
    path: Path,
) -> dict[str, dict[str, list[tuple[int, int]]]]:
    by_label: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            by_label[fields[3].lower()][fields[0]].append(
                (int(fields[1]), int(fields[2]))
            )
    return {
        label: {
            chrom: sorted(values) for chrom, values in chrom_intervals.items()
        }
        for label, chrom_intervals in by_label.items()
    }


def _contig_candidates(row: pd.Series, side: str) -> list[str]:
    donor = str(row["donor_id"])
    contig = str(row[f"{side}_contig"])
    track = str(row.get(f"{side}_track_contig", ""))
    haplotype = "1" if side == "h1" else "2"
    values = [track, f"{donor}#{haplotype}#{contig}", contig]
    return list(dict.fromkeys(value for value in values if value and value != "nan"))


def _bigbed_fraction(
    handle: object,
    candidates: list[str],
    start: int,
    end: int,
) -> tuple[float, str]:
    chroms = handle.chroms()
    for chrom in candidates:
        if chrom not in chroms:
            continue
        entries = handle.entries(chrom, start, min(end, int(chroms[chrom]))) or []
        return interval_overlap_fraction(
            [(int(left), int(right)) for left, right, *_ in entries], start, end
        ), chrom
    return float("nan"), ""


def _bed_fraction(
    intervals: dict[str, list[tuple[int, int]]],
    candidates: list[str],
    start: int,
    end: int,
) -> tuple[float, str]:
    for chrom in candidates:
        if chrom in intervals:
            return interval_overlap_fraction(intervals[chrom], start, end), chrom
    return float("nan"), ""


def annotate_paired_windows(
    *,
    pairs_path: str | Path,
    epigenome_root: str | Path,
    out_path: str | Path,
    strict: bool = True,
) -> dict[str, object]:
    pairs = pd.read_csv(pairs_path, compression="infer")
    epigenome_root = Path(epigenome_root)
    annotated: list[pd.DataFrame] = []
    source_rows: list[dict[str, object]] = []

    for donor, frame in pairs.groupby("donor_id", sort=False):
        donor_dir = epigenome_root / str(donor)
        repeat_paths = {
            "h1": donor_dir / "RepeatMasker.hap1.bb",
            "h2": donor_dir / "RepeatMasker.hap2.bb",
        }
        hmm_path = donor_dir / "HMMFlagger.PacBio.bed.gz"
        missing = [
            str(path)
            for path in [*repeat_paths.values(), hmm_path]
            if not path.exists()
        ]
        if missing and strict:
            raise FileNotFoundError(
                f"Missing annotations for donor {donor}: {missing}"
            )
        repeat_handles = {}
        if pyBigWig is None and any(path.exists() for path in repeat_paths.values()):
            raise RuntimeError("pyBigWig is required to read RepeatMasker bigBed files.")
        for side, path in repeat_paths.items():
            if path.exists():
                repeat_handles[side] = pyBigWig.open(str(path))
        hmm = _load_bed(hmm_path) if hmm_path.exists() else {}
        hmm_by_label = (
            _load_labeled_bed(hmm_path) if hmm_path.exists() else {}
        )

        donor_frame = frame.copy()
        for side in ("h1", "h2"):
            repeat_values: list[float] = []
            repeat_contigs: list[str] = []
            hmm_values: list[float] = []
            hmm_contigs: list[str] = []
            hmm_label_values: dict[str, list[float]] = {
                label: [] for label in ("dup", "col", "err", "nnn")
            }
            for _, row in donor_frame.iterrows():
                start = int(row[f"{side}_window_start"])
                end = start + int(row.get(f"{side}_path_bp", 10_000))
                candidates = _contig_candidates(row, side)
                if side in repeat_handles:
                    repeat_fraction, repeat_contig = _bigbed_fraction(
                        repeat_handles[side], candidates, start, end
                    )
                else:
                    repeat_fraction, repeat_contig = float("nan"), ""
                hmm_fraction, hmm_contig = _bed_fraction(
                    hmm, candidates, start, end
                )
                repeat_values.append(repeat_fraction)
                repeat_contigs.append(repeat_contig)
                hmm_values.append(hmm_fraction)
                hmm_contigs.append(hmm_contig)
                for label, values in hmm_label_values.items():
                    value, _ = _bed_fraction(
                        hmm_by_label.get(label, {}), candidates, start, end
                    )
                    values.append(value)
            donor_frame[f"{side}_repeat_fraction"] = repeat_values
            donor_frame[f"{side}_repeat_contig"] = repeat_contigs
            donor_frame[f"{side}_hmmflagger_fraction"] = hmm_values
            donor_frame[f"{side}_hmmflagger_contig"] = hmm_contigs
            for label, values in hmm_label_values.items():
                donor_frame[
                    f"{side}_hmmflagger_{label}_fraction"
                ] = values
            multiplicity = donor_frame.get(
                f"{side}_mean_path_node_multiplicity",
                pd.Series(np.ones(len(donor_frame)), index=donor_frame.index),
            )
            donor_frame[f"{side}_copy_number_proxy"] = multiplicity.astype(float)
            multicopy = donor_frame.get(
                f"{side}_multicopy_fraction",
                pd.Series(np.zeros(len(donor_frame)), index=donor_frame.index),
            )
            donor_frame[f"{side}_graph_mappability_proxy"] = (
                1.0 - multicopy.astype(float).clip(0.0, 1.0)
            )
        for handle in repeat_handles.values():
            handle.close()

        donor_frame["delta_repeat_fraction"] = (
            donor_frame["h1_repeat_fraction"] - donor_frame["h2_repeat_fraction"]
        )
        donor_frame["delta_hmmflagger_fraction"] = (
            donor_frame["h1_hmmflagger_fraction"]
            - donor_frame["h2_hmmflagger_fraction"]
        )
        for label in ("dup", "col", "err", "nnn"):
            donor_frame[f"delta_hmmflagger_{label}_fraction"] = (
                donor_frame[f"h1_hmmflagger_{label}_fraction"]
                - donor_frame[f"h2_hmmflagger_{label}_fraction"]
            )
        donor_frame["delta_copy_number_proxy"] = (
            donor_frame["h1_copy_number_proxy"]
            - donor_frame["h2_copy_number_proxy"]
        )
        donor_frame["delta_graph_mappability_proxy"] = (
            donor_frame["h1_graph_mappability_proxy"]
            - donor_frame["h2_graph_mappability_proxy"]
        )
        donor_frame["sv_context_score"] = 1.0 - donor_frame["node_jaccard"]
        donor_frame["sv_length_delta_bp"] = (
            donor_frame.get("h1_path_bp", 10_000)
            - donor_frame.get("h2_path_bp", 10_000)
        ).abs()
        donor_frame["sv_event_proxy"] = (
            donor_frame["sv_context_score"].ge(0.10)
            | donor_frame["sv_length_delta_bp"].ge(50)
        ).astype(int)
        annotated.append(donor_frame)
        source_rows.append(
            {
                "donor_id": str(donor),
                "repeatmasker_available": all(
                    path.exists() for path in repeat_paths.values()
                ),
                "hmmflagger_available": hmm_path.exists(),
            }
        )

    result = pd.concat(annotated, ignore_index=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, compression="infer")
    summary = {
        "task": "paired_window_context_annotation",
        "n_pairs": int(len(result)),
        "n_donors": int(result["donor_id"].nunique()),
        "exact_annotations": {
            "repeat": "donor/haplotype-specific RepeatMasker bigBed overlap",
            "assembly_qc": (
                "donor/haplotype-specific PacBio HMMFlagger any, duplication, "
                "collapse, error, and N-run interval overlaps"
            ),
        },
        "proxies_not_ground_truth": {
            "mappability": "1 - graph path multicopy fraction",
            "copy_number": "mean graph-node multiplicity along the haplotype path",
            "sv": "node-set divergence and paired path-length difference",
        },
        "sources": source_rows,
        "output": str(out_path),
    }
    out_path.with_suffix("").with_suffix(".annotations.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--epigenome-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    summary = annotate_paired_windows(
        pairs_path=args.pairs,
        epigenome_root=args.epigenome_root,
        out_path=args.out,
        strict=not args.allow_missing,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
