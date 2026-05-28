#!/usr/bin/env python3
"""Audit how much of the cCRE-labeled node universe appears in benchmark slices.

The logistic cCRE baselines evaluate all labeled GRCh38 nodes, while the
current GAT experiments evaluate only labeled nodes that appear inside
benchmark slices. This script quantifies that mismatch so the paper can state
the caveat precisely.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(sys.maxsize)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open("r", newline="")


def _segid_from_slice_id(raw: str) -> int | None:
    if raw.startswith("s") and raw[1:].isdigit():
        return int(raw[1:])
    if raw.isdigit():
        return int(raw)
    return None


def _load_labels(path: Path) -> dict[int, dict[str, str]]:
    with _open_text(path) as fh:
        rows = csv.DictReader(fh)
        return {
            int(row["segid"]): {
                "chrom": row["chrom"],
                "ccre_class": row["ccre_class"],
            }
            for row in rows
        }


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _slice_labeled_segids(path: Path, labels: dict[int, dict[str, str]]) -> set[int]:
    labeled: set[int] = set()
    with _open_text(path) as fh:
        rows = csv.DictReader(fh)
        for row in rows:
            segid = _segid_from_slice_id(row["id"])
            if segid is not None and segid in labels:
                labeled.add(segid)
    return labeled


def _split_for_target(target_sn: str, val_chrs: set[str], test_chrs: set[str]) -> str:
    if target_sn in val_chrs:
        return "val"
    if target_sn in test_chrs:
        return "test"
    return "train"


def _pct(num: int, den: int) -> str:
    if den == 0:
        return "0.00"
    return f"{100.0 * num / den:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default="data/hprc/benchmark_paper/manifest.csv",
        help="Benchmark manifest used by current cCRE GAT experiments.",
    )
    ap.add_argument(
        "--node-labels",
        default="data/hprc/ccre/run_003/node_labels.csv.gz",
        help="Mapped cCRE node labels.",
    )
    ap.add_argument(
        "--out-dir",
        default="results/hprc/ccre_window_coverage",
        help="Output directory for coverage audit artifacts.",
    )
    ap.add_argument(
        "--val-chrs",
        nargs="+",
        default=["GRCh38#0#chr16"],
        help="Validation target SN strings.",
    )
    ap.add_argument(
        "--test-chrs",
        nargs="+",
        default=["GRCh38#0#chr8", "GRCh38#0#chr19", "GRCh38#0#chr22"],
        help="Test target SN strings used by current cCRE runs.",
    )
    args = ap.parse_args()

    manifest_path = ROOT / args.manifest
    label_path = ROOT / args.node_labels
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = _load_labels(label_path)
    manifest = _load_manifest(manifest_path)
    val_chrs = set(args.val_chrs)
    test_chrs = set(args.test_chrs)

    split_closure_sets: dict[tuple[str, str], set[int]] = defaultdict(set)
    split_any_sets: dict[str, set[int]] = defaultdict(set)
    target_closure_sets: dict[tuple[str, str], set[int]] = defaultdict(set)
    occurrences: dict[tuple[str, str], int] = defaultdict(int)
    slice_counts: dict[tuple[str, str], int] = defaultdict(int)

    for row in manifest:
        target = row["target_sn"]
        closure = row["closure"]
        split = _split_for_target(target, val_chrs, test_chrs)
        labeled = _slice_labeled_segids(ROOT / row["segments_path"], labels)
        key = (split, closure)
        split_closure_sets[key].update(labeled)
        split_any_sets[split].update(labeled)
        target_closure_sets[(target, closure)].update(labeled)
        occurrences[key] += len(labeled)
        slice_counts[key] += 1

    total_labeled = len(labels)
    all_test_labeled = {
        segid
        for segid, meta in labels.items()
        if f"GRCh38#0#{meta['chrom']}" in test_chrs
    }

    rows = []
    for split in ["train", "val", "test"]:
        for closure in ["strict", "1hop"]:
            key = (split, closure)
            rows.append(
                {
                    "split": split,
                    "closure": closure,
                    "n_slices": slice_counts[key],
                    "unique_labeled_nodes": len(split_closure_sets[key]),
                    "labeled_node_occurrences": occurrences[key],
                    "fraction_of_all_labeled": (
                        len(split_closure_sets[key]) / total_labeled
                        if total_labeled
                        else 0.0
                    ),
                }
            )
    rows.append(
        {
            "split": "test",
            "closure": "strict_or_1hop",
            "n_slices": sum(slice_counts[("test", c)] for c in ["strict", "1hop"]),
            "unique_labeled_nodes": len(split_any_sets["test"]),
            "labeled_node_occurrences": sum(
                occurrences[("test", c)] for c in ["strict", "1hop"]
            ),
            "fraction_of_all_labeled": (
                len(split_any_sets["test"]) / total_labeled if total_labeled else 0.0
            ),
        }
    )

    csv_path = out_dir / "coverage_by_split_closure.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_target_rows = []
    for target in sorted({row["target_sn"] for row in manifest}):
        split = _split_for_target(target, val_chrs, test_chrs)
        for closure in ["strict", "1hop"]:
            covered = target_closure_sets[(target, closure)]
            by_target_rows.append(
                {
                    "target_sn": target,
                    "split": split,
                    "closure": closure,
                    "unique_labeled_nodes": len(covered),
                }
            )

    target_csv_path = out_dir / "coverage_by_target.csv"
    with target_csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(by_target_rows[0].keys()))
        writer.writeheader()
        writer.writerows(by_target_rows)

    test_window_unique = len(split_any_sets["test"])
    md_lines = [
        "# cCRE Benchmark-Window Coverage Audit",
        "",
        "This audit quantifies why the current cCRE GAT evaluation is not yet",
        "directly comparable to the all-node logistic baseline.",
        "",
        f"- Node label file: `{args.node_labels}`",
        f"- Benchmark manifest: `{args.manifest}`",
        f"- Total labeled GRCh38 nodes: {total_labeled:,}",
        f"- Logistic test chromosomes: {', '.join(sorted(test_chrs))}",
        f"- All-node logistic test labeled nodes: {len(all_test_labeled):,}",
        f"- Unique test labeled nodes covered by current benchmark windows: {test_window_unique:,}",
        f"- Window coverage of logistic test universe: {_pct(test_window_unique, len(all_test_labeled))}%",
        "",
        "## Coverage By Split And Closure",
        "",
        "| split | closure | slices | unique labeled nodes | labeled occurrences | % of all labeled nodes |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            "| {split} | {closure} | {n_slices} | {unique_labeled_nodes:,} | "
            "{labeled_node_occurrences:,} | {pct} |".format(
                **row,
                pct=f"{100.0 * row['fraction_of_all_labeled']:.2f}",
            )
        )
    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current GAT benchmark-window evaluation covers only a subset of",
            "the labeled nodes used by the logistic cCRE baseline. The next fair",
            "comparison should either build cCRE-specific windows that cover most",
            "labeled GRCh38 nodes or restrict the logistic baseline to the same",
            "window-covered node universe.",
            "",
            f"CSV outputs: `{args.out_dir}/coverage_by_split_closure.csv` and "
            f"`{args.out_dir}/coverage_by_target.csv`.",
            "",
        ]
    )
    md_path = out_dir / "coverage_audit.md"
    md_path.write_text("\n".join(md_lines))

    print(f"wrote {csv_path.relative_to(ROOT)}")
    print(f"wrote {target_csv_path.relative_to(ROOT)}")
    print(f"wrote {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
