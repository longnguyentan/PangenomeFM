"""Construction-normalize cleaned graph tables by uniformly chopping nodes.

The transformation preserves each original oriented edge and inserts the two
oriented traversal directions through every chopped segment. It is intentionally
table-native so the same policy can be applied to HPRC and HGSVC inputs even
when vg/odgi are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import IO, Any


SEGMENT_COLUMNS = ["id", "name", "seq", "LN", "SN", "SO", "SR"]
LINK_COLUMNS = [
    "from_seg",
    "from_orient",
    "to_seg",
    "to_orient",
    "overlap",
    "SR",
    "L1",
    "L2",
]
MAP_COLUMNS = ["parent_name", "chunk_name", "chunk_id", "chunk_index", "chunk_count"]

# Whole-genome graph segments can exceed Python's conservative CSV field limit.
csv.field_size_limit(sys.maxsize)


def _open_text(path: Path, mode: str) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", newline="", encoding="utf-8")
    return path.open(mode, newline="", encoding="utf-8")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _chunk_name(parent: str, index: int) -> str:
    return f"{parent}~chop{index:06d}"


def canonicalize_graph_tables(
    *,
    segments_path: str | Path,
    links_path: str | Path,
    out_dir: str | Path,
    max_node_length: int = 1024,
) -> dict[str, Any]:
    if max_node_length < 1:
        raise ValueError("max_node_length must be positive")

    segments_path = Path(segments_path)
    links_path = Path(links_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segments_out = out_dir / "full_segments.csv.gz"
    links_out = out_dir / "full_links.csv.gz"
    map_out = out_dir / "segment_chop_map.csv.gz"
    summary_out = out_dir / "canonicalization_summary.json"

    endpoints: dict[str, tuple[str, str, int, int]] = {}
    stats: dict[str, Any] = {
        "max_node_length": int(max_node_length),
        "input_segments": 0,
        "output_segments": 0,
        "segments_chopped": 0,
        "input_links": 0,
        "output_links": 0,
        "internal_links": 0,
        "missing_link_endpoints": 0,
    }

    # Internal links are written first. Original links are appended in pass 2.
    with (
        _open_text(segments_path, "r") as in_fh,
        _open_text(segments_out, "w") as seg_fh,
        _open_text(map_out, "w") as map_fh,
        _open_text(links_out, "w") as link_fh,
    ):
        reader = csv.DictReader(in_fh)
        missing = set(SEGMENT_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Segment table missing columns: {sorted(missing)}")
        seg_writer = csv.DictWriter(seg_fh, fieldnames=SEGMENT_COLUMNS)
        map_writer = csv.DictWriter(map_fh, fieldnames=MAP_COLUMNS)
        link_writer = csv.DictWriter(link_fh, fieldnames=LINK_COLUMNS)
        seg_writer.writeheader()
        map_writer.writeheader()
        link_writer.writeheader()

        next_id = 0
        for row in reader:
            stats["input_segments"] += 1
            parent = str(row["name"])
            sequence = str(row["seq"])
            declared_length = _as_int(row.get("LN"), len(sequence))
            if sequence == "*":
                raise ValueError(
                    f"Cannot sequence-preservingly chop segment {parent!r} with seq='*'."
                )
            if declared_length != len(sequence):
                raise ValueError(
                    f"Segment {parent!r} has LN={declared_length}, sequence length={len(sequence)}."
                )
            n_chunks = max(1, (declared_length + max_node_length - 1) // max_node_length)
            stats["segments_chopped"] += int(n_chunks > 1)
            first_name = _chunk_name(parent, 0)
            last_name = _chunk_name(parent, n_chunks - 1)
            first_length = min(declared_length, max_node_length)
            last_length = declared_length - (n_chunks - 1) * max_node_length
            endpoints[parent] = (first_name, last_name, first_length, last_length)

            previous_name: str | None = None
            previous_length = 0
            for chunk_index in range(n_chunks):
                offset = chunk_index * max_node_length
                chunk_sequence = sequence[offset : offset + max_node_length]
                chunk_name = _chunk_name(parent, chunk_index)
                chunk_length = len(chunk_sequence)
                seg_writer.writerow(
                    {
                        "id": next_id,
                        "name": chunk_name,
                        "seq": chunk_sequence,
                        "LN": chunk_length,
                        "SN": row.get("SN", ""),
                        "SO": _as_int(row.get("SO")) + offset,
                        "SR": row.get("SR", 0),
                    }
                )
                map_writer.writerow(
                    {
                        "parent_name": parent,
                        "chunk_name": chunk_name,
                        "chunk_id": next_id,
                        "chunk_index": chunk_index,
                        "chunk_count": n_chunks,
                    }
                )
                next_id += 1
                stats["output_segments"] += 1

                if previous_name is not None:
                    # Forward traversal and its reverse-oriented counterpart.
                    link_writer.writerow(
                        {
                            "from_seg": previous_name,
                            "from_orient": "+",
                            "to_seg": chunk_name,
                            "to_orient": "+",
                            "overlap": "0M",
                            "SR": row.get("SR", 0),
                            "L1": previous_length,
                            "L2": chunk_length,
                        }
                    )
                    link_writer.writerow(
                        {
                            "from_seg": chunk_name,
                            "from_orient": "-",
                            "to_seg": previous_name,
                            "to_orient": "-",
                            "overlap": "0M",
                            "SR": row.get("SR", 0),
                            "L1": chunk_length,
                            "L2": previous_length,
                        }
                    )
                    stats["internal_links"] += 2
                    stats["output_links"] += 2
                previous_name = chunk_name
                previous_length = chunk_length

        with _open_text(links_path, "r") as original_link_fh:
            link_reader = csv.DictReader(original_link_fh)
            missing = set(LINK_COLUMNS[:5]) - set(link_reader.fieldnames or [])
            if missing:
                raise ValueError(f"Link table missing columns: {sorted(missing)}")
            for row in link_reader:
                stats["input_links"] += 1
                source = endpoints.get(str(row["from_seg"]))
                target = endpoints.get(str(row["to_seg"]))
                if source is None or target is None:
                    stats["missing_link_endpoints"] += 1
                    continue
                from_orient = str(row["from_orient"])
                to_orient = str(row["to_orient"])
                from_name = source[1] if from_orient == "+" else source[0]
                from_length = source[3] if from_orient == "+" else source[2]
                to_name = target[0] if to_orient == "+" else target[1]
                to_length = target[2] if to_orient == "+" else target[3]
                link_writer.writerow(
                    {
                        "from_seg": from_name,
                        "from_orient": from_orient,
                        "to_seg": to_name,
                        "to_orient": to_orient,
                        "overlap": row.get("overlap", "0M"),
                        "SR": row.get("SR", 0),
                        "L1": from_length,
                        "L2": to_length,
                    }
                )
                stats["output_links"] += 1

    if stats["missing_link_endpoints"]:
        raise ValueError(
            f"{stats['missing_link_endpoints']} links referenced absent segments; "
            "outputs were written for diagnosis but are not valid."
        )
    stats["segment_expansion_factor"] = (
        stats["output_segments"] / stats["input_segments"]
        if stats["input_segments"]
        else 0.0
    )
    stats["inputs"] = {
        "segments": str(segments_path),
        "links": str(links_path),
    }
    stats["outputs"] = {
        "segments": str(segments_out),
        "links": str(links_out),
        "mapping": str(map_out),
    }
    summary_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--links", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-node-length", type=int, default=1024)
    args = parser.parse_args()
    result = canonicalize_graph_tables(
        segments_path=args.segments,
        links_path=args.links,
        out_dir=args.out_dir,
        max_node_length=args.max_node_length,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
