from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import TextIO


SEGMENT_COLUMNS = ["id", "name", "seq", "LN", "SN", "SO", "SR"]
LINK_COLUMNS = ["from_seg", "from_orient", "to_seg", "to_orient", "overlap", "SR", "L1", "L2"]


def _open_text(path: str | Path, mode: str) -> TextIO:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", newline="", encoding="utf-8")  # type: ignore[return-value]
    return path.open(mode, newline="", encoding="utf-8")


def _parse_tags(fields: list[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for field in fields:
        parts = field.split(":", 2)
        if len(parts) == 3:
            tags[parts[0]] = parts[2]
    return tags


def _as_int(value: str | None, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def parse_gfa_to_tables(
    *,
    gfa_path: str | Path,
    segments_out: str | Path,
    links_out: str | Path,
    summary_out: str | Path | None = None,
    max_lines: int | None = None,
    target_sns: set[str] | None = None,
    include_link_neighbors: bool = False,
) -> dict[str, int]:
    """Convert S/L records from a GFA/rGFA file to project CSV tables.

    The parser streams line by line so it can handle large public graph files.
    It preserves the table schema used by the existing `pgb` training code.
    Path records are intentionally not materialized yet; path-aware extraction
    should become a separate parser stage once the exact external graph format
    is selected.
    """
    gfa_path = Path(gfa_path)
    segments_out = Path(segments_out)
    links_out = Path(links_out)
    segments_out.parent.mkdir(parents=True, exist_ok=True)
    links_out.parent.mkdir(parents=True, exist_ok=True)
    if summary_out:
        Path(summary_out).parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "lines_read": 0,
        "segments": 0,
        "links": 0,
        "paths": 0,
        "walks": 0,
        "segments_missing_sn": 0,
        "segments_missing_so": 0,
        "segments_missing_sr": 0,
    }

    if target_sns:
        stats["target_sns"] = len(target_sns)
        stats["include_link_neighbors"] = int(include_link_neighbors)
        _parse_targeted_gfa(
            gfa_path=gfa_path,
            segments_out=segments_out,
            links_out=links_out,
            stats=stats,
            max_lines=max_lines,
            target_sns=target_sns,
            include_link_neighbors=include_link_neighbors,
        )
    else:
        _parse_full_gfa(
            gfa_path=gfa_path,
            segments_out=segments_out,
            links_out=links_out,
            stats=stats,
            max_lines=max_lines,
        )

    if summary_out:
        Path(summary_out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def _write_segment(seg_writer, name: str, seq: str, tags: dict[str, str], stats: dict[str, int]) -> None:
    ln = _as_int(tags.get("LN"), len(seq) if seq != "*" else 0)
    sn = tags.get("SN", "")
    so = _as_int(tags.get("SO"), 0)
    sr = _as_int(tags.get("SR"), 0)
    stats["segments"] += 1
    stats["segments_missing_sn"] += int(not sn)
    stats["segments_missing_so"] += int("SO" not in tags)
    stats["segments_missing_sr"] += int("SR" not in tags)
    seg_writer.writerow(
        {
            "id": stats["segments"] - 1,
            "name": name,
            "seq": seq,
            "LN": ln,
            "SN": sn,
            "SO": so,
            "SR": sr,
        }
    )


def _write_link(link_writer, fields: list[str], stats: dict[str, int]) -> None:
    tags = _parse_tags(fields[6:])
    stats["links"] += 1
    link_writer.writerow(
        {
            "from_seg": fields[1],
            "from_orient": fields[2],
            "to_seg": fields[3],
            "to_orient": fields[4],
            "overlap": fields[5],
            "SR": _as_int(tags.get("SR"), 0),
            "L1": tags.get("L1", ""),
            "L2": tags.get("L2", ""),
        }
    )


def _parse_full_gfa(
    *,
    gfa_path: Path,
    segments_out: Path,
    links_out: Path,
    stats: dict[str, int],
    max_lines: int | None,
) -> None:
    with (
        _open_text(gfa_path, "r") as gfa,
        _open_text(segments_out, "w") as seg_fh,
        _open_text(links_out, "w") as link_fh,
    ):
        seg_writer = csv.DictWriter(seg_fh, fieldnames=SEGMENT_COLUMNS)
        link_writer = csv.DictWriter(link_fh, fieldnames=LINK_COLUMNS)
        seg_writer.writeheader()
        link_writer.writeheader()

        for line in gfa:
            if max_lines is not None and stats["lines_read"] >= max_lines:
                break
            stats["lines_read"] += 1
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if not fields:
                continue
            rec_type = fields[0]

            if rec_type == "S":
                if len(fields) < 3:
                    continue
                name = fields[1]
                seq = fields[2]
                tags = _parse_tags(fields[3:])
                _write_segment(seg_writer, name, seq, tags, stats)
            elif rec_type == "L":
                if len(fields) < 6:
                    continue
                _write_link(link_writer, fields, stats)
            elif rec_type == "P":
                stats["paths"] += 1
            elif rec_type == "W":
                stats["walks"] += 1


def _parse_targeted_gfa(
    *,
    gfa_path: Path,
    segments_out: Path,
    links_out: Path,
    stats: dict[str, int],
    max_lines: int | None,
    target_sns: set[str],
    include_link_neighbors: bool,
) -> None:
    target_names: set[str] = set()
    keep_names: set[str] = set()

    with _open_text(gfa_path, "r") as gfa:
        for line in gfa:
            if max_lines is not None and stats["lines_read"] >= max_lines:
                break
            stats["lines_read"] += 1
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if not fields:
                continue
            rec_type = fields[0]
            if rec_type == "S" and len(fields) >= 3:
                tags = _parse_tags(fields[3:])
                if tags.get("SN", "") in target_sns:
                    target_names.add(fields[1])
                    keep_names.add(fields[1])
            elif rec_type == "P":
                stats["paths"] += 1
            elif rec_type == "W":
                stats["walks"] += 1

    if include_link_neighbors:
        lines_read_pass2 = 0
        with _open_text(gfa_path, "r") as gfa:
            for line in gfa:
                if max_lines is not None and lines_read_pass2 >= max_lines:
                    break
                lines_read_pass2 += 1
                if not line or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 6 or fields[0] != "L":
                    continue
                if fields[1] in target_names or fields[3] in target_names:
                    keep_names.add(fields[1])
                    keep_names.add(fields[3])

    lines_read_pass3 = 0
    with (
        _open_text(gfa_path, "r") as gfa,
        _open_text(segments_out, "w") as seg_fh,
        _open_text(links_out, "w") as link_fh,
    ):
        seg_writer = csv.DictWriter(seg_fh, fieldnames=SEGMENT_COLUMNS)
        link_writer = csv.DictWriter(link_fh, fieldnames=LINK_COLUMNS)
        seg_writer.writeheader()
        link_writer.writeheader()

        for line in gfa:
            if max_lines is not None and lines_read_pass3 >= max_lines:
                break
            lines_read_pass3 += 1
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if not fields:
                continue
            rec_type = fields[0]
            if rec_type == "S" and len(fields) >= 3 and fields[1] in keep_names:
                _write_segment(seg_writer, fields[1], fields[2], _parse_tags(fields[3:]), stats)
            elif rec_type == "L" and len(fields) >= 6:
                if fields[1] in keep_names and fields[3] in keep_names:
                    _write_link(link_writer, fields, stats)

    stats["target_segments"] = len(target_names)
    stats["kept_segment_names"] = len(keep_names)
