#!/usr/bin/env python3
"""Read-only preservation check for original and existing LaTeX comments.

Run from any directory. An optional --manuscript-dir checks a copied bundle;
--original-source and --followup-source verify other copies of the supplied
attachments. Each pinned attachment is also verified automatically when its
recorded local path exists. Follow-up response R08 must remain evidence-open.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def comment_start(line: str) -> int | None:
    """Find the first percent sign not escaped by an odd backslash run."""
    for index, char in enumerate(line):
        if char != "%":
            continue
        previous = index - 1
        while previous >= 0 and line[previous] == "\\":
            previous -= 1
        if (index - previous - 1) % 2 == 0:
            return index
    return None


def comment_records(text: str) -> list[dict]:
    records = []
    for number, line in enumerate(text.splitlines(), 1):
        start = comment_start(line)
        if start is not None:
            records.append({"line": number, "comment": line[start:]})
    return records


def read_text_exact(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def without_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        start = comment_start(line)
        lines.append(line if start is None else line[:start])
    return "\n".join(lines)


def counts(records: list[dict]) -> Counter:
    return Counter(record["comment"] for record in records)


def check(args: argparse.Namespace) -> int:
    directory = args.manuscript_dir.resolve()
    manifest_path = directory / "editorial_comment_manifest.json"
    manifest = json.loads(read_text_exact(manifest_path))
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported editorial comment manifest schema")
    errors = []
    original = manifest["original_source"]
    expected_original = original["comments"]
    if len(expected_original) != original["comment_count"] or len(expected_original) != 118:
        errors.append("Manifest must record all 118 original comment occurrences")

    source = args.original_source or Path(original["path"])
    if source.exists():
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != original["sha256"]:
            errors.append(f"Original attachment SHA-256 mismatch: {source}")
        if comment_records(raw.decode("utf-8")) != expected_original:
            errors.append("Original attachment comments/line anchors differ from manifest")
        source_status = "attachment SHA-256 and original line anchors verified"
    elif args.original_source is not None:
        errors.append(f"Requested original attachment does not exist: {source}")
        source_status = "requested attachment unavailable"
    else:
        source_status = "local attachment unavailable; checked pinned manifest records"

    followup = manifest["followup_source"]
    expected_followup = followup["comments"]
    if len(expected_followup) != followup["comment_count"] or len(expected_followup) != 25:
        errors.append("Manifest must record all 25 follow-up comment occurrences")
    followup_source = args.followup_source or Path(followup["path"])
    if followup_source.exists():
        raw = followup_source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != followup["sha256"]:
            errors.append(f"Follow-up attachment SHA-256 mismatch: {followup_source}")
        if comment_records(raw.decode("utf-8")) != expected_followup:
            errors.append("Follow-up attachment comments/line anchors differ from manifest")
        followup_status = "follow-up attachment SHA-256 and line anchors verified"
    elif args.followup_source is not None:
        errors.append(f"Requested follow-up attachment does not exist: {followup_source}")
        followup_status = "requested follow-up attachment unavailable"
    else:
        followup_status = "local follow-up attachment unavailable; checked pinned manifest records"

    response_groups = followup["response_groups"]
    required_ids = [f"R{number:02}" for number in range(1, 14)]
    if [group["id"] for group in response_groups] != required_ids:
        errors.append("Follow-up response groups must identify R01 through R13 exactly once in order")
    grouped_lines = []
    for group in response_groups:
        grouped_lines.extend(group["comment_lines"])
        start, end = group["line_range"]
        if any(not start <= line <= end for line in group["comment_lines"]):
            errors.append(f"{group['id']}: comment anchors fall outside the recorded source-line range")
        status = "Evidence open" if group["id"] == "R08" else "Done"
        if group["required_status"] != status:
            errors.append(f"{group['id']}: required status must remain {status!r}")
    if Counter(grouped_lines) != Counter(record["line"] for record in expected_followup):
        errors.append("Follow-up groups must cover each of the 25 comment lines exactly once")

    single_file = manifest.get("layout") == "single_file"
    current_main = read_text_exact(directory / "main.tex")
    if single_file:
        begin, end = manifest["inline_archive_markers"]
        if current_main.count(begin) != 1 or current_main.count(end) != 1:
            raise ValueError("Expected exactly one preserved inline archive")
        archive_text = current_main.split(begin, 1)[1].split(end, 1)[0]
    else:
        archive_text = read_text_exact(directory / manifest["archive_file"])
    archive_lines = archive_text.splitlines()
    for number, line in enumerate(archive_lines, 1):
        if line.strip() and not line.lstrip().startswith("%"):
            errors.append(f"Archive contains active TeX at line {number}")
    anchor_pattern = re.compile(r"^% Long Note: Original attachment line (\d+)\.$")
    anchored = []
    for index, line in enumerate(archive_lines):
        match = anchor_pattern.fullmatch(line)
        if match:
            following = archive_lines[index + 1] if index + 1 < len(archive_lines) else ""
            anchored.append({"line": int(match.group(1)), "comment": following})
    if anchored != expected_original:
        errors.append("Archive must retain every exact original comment suffix at its source-line anchor, in order")
    archived_counts = counts(comment_records(archive_text))
    for comment, expected_count in counts(expected_original).items():
        if archived_counts[comment] != expected_count:
            errors.append(f"Archive occurrence mismatch ({expected_count} required): {comment!r}")
    for comment, missing_count in (counts(expected_followup) - archived_counts).items():
        errors.append(f"Archive lacks {missing_count} exact follow-up comment occurrence(s): {comment!r}")

    response_files = ["main.tex", manifest["archive_file"]]
    if followup["required_response_files"] != response_files:
        errors.append("Follow-up responses must be verified in both main.tex and the editorial archive")
    response_pattern = re.compile(r"^% Long Note:\s*(.*?)\s*\[(R\d{2})\](?=\s|[.:;]|$)")
    for filename in response_files:
        responses = {}
        for record in comment_records(archive_text if single_file and filename == manifest["archive_file"] else read_text_exact(directory / filename)):
            match = response_pattern.match(record["comment"])
            if match:
                status, response_id = match.groups()
                responses.setdefault(response_id, []).append(status)
        for response_id in required_ids:
            expected_status = "Evidence open" if response_id == "R08" else "Done"
            if response_id not in responses:
                errors.append(f"{filename}: missing % Long Note: {expected_status} [{response_id}]")
            elif any(status != expected_status for status in responses[response_id]):
                errors.append(f"{filename}: {response_id} must be marked {expected_status!r}, never a conflicting status")
        for response_id in responses.keys() - set(required_ids):
            errors.append(f"{filename}: unknown follow-up response ID {response_id}")

    total_baseline = 0
    total_current = 0
    note_count = 0
    prefix = manifest["editorial_note_prefix"]
    for baseline in manifest["baseline_files"]:
        path = directory / ("main.tex" if single_file else baseline["path"])
        expected = counts(baseline["comments"])
        total_baseline += sum(expected.values())
        if not path.is_file():
            errors.append(f"Baseline source file is missing: {baseline['path']}")
            continue
        current = comment_records(read_text_exact(path))
        actual = counts(current)
        total_current += len(current)
        note_count += sum(record["comment"].startswith(prefix) for record in current)
        for comment, missing_count in (expected - actual).items():
            errors.append(f"{baseline['path']}: removed {missing_count} existing occurrence(s) of {comment!r}")
        if args.verify_git_baseline:
            git_path = f"{manifest['baseline_revision']}:{baseline['git_path']}"
            result = subprocess.run(
                ["git", "show", git_path], cwd=directory, capture_output=True, check=False
            )
            if result.returncode:
                errors.append(f"Cannot read pinned Git baseline: {git_path}")
            elif (
                hashlib.sha256(result.stdout).hexdigest() != baseline["sha256"]
                or comment_records(result.stdout.decode("utf-8")) != baseline["comments"]
            ):
                errors.append(f"Pinned Git baseline differs from manifest: {baseline['path']}")

    main_text = without_comments(current_main)
    if single_file:
        if re.search(r"\\(?:input|include|externaldocument)\s*\{", main_text):
            errors.append("Single-file manuscript must not require external TeX files")
        bibliography = manifest.get("external_bibliography")
        if bibliography:
            if not (directory / bibliography).is_file():
                errors.append(f"External bibliography missing: {bibliography}")
            if re.search(r"\\(?:bibitem|begin\{thebibliography\})", main_text):
                errors.append("References must live in the external .bib, not main.tex")
        elif re.search(r"\\bibliography\s*\{", main_text):
            errors.append("Undeclared external bibliography")
        for source_group in manifest.get("consolidation_comment_sets", []):
            expected = counts(source_group["comments"])
            for comment, missing in (expected - counts(comment_records(current_main))).items():
                errors.append(f"{source_group['name']}: missing {missing} occurrence(s): {comment!r}")
        total_current = len(comment_records(current_main))
        note_count = sum(r["comment"].startswith(prefix) for r in comment_records(current_main))
    else:
        archive_stem = re.escape(Path(manifest["archive_file"]).stem)
        archive_input = re.compile(r"\\input\s*\{\s*" + archive_stem + r"(?:\.tex)?\s*\}")
        if not archive_input.search(main_text):
            errors.append("main.tex must include the comment-only editorial archive with an active input command")
    if not note_count:
        errors.append("Current manuscript source needs at least one new % Long Note: editorial response outside the archive")

    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"PASS: all {len(expected_original)} original comment occurrences preserved verbatim in the comment-only archive")
    print(f"PASS: all {len(expected_followup)} follow-up comment occurrences covered verbatim in the archive")
    print("PASS: R01–R13 responses present in main and archive; R08 remains Evidence open and the other twelve are Done")
    print(f"PASS: all {total_baseline} baseline comments retained; layout: {manifest.get('layout', 'modular')}")
    print(f"PASS: archive is preserved; {note_count} Long Note response lines among {total_current} current source comments")
    print(f"PASS: {source_status}")
    print(f"PASS: {followup_status}")
    if args.verify_git_baseline:
        print(f"PASS: pinned Git baseline {manifest['baseline_revision']} verified")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--original-source", type=Path)
    parser.add_argument("--followup-source", type=Path)
    parser.add_argument("--verify-git-baseline", action="store_true", help="Also verify the manifest against its pinned Git revision")
    args = parser.parse_args()
    try:
        return check(args)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
