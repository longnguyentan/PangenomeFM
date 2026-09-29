#!/usr/bin/env python3
"""Check external citation coverage and reviewed bibliography integrity."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from check_editorial_comments import without_comments


def entries(text: str) -> dict[str, str]:
    """Read this project's braced BibTeX entries, retaining nested TeX braces."""
    result = {}
    for match in re.finditer(r"(?m)^@(\w+)\s*\{([^,]+),", text):
        key = match[2].strip()
        if key in result:
            raise ValueError(f"Duplicate bibliography key: {key}")
        depth = 1
        end = match.end()
        while end < len(text) and depth:
            if text[end] in "{}" and (end == 0 or text[end - 1] != "\\"):
                depth += 1 if text[end] == "{" else -1
            end += 1
        if depth:
            raise ValueError(f"Unbalanced bibliography entry: {key}")
        result[key] = text[match.end():end - 1].strip()
    return result


def check(directory: Path) -> dict:
    raw_bib = (directory / 'references.bib').read_bytes()
    bibliography = entries(raw_bib.decode())
    source = without_comments((directory / 'main.tex').read_text()).split(
        r'\end{document}', 1)[0]
    if re.search(r'\\(?:bibitem|begin\{thebibliography\})', source):
        raise ValueError('Bibliography entries must remain outside main.tex')
    if source.count(r'\putbib[references]') != 2:
        raise ValueError('Expected two external, independently numbered reference lists')
    keys = set()
    for match in re.finditer(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}', source):
        keys.update(key.strip() for key in match[1].split(','))
    missing = keys - bibliography.keys()
    if missing:
        raise ValueError(f'Unresolved citations: {sorted(missing)}')
    unused = bibliography.keys() - keys
    if unused:
        raise ValueError(f'Unused bibliography entries: {sorted(unused)}')
    aliases = []
    for key, body in bibliography.items():
        base = key.removeprefix('supp__')
        if key != base and base in bibliography:
            if body != bibliography[base]:
                raise ValueError(f'Main/Supplementary metadata differ: {key}')
            aliases.append(key)
    audit = json.loads((directory / 'bibliography_audit_20260929.json').read_text())
    canonical = keys - set(aliases)
    reviewed = [row['key'] for row in audit['works']]
    if Counter(reviewed) != Counter(canonical):
        raise ValueError('Review manifest does not cover each distinct work exactly once')
    if audit['bibliography_sha256'] != hashlib.sha256(raw_bib).hexdigest():
        raise ValueError('Bibliography changed since the recorded metadata review')
    if not all(row.get('source') and row.get('review') for row in audit['works']):
        raise ValueError('Every work needs an explicit source and review disposition')
    return {'citation_keys': len(keys), 'distinct_works': len(canonical),
            'intentional_supplementary_aliases': len(aliases),
            'missing_citations': 0, 'inline_entries': 0,
            'scope': 'Citation integrity and pinned metadata review; not automatic claim validation'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(check(args.manuscript_dir.resolve()), indent=2))
