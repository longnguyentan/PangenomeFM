#!/usr/bin/env python3
"""Build machine-readable repository, dependency, checkpoint, and path audits."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _declared_dependencies(root: Path) -> set[str]:
    dependencies: list[str] = []
    requirements = root / "requirements.txt"
    if requirements.exists():
        dependencies.extend(requirements.read_text(encoding="utf-8").splitlines())
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = payload.get("project", {})
        dependencies.extend(project.get("dependencies", []))
        for values in project.get("optional-dependencies", {}).values():
            dependencies.extend(values)
    return {
        re.split(r"[<>=~!\s\[]", value.strip().lower(), maxsplit=1)[0]
        for value in dependencies
        if value.strip() and not value.lstrip().startswith("#")
    }


def _imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def build(root: Path, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts]
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    for path in files:
        relative = path.relative_to(root)
        key = "/".join(relative.parts[:2]) if len(relative.parts) > 1 else relative.parts[0]
        grouped[key]["files"] += 1
        grouped[key]["bytes"] += path.stat().st_size
    pd.DataFrame(
        [{"repository_group": key, **value} for key, value in sorted(grouped.items())]
    ).to_csv(out_dir / "repository_file_inventory.csv", index=False)

    checkpoint_rows = []
    for path in sorted((root / "results").rglob("*.pt")):
        relative = path.relative_to(root)
        checkpoint_rows.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "category": (
                    "ccre" if "ccre" in str(relative).lower() else "link_pretraining"
                ),
                "masked_query_training": "masked" in str(relative).lower()
                or "maskedq" in path.name,
            }
        )
    pd.DataFrame(checkpoint_rows).to_csv(out_dir / "checkpoint_inventory.csv", index=False)

    source_files = list((root / "src").rglob("*.py")) + list((root / "scripts").rglob("*.py"))
    module_to_files: dict[str, set[str]] = defaultdict(set)
    for path in source_files:
        for module in _imports(path):
            module_to_files[module].add(str(path.relative_to(root)))
    declared = _declared_dependencies(root)
    standard = set(getattr(sys, "stdlib_module_names", set()))
    distribution_aliases = {
        "sklearn": "scikit-learn",
        "umap": "umap-learn",
        "Bio": "biopython",
        "PIL": "pillow",
        "yaml": "pyyaml",
    }
    dependency_rows = []
    local_modules = {
        path.name if path.is_dir() else path.stem
        for path in (root / "src").iterdir()
        if path.is_dir() or path.suffix == ".py"
    }
    for module, consumers in sorted(module_to_files.items()):
        normalized = module.lower().replace("_", "-")
        declared_name = distribution_aliases.get(module, normalized).lower()
        local = module in local_modules
        installed = local or module in standard or importlib.util.find_spec(module) is not None
        dependency_rows.append(
            {
                "module": module,
                "local_or_stdlib": local or module in standard,
                "importable_current_environment": installed,
                "mentioned_in_requirements_or_pyproject": any(
                    declared_name == item or declared_name in item or item in declared_name
                    for item in declared
                ),
                "consumer_count": len(consumers),
                "example_consumer": sorted(consumers)[0],
            }
        )
    pd.DataFrame(dependency_rows).to_csv(out_dir / "dependency_audit.csv", index=False)

    hardcoded_rows = []
    incomplete_rows = []
    absolute_pattern = re.compile(r"/(?:Users|home|mnt|scratch|gpfs)/[^\"'\s)]+")
    incomplete_pattern = re.compile(r"\b(?:TODO|FIXME|NotImplementedError)\b|^\s*pass\s*(?:#.*)?$", re.MULTILINE)
    for path in source_files + list((root / "configs").rglob("*.json")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in absolute_pattern.finditer(line):
                hardcoded_rows.append(
                    {
                        "path": str(path.relative_to(root)),
                        "line": line_number,
                        "absolute_path": match.group(0),
                    }
                )
            if incomplete_pattern.search(line):
                incomplete_rows.append(
                    {
                        "path": str(path.relative_to(root)),
                        "line": line_number,
                        "text": line.strip(),
                    }
                )
    pd.DataFrame(hardcoded_rows, columns=["path", "line", "absolute_path"]).to_csv(
        out_dir / "hardcoded_path_audit.csv", index=False
    )
    pd.DataFrame(incomplete_rows, columns=["path", "line", "text"]).to_csv(
        out_dir / "potential_incomplete_code.csv", index=False
    )

    summaries = []
    for path in sorted((root / "results").rglob("summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            parse_status = "valid_json"
            top_keys = ";".join(sorted(payload)[:20]) if isinstance(payload, dict) else type(payload).__name__
        except json.JSONDecodeError:
            parse_status = "invalid_json"
            top_keys = ""
        summaries.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "parse_status": parse_status,
                "top_level_keys": top_keys,
            }
        )
    pd.DataFrame(summaries).to_csv(out_dir / "result_summary_inventory.csv", index=False)

    summary = {
        "files_excluding_git": len(files),
        "bytes_excluding_git": sum(path.stat().st_size for path in files),
        "python_source_files": len(source_files),
        "checkpoints": len(checkpoint_rows),
        "checkpoint_bytes": sum(row["bytes"] for row in checkpoint_rows),
        "result_summary_json_files": len(summaries),
        "hardcoded_absolute_paths": len(hardcoded_rows),
        "potential_incomplete_markers": len(incomplete_rows),
        "dependency_modules": len(dependency_rows),
        "dependency_modules_not_importable": [
            row["module"] for row in dependency_rows if not row["importable_current_environment"]
        ],
    }
    (out_dir / "repository_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.out_dir), indent=2))


if __name__ == "__main__":
    main()
