#!/usr/bin/env python3
"""Validate the pinned data profiles and count every configured matrix shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from download_full_data import load_manifest, select_profile


REPO_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/server_full_multicohort_20260806.json")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "configs/server_full_data_manifest.tsv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resources = load_manifest(args.manifest)
    profile_rows = {}
    for profile in ["sv-core", "analysis-full", "archive-full-resolution"]:
        selected = select_profile(resources, profile)
        profile_rows[profile] = {
            "resources": len(selected),
            "known_bytes": sum(resource.expected_bytes or 0 for resource in selected),
            "unknown_size_resources": [resource.resource_id for resource in selected if resource.expected_bytes is None],
            "datasets": sorted({resource.dataset for resource in selected}),
        }
    seeds = len(config["training"]["seeds"])
    contexts = len(config["training"]["contexts"])
    matrix = config["experiment_matrix"]
    counts = {
        "final": len(matrix["final_regimes"]) * seeds * contexts,
        "folds": len(matrix["fold_regimes"]) * len(config["rotating_chromosome_folds"]) * seeds * contexts,
        "transfer": len(matrix["cohort_transfer_pairs"]) * seeds * contexts,
        "release": len(matrix["release_transfer_pairs"]) * seeds * contexts,
    }
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "profiles": profile_rows,
        "datasets": list(config["datasets"]),
        "chromosomes": config["primary_chromosomes"],
        "chromosome_count": len(config["primary_chromosomes"]),
        "matrix_jobs": {**counts, "total": sum(counts.values())},
        "seeds": config["training"]["seeds"],
        "contexts": config["training"]["contexts"],
        "release_3_policy": config["release_3_policy"],
        "status": "validated_plan",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
