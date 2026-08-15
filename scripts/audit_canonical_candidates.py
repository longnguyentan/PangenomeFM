#!/usr/bin/env python3
"""Audit exact and reverse-complement candidate identity across a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from graph.neg_sampling import canonical_oriented_pair


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: object, manifest: Path) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    candidate = manifest.parent / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"cannot resolve {value!r} from {manifest}")


def audit_manifest(manifest_path: Path) -> tuple[dict[str, object], pd.DataFrame]:
    manifest = pd.read_csv(manifest_path)
    if "edge_pred_path" not in manifest:
        raise ValueError("manifest misses edge_pred_path")
    details: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        path = resolve_path(row.edge_pred_path, manifest_path)
        candidates = pd.read_csv(path, compression="infer")
        missing = {"u_oid", "v_oid", "label"} - set(candidates)
        if missing:
            raise ValueError(f"{path} misses columns: {sorted(missing)}")
        labels = pd.to_numeric(candidates["label"], errors="coerce")
        canonical = [
            canonical_oriented_pair(u_oid, v_oid)
            for u_oid, v_oid in zip(candidates["u_oid"], candidates["v_oid"])
        ]
        work = pd.DataFrame(
            {
                "canonical_u": [pair[0] for pair in canonical],
                "canonical_v": [pair[1] for pair in canonical],
                "label": labels,
            }
        )
        keys = ["canonical_u", "canonical_v"]
        label_counts = work.groupby(keys, sort=False)["label"].nunique(dropna=True)
        conflicts = int((label_counts > 1).sum())
        details.append(
            {
                "slice_id": str(
                    getattr(row, "slice_id", None) or getattr(row, "name", path.stem)
                ),
                "candidate_path": str(path.resolve()),
                "candidate_rows": int(len(work)),
                "positive_rows": int((labels == 1).sum()),
                "negative_rows": int((labels == 0).sum()),
                "invalid_label_rows": int((~labels.isin([0, 1])).sum()),
                "exact_duplicate_rows": int(
                    candidates.duplicated(["u_oid", "v_oid"]).sum()
                ),
                "reverse_equivalent_duplicate_rows": int(work.duplicated(keys).sum()),
                "orientation_equivalent_label_conflicts": conflicts,
            }
        )
    detail_frame = pd.DataFrame(details)
    failure_columns = [
        "invalid_label_rows",
        "exact_duplicate_rows",
        "reverse_equivalent_duplicate_rows",
        "orientation_equivalent_label_conflicts",
    ]
    totals = {
        column: int(detail_frame[column].sum()) for column in failure_columns
    }
    audit = {
        "schema_version": 1,
        "status": "PASS" if not any(totals.values()) else "FAIL",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256sum(manifest_path),
        "slice_files": int(len(detail_frame)),
        "candidate_rows": int(detail_frame["candidate_rows"].sum()),
        "positive_rows": int(detail_frame["positive_rows"].sum()),
        "negative_rows": int(detail_frame["negative_rows"].sum()),
        "candidate_identity_policy": "(u,v) == (v^1,u^1)",
        "failure_totals": totals,
        "failing_slices": int(
            detail_frame[failure_columns].ne(0).any(axis=1).sum()
        ),
    }
    return audit, detail_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    audit_path = args.out_dir / "canonical_candidate_audit.json"
    if audit_path.exists() and not args.overwrite:
        raise FileExistsError(audit_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit, details = audit_manifest(args.manifest)
    details.to_csv(
        args.out_dir / "canonical_candidate_slice_audit.csv.gz",
        index=False,
        compression="gzip",
    )
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
