#!/usr/bin/env python3
"""Audit the experiment gates required before rewriting the manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.modality_factorial import load_frozen_node_embedding_cache


def matrix_gate(name: str, root: Path, aggregate: Path) -> dict[str, object]:
    matrix_path = root / "matrix_summary.json"
    if not matrix_path.exists():
        return {"name": name, "status": "fail", "reason": "matrix_summary.json missing"}
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    audits = list(root.glob("fold_*/seed_*/*/audit.json"))
    complete = sum(
        json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
        for path in audits
    )
    aggregate_audit = aggregate / "audit.json"
    contribution_paths = sorted(aggregate.glob("*_modality_contributions.csv"))
    contribution_rows = (
        sum(len(pd.read_csv(path)) for path in contribution_paths)
        if contribution_paths
        else 0
    )
    passed = all(
        [
            matrix.get("jobs_requested") == 30,
            matrix.get("jobs_recorded") == 30,
            matrix.get("failures") == 0,
            complete == 30,
            aggregate_audit.exists(),
            contribution_rows > 0,
        ]
    )
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "jobs_requested": matrix.get("jobs_requested"),
        "jobs_recorded": matrix.get("jobs_recorded"),
        "failures": matrix.get("failures"),
        "complete_fold_audits": complete,
        "aggregate_audit": str(aggregate_audit),
        "modality_contribution_rows": contribution_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-sensitivity-audit", type=Path, required=True)
    parser.add_argument("--ccre-root", type=Path, required=True)
    parser.add_argument("--ccre-aggregate", type=Path, required=True)
    parser.add_argument("--sv-root", type=Path, required=True)
    parser.add_argument("--sv-aggregate", type=Path, required=True)
    parser.add_argument("--sequence-cache", type=Path)
    parser.add_argument("--path-branch-audit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    gates: list[dict[str, object]] = []
    canonical = json.loads(
        args.canonical_sensitivity_audit.read_text(encoding="utf-8")
    )
    gates.append(
        {
            "name": "canonical_affected_slice_sensitivity",
            "status": (
                "pass"
                if canonical.get("status") == "complete"
                and canonical.get("sensitivity_gate") == "pass"
                else "fail"
            ),
            "affected_loci": canonical.get(
                "canonical_identity_affected_genomic_loci"
            ),
            "failed_primary_gate_rows": canonical.get("failed_primary_gate_rows"),
        }
    )
    gates.append(matrix_gate("ccre_modality_factorial", args.ccre_root, args.ccre_aggregate))
    gates.append(matrix_gate("sv_modality_factorial", args.sv_root, args.sv_aggregate))
    if args.sequence_cache is not None:
        segids, embeddings, audit = load_frozen_node_embedding_cache(args.sequence_cache)
        gates.append(
            {
                "name": "frozen_sequence_model_cache",
                "status": (
                    "pass"
                    if audit.get("coverage_fraction") == 1.0
                    and len(segids) == len(embeddings)
                    else "fail"
                ),
                "nodes": int(len(segids)),
                "dimension": int(embeddings.shape[1]),
                "model": audit.get("model_name"),
                "revision": audit.get("resolved_revision"),
            }
        )
    if args.path_branch_audit is not None:
        path_audit = json.loads(args.path_branch_audit.read_text(encoding="utf-8"))
        gates.append(
            {
                "name": "path_branch_choice_candidates",
                "status": (
                    "pass"
                    if path_audit.get("status") == "complete"
                    and path_audit.get("candidate_groups", 0) > 0
                    and path_audit.get("observed_edges_absent_from_gfa") == 0
                    else "fail"
                ),
                "candidate_groups": path_audit.get("candidate_groups"),
                "negative_rows": path_audit.get("negative_rows"),
            }
        )
    overall = "pass" if all(gate["status"] == "pass" for gate in gates) else "fail"
    payload = {"schema_version": 1, "status": overall, "gates": gates}
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
