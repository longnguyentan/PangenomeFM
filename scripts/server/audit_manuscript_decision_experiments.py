#!/usr/bin/env python3
"""Audit the experiment gates required before rewriting the manuscript."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from evaluation.modality_factorial import load_frozen_node_embedding_cache
from scripts.server.aggregate_ccre_frozen_probes import (
    PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST,
)


def matrix_gate(
    name: str,
    root: Path,
    aggregate: Path,
    *,
    expected_bootstrap: int,
) -> dict[str, object]:
    """Audit a complete matrix and the named manuscript-primary contrast."""

    matrix_path = root / "matrix_summary.json"
    if not matrix_path.exists():
        return {"name": name, "status": "fail", "reason": "matrix_summary.json missing"}
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    audits = list(root.glob("fold_*/seed_*/*/audit.json"))
    complete = sum(
        json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
        for path in audits
    )
    aggregate_audit_path = aggregate / "audit.json"
    aggregate_audit = (
        json.loads(aggregate_audit_path.read_text(encoding="utf-8"))
        if aggregate_audit_path.exists()
        else {}
    )
    contribution_paths = sorted(aggregate.glob("*_modality_contributions.csv"))
    contributions = (
        pd.concat([pd.read_csv(path) for path in contribution_paths], ignore_index=True)
        if contribution_paths
        else pd.DataFrame()
    )
    contribution_rows = len(contributions)
    primary = (
        contributions.loc[
            contributions.get("contrast", pd.Series(dtype=str)).eq(
                PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST
            )
            & contributions.get("metric", pd.Series(dtype=str)).eq("auprc")
        ].copy()
        if not contributions.empty
        else pd.DataFrame()
    )
    primary_contexts = set(primary.get("closure", pd.Series(dtype=str)))
    primary_complete = bool(
        len(primary) == 2
        and primary_contexts == {"strict", "1hop"}
        and primary["mean_gain"].notna().all()
        and primary["ci95_low"].notna().all()
        and primary["ci95_high"].notna().all()
        and primary["n_paired_runs"].eq(15).all()
        and primary["n_folds"].eq(5).all()
        and primary["n_seeds"].eq(3).all()
    )
    context_paths = sorted(aggregate.glob("*_context_contributions.csv"))
    contexts = (
        pd.concat([pd.read_csv(path) for path in context_paths], ignore_index=True)
        if context_paths
        else pd.DataFrame()
    )
    interaction_name = (
        f"one_hop_minus_strict__{PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST}"
    )
    primary_interaction = (
        contexts.loc[
            contexts.get("contrast", pd.Series(dtype=str)).eq(interaction_name)
            & contexts.get("metric", pd.Series(dtype=str)).eq("auprc")
        ]
        if not contexts.empty
        else pd.DataFrame()
    )
    interaction_complete = bool(
        len(primary_interaction) == 1
        and primary_interaction["n_paired_runs"].eq(15).all()
        and primary_interaction["n_folds"].eq(5).all()
        and primary_interaction["n_seeds"].eq(3).all()
    )
    fold_metric_paths = [
        path
        for path in sorted(aggregate.glob("*_fold_metrics.csv"))
        if "stratified" not in path.name
    ]
    fold_metrics = (
        pd.concat([pd.read_csv(path) for path in fold_metric_paths], ignore_index=True)
        if fold_metric_paths
        else pd.DataFrame()
    )
    exact_count_audit = False
    if not fold_metrics.empty:
        count_columns = ["n_train", "n_validation", "n_test"]
        exact_count_audit = all(column in fold_metrics for column in count_columns)
        if exact_count_audit:
            distinct = fold_metrics.groupby(["fold", "seed", "closure"])[
                count_columns
            ].nunique()
            exact_count_audit = bool(distinct.le(1).all().all())
    passed = all(
        [
            matrix.get("jobs_requested") == 30,
            matrix.get("jobs_recorded") == 30,
            matrix.get("failures") == 0,
            complete == 30,
            aggregate_audit.get("status") == "complete",
            aggregate_audit.get("files") == 30,
            aggregate_audit.get("n_bootstrap") == expected_bootstrap,
            aggregate_audit.get("exact_feature_universe", {}).get("status") == "pass",
            contribution_rows > 0,
            primary_complete,
            interaction_complete,
            exact_count_audit,
        ]
    )
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "jobs_requested": matrix.get("jobs_requested"),
        "jobs_recorded": matrix.get("jobs_recorded"),
        "failures": matrix.get("failures"),
        "complete_fold_audits": complete,
        "aggregate_audit": str(aggregate_audit_path),
        "aggregate_status": aggregate_audit.get("status"),
        "bootstrap_replicates": aggregate_audit.get("n_bootstrap"),
        "modality_contribution_rows": contribution_rows,
        "required_primary_contrast": PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST,
        "primary_auprc_rows": int(len(primary)),
        "primary_contexts": sorted(primary_contexts),
        "primary_paired_runs": (
            sorted(int(value) for value in primary["n_paired_runs"].unique())
            if not primary.empty
            else []
        ),
        "primary_context_interaction_rows": int(len(primary_interaction)),
        "exact_feature_universe": exact_count_audit,
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
    parser.add_argument("--expected-bootstrap", type=int, default=10_000)
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
    gates.append(
        matrix_gate(
            "ccre_modality_factorial",
            args.ccre_root,
            args.ccre_aggregate,
            expected_bootstrap=args.expected_bootstrap,
        )
    )
    gates.append(
        matrix_gate(
            "sv_modality_factorial",
            args.sv_root,
            args.sv_aggregate,
            expected_bootstrap=args.expected_bootstrap,
        )
    )
    if args.sequence_cache is not None:
        segids, embeddings, audit = load_frozen_node_embedding_cache(args.sequence_cache)
        revision = str(audit.get("resolved_revision", ""))
        cache_passed = all(
            [
                audit.get("status") == "complete",
                audit.get("coverage_fraction") == 1.0,
                len(segids) == len(embeddings),
                audit.get("downstream_label_access") == "none",
                audit.get("model_parameters_frozen") is True,
                audit.get("fine_tuned") is False,
                bool(re.fullmatch(r"[0-9a-f]{40}", revision)),
            ]
        )
        gates.append(
            {
                "name": "frozen_sequence_model_cache",
                "status": "pass" if cache_passed else "fail",
                "nodes": int(len(segids)),
                "dimension": int(embeddings.shape[1]),
                "model": audit.get("model_name"),
                "revision": revision,
                "coverage_fraction": audit.get("coverage_fraction"),
                "downstream_label_access": audit.get("downstream_label_access"),
                "model_parameters_frozen": audit.get("model_parameters_frozen"),
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
