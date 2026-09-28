"""Supplemental paired chromosome comparison with converged original-v1 probes."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.paired_inference import bh_adjust, fold_sign_flip
from tasks.entex.prepare import fingerprint
from tasks.transfer.hr_control_report import audited_run
from tasks.transfer.masked_junction_reference import compare as paired_compare
from tasks.transfer.masked_replication import chromosome_estimate, validate_matrix, verify_saved_probes
from tasks.transfer.traitgym import write_json


def load_candidate_replication(root: Path) -> tuple[pd.DataFrame, dict]:
    """Read only the candidate table whose exact bytes the completed audit binds.

    Use the report-local file so an intact report remains portable. Hash and
    parse the same bytes rather than reopening a file after its checksum check.
    Older unbound reports must be regenerated from their completed raw runs.
    """
    audit_path = root / "audit.json"
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    if (
        audit.get("status") != "complete"
        or audit.get("all_saved_predictions_replayed") is not True
        or audit.get("all_probes_converged") is not True
    ):
        raise ValueError("Candidate replication is incomplete")
    expected = audit.get("audited_per_run")
    if not isinstance(expected, dict) or not expected.get("sha256"):
        raise ValueError(
            "Candidate table fingerprint is missing; regenerate the replication "
            "report from completed source runs"
        )
    table_path = root / "audited_per_run.csv"
    table_bytes = table_path.read_bytes()
    table_sha256 = hashlib.sha256(table_bytes).hexdigest()
    if table_sha256 != expected["sha256"] or len(table_bytes) != expected.get("bytes"):
        raise ValueError("Candidate table differs from its completed replication audit")
    frame = pd.read_csv(BytesIO(table_bytes), float_precision="round_trip")
    sources = {
        "candidate_audit": dict(path=str(audit_path.resolve()), bytes=len(audit_bytes),
            sha256=hashlib.sha256(audit_bytes).hexdigest()),
        "candidate_table": dict(path=str(table_path.resolve()), bytes=len(table_bytes),
            sha256=table_sha256),
    }
    return frame, sources


def compare_reference(
    candidate: pd.DataFrame, reference: pd.DataFrame, plan: dict, replication: dict
):
    validate_matrix(candidate, replication)
    if not np.isfinite(reference[plan["metrics"]].to_numpy()).all():
        raise ValueError("Nonfinite original-v1 reference metrics")
    keys = ["fold", "seed", "task", "feature"]
    expected = set(
        product(
            plan["folds"], plan["seeds"], plan["tasks"], ["cs", "csh", "cst", "csht"]
        )
    )
    if (
        reference.duplicated(keys).any()
        or set(reference[keys].itertuples(index=False, name=None)) != expected
        or not reference.model.eq("v1").all()
    ):
        raise ValueError("Missing/duplicate or non-v1 reference evaluation")
    pieces = []
    for fold in plan["folds"]:
        part = paired_compare(
            candidate.loc[candidate.fold.eq(fold)],
            reference.loc[reference.fold.eq(fold)],
            dict(plan, reference_model="v1"),
        ).assign(fold=fold)
        pieces.append(part.rename(columns={"junction_q_value": "original_v1_value"}))
    paired = pd.concat(pieces, ignore_index=True)
    rows = []
    scopes = dict(
        development_excluded_four_folds=replication["primary_summary_folds"],
        all_five_development_exposed=plan["folds"],
        development_exposed_fold_only=[replication["development_exposed_test_fold"]],
    )
    for scope, folds in scopes.items():
        for key, group in paired.loc[paired.fold.isin(folds)].groupby(
            ["task", "feature", "metric", "model"]
        ):
            group = group.rename(columns={"difference": "gain"})
            rows.append(
                dict(
                    zip(["task", "feature", "metric", "model"], key),
                    scope=scope,
                    **chromosome_estimate(group, "gain", replication),
                    sign_flip_p=fold_sign_flip(group),
                )
            )
    summary = pd.DataFrame(rows)
    summary["bh_q_within_scope_metric"] = summary.groupby(
        ["scope", "metric"]
    ).sign_flip_p.transform(bh_adjust)
    return paired, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.config.read_text())
    rep = json.loads(Path(plan["replication_protocol"]).read_text())
    candidate_root = Path(plan["replication_root"] + "_analysis")
    candidate, candidate_sources = load_candidate_replication(candidate_root)
    records, artifacts = [], []
    for fold, seed in product(plan["folds"], plan["seeds"]):
        root = args.reference_root / fold / f"seed_{seed}"
        status = json.loads((root / "status.json").read_text())
        if (
            status["status"] != "complete"
            or status["models"] != ["v1"]
            or status["completed_commands"] != len(status["commands"])
            or status["probe_max_iter_override"] != plan["probe_max_iter"]
            or not status["encoders_frozen"]
            or status["evaluation_partition"] != "test"
        ):
            raise ValueError("Reference execution/protocol mismatch")
        for task in plan["tasks"]:
            output = root / "probes/v1" / task / plan["context"]
            m = pd.read_csv(output / "metrics.csv")
            if (
                not m.probe_converged.all()
                or not m.probe_max_iter.eq(plan["probe_max_iter"]).all()
            ):
                raise ValueError("Reference convergence incomplete")
            records.append(audited_run(output, "v1", task, fold, seed, plan["context"]))
            artifacts.extend(
                dict(fold=fold, seed=seed, task=task, **r)
                for r in verify_saved_probes(output, m)
            )
    reference = pd.concat(records, ignore_index=True)
    paired, summary = compare_reference(candidate, reference, plan, rep)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in [
        ("reference_per_run", reference),
        ("paired_per_run", paired),
        ("contrasts", summary),
        ("fitted_probe_artifacts", pd.DataFrame(artifacts)),
    ]:
        frame.to_csv(args.out_dir / (name + ".csv"), index=False)
    write_json(
        args.out_dir / "audit.json",
        dict(
            status="complete",
            plan=plan,
            reference_probe_runs=len(records),
            all_metrics_replayed=True,
            all_probes_converged=True,
            all_baselines_bitwise_identical=True,
            n_verified_saved_probes=len(artifacts),
            **candidate_sources,
            implementation=fingerprint(Path(__file__)),
            scope="Supplemental fixed old/new pipeline comparison; no architecture promotion",
        ),
    )


if __name__ == "__main__":
    main()
