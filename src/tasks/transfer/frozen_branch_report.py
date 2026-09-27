"""Audit all four frozen-branch combinations and both single-branch references."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import pandas as pd

from tasks.entex.prepare import fingerprint
from tasks.transfer.development_report import FEATURES, validate_development_pairs
from tasks.transfer.hr_control_report import audited_run
from tasks.transfer.report import save_figure

COMPOSITES = ["T_Q", "T_Rq", "Rt_Q", "Rt_Rq"]
MODELS = [*COMPOSITES, "T", "Q"]


def compare(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mapping = dict(zip(COMPOSITES, ["v2", "v1", "v2_random", "random"]))
    validate_development_pairs(frame.loc[frame.model.isin(COMPOSITES)].assign(
        model=lambda x: x.model.map(mapping)))
    expected = set(product(["sv", "ccre"], MODELS, FEATURES))
    if frame.duplicated(["task", "model", "feature"]).any() or set(
            frame[["task", "model", "feature"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete six-model comparison")
    if not frame.n_test.eq(0).all() or not frame.evaluation_partition.eq("development_validation").all():
        raise ValueError("Unexpected test evidence")
    if any(frame[k].nunique() != 1 for k in ["fold", "seed", "context"]):
        raise ValueError("Different partitions")
    rows, checks = [], []
    for task, group in frame.groupby("task"):
        for key in ["targets_sha256", "n_train", "n_val", "n_evaluated", "positive_prevalence"]:
            if group[key].nunique() != 1:
                raise ValueError("Single-branch reference universe differs")
        for base in ["cs", "csh"]:
            if group.loc[group.feature.eq(base), "scores_sha256"].nunique() != 1:
                raise ValueError("Baseline predictions differ")
        for feature, metric in product(["cst", "csht"], ["auprc", "auroc", "normalized_ap", "balanced_accuracy", "f1"]):
            values = group.loc[group.feature.eq(feature)].set_index("model")[metric]
            for control in MODELS[1:]:
                rows.append(dict(task=task, feature=feature, metric=metric, comparison="T_Q minus " + control,
                    candidate=float(values["T_Q"]), control=float(values[control]),
                    difference=float(values["T_Q"] - values[control]), scope="development validation; no CI"))
        ap = group.set_index(["model", "feature"]).auprc
        control_gains = {m: float(ap["T_Q", "csht"] - ap[m, "csht"]) for m in COMPOSITES[1:]}
        tolerances = {f: float(ap["T_Q", f] - max(ap["T", f], ap["Q", f])) for f in ["cst", "csht"]}
        checks.append(dict(task=task, after_H_control_gains=control_gains, difference_from_better_single_branch=tolerances,
            passes=all(v > 0 for v in control_gains.values()) and all(v >= -.005 for v in tolerances.values())))
    gate = dict(status="eligible_for_replication" if all(c["passes"] for c in checks) else "not_promoted",
                checks=checks, tolerance=.005, single_branch_tolerance_applied_to_both_feature_sets=True,
                limitation="Adaptively motivated one-fold development; no confirmatory superiority claim",
                strict_context_eligible=False, test_predictions_produced=False)
    return pd.DataFrame(rows), gate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--topology-root", type=Path, required=True)
    ap.add_argument("--sequence-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    roots = [args.root, args.topology_root, args.sequence_root]
    receipts = [json.loads((r / "status.json").read_text()) for r in roots]
    for receipt in receipts:
        if (receipt["status"] != "complete" or receipt["completed_commands"] != len(receipt["commands"])
                or not receipt["encoders_frozen"] or receipt["evaluation_partition"] != "development_validation"):
            raise ValueError("Incomplete or incorrectly scoped source")
        for key in ["fold", "seed", "graph_sha256", "extraction_candidate_policy"]:
            if receipt[key] != receipts[0][key]:
                raise ValueError("Source protocol differs")
    receipt = receipts[0]
    records, branch_evidence = [], []
    for model, task in product(MODELS, ["sv", "ccre"]):
        root, name = ((args.topology_root, "v2") if model == "T" else
                      (args.sequence_root, "nt") if model == "Q" else (args.root, model))
        path = root / "probes" / name / task / "1hop"
        records.append(audited_run(path, model, task, receipt["fold"]["name"],
                                   receipt["seed"], "1hop", validation_only=True))
        if model in COMPOSITES:
            audit = json.loads((path / "audit.json").read_text())["canonical_candidate_audit"]
            if not audit.get("exact_branch_preservation") or [b["dimension"] for b in audit["branches"]] != [48, 48]:
                raise ValueError("Branches not preserved at the prespecified dimensions")
            hashes = [b["checkpoint"]["sha256"] for b in audit["branches"]]
            refs = [args.topology_root / "probes" / ("v2_random" if model.startswith("Rt") else "v2") / task / "1hop/audit.json",
                    args.sequence_root / "probes" / ("nt_random" if model.endswith("Rq") else "nt") / task / "1hop/audit.json"]
            if hashes != [json.loads(p.read_text())["checkpoint_sha256"] for p in refs]:
                raise ValueError("Composite used unexpected branch checkpoints")
            branch_evidence.append(dict(model=model, task=task, checkpoint_hashes=hashes))
    frame = pd.concat(records, ignore_index=True)
    frame["embedding"] = frame.model
    differences, gate = compare(frame)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.out_dir / "audited_per_run.csv", index=False)
    differences.to_csv(args.out_dir / "paired_differences.csv", index=False)
    (args.out_dir / "development_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    (args.out_dir / "audit.json").write_text(json.dumps(dict(status="complete",
        sources=[fingerprint(r / "status.json") for r in roots], branch_evidence=branch_evidence,
        arithmetic="metrics replayed from actual identical validation examples",
        dimensions="48+48 in every composite; each single-branch reference is 48",
        scope="one development fold/seed; encoders frozen; no test predictions", n_runs=12), indent=2) + "\n")
    # All inherited SV bins remain available in the original per-run outputs.
    strata = []
    for model in MODELS:
        root, name = ((args.topology_root, "v2") if model == "T" else
                      (args.sequence_root, "nt") if model == "Q" else (args.root, model))
        strata.append(pd.read_csv(root / "probes" / name / "sv/1hop/stratified_metrics.csv").assign(model=model))
    pd.concat(strata, ignore_index=True).to_csv(args.out_dir / "all_sv_strata.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    labels = ["T+Q", "T+random Q", "random T+Q", "both random", "T only", "Q only"]
    for ax, task in zip(axes, ["sv", "ccre"]):
        for feature, marker, label in [("cst", "o", "C+S+E"), ("csht", "s", "C+S+H+E")]:
            values = frame.loc[frame.task.eq(task) & frame.feature.eq(feature)].set_index("model").loc[MODELS, "auprc"]
            ax.plot(range(6), values, marker=marker, label=label)
        ax.set(xticks=range(6), xticklabels=labels, ylim=(0, 1), ylabel="Validation AUPRC",
               title="SV insertion/deletion" if task == "sv" else "cCRE")
        ax.tick_params(axis="x", rotation=35)
    axes[0].legend(frameon=False)
    fig.suptitle("Frozen branch preservation: one development fold/seed")
    save_figure(fig, args.out_dir, "frozen_branches")
    plt.close(fig)


if __name__ == "__main__":
    main()
