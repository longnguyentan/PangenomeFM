"""Summarize all three fixed development seeds without treating seeds as folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tasks.entex.prepare import fingerprint
from tasks.transfer.frozen_branch_report import MODELS, compare
from tasks.transfer.report import save_figure

SEEDS = {42, 314159, 20260806}
METRICS = ["auprc", "auroc", "normalized_ap", "balanced_accuracy", "f1"]


def replay_solver_gate(performance: dict, saved: dict, optimizer: pd.DataFrame | None) -> dict:
    """Retain an unsuccessful numerical gate instead of mistaking it for bad arithmetic."""
    gate = dict(performance)
    if optimizer is None:
        if saved.get("probe_optimization_fully_recorded") or saved.get("all_probes_converged") is not None:
            raise ValueError("Missing optimizer evidence for the saved gate")
        gate.update(probe_optimization_fully_recorded=False, all_probes_converged=None,
                    probe_max_iter=None)
    else:
        if (len(optimizer) != 48 or optimizer.duplicated(["model", "task", "feature_set"]).any()
                or optimizer.groupby(["model", "task"]).size().ne(4).any()
                or set(optimizer.model) != set(MODELS) or set(optimizer.task) != {"sv", "ccre"}
                or optimizer.probe_max_iter.nunique() != 1 or optimizer.probe_solver.nunique() != 1
                or not optimizer.probe_converged.isin([True, False]).all()):
            raise ValueError("Incomplete or inconsistent optimizer evidence")
        converged = bool(optimizer.probe_converged.eq(True).all())
        if saved.get("all_probes_converged") is not converged or not saved.get("probe_optimization_fully_recorded"):
            raise ValueError("Saved numerical gate disagrees with optimizer evidence")
        gate.update(probe_optimization_fully_recorded=True, all_probes_converged=converged,
                    probe_max_iter=int(optimizer.probe_max_iter.iloc[0]))
        if not converged:
            gate.update(performance_gate_before_solver_check=gate["status"], status="optimization_incomplete")
    if gate["status"] != saved["status"]:
        raise ValueError("Stored gate differs from recomputed metrics/optimizer")
    return gate


def summarize(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if len(frames) != 3 or any(f.seed.nunique() != 1 for f in frames):
        raise ValueError("Require exactly one complete source per prespecified seed")
    if {int(f.seed.iloc[0]) for f in frames} != SEEDS:
        raise ValueError("Missing, duplicate or unexpected replication seed")
    all_runs, contrasts, gates = [], [], []
    for frame in frames:
        if not np.isfinite(frame[METRICS].to_numpy()).all():
            raise ValueError("Nonfinite primary run metrics")
        differences, gate = compare(frame)
        seed = int(frame.seed.iloc[0])
        all_runs.append(frame)
        contrasts.append(differences.assign(seed=seed))
        gates.append(dict(seed=seed, **gate))
    runs = pd.concat(all_runs, ignore_index=True)
    if not runs.fold.eq("fold_a").all() or not runs.context.eq("1hop").all():
        raise ValueError("Replication changed the prespecified fold or context")
    for _, group in runs.groupby("task"):
        for key in ["targets_sha256", "n_train", "n_val", "n_evaluated", "positive_prevalence"]:
            if group[key].nunique() != 1:
                raise ValueError(f"Example universe changed across seeds: {key}")
    for _, group in runs.groupby(["task", "model", "feature"]):
        if group.checkpoint_sha256.nunique() != 3:
            raise ValueError("A branch checkpoint was reused across initialization seeds")
    per_seed = pd.concat(contrasts, ignore_index=True)
    summary = per_seed.groupby(["task", "feature", "metric", "comparison"], sort=True).agg(
        mean=("difference", "mean"), std=("difference", "std"), minimum=("difference", "min"),
        maximum=("difference", "max"), n_seeds=("seed", "nunique"),
        n_positive=("difference", lambda x: int((x > 0).sum())),
        candidate_mean=("candidate", "mean"), control_mean=("control", "mean")).reset_index()
    if not summary.n_seeds.eq(3).all():
        raise ValueError("Incomplete paired contrasts")
    audit = dict(status="complete", seeds=sorted(SEEDS), fold="fold_a", context="1hop",
        scope="Initialization stability on one already inspected development fold",
        confirmatory_ci=False, independent_chromosome_replication=False,
        source_gates=gates, all_seed_development_gates_pass=all(
            g["status"] == "eligible_for_replication" for g in gates),
        interpretation="All seeds and contrasts retained; mean/SD/range describe seed variation, not chromosome uncertainty")
    return runs, per_seed, summary, audit


def plot(per_seed: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    controls = ["T_Rq", "Rt_Q", "Rt_Rq", "T", "Q"]
    labels = ["T + random Q", "random T + Q", "both random", "T only", "Q only"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained", sharey=True)
    for ax, task in zip(axes, ["sv", "ccre"]):
        selected = per_seed.loc[per_seed.task.eq(task) & per_seed.feature.eq("csht") & per_seed.metric.eq("auprc")]
        for seed, offset, color in zip(sorted(SEEDS), [-.15, 0, .15], ["#245a81", "#b86b35", "#42794c"]):
            values = selected.loc[selected.seed.eq(seed)].set_index("comparison").loc[
                ["T_Q minus " + c for c in controls], "difference"]
            ax.scatter(values, np.arange(5) + offset, label=str(seed), color=color, s=30)
        ax.axvline(0, color=".5", linewidth=.8)
        ax.set(yticks=range(5), yticklabels=labels, ylim=(4.6, -.6),
               xlabel="T+Q minus comparator: validation Δ AUPRC",
               title="SV insertion/deletion" if task == "sv" else "cCRE")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[0].legend(title="Initialization seed", frameon=False, loc="best")
    fig.suptitle("After C+S+H: three seeds, one development fold; no confidence interval")
    save_figure(fig, out, "frozen_branch_seed_stability")
    plt.close(fig)


def build(roots: list[Path], out: Path) -> None:
    sources, frames, numerical_gates = [], [], []
    for root in roots:
        audit = json.loads((root / "audit.json").read_text())
        if audit.get("status") != "complete":
            raise ValueError("Incomplete native source audit")
        for name in ["audit.json", "audited_per_run.csv", "paired_differences.csv", "development_gate.json"]:
            sources.append(fingerprint(root / name))
        frame = pd.read_csv(root / "audited_per_run.csv")
        expected, gate = compare(frame)
        saved = pd.read_csv(root / "paired_differences.csv")
        keys = ["task", "feature", "metric", "comparison"]
        pd.testing.assert_frame_equal(expected.sort_values(keys).reset_index(drop=True),
            saved.sort_values(keys).reset_index(drop=True), check_exact=False, atol=1e-12, rtol=0)
        previous = json.loads((root / "development_gate.json").read_text())
        optimizer_path = root / "probe_optimization.csv"
        optimizer = pd.read_csv(optimizer_path) if optimizer_path.is_file() else None
        if optimizer is not None:
            sources.append(fingerprint(optimizer_path))
        numerical_gates.append(dict(seed=int(frame.seed.iloc[0]), **replay_solver_gate(gate, previous, optimizer)))
        frames.append(frame)
    if len({g["probe_max_iter"] for g in numerical_gates}) != 1:
        raise ValueError("Cannot combine different or incompletely recorded optimizer budgets")
    runs, per_seed, summary, audit = summarize(frames)
    audit.update(source_gates=numerical_gates, all_seed_development_gates_pass=all(
        g["status"] == "eligible_for_replication" for g in numerical_gates),
        probe_max_iter=numerical_gates[0]["probe_max_iter"],
        all_probes_converged=(all(g["all_probes_converged"] for g in numerical_gates)
                             if numerical_gates[0]["probe_max_iter"] is not None else None))
    out.mkdir(parents=True, exist_ok=False)
    runs.to_csv(out / "audited_per_run.csv", index=False)
    per_seed.to_csv(out / "paired_per_seed.csv", index=False)
    summary.to_csv(out / "seed_summary.csv", index=False)
    audit["sources"] = sources
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    plot(per_seed, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs=3, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.sources, args.out_dir)
