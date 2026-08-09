#!/usr/bin/env python3
"""Create evidence-safe paper tables from a downloaded full server run.

This analysis deliberately treats rotating chromosome holdouts as the primary
within-graph evaluation, keeps HPRC and HGSVC results separate when a shared
model was trained, and uses chromosome-level bootstrap intervals to avoid
presenting thousands of graph slices as independent biological replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


METRICS = ["auroc", "auprc", "accuracy", "precision", "recall", "f1", "brier", "ece_10bin"]
CHROMOSOME_ORDER = [f"chr{index}" for index in range(1, 23)] + ["chrX", "chrY"]


def bootstrap_mean(
    values: np.ndarray, *, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Return a percentile interval for a mean over independent units."""

    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return math.nan, math.nan
    draws = rng.choice(clean, size=(n_boot, len(clean)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def bh_adjust(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values preserving the input index."""

    values = p_values.to_numpy(dtype=float)
    finite = np.isfinite(values)
    adjusted = np.full(len(values), np.nan)
    if not finite.any():
        return pd.Series(adjusted, index=p_values.index)
    selected = values[finite]
    order = np.argsort(selected)
    ranked = selected[order]
    correction = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    correction = np.minimum.accumulate(correction[::-1])[::-1]
    restored = np.empty_like(correction)
    restored[order] = np.clip(correction, 0.0, 1.0)
    adjusted[np.flatnonzero(finite)] = restored
    return pd.Series(adjusted, index=p_values.index)


def rank_biserial(values: np.ndarray) -> float:
    """Matched-pairs rank-biserial effect size for nonzero differences."""

    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean) & (clean != 0)]
    if not len(clean):
        return 0.0
    ranks = rankdata(np.abs(clean))
    total = ranks.sum()
    return float((ranks[clean > 0].sum() - ranks[clean < 0].sum()) / total)


def normalize_chromosome(value: object) -> str:
    # Match two-digit autosomes before single-digit alternatives and prohibit
    # trailing digits, so chr10 is never collapsed to chr1.
    match = re.search(r"chr(?:2[0-2]|1[0-9]|[1-9]|X|Y)(?![0-9])", str(value))
    return match.group(0) if match else str(value)


def chromosome_bootstrap_summary(
    primary_slices: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_columns = ["regime", "dataset", "closure", "seed"]
    per_seed = (
        primary_slices.groupby(group_columns, dropna=False)
        .agg(
            n_chromosomes=("chromosome", "nunique"),
            n_slices=("slice", "size"),
            n_targets=("n_targets", "sum"),
            **{f"mean_{metric}": (metric, "mean") for metric in METRICS},
        )
        .reset_index()
    )

    chromosome_seed = (
        primary_slices.groupby(
            ["regime", "dataset", "closure", "seed", "chromosome"],
            dropna=False,
        )
        .agg(
            n_slices=("slice", "size"),
            n_targets=("n_targets", "sum"),
            **{f"mean_{metric}": (metric, "mean") for metric in METRICS},
        )
        .reset_index()
    )
    chromosome_summary = (
        chromosome_seed.groupby(
            ["regime", "dataset", "closure", "chromosome"], dropna=False
        )
        .agg(
            seeds=("seed", "nunique"),
            n_slices=("n_slices", "sum"),
            n_targets=("n_targets", "sum"),
            **{
                f"mean_{metric}": (f"mean_{metric}", "mean")
                for metric in METRICS
            },
            **{
                f"sd_{metric}": (f"mean_{metric}", "std")
                for metric in METRICS
            },
        )
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for keys, group in chromosome_summary.groupby(
        ["regime", "dataset", "closure"], dropna=False
    ):
        regime, dataset, closure = keys
        seed_group = per_seed[
            (per_seed["regime"] == regime)
            & (per_seed["dataset"] == dataset)
            & (per_seed["closure"] == closure)
        ]
        row: dict[str, object] = {
            "regime": regime,
            "dataset": dataset,
            "closure": closure,
            "seeds": int(seed_group["seed"].nunique()),
            "chromosomes": int(group["chromosome"].nunique()),
            "mean_slices_per_seed": float(seed_group["n_slices"].mean()),
            "mean_targets_per_seed": float(seed_group["n_targets"].mean()),
            "confidence_interval_unit": "chromosome",
            "bootstrap_replicates": n_boot,
        }
        for metric in METRICS:
            per_seed_values = seed_group[f"mean_{metric}"].to_numpy(dtype=float)
            chromosome_values = group[f"mean_{metric}"].to_numpy(dtype=float)
            lower, upper = bootstrap_mean(
                chromosome_values, n_boot=n_boot, rng=rng
            )
            row[f"{metric}_mean"] = float(np.nanmean(per_seed_values))
            row[f"{metric}_sd_across_seeds"] = float(
                np.nanstd(per_seed_values, ddof=1)
            )
            row[f"{metric}_chromosome_bootstrap_ci_lower"] = lower
            row[f"{metric}_chromosome_bootstrap_ci_upper"] = upper
        rows.append(row)
    return per_seed, chromosome_seed, pd.DataFrame(rows)


def summarize_transfer(seed_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = seed_metrics[
        (seed_metrics["metric_scope"] == "split")
        & (seed_metrics["split"] == "all")
        & seed_metrics["experiment_type"].isin(
            ["cohort_transfer", "release_transfer"]
        )
    ].copy()
    columns = [
        "experiment_type",
        "regime",
        "transfer_pair",
        "closure",
        "seed",
        "n_targets",
        "positive_fraction",
        *METRICS,
    ]
    per_seed = selected[columns].sort_values(
        ["experiment_type", "regime", "closure", "seed"]
    )
    summary = (
        per_seed.groupby(
            ["experiment_type", "regime", "transfer_pair", "closure"],
            dropna=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            mean_targets=("n_targets", "mean"),
            **{f"{metric}_mean": (metric, "mean") for metric in METRICS},
            **{
                f"{metric}_sd_across_seeds": (metric, "std")
                for metric in METRICS
            },
        )
        .reset_index()
    )
    return per_seed, summary


def summarize_context_deltas(
    matched: pd.DataFrame, *, n_boot: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = matched[
        (matched["experiment_type"] == "rotating_folds")
        & (matched["split"] == "heldout_chr_test")
    ].copy()
    delta_columns = [
        "delta_auroc_1hop_minus_strict",
        "delta_auprc_1hop_minus_strict",
        "delta_brier_1hop_minus_strict",
        "delta_ece_10bin_1hop_minus_strict",
    ]
    per_chromosome = (
        selected.groupby(
            ["regime", "dataset", "chromosome"], dropna=False
        )[delta_columns]
        .mean()
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for keys, group in per_chromosome.groupby(
        ["regime", "dataset"], dropna=False
    ):
        regime, dataset = keys
        row: dict[str, object] = {
            "regime": regime,
            "dataset": dataset,
            "chromosomes": int(group["chromosome"].nunique()),
            "matching_scope": "coordinate_interval_only",
            "exposure_matched": False,
            "interpretation": (
                "Context and targets remain exposure-confounded; this is not "
                "a causal representation effect."
            ),
        }
        for column in delta_columns:
            values = group[column].to_numpy(dtype=float)
            lower, upper = bootstrap_mean(values, n_boot=n_boot, rng=rng)
            try:
                p_value = float(wilcoxon(values, zero_method="wilcox").pvalue)
            except ValueError:
                p_value = 1.0
            prefix = column.removeprefix("delta_").removesuffix(
                "_1hop_minus_strict"
            )
            row[f"delta_{prefix}_mean"] = float(np.nanmean(values))
            row[f"delta_{prefix}_ci_lower"] = lower
            row[f"delta_{prefix}_ci_upper"] = upper
            row[f"delta_{prefix}_wilcoxon_p"] = p_value
            row[f"delta_{prefix}_rank_biserial"] = rank_biserial(values)
        rows.append(row)
    summary = pd.DataFrame(rows)
    p_columns = [column for column in summary if column.endswith("_wilcoxon_p")]
    for column in p_columns:
        summary[column.replace("_p", "_bh_q")] = bh_adjust(summary[column])
    return per_chromosome, summary


def paired_model_comparisons(
    chromosome_seed: pd.DataFrame,
) -> pd.DataFrame:
    chrom = (
        chromosome_seed.groupby(
            ["regime", "dataset", "closure", "chromosome"], dropna=False
        )[[f"mean_{metric}" for metric in METRICS]]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    comparisons = [
        ("hprc_r2", "combined_hprc_r2_hgsvc3", "hprc_r2"),
        ("hgsvc3", "combined_hprc_r2_hgsvc3", "hgsvc3"),
    ]
    for baseline, combined, dataset in comparisons:
        for closure in ["strict", "1hop"]:
            left = chrom[
                (chrom["regime"] == baseline)
                & (chrom["dataset"] == dataset)
                & (chrom["closure"] == closure)
            ]
            right = chrom[
                (chrom["regime"] == combined)
                & (chrom["dataset"] == dataset)
                & (chrom["closure"] == closure)
            ]
            paired = left.merge(
                right,
                on=["dataset", "closure", "chromosome"],
                suffixes=("_baseline", "_combined"),
            )
            for metric in ["auroc", "auprc", "brier"]:
                values = (
                    paired[f"mean_{metric}_combined"]
                    - paired[f"mean_{metric}_baseline"]
                ).to_numpy(dtype=float)
                try:
                    p_value = float(wilcoxon(values).pvalue)
                except ValueError:
                    p_value = 1.0
                rows.append(
                    {
                        "dataset": dataset,
                        "closure": closure,
                        "baseline_regime": baseline,
                        "comparison_regime": combined,
                        "metric": metric,
                        "chromosomes": len(values),
                        "mean_delta_combined_minus_baseline": float(
                            np.mean(values)
                        ),
                        "median_delta_combined_minus_baseline": float(
                            np.median(values)
                        ),
                        "rank_biserial": rank_biserial(values),
                        "wilcoxon_p": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    result["bh_q_value"] = bh_adjust(result["wilcoxon_p"])
    return result


def corrected_scaling_table(
    results_root: Path, benchmark_coverage: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in ["hprc_r2", "hgsvc3"]:
        path = results_root / "scaling" / dataset / "scaling_benchmark.csv"
        if not path.exists():
            continue
        probes = pd.read_csv(path)
        probes = probes[probes["returncode"] == 0].copy()
        coverage = benchmark_coverage[
            (benchmark_coverage["dataset"] == dataset)
            & (benchmark_coverage["benchmark_kind"] == "pretrain_benchmark")
            & (benchmark_coverage["closure"] == "1hop")
        ]
        full_slices = int(coverage["slices"].iloc[0])
        slope, intercept = np.polyfit(
            probes["train_slices"].to_numpy(dtype=float),
            probes["seconds_per_epoch"].to_numpy(dtype=float),
            deg=1,
        )
        slope = max(float(slope), 0.0)
        intercept = max(float(intercept), 0.0)
        seconds_per_epoch = intercept + slope * full_slices
        rows.append(
            {
                "dataset": dataset,
                "successful_probes": len(probes),
                "largest_probe_train_slices": int(probes["train_slices"].max()),
                "full_training_slices_selected_closure": full_slices,
                "fitted_fixed_seconds_per_epoch": intercept,
                "fitted_seconds_per_training_slice": slope,
                "estimated_full_seconds_per_epoch": seconds_per_epoch,
                "estimated_100_epoch_hours": seconds_per_epoch * 100 / 3600,
                "measured_peak_resident_gib": float(
                    probes["maximum_resident_bytes"].max() / 1024**3
                ),
                "measured_peak_gpu_memory_mib": float(
                    probes["maximum_gpu_memory_mib"].max()
                ),
                "projection_status": (
                    "planning estimate; early stopping and graph-size "
                    "nonlinearity are not modeled"
                ),
            }
        )
    return pd.DataFrame(rows)


def execution_audit(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    queue_rows: list[dict[str, object]] = []
    for phase in ["final", "folds", "transfer", "release"]:
        path = results_root / f"gpu_queue_{phase}_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        queue_rows.append(
            {
                "phase": phase,
                "jobs_requested": payload["jobs_requested"],
                "jobs_recorded": payload["jobs_recorded"],
                "failures": payload["failures"],
                "verified_complete": (
                    payload["jobs_requested"] == payload["jobs_recorded"]
                    and payload["failures"] == 0
                ),
            }
        )

    step_rows: list[dict[str, object]] = []
    for path in results_root.glob("**/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        steps = payload.get("steps")
        if not isinstance(steps, dict):
            continue
        for name, step in steps.items():
            if not isinstance(step, dict) or name == "preflight":
                continue
            relative = path.relative_to(results_root)
            queue_phase = (
                relative.parts[1]
                if relative.parts and relative.parts[0] == "gpu_queue_states"
                else "preparation"
            )
            step_rows.append(
                {
                    "queue_phase": queue_phase,
                    "state_file": str(relative),
                    "step": name,
                    "status": step.get("status"),
                    "wall_seconds": step.get("wall_seconds", 0.0),
                    "started_at": step.get("started_at"),
                    "finished_at": step.get("finished_at"),
                }
            )
    steps = pd.DataFrame(step_rows).drop_duplicates(
        ["state_file", "step"], keep="last"
    )
    return pd.DataFrame(queue_rows), steps


def plot_primary_chromosomes(chromosome_seed: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        chromosome_seed.groupby(
            ["regime", "dataset", "closure", "chromosome"], dropna=False
        )["mean_auprc"]
        .mean()
        .reset_index()
    )
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    panels = [
        ("hprc_r2", "HPRC R2 held-out chromosomes"),
        ("hgsvc3", "HGSVC3 held-out chromosomes"),
    ]
    colors = {"strict": "#3B6FB6", "1hop": "#D8583C"}
    styles = {"standalone": "-", "combined": "--"}
    for axis, (dataset, title) in zip(axes, panels):
        dataset_frame = summary[summary["dataset"] == dataset]
        standalone = dataset
        combined = "combined_hprc_r2_hgsvc3"
        for regime, regime_label in [
            (standalone, "standalone"),
            (combined, "combined"),
        ]:
            for closure in ["strict", "1hop"]:
                group = dataset_frame[
                    (dataset_frame["regime"] == regime)
                    & (dataset_frame["closure"] == closure)
                ].copy()
                group["chromosome"] = pd.Categorical(
                    group["chromosome"], categories=CHROMOSOME_ORDER, ordered=True
                )
                group = group.sort_values("chromosome")
                axis.plot(
                    range(len(group)),
                    group["mean_auprc"],
                    color=colors[closure],
                    linestyle=styles[regime_label],
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                    label=f"{regime_label}, {closure}",
                )
        axis.set_title(title)
        axis.set_ylabel("Mean slice AUPRC")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(ncol=2, fontsize=8, loc="lower right")
    axes[-1].set_xticks(range(len(CHROMOSOME_ORDER)), CHROMOSOME_ORDER, rotation=90)
    axes[-1].set_xlabel("Chromosome held out from representation training")
    figure.tight_layout()
    figure.savefig(out_dir / "primary_all_chromosome_auprc.png", dpi=240)
    figure.savefig(out_dir / "primary_all_chromosome_auprc.pdf")
    plt.close(figure)


def plot_transfer(transfer: pd.DataFrame, out_dir: Path) -> None:
    cohort = transfer[transfer["experiment_type"] == "cohort_transfer"].copy()
    cohort["closure_order"] = pd.Categorical(
        cohort["closure"], categories=["strict", "1hop"], ordered=True
    )
    cohort = cohort.sort_values(["regime", "closure_order"])
    regime_labels = {
        "hgsvc3_hprc1_combined_to_hprc_r2": "Integrated graph → HPRC R2",
        "hgsvc3_to_hprc_r2": "HGSVC3 → HPRC R2",
        "hprc_r2_to_hgsvc3": "HPRC R2 → HGSVC3",
    }
    labels = [
        f"{regime_labels.get(regime, regime)}\n{closure}"
        for regime, closure in zip(cohort["regime"], cohort["closure"])
    ]
    positions = np.arange(len(cohort))
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, metric in zip(axes, ["auroc", "auprc"]):
        axis.bar(
            positions,
            cohort[f"{metric}_mean"],
            yerr=cohort[f"{metric}_sd_across_seeds"],
            color=["#D8583C" if value == "1hop" else "#3B6FB6" for value in cohort["closure"]],
            capsize=3,
        )
        axis.axhline(0.5, color="black", linewidth=1, linestyle=":")
        axis.set_xticks(positions, labels, rotation=55, ha="right", fontsize=8)
        axis.set_title(metric.upper())
        axis.set_ylim(0.5, 1.0)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Target-pooled performance (mean ± SD, 3 seeds)")
    figure.suptitle("Frozen cross-graph transfer on all 24 chromosomes")
    figure.tight_layout()
    figure.savefig(out_dir / "cohort_transfer_all_chromosomes.png", dpi=240)
    figure.savefig(out_dir / "cohort_transfer_all_chromosomes.pdf")
    plt.close(figure)


def write_evidence_tables(out_dir: Path) -> None:
    """Write manuscript-facing status, claim, experiment, and risk matrices."""

    project_status = pd.DataFrame(
        [
            {
                "component": "HPRC R2 full released SV graph",
                "claimed_status": "Primary full dataset",
                "verified_status": "Fully implemented and executed",
                "evidence": "231 donors, 462 phased haplotypes, 24 chromosomes; validation pass",
                "problems": "Aggregate graph is not donor-resolved evaluation",
                "required_action": "Retain as primary topology graph; build path-resolved donor examples for donor holdout",
            },
            {
                "component": "HGSVC3 full released SV graph",
                "claimed_status": "Independent cohort",
                "verified_status": "Fully implemented and executed",
                "evidence": "65 donors, 130 phased haplotypes, 24 chromosomes; validation pass",
                "problems": "Four donors overlap HPRC R2; CHM13 construction differs",
                "required_action": "Describe as cross-graph transfer, not strictly donor-independent transfer",
            },
            {
                "component": "HPRC R1.1 release transfer",
                "claimed_status": "Release-transfer control",
                "verified_status": "Fully implemented and executed",
                "evidence": "12/12 release jobs completed across two directions, three seeds, two contexts",
                "problems": "R1.1 overlaps/superseded by R2 and is not an independent cohort",
                "required_action": "Use only as release sensitivity analysis",
            },
            {
                "component": "Official HGSVC3 plus HPRC-v1 integrated graph",
                "claimed_status": "Integrated-graph stress test",
                "verified_status": "Fully implemented and executed",
                "evidence": "107 donors/214 nonreference haplotypes detected; 216 haplotypes including references per README",
                "problems": "Contains HPRC-v1, not R2; cannot be counted as another cohort",
                "required_action": "Keep as construction/integration stress test",
            },
            {
                "component": "All-chromosome rotating holdouts",
                "claimed_status": "Genome-wide generalization",
                "verified_status": "Fully implemented and executed",
                "evidence": "120/120 jobs completed; five folds cover chr1-chr22, chrX, chrY; three seeds; two contexts",
                "problems": "Chromosome-held-out but not donor- or haplotype-held-out",
                "required_action": "Call this all-chromosome topology generalization, not genome-wide biological validation",
            },
            {
                "component": "Cross-graph transfer",
                "claimed_status": "Cross-cohort transfer",
                "verified_status": "Fully implemented and executed",
                "evidence": "18/18 jobs completed for three directions, three seeds, two contexts",
                "problems": "Four overlapping donors and construction/reference confounding",
                "required_action": "Qualify as frozen cross-graph transfer",
            },
            {
                "component": "Context-matched strict versus one-hop analysis",
                "claimed_status": "Expanded context improves representation",
                "verified_status": "Coordinate-matched only; exposure matching failed",
                "evidence": "134,775 coordinate-matched intervals; zero pairs within 10% exposure caliper",
                "problems": "One-hop exposes roughly 1.56-1.59x nodes and 1.84-1.89x links in full-coverage benchmarks",
                "required_action": "Implement target- and exposure-matched resampling before causal context claims",
            },
            {
                "component": "Donor/haplotype-held-out topology evaluation",
                "claimed_status": "Requested leakage control",
                "verified_status": "Proposed but absent",
                "evidence": "Current examples are drawn from released aggregate graph topology",
                "problems": "Donor and H1/H2 contributions are pooled before example construction",
                "required_action": "Construct path-resolved donor/haplotype examples",
            },
            {
                "component": "cCRE, methylation, haplotype, SV, and QTL/GWAS downstream tasks",
                "claimed_status": "Foundation-model biological validation",
                "verified_status": "Not executed in this full server run",
                "evidence": "Downloaded bundle contains topology, transfer, release, scaling, and context outputs only",
                "problems": "No new multi-task biological evidence from the all-chromosome model checkpoints",
                "required_action": "Run checkpoint-based downstream probes with donor-held-out splits and fair baselines",
            },
        ]
    )
    project_status.to_csv(out_dir / "project_status.csv", index=False)

    claims = pd.DataFrame(
        [
            ["Query-edge-masked local topology is learnable across all chromosomes", "Strongly supported", "Five rotating folds, 24 chromosomes, three seeds, two cohorts", "Primary claim"],
            ["A shared HPRC R2 plus HGSVC3 model improves within-graph reconstruction", "Strongly supported but modest outside HPRC strict", "Positive paired chromosome deltas in all 24 chromosomes; BH-adjusted Wilcoxon q<1e-6", "State effect sizes and shared-donor caveat"],
            ["Representations transfer across released aggregate graphs", "Moderately supported", "Frozen one-hop AUPRC 0.941 HPRC→HGSVC3 and 0.890 HGSVC3→HPRC R2", "Use cross-graph wording"],
            ["HGSVC3 is intrinsically weaker than HPRC R2", "Contradicted by the new within-graph results", "Standalone HGSVC3 exceeds HPRC R2 in strict chromosome-held-out AUPRC", "Remove or reframe as directional transfer/construction sensitivity"],
            ["One-hop context improves representation independently of exposure", "Unsupported", "Zero exposure-caliper-matched pairs; node/link exposure differs markedly", "Do not make causal context claim"],
            ["The model is donor-independent", "Unsupported", "Aggregate graphs pool donors and four donors overlap cohorts", "Implement path-resolved donor holdout"],
            ["PangenomeFM is a general-purpose biological foundation model", "Unsupported", "Full run evaluates one structural objective plus transfer; no new functional tasks", "Use pretrained pangenome-topology encoder"],
            ["The full run supports haplotype-specific functional prediction", "Unsupported", "No methylation/expression/cCRE task executed from these checkpoints", "Keep exploratory until donor-held-out biological gains exist"],
            ["The full run establishes SV or disease relevance", "Unsupported", "No supervised SV, GWAS, QTL enrichment, fine-mapping, or colocalization task in bundle", "Requires labeled resources and matched-background tests"],
        ],
        columns=["claim", "evidence_class", "verified_evidence", "required_wording_or_action"],
    )
    claims.to_csv(out_dir / "claim_evidence_matrix.csv", index=False)

    experiments = pd.DataFrame(
        [
            ["Four released graph inventories and validation", "Partial/selected data", "Full GFA/GBZ download, path indexing, validation", "Completed", "All four graphs pass fatal validation", "Data readiness established", "Freeze official release policy confirmation"],
            ["Deterministic full-coordinate pretraining benchmarks", "Absent", "5 Mb tiling, paired-subsample negatives, matched intervals", "Completed", "608 HPRC and 635 HGSVC slices per context", "All chromosomes represented", "Add target/exposure matching"],
            ["Rotating chromosome holdouts", "Four selected chromosomes", "Five folds × four regimes × three seeds × two contexts", "120/120 completed", "High all-chromosome masked reconstruction", "Strong topology generalization; not donor holdout", "Use as primary Figure 2"],
            ["Final all-data models", "Selected graphs/seeds", "Five regimes × three seeds × two contexts", "30/30 completed", "Checkpoints available", "Ready for downstream probes", "Do not use in-distribution metrics as primary generalization"],
            ["Cohort/cross-graph transfer", "Targeted chromosomes", "Three directions × three seeds × two contexts", "18/18 completed", "One-hop transfer substantially stronger than strict", "Moderate transfer evidence with construction confounding", "Exclude shared donors in a path-resolved design"],
            ["Graph-release transfer", "Absent", "R1.1↔R2, three seeds, two contexts", "12/12 completed", "Directional and seed-variable", "Useful stress test, not independent cohort", "Supplementary analysis"],
            ["Scaling benchmark", "CPU/local pilot", "Four one-epoch GPU probes per primary cohort", "8/8 completed", "Peak 1.6-2.5 GiB GPU; corrected 3.0-3.3 h/100-epoch planning projection", "Current model is inexpensive at this representation scale", "Validate projection with multi-epoch run"],
            ["Exposure-adjusted context comparison", "Absent", "Coordinate pairing, 10% caliper, regression for external benchmarks", "Executed but no caliper matches", "0/134,775 exposure-matched interval pairs", "Negative diagnostic; context claim remains blocked", "Redesign benchmark"],
            ["Biological downstream validation", "Small prior pilots", "No new server implementation in this matrix", "Not executed", "No new result", "Foundation-model scope remains unsupported", "Prioritize cCRE and donor-held-out functional probes"],
        ],
        columns=["experiment", "status_before", "implementation_performed", "execution_status", "main_result", "interpretation", "next_action"],
    )
    experiments.to_csv(out_dir / "experiment_table.csv", index=False)

    risks = pd.DataFrame(
        [
            ["Expanded-context advantage may be exposure-driven", "Critical", "Zero 10% exposure-caliper matches and 1.8-1.9x link exposure", "Redesign target/exposure-matched evaluation"],
            ["No donor/haplotype holdout", "Critical", "Released aggregate topology pools all paths", "Build path-resolved examples and split before graph aggregation"],
            ["Foundation-model terminology exceeds evidence", "Critical", "Only structural pretraining/transfer is newly demonstrated", "Use pangenome-topology encoder language"],
            ["Very high within-graph scores may reflect task shortcuts", "High", "No full coordinate-only, graph-only, sequence-only, or heuristic suite in this server matrix", "Run matched ablations on identical rotating folds"],
            ["Cross-graph transfer is not strictly cohort independent", "High", "Four shared donors plus reference/construction differences", "Report overlap and run overlap-excluded path design"],
            ["Calibration is weak despite ranking performance", "High", "ECE about 0.14-0.19 in primary chromosome holdouts", "Fit calibration on validation chromosomes only"],
            ["HPRC R2 chrY is a failure/stress chromosome", "Moderate", "Standalone strict chrY AUPRC about 0.868", "Report rather than hide; stratify graph complexity and sex chromosomes"],
            ["DANN lacks chromosome-held-out evaluation", "Moderate", "DANN was run only in final all-data matrix and underperformed in-distribution strict", "Treat as negative exploratory ablation or run rotating folds"],
            ["Biological validation remains disconnected from full checkpoints", "Critical", "No cCRE, methylation, SV, or QTL task in bundle", "Run frozen probes before any journal/foundation claim"],
        ],
        columns=["reviewer_risk", "severity", "evidence", "mitigation"],
    )
    risks.to_csv(out_dir / "reviewer_risk_table.csv", index=False)


def write_artifact_manifest(out_dir: Path) -> None:
    """Checksum every generated deliverable except the manifest itself."""

    rows: list[dict[str, object]] = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path == out_dir / "artifact_manifest.tsv":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        rows.append(
            {
                "file": str(path.relative_to(out_dir)),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    pd.DataFrame(rows).to_csv(
        out_dir / "artifact_manifest.tsv", sep="\t", index=False
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source = results_root / "paper_source_data"

    slices = pd.read_csv(source / "per_slice_metrics.csv.gz", low_memory=False)
    slices["chromosome"] = slices["chromosome"].map(normalize_chromosome)
    primary = slices[
        (slices["experiment_type"] == "rotating_folds")
        & (slices["split"] == "heldout_chr_test")
    ].copy()
    per_seed, chromosome_seed, primary_summary = chromosome_bootstrap_summary(
        primary, n_boot=args.n_boot, seed=args.seed
    )
    per_seed.to_csv(out_dir / "primary_cross_chromosome_per_seed.csv", index=False)
    chromosome_seed.to_csv(
        out_dir / "primary_cross_chromosome_per_seed_chromosome.csv", index=False
    )
    primary_summary.to_csv(
        out_dir / "primary_cross_chromosome_summary.csv", index=False
    )

    chromosome_primary = (
        chromosome_seed.groupby(
            ["regime", "dataset", "closure", "chromosome"], dropna=False
        )
        .agg(
            seeds=("seed", "nunique"),
            n_slices=("n_slices", "sum"),
            n_targets=("n_targets", "sum"),
            **{
                f"{metric}_mean": (f"mean_{metric}", "mean")
                for metric in METRICS
            },
            **{
                f"{metric}_sd_across_seeds": (f"mean_{metric}", "std")
                for metric in METRICS
            },
        )
        .reset_index()
    )
    chromosome_primary["chromosome"] = pd.Categorical(
        chromosome_primary["chromosome"], categories=CHROMOSOME_ORDER, ordered=True
    )
    chromosome_primary = chromosome_primary.sort_values(
        ["regime", "dataset", "closure", "chromosome"]
    )
    chromosome_primary.to_csv(
        out_dir / "primary_per_chromosome_metrics.csv", index=False
    )
    chromosome_primary.sort_values(["auprc_mean", "auroc_mean"]).to_csv(
        out_dir / "primary_failure_chromosomes.csv", index=False
    )

    seed_metrics = pd.read_csv(source / "experiment_seed_metrics.csv")
    transfer_per_seed, transfer_summary = summarize_transfer(seed_metrics)
    transfer_per_seed.to_csv(out_dir / "transfer_per_seed.csv", index=False)
    transfer_summary.to_csv(out_dir / "transfer_summary.csv", index=False)

    matched = pd.read_csv(source / "matched_context_slice_deltas.csv", low_memory=False)
    matched["chromosome"] = matched["chromosome"].map(normalize_chromosome)
    context_chromosome, context_summary = summarize_context_deltas(
        matched, n_boot=args.n_boot, seed=args.seed + 1
    )
    context_chromosome.to_csv(
        out_dir / "coordinate_matched_context_deltas_per_chromosome.csv",
        index=False,
    )
    context_summary.to_csv(
        out_dir / "coordinate_matched_context_summary.csv", index=False
    )

    model_comparisons = paired_model_comparisons(chromosome_seed)
    model_comparisons.to_csv(
        out_dir / "paired_combined_vs_standalone_tests.csv", index=False
    )

    benchmark_coverage = pd.read_csv(
        results_root / "resource_inventory" / "benchmark_coverage.csv"
    )
    exposure = benchmark_coverage.pivot_table(
        index=["dataset", "benchmark_kind"],
        columns="closure",
        values=["slices", "visible_nodes_sum", "visible_links_sum", "coordinate_bp_sum"],
        aggfunc="first",
    )
    exposure.columns = [f"{metric}_{closure}" for metric, closure in exposure.columns]
    exposure = exposure.reset_index()
    for metric in ["visible_nodes_sum", "visible_links_sum", "coordinate_bp_sum"]:
        exposure[f"{metric}_1hop_to_strict_ratio"] = (
            exposure[f"{metric}_1hop"] / exposure[f"{metric}_strict"]
        )
    exposure["exposure_caliper"] = 0.10
    exposure["exposure_matched_pair_count"] = 0
    exposure["exposure_matched_claim_allowed"] = False
    exposure.to_csv(out_dir / "context_exposure_audit.csv", index=False)

    scaling = corrected_scaling_table(results_root, benchmark_coverage)
    scaling.to_csv(out_dir / "corrected_scaling_projection.csv", index=False)

    queue_status, execution_steps = execution_audit(results_root)
    queue_status.to_csv(out_dir / "verified_queue_status.csv", index=False)
    execution_steps.to_csv(out_dir / "verified_execution_steps.csv", index=False)
    runtime = (
        execution_steps.groupby(["queue_phase", "status"], dropna=False)
        .agg(
            steps=("step", "size"),
            compute_wall_seconds=("wall_seconds", "sum"),
        )
        .reset_index()
    )
    runtime["compute_wall_hours"] = runtime["compute_wall_seconds"] / 3600
    runtime.to_csv(out_dir / "verified_execution_runtime.csv", index=False)

    dataset_manifest = pd.read_csv(
        results_root / "resource_inventory" / "dataset_manifest.csv"
    )
    dataset_manifest.to_csv(out_dir / "verified_dataset_manifest.csv", index=False)

    write_evidence_tables(out_dir)

    plot_primary_chromosomes(chromosome_seed, out_dir)
    plot_transfer(transfer_summary, out_dir)

    context_source_summary = json.loads(
        (
            results_root
            / "context_adjusted_performance"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    package_files = sum(1 for path in results_root.rglob("*") if path.is_file())
    checkpoint_files = sum(1 for _ in results_root.rglob("ckpt_*.pt"))
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results_root": str(results_root),
        "primary_evaluation": "five-fold rotating chromosome holdout covering chr1-chr22, chrX, chrY",
        "primary_split": "heldout_chr_test",
        "seeds": sorted(int(value) for value in primary["seed"].unique()),
        "datasets": sorted(str(value) for value in dataset_manifest["dataset"]),
        "combined_unique_donors": 292,
        "prediction_files_aggregated": 186,
        "locally_available_final_checkpoints": checkpoint_files,
        "files_after_checkpoint_extraction": package_files,
        "queue_status_all_complete": bool(queue_status["verified_complete"].all()),
        "context_exposure_matched_pairs": int(
            context_source_summary["exposure_matched_pairs"]
        ),
        "context_exposure_matched_fraction": float(
            context_source_summary["exposure_matched_fraction"]
        ),
        "claim_boundary": (
            "All-chromosome masked topology reconstruction and cross-graph "
            "transfer are supported. A context-independent representation "
            "gain, donor-held-out transfer, functional foundation-model "
            "breadth, and disease relevance are not established."
        ),
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_artifact_manifest(out_dir)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
