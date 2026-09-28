#!/usr/bin/env python3
"""Collect completed frozen-probe tasks without selecting favorable outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tasks.entex.prepare import fingerprint
from tasks.transfer.report import save_figure

CS = "coordinate_plus_frozen_sequence_fm"
CST = CS + "_plus_frozen_pangenomefm"


def plot_scorecard(frame: pd.DataFrame, out: Path) -> None:
    """Show every recorded context, keeping the small external dataset separate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    entex = frame.task.str.startswith(("P0", "P1", "P2", "RNA"))
    external = frame.task.str.startswith("HG008")
    groups = [frame.loc[entex], frame.loc[~entex & ~external], frame.loc[external]]
    titles = ["EN-TEx: all prespecified tasks, sensitivities and tissues",
              "ENCODE and HGSVC3: binary tasks and recorded cCRE strata",
              "HG008: prospective refits; 69 variants in one genome"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), layout="constrained",
        gridspec_kw={"height_ratios": [16, 9, 2]})
    for ax, data, title in zip(axes, groups, titles):
        names = list(data.task.unique())
        for context, offset, color in [("strict", -.15, "#245a81"), ("1hop", .15, "#b86b35")]:
            part = data.loc[data.context.eq(context)].set_index("task").loc[names]
            y = np.arange(len(names)) + offset
            # Intervals need not contain the point estimate: draw endpoints
            # directly instead of passing potentially negative xerr values.
            ax.hlines(y, part.ci95_low, part.ci95_high, color=color, linewidth=1.1)
            ax.scatter(part.delta_t, y, s=23, color=color, label=context, zorder=3)
        ax.axvline(0, color=".45", linewidth=.8)
        ax.set(yticks=range(len(names)), yticklabels=names, ylim=(len(names)-.4, -.7),
               title=title, xlabel="Paired Δ AUPRC with source bootstrap 95% interval")
        ax.grid(axis="x", color=".92", linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="best", ncols=2)
    fig.suptitle("Historical frozen v1 downstream evidence — all 52 context rows\n"
                 "Horizontal scales differ between panels; overlapping subgroups are not independent benchmarks",
                 fontsize=11)
    save_figure(fig, out, "downstream_topology_gains")
    plt.close(fig)


def paired_row(task, context, baseline, augmented, gain, low, high, n_runs, source, **extra):
    if not np.isclose(augmented - baseline, gain, atol=1e-10, rtol=0):
        raise ValueError(f"Paired arithmetic differs: {task}/{context}")
    if not (0 <= baseline <= 1 and 0 <= augmented <= 1 and low <= high):
        raise ValueError("Invalid AP or interval")
    return dict(task=task, context=context, cs_auprc=baseline, cst_auprc=augmented,
                delta_t= gain, ci95_low=low, ci95_high=high, n_runs=n_runs,
                interpretation="positive interval" if low > 0 else "negative interval" if high < 0 else "inconclusive",
                scope="chromosome-held-out; frozen v1 encoders", source=source, **extra)


def build(root: Path, out: Path) -> None:
    rows, sources = [], set()

    def read(relative):
        path = root / relative
        sources.add(path)
        return pd.read_csv(path)

    absolute = read("data/manuscript_gap_sources_20260830/figure5_absolute.csv")
    gains = read("data/manuscript_gap_sources_20260830/figure5_contributions.csv")
    for task, title in [("cCRE", "ENCODE binary cCRE"), ("SV", "HGSVC3 INS versus DEL")]:
        for context in ["strict", "1hop"]:
            a = absolute.query("task == @task and closure == @context and metric == 'auprc'").set_index("feature_set")
            g = gains.query("task == @task and closure == @context and contrast == 'topology_given_coordinate_and_frozen_sequence_fm'").iloc[0]
            suffix = "_pair" if task == "SV" else ""
            rows.append(paired_row(title, context, a.loc[CS+suffix, "mean"], a.loc[CST+suffix, "mean"],
                g.mean_gain, g.ci95_low, g.ci95_high, g.n_paired_runs,
                "data/manuscript_gap_sources_20260830/figure5_contributions.csv",
                analysis="original manuscript cached results"))
    base = "results/foundation_evidence_20260927/ccre_subclass_source"
    a = read(base + "/ccre_stratum_absolute_summary.csv")
    g = read(base + "/ccre_stratum_topology_gains.csv")
    for r in g.itertuples():
        group = a.loc[a.stratification.eq(r.stratification) & a.stratum.eq(r.stratum) & a.closure.eq(r.closure)].set_index("feature_set")
        rows.append(paired_row("ENCODE " + r.stratum, r.closure, group.loc[CS, "mean"], group.loc[CST, "mean"],
            r.mean_gain, r.ci95_low, r.ci95_high, r.n_paired_runs, base + "/ccre_stratum_topology_gains.csv",
            analysis="stratified binary-probe predictions; not independently trained subtype probes"))
    entex = [
        ("P0 AS-prone cCRE", "v1/p0_analysis", ""),
        ("P0 exposure matched", "v1/p0_exposure_matched_analysis", ""),
        ("P0 H3K27ac only", "v1/p0_h3k27ac_analysis", ""),
        ("P0 CTCF only", "v1/p0_ctcf_analysis", ""),
        ("P1 enhancer tissue macro", "v1/p1_analysis", "macro_"),
        ("P2 CTCF SNV", "v1/p2_analysis/ctcf", ""),
        ("P2 H3K27ac SNV", "v1/p2_analysis/h3k27ac", ""),
        ("RNA ASE SNV", "meeting_20260929/rna_analysis", ""),
        *[(f"P2 {assay.upper()} SNV", f"extension_20260927/analysis/{assay}", "") for assay in ["atac", "h3k4me3", "h3k27me3"]],
        *[("P1 enhancer " + tissue, f"v1/p1_analysis/{tissue}", "") for tissue in
          ["Peyers_patch", "body_of_pancreas", "gastroesophageal_sphincter", "thyroid_gland", "tibial_nerve"]],
    ]
    for title, relative, prefix in entex:
        base = "results/entex/" + relative
        a = read(base + "/" + prefix + "summary.csv")
        g = read(base + "/" + prefix + "paired_gains.csv")
        for r in g.loc[g.comparison.eq("Delta_T_given_C_S")].itertuples():
            group = a.loc[a.context.eq(r.context) & a.metric.eq("auprc")].set_index("feature_set")
            rows.append(paired_row(title, r.context, group.loc[CS, "mean"], group.loc[CST, "mean"],
                r.mean, r.ci95_low, r.ci95_high, r.n_runs, base + "/" + prefix + "paired_gains.csv",
                analysis="supplied AS calls" if not title.startswith("P1") else "active versus explicitly repressed dELS"))
    base = "results/foundation_evidence_20260927/hg008_prospective_refit/analysis"
    a, g = read(base + "/summary.csv"), read(base + "/paired_gains.csv")
    for r in g.loc[g.comparison.eq("Delta_T_given_C_S") & g.metric.eq("auprc") & g.scope.eq("all")].itertuples():
        group = a.loc[a.context.eq(r.context) & a.metric.eq("auprc") & a.scope.eq("all")].set_index("feature_set")
        row = paired_row("HG008 clonal INS versus DEL", r.context, group.loc[CS, "mean"], group.loc[CST, "mean"],
                        r.mean, r.ci95_low, r.ci95_high, r.n_runs, base + "/paired_gains.csv",
                        analysis="prospective refits; 69 variants in one genome; not historical replay")
        row["scope"] = "external genome; no HG008 training"
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.duplicated(["task", "context"]).any():
        raise ValueError("Duplicate task/context rows")
    out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(out / "completed_downstream_tasks.csv", index=False)
    lines = ["# Completed downstream task scorecard", "",
             "AP values and paired 95% intervals come from the indicated saved source tables. "
             "Intervals use the source hierarchical fold/seed bootstrap. These are not independent "
             "tests of architecture choices made after viewing these results. Positive intervals "
             "are not equivalent to multiplicity-adjusted significance.", "",
             "C = manuscript coordinate/structural features; S = frozen NT; T = frozen topology v1. "
             "Rows marked ENCODE subtype/complexity are subsets of the original binary predictions.", "",
             "| Task | Context | C+S AP | C+S+T AP | ΔT [95% CI] | Runs |",
             "|---|---|---:|---:|---:|---:|"]
    for r in frame.itertuples():
        lines.append(f"| {r.task} | {r.context} | {r.cs_auprc:.6f} | {r.cst_auprc:.6f} | "
                     f"{r.delta_t:+.6f} [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}] | {int(r.n_runs)} |")
    lines += ["", "Source paths and analysis scope are retained in completed_downstream_tasks.csv. "
              "Development-only v2/NT/composite results and unresolved tasks are documented separately; "
              "no incomplete task receives a fabricated score.", ""]
    (out / "completed_downstream_tasks.md").write_text("\n".join(lines))
    (out / "audit.json").write_text(json.dumps(dict(status="complete", rows=len(frame),
        sources=[fingerprint(p) for p in sorted(sources)], arithmetic="C+S+T minus C+S checked to 1e-10",
        best_result_selection=False, scope="source-table consolidation; not re-training or fresh prediction replay"), indent=2) + "\n")
    plot_scorecard(frame, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.out_dir)
