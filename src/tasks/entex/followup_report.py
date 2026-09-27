"""Plot all prespecified EN-TEx measurement follow-ups, retaining null outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tasks.entex.snv import ASSAYS
from tasks.entex.analyze import BASE, FULL
from evaluation.paired_inference import bh_adjust, fold_sign_flip


def multiplicity_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = runs.loc[runs.analysis.isin(["measurement_weight", "equal_locus_weight"])]
    for (assay, analysis, context), group in selected.groupby(["assay", "analysis", "closure"]):
        for metric in ["auprc", "auroc", "normalized_ap"]:
            wide = group.pivot(index=["fold", "seed"], columns="feature_set", values=metric)
            if set(wide.columns) != {BASE, FULL} or wide.isna().any().any():
                raise ValueError("Unpaired multiplicity comparison")
            gains = (wide[FULL] - wide[BASE]).rename("gain").reset_index()
            rows.append(dict(assay=assay, analysis=analysis, context=context, metric=metric,
                             mean_gain=gains.gain.mean(), sign_flip_p=fold_sign_flip(gains)))
    out = pd.DataFrame(rows)
    out["bh_q"] = out.groupby(["analysis", "metric"]).sign_flip_p.transform(bh_adjust)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--assays", nargs="+", choices=list(ASSAYS), default=["ctcf", "h3k27ac"])
    args = ap.parse_args()
    if len(args.assays) != len(set(args.assays)):
        raise ValueError("Duplicate requested assays")
    tables, runs = [], []
    for assay in args.assays:
        audit = json.loads((args.root / assay / "audit.json").read_text())
        if audit["status"] != "complete" or audit["n_prediction_runs"] != 30:
            raise ValueError("All complete assay matrices required")
        tables.append(pd.read_csv(args.root / assay / "paired_gains.csv"))
        runs.append(pd.read_csv(args.root / assay / "per_run.csv"))
    frame = pd.concat(tables, ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.out_dir / "all_paired_gains.csv", index=False)
    multiplicity_table(pd.concat(runs, ignore_index=True)).to_csv(
        args.out_dir / "multiplicity_sensitivity.csv", index=False)
    main_rows = frame.loc[
        frame.analysis.isin(
            ["measurement_weight", "equal_locus_weight", "donor_macro", "tissue_macro"]
        )
    ].copy()
    main_rows.to_csv(args.out_dir / "main_comparisons.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42})
    labels = {
        "measurement_weight": "Measurement weighted",
        "equal_locus_weight": "Equal locus weight",
        "donor_macro": "Donor macro",
        "tissue_macro": "Tissue macro",
    }
    for metric in ["auprc", "auroc", "normalized_ap"]:
        fig, axes = plt.subplots(
            1, len(args.assays), figsize=(5 * len(args.assays), 3.5), layout="constrained", sharey=True, squeeze=False
        )
        axes = axes[0]
        for ax, assay in zip(axes, args.assays):
            for context, offset, color in [
                ("strict", -0.12, "#245a81"),
                ("1hop", 0.12, "#a36c26"),
            ]:
                values = (
                    main_rows.loc[
                        main_rows.assay.eq(assay)
                        & main_rows.metric.eq(metric)
                        & main_rows.context.eq(context)
                    ]
                    .set_index("analysis")
                    .loc[list(labels)]
                )
                for i, row in enumerate(values.itertuples()):
                    ax.plot(
                        row.mean,
                        i + offset,
                        "o",
                        color=color,
                        label=context if i == 0 else None,
                    )
                    ax.hlines(i + offset, row.ci95_low, row.ci95_high, color=color)
            ax.axvline(0, color="0.6", linewidth=0.8)
            ax.set_yticks(range(len(labels)), list(labels.values()))
            ax.set_title(assay.upper())
            ax.set_xlabel(f"Δ {metric.upper()} (C+S+T − C+S)")
        axes[0].invert_yaxis()
        axes[-1].legend(frameon=False)
        fig.suptitle("Exploratory EN-TEx follow-ups: pointwise 95% fold/seed intervals")
        for extension in ["pdf", "svg", "png"]:
            fig.savefig(args.out_dir / f"weighting_{metric}.{extension}", dpi=300)
        plt.close(fig)
    for stratum in ["donor", "tissue"]:
        selected = frame.loc[frame.analysis.eq(stratum) & frame.metric.eq("auprc")]
        names = sorted(selected.group.unique())
        fig, axes = plt.subplots(
            1,
            len(args.assays),
            figsize=(5.5 * len(args.assays), max(3, len(names) * 0.28 + 1.5)),
            sharey=True,
            layout="constrained",
            squeeze=False,
        )
        axes = axes[0]
        for ax, assay in zip(axes, args.assays):
            for context, offset, color in [
                ("strict", -0.13, "#245a81"),
                ("1hop", 0.13, "#a36c26"),
            ]:
                values = selected.loc[
                    selected.assay.eq(assay) & selected.context.eq(context)
                ].set_index("group")
                for i, name in enumerate(names):
                    if name not in values.index:
                        continue
                    row = values.loc[name]
                    ax.plot(row["mean"], i + offset, "o", color=color, markersize=3)
                    ax.hlines(
                        i + offset,
                        row.ci95_low,
                        row.ci95_high,
                        color=color,
                        linewidth=0.8,
                    )
            ax.axvline(0, color="0.6", linewidth=0.8)
            ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names])
            ax.set_title(assay.upper())
            ax.set_xlabel("Δ AUPRC (C+S+T − C+S)")
        axes[0].invert_yaxis()
        fig.suptitle(
            f"All eligible {stratum}s; blue=strict, ochre=one-hop\nChromosome-held-out predictions; not {stratum}-held-out fitting"
        )
        for extension in ["pdf", "svg", "png"]:
            fig.savefig(args.out_dir / f"{stratum}_gains.{extension}", dpi=300)
        plt.close(fig)
    (args.out_dir / "audit.json").write_text(
        json.dumps(
            dict(
                status="complete",
                assays=args.assays,
                selection="All count-eligible groups, alphabetic order; no selection by performance",
                intervals="Pointwise exploratory hierarchical fold/seed bootstrap, not multiplicity-adjusted",
                multiplicity="Exploratory fold sign-flip with BH across all requested assays/contexts, separately per metric and weighting; five folds limit p-value resolution",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
