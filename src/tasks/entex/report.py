"""Collect completed primary task comparisons without choosing contexts by outcome."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from tasks.entex.analyze import BASE, FULL


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/entex/v1"))
    args = ap.parse_args()
    sources = {
        "P0 AS-prone cCRE": ("p0_analysis", "summary.csv", "paired_gains.csv"),
        "P1 enhancer tissue macro": (
            "p1_analysis",
            "macro_summary.csv",
            "macro_paired_gains.csv",
        ),
        "P2 CTCF SNV": ("p2_analysis/ctcf", "summary.csv", "paired_gains.csv"),
        "P2 H3K27ac SNV": ("p2_analysis/h3k27ac", "summary.csv", "paired_gains.csv"),
    }
    rows = []
    for task, (folder, sfile, gfile) in sources.items():
        summary = pd.read_csv(args.root / folder / sfile)
        gains = pd.read_csv(args.root / folder / gfile)
        for context in ["strict", "1hop"]:
            gain = gains.loc[
                gains.context.eq(context) & gains.comparison.eq("Delta_T_given_C_S")
            ]
            if len(gain) != 1 or gain.n_runs.iloc[0] != 15 or gain.n_folds.iloc[0] != 5:
                raise ValueError(f"Incomplete paired result: {task}/{context}")
            row = dict(
                task=task,
                context=context,
                **{
                    k: gain.iloc[0][k] for k in ["mean", "std", "ci95_low", "ci95_high"]
                },
            )
            for feature, label in [(BASE, "C_S"), (FULL, "C_S_T")]:
                for metric in ["auprc", "auroc"]:
                    match = summary.loc[
                        summary.context.eq(context)
                        & summary.feature_set.eq(feature)
                        & summary.metric.eq(metric)
                    ]
                    if len(match) != 1:
                        raise ValueError("Missing or duplicate metric")
                    row[f"{metric}_{label}"] = float(match["mean"].iloc[0])
            rows.append(row)
    frame = pd.DataFrame(rows)
    out = args.root / "final_report"
    out.mkdir(exist_ok=True)
    frame.to_csv(out / "main_comparisons.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4), sharex=True, sharey=True, layout="constrained"
    )
    for ax, context in zip(axes, ["strict", "1hop"]):
        selected = frame.loc[frame.context.eq(context)]
        for i, row in enumerate(selected.itertuples()):
            ax.plot(row.mean, i, "o", color="#245a81")
            ax.hlines(i, row.ci95_low, row.ci95_high, color="#245a81")
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_yticks(range(len(selected)), selected.task)
        ax.set_title(context)
        ax.set_xlabel("Δ AUPRC (C+S+T − C+S), 95% CI")
    for ext in ["png", "svg"]:
        fig.savefig(out / f"main_topology_comparisons.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
