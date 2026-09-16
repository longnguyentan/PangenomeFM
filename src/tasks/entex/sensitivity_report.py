"""Present every predefined P0 sensitivity together, without selecting by outcome."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tasks.entex.analyze import BASE, FULL

CASES = {
    "primary": "p0_analysis",
    "exposure_matched": "p0_exposure_matched_analysis",
    "h3k27ac": "p0_h3k27ac_analysis",
    "ctcf": "p0_ctcf_analysis",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/entex/v1"))
    ap.add_argument(
        "--out-dir", type=Path, default=Path("results/entex/v1/p0_sensitivity_report")
    )
    args = ap.parse_args()
    rows = []
    reference_keys = None
    for name, directory in CASES.items():
        folder = args.root / directory
        runs = pd.read_csv(folder / "per_run.csv")
        keys = set(
            runs[["closure", "fold", "seed", "feature_set"]].itertuples(
                index=False, name=None
            )
        )
        if len(keys) != 210 or len(runs) != 210:
            raise ValueError(f"Incomplete 30-run seven-feature matrix: {name}")
        if reference_keys is not None and keys != reference_keys:
            raise ValueError("Different fold/seed/feature keys between sensitivities")
        reference_keys = keys
        gains = pd.read_csv(folder / "paired_gains.csv")
        for row in gains.loc[gains.comparison.eq("Delta_T_given_C_S")].to_dict(
            "records"
        ):
            selected = runs.loc[runs.closure.eq(row["context"])]
            means = selected.groupby("feature_set").auprc.mean()
            rows.append(
                dict(
                    task=name,
                    **row,
                    auprc_C=means["coordinate"],
                    auprc_C_S=means[BASE],
                    auprc_C_S_T=means[FULL],
                    positive_prevalence_mean=selected.positive_prevalence.mean(),
                )
            )
    result = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_dir / "all_predefined_comparisons.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    for ax, context in zip(axes, ["strict", "1hop"]):
        frame = result.loc[result.context.eq(context)]
        for i, row in enumerate(frame.itertuples()):
            ax.plot(row.mean, i, "o", color="#245a81")
            ax.hlines(i, row.ci95_low, row.ci95_high, color="#245a81")
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_yticks(range(len(frame)), [s.replace("_", " ") for s in frame.task])
        ax.set_title(context)
        ax.set_xlabel("Δ AUPRC (C+S+T − C+S), 95% CI")
    for ext in ["png", "svg"]:
        fig.savefig(args.out_dir / f"predefined_sensitivities.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
