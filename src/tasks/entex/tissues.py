"""Summarize prespecified P1 tissues and equally weighted tissue-macro gains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from tasks.entex.analyze import BASE, CT, estimate, paired


def macro_runs(frame: pd.DataFrame, expected_tissues: list[str]) -> pd.DataFrame:
    keys = ["fold", "seed", "closure", "feature_set"]
    if frame.duplicated(keys + ["subtask"]).any():
        raise ValueError("Duplicate tissue/run result")
    if set(frame.subtask) != set(expected_tissues):
        raise ValueError("Missing or unexpected tissue")
    if not frame.groupby(keys).subtask.nunique().eq(len(expected_tissues)).all():
        raise ValueError("Unpaired tissue coverage between runs/features")
    metrics = [
        "auprc",
        "auroc",
        "normalized_ap",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
    ]
    if frame[metrics].isna().any().any():
        raise ValueError(
            "Undefined tissue metric; cannot silently change macro universe"
        )
    result = frame.groupby(keys, as_index=False)[metrics].mean()
    result["n_tissues"] = len(expected_tissues)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-root", type=Path, required=True)
    ap.add_argument("--preparation-audit", type=Path, required=True)
    ap.add_argument("--complexity", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    tissues = json.loads(args.preparation_audit.read_text())["selected_tissues"]
    frames = []
    gains = []
    for tissue in tissues:
        target = args.out_dir / tissue
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tasks.entex.analyze",
                "--probe-root",
                str(args.probe_root / tissue),
                "--complexity",
                str(args.complexity),
                "--out-dir",
                str(target),
            ],
            check=True,
        )
        frames.append(pd.read_csv(target / "per_run.csv"))
        gains.append(pd.read_csv(target / "paired_gains.csv").assign(tissue=tissue))
    all_runs = pd.concat(frames, ignore_index=True)
    macro = macro_runs(all_runs, tissues)
    summary = []
    macro_gains = []
    for (context, feature), g in macro.groupby(["closure", "feature_set"]):
        for metric in [
            "auprc",
            "auroc",
            "normalized_ap",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
        ]:
            summary.append(
                dict(
                    context=context,
                    feature_set=feature,
                    metric=metric,
                    **estimate(g, metric, 10000, 20260806),
                )
            )
    for context, g in macro.groupby("closure"):
        for base, name in [(BASE, "Delta_T_given_C_S"), (CT, "Delta_S_given_C_T")]:
            macro_gains.append(
                dict(
                    context=context, comparison=name, **paired(g, base, 10000, 20260806)
                )
            )
    pd.concat(gains).to_csv(args.out_dir / "per_tissue_paired_gains.csv", index=False)
    all_runs.to_csv(args.out_dir / "all_tissue_runs.csv", index=False)
    macro.to_csv(args.out_dir / "macro_runs.csv", index=False)
    pd.DataFrame(summary).to_csv(args.out_dir / "macro_summary.csv", index=False)
    pd.DataFrame(macro_gains).to_csv(
        args.out_dir / "macro_paired_gains.csv", index=False
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot = pd.concat(
        [pd.concat(gains), pd.DataFrame(macro_gains).assign(tissue="macro")],
        ignore_index=True,
    )
    for context, g in plot.loc[plot.comparison.eq("Delta_T_given_C_S")].groupby(
        "context"
    ):
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        for i, row in enumerate(g.itertuples()):
            ax.plot(i, row.mean, "o", color="#245a81")
            ax.vlines(i, row.ci95_low, row.ci95_high, color="#245a81")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks(range(len(g)), g.tissue, rotation=25, ha="right")
        ax.set_ylabel("Δ AUPRC (C+S+T − C+S)")
        ax.set_title(f"Active/repressed distal enhancers ({context})")
        for ext in ["png", "svg"]:
            fig.savefig(args.out_dir / f"tissue_topology_gain_{context}.{ext}", dpi=300)
        plt.close(fig)
    (args.out_dir / "audit.json").write_text(
        json.dumps(
            dict(
                tissues=tissues,
                macro_definition="equal-weight mean over the fixed selected tissues within each fold/seed, then fold/seed bootstrap; no tissue resampling",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
