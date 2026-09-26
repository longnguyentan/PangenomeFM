"""Secondary scaling comparisons and native SV-complexity reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tasks.entex.analyze import estimate
from tasks.transfer.scaling_bio_summary import canonical_feature, summarize
from tasks.transfer.sv_strata import annotate, run_metrics


def complexity_runs(
    frame: pd.DataFrame, examples: Path, complexity: Path
) -> pd.DataFrame:
    metadata = annotate(pd.read_csv(examples), complexity)
    if metadata.complexity.ne("unassigned").mean() < 0.95:
        raise ValueError("Native SV complexity coverage below 95%")
    config = json.loads(Path("configs/structural_mechanism_v1.json").read_text())
    results = []
    for source, values in frame.loc[frame.task.eq("sv")].groupby("source"):
        predictions = pd.read_csv(Path(source).parent / "test_predictions.csv.gz")
        predictions["feature_set"] = predictions.feature_set.str.removesuffix("_pair")
        result = run_metrics(predictions, metadata, config)
        result = result.loc[result.family.eq("complexity")].copy()
        result["feature_set"] = result.feature_set.map(canonical_feature)
        result["task"] = "sv_complexity_" + result.stratum
        result["fraction"] = values.fraction.iloc[0]
        result["n_test"] = result.n
        for column in ["n_train", "n_val"]:
            result[column] = result.feature_set.map(
                values.set_index("feature_set")[column]
            )
        results.append(result)
    return pd.concat(results, ignore_index=True)


def contrasts(frame: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    for (task, context, feature), group in frame.groupby(
        ["task", "context", "feature_set"]
    ):
        for metric in ["auprc", "auroc"]:
            wide = group.pivot(
                index=["fold", "seed"], columns="fraction", values=metric
            )
            if 1.0 not in wide or wide.isna().any().any():
                raise ValueError("Unpaired scaling contrast")
            for fraction in wide:
                if fraction == 1.0:
                    continue
                gains = (wide[1.0] - wide[fraction]).rename("gain").reset_index()
                rows.append(
                    dict(
                        task=task,
                        context=context,
                        feature_set=feature,
                        metric=metric,
                        contrast=f"1 minus {fraction:g}",
                        fraction=fraction,
                        **estimate(gains, "gain", n_bootstrap, seed),
                    )
                )
    return pd.DataFrame(rows)


def figures(absolute: pd.DataFrame, paired: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42})
    for task, rows in absolute.loc[absolute.metric.eq("auprc")].groupby("task"):
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
        for j, context in enumerate(["strict", "1hop"]):
            for feature, color in [
                ("C+S", "#245a81"),
                ("C+T", "#a36c26"),
                ("C+S+T", "#388061"),
            ]:
                values = rows.loc[
                    rows.context.eq(context) & rows.feature_set.eq(feature)
                ].sort_values("fraction")
                axes[0, j].plot(
                    values.fraction, values["mean"], "o-", color=color, label=feature
                )
                axes[0, j].fill_between(
                    values.fraction,
                    values.ci95_low,
                    values.ci95_high,
                    color=color,
                    alpha=0.12,
                )
            gain = paired.loc[
                paired.task.eq(task)
                & paired.context.eq(context)
                & paired.contrast.eq("C+S+T minus C+S")
            ].sort_values("fraction")
            axes[1, j].plot(gain.fraction, gain["mean"], "o-", color="#388061")
            axes[1, j].fill_between(
                gain.fraction,
                gain.ci95_low,
                gain.ci95_high,
                color="#388061",
                alpha=0.15,
            )
            axes[1, j].axhline(0, color="0.5", linewidth=0.8)
            axes[0, j].set_title(context)
            axes[0, j].set_ylim(0, 1)
            axes[0, j].set_ylabel("AUPRC")
            axes[1, j].set_ylabel("Δ AUPRC (C+S+T − C+S)")
            for ax in axes[:, j]:
                ax.set_xticks([0.125, 0.25, 0.5, 1], ["12.5%", "25%", "50%", "100%"])
                ax.set_xlabel("Requested pretraining-window fraction")
        axes[0, 0].legend(frameon=False)
        fig.suptitle(f"Frozen biological scaling: {task}")
        for extension in ["pdf", "svg", "png"]:
            fig.savefig(out / f"scaling_{task}.{extension}", dpi=300)
        plt.close(fig)


def write_report(
    frame: pd.DataFrame,
    out: Path,
    examples: Path,
    complexity: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    from tasks.entex.prepare import fingerprint

    strata = complexity_runs(frame, examples, complexity)
    strata.to_csv(out / "sv_complexity_per_run.csv", index=False)
    absolute, paired = summarize(strata, n_bootstrap, seed)
    absolute.to_csv(out / "sv_complexity_summary.csv", index=False)
    paired.to_csv(out / "sv_complexity_paired_gains.csv", index=False)
    all_runs = pd.concat([frame, strata], ignore_index=True)
    contrasts(all_runs, n_bootstrap, seed).to_csv(
        out / "paired_vs_full.csv", index=False
    )
    full_absolute, full_paired = summarize(frame, n_bootstrap, seed)
    figures(pd.concat([full_absolute, absolute]), pd.concat([full_paired, paired]), out)
    (out / "complexity_audit.json").write_text(
        json.dumps(
            dict(
                examples=fingerprint(examples),
                complexity=fingerprint(complexity),
                definition="Unmodified native SV strata: maximum complexity at the two legacy endpoint coordinates",
                caution="Pointwise exploratory intervals; no new bins selected using outcomes",
            ),
            indent=2,
        )
        + "\n"
    )
