"""Secondary metrics and figures from completed SV strata and donor predictions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from evaluation.paired_inference import bh_adjust, fold_sign_flip
from tasks.entex.analyze import BASE, FULL, estimate
from tasks.entex.prepare import fingerprint
from tasks.entex.probe import FEATURES


def paired_rows(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    keys = ["fold", "seed"]
    if frame.duplicated(keys + ["feature_set"]).any():
        raise ValueError("Duplicate paired result")
    selected = frame.loc[frame.feature_set.isin([BASE, FULL])]
    if not selected.groupby(keys).feature_set.nunique().eq(2).all():
        raise ValueError("Missing paired comparator")
    wide = selected.pivot(index=keys, columns="feature_set", values=metric)
    if BASE not in wide or FULL not in wide:
        raise ValueError("Missing paired comparator")
    if (wide[BASE].isna() != wide[FULL].isna()).any():
        raise ValueError("Asymmetric undefined paired metric")
    for column in ["n", "positive_prevalence"]:
        if column in selected:
            counts = selected.pivot(index=keys, columns="feature_set", values=column)
            if not counts[BASE].eq(counts[FULL]).all():
                raise ValueError("Paired example counts/prevalence differ")
    return (wide[FULL] - wide[BASE]).rename("gain").reset_index()


def strata_tables(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    absolute, contrasts = [], []
    for (family, stratum, context), group in frame.groupby(
        ["family", "stratum", "context"]
    ):
        if set(group.feature_set) != set(FEATURES):
            raise ValueError("Missing one of the seven feature sets")
        columns = (
            ["class_recall"]
            if family == "sv_type"
            else [
                "auprc",
                "auroc",
                "normalized_ap",
                "balanced_accuracy",
                "f1",
                "precision",
                "recall",
            ]
        )
        meta = dict(family=family, stratum=stratum, context=context)
        base = group.loc[group.feature_set.eq(BASE)]
        meta.update(
            mean_n=float(base.n.mean()),
            mean_prevalence=float(base.positive_prevalence.mean()),
            sufficiently_powered=bool(base.sufficiently_powered.all()),
        )
        for metric in columns:
            for feature, values in group.groupby("feature_set"):
                absolute.append(
                    dict(
                        **meta,
                        metric=metric,
                        feature_set=feature,
                        **estimate(values, metric, cfg["n_bootstrap"], cfg["seed"]),
                    )
                )
            gains = paired_rows(group, metric)
            contrasts.append(
                dict(
                    **meta,
                    metric=metric,
                    p_fold_signflip=fold_sign_flip(gains),
                    **estimate(gains, "gain", cfg["n_bootstrap"], cfg["seed"]),
                )
            )
    comparisons = pd.DataFrame(contrasts)
    comparisons["q_bh"] = comparisons.groupby(
        ["family", "context", "metric"]
    ).p_fold_signflip.transform(bh_adjust)
    return pd.DataFrame(absolute), comparisons


def save_figure(fig, out: Path, name: str) -> None:
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(out / f"{name}.{extension}", dpi=300, bbox_inches="tight")


def figures(contrasts: pd.DataFrame, donors: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"strict": "#245a81", "1hop": "#be6831"}
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ["auprc", "auroc", "normalized_ap"],
        ["AUPRC", "AUROC", "Prevalence-normalized AP"],
    ):
        for i, context in enumerate(colors):
            data = contrasts.loc[
                contrasts.family.eq("complexity")
                & contrasts.context.eq(context)
                & contrasts.metric.eq(metric)
            ].set_index("stratum")
            for j, category in enumerate(["low", "medium", "high"]):
                r = data.loc[category]
                x = j + (i - 0.5) * 0.16
                ax.plot(
                    x,
                    r["mean"],
                    "o",
                    color=colors[context],
                    label=context if j == 0 else None,
                )
                ax.vlines(x, r.ci95_low, r.ci95_high, color=colors[context])
        ax.axhline(0, color="0.5", linewidth=0.7)
        ax.set(
            xticks=range(3),
            xticklabels=["Low", "Medium", "High"],
            ylabel=f"Paired Δ {title}",
            title=title,
        )
    axes[0].legend(frameon=False)
    fig.suptitle(
        "SV insertion versus deletion: topology gain by native graph complexity"
    )
    save_figure(fig, out, "sv_complexity")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, family, title in zip(
        axes, ["length", "allele_frequency"], ["Variant length", "Allele frequency"]
    ):
        subset = contrasts.loc[
            contrasts.family.eq(family) & contrasts.metric.eq("auprc")
        ]
        # Use the inherited categorical labels, never reorder by observed gain.
        order = sorted(
            subset.stratum.unique(),
            key=lambda label: float(
                re.search(r"[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?", label).group()
            ),
        )
        for i, context in enumerate(colors):
            data = subset.loc[subset.context.eq(context)].set_index("stratum")
            for j, category in enumerate(order):
                r = data.loc[category]
                y = j + (i - 0.5) * 0.18
                color = colors[context] if r.sufficiently_powered else "0.65"
                ax.plot(
                    r["mean"], y, "o" if r.sufficiently_powered else "x", color=color
                )
                ax.hlines(y, r.ci95_low, r.ci95_high, color=color)
        ax.axvline(0, color="0.5", linewidth=0.7)
        ax.set(
            yticks=range(len(order)),
            yticklabels=order,
            xlabel="Paired Δ AUPRC",
            title=title,
        )
        ax.invert_yaxis()
    fig.suptitle("SV strata: blue strict; orange 1-hop; grey × underpowered")
    save_figure(fig, out, "sv_size_frequency")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    selected = donors.loc[donors.unit.eq("donor") & donors.metric.eq("auprc")]
    names = sorted(selected["sample"].unique())
    for ax, context in zip(axes, colors):
        data = selected.loc[selected.context.eq(context)].set_index("sample").loc[names]
        for j, (_, r) in enumerate(data.iterrows()):
            color = "#be6831" if r.in_hprc else colors["strict"]
            ax.plot(j, r["mean"], "s" if r.in_hprc else "o", markersize=3, color=color)
            ax.vlines(j, r.ci95_low, r.ci95_high, color=color, linewidth=0.6)
        ax.axhline(0, color="0.5", linewidth=0.7)
        ax.set(
            title=context,
            xlabel="Donors in accession order (n=65)",
            ylabel="Paired Δ AUPRC",
        )
    fig.suptitle("Donor-stratified chromosome holdout; orange squares overlap HPRC")
    save_figure(fig, out, "donor_stratified_sv")
    plt.close(fig)


def donor_tables(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    absolute, contrasts = [], []
    for (sample, unit, context, in_hprc), group in frame.groupby(
        ["sample", "unit", "context", "in_hprc"]
    ):
        meta = dict(sample=sample, unit=unit, context=context, in_hprc=bool(in_hprc))
        base = group.loc[group.feature_set.eq(BASE)]
        meta.update(
            mean_n=float(base.n.mean()),
            mean_prevalence=float(base.positive_fraction.mean()),
        )
        for metric in ["auprc", "auroc"]:
            gains = paired_rows(group, metric)
            contrasts.append(
                dict(
                    **meta,
                    metric=metric,
                    **estimate(gains, "gain", cfg["n_bootstrap"], cfg["seed"]),
                )
            )
            for feature, values in group.groupby("feature_set"):
                absolute.append(
                    dict(
                        **meta,
                        metric=metric,
                        feature_set=feature,
                        **estimate(values, metric, cfg["n_bootstrap"], cfg["seed"]),
                    )
                )
    return pd.DataFrame(absolute), pd.DataFrame(contrasts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strata", type=Path, required=True)
    ap.add_argument("--donors", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(Path("configs/structural_mechanism_v1.json").read_text())
    a, c = strata_tables(pd.read_csv(args.strata / "per_run.csv"), cfg)
    d, g = donor_tables(pd.read_csv(args.donors / "per_run.csv"), cfg)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, values in [
        ("strata_absolute", a),
        ("strata_paired", c),
        ("donor_absolute", d),
        ("donor_paired", g),
    ]:
        values.to_csv(args.out_dir / f"{name}.csv", index=False)
    figures(c, g, args.out_dir)
    (args.out_dir / "audit.json").write_text(
        json.dumps(
            dict(
                status="complete",
                config=cfg,
                inputs=[
                    fingerprint(p / "per_run.csv") for p in [args.strata, args.donors]
                ],
                fitting=False,
                normalized_ap="(AP - prevalence) / (1 - prevalence)",
                donor_inference="Descriptive; correlated carriers of shared variants, chromosome-held-out pooled probe",
                confidence_intervals="Pointwise hierarchical fold/seed bootstrap; not multiplicity-adjusted",
                q_values="BH separately within family/context/metric; sign flips of five seed-averaged folds",
                class_metrics="Within INS or DEL, class recall replaces undefined discrimination metrics",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
