"""Rebuild revision tables and reconstruction intervals from bundled evidence."""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def reconstruction_intervals(frame, n_bootstrap=5000):
    """Resample chromosome blocks while retaining the window-weighted estimand."""
    rows = []
    for key, group in frame.groupby(["regime", "dataset", "closure"]):
        if group.duplicated(["chromosome", "seed"]).any():
            raise ValueError("Duplicate reconstruction block")
        values = group.pivot(index="seed", columns="chromosome", values="mean_auprc")
        counts = group.pivot(index="seed", columns="chromosome", values="n_slices")
        if (
            values.isna().any().any()
            or counts.isna().any().any()
            or counts.le(0).any().any()
        ):
            raise ValueError("Incomplete reconstruction block matrix")
        a, w = values.to_numpy(), counts.to_numpy()
        point = np.mean((a * w).sum(axis=1) / w.sum(axis=1))
        rng = np.random.default_rng(20260924)
        index = rng.integers(a.shape[1], size=(n_bootstrap, a.shape[1]))
        draws = (
            (a[:, index] * w[:, index]).sum(axis=2) / w[:, index].sum(axis=2)
        ).mean(axis=0)
        rows.append(
            dict(
                regime=key[0],
                dataset=key[1],
                closure=key[2],
                auprc_mean=point,
                ci95_low=np.quantile(draws, 0.025),
                ci95_high=np.quantile(draws, 0.975),
                n_chromosomes=a.shape[1],
                n_seeds=a.shape[0],
                bootstrap_replicates=n_bootstrap,
            )
        )
    return pd.DataFrame(rows)


def build():
    evidence = ROOT / "evidence"
    sources = [
        "primary_cross_chromosome_per_seed_chromosome.csv",
        "figure3_reconstruction.csv",
        "figure3_transfer.csv",
        "main_comparisons.csv",
    ]
    original = pd.read_csv(evidence / "figure3_reconstruction.csv")
    intervals = reconstruction_intervals(pd.read_csv(evidence / sources[0]))
    joined = intervals.merge(
        original[["regime", "dataset", "closure", "auprc_mean"]],
        on=["regime", "dataset", "closure"],
        suffixes=("", "_original"),
        validate="one_to_one",
    )
    if len(joined) != len(original) or not np.allclose(
        joined.auprc_mean, joined.auprc_mean_original, atol=1e-12, rtol=0
    ):
        raise ValueError("Headline reconstruction mean changed")
    intervals.to_csv(evidence / "reconstruction_intervals_corrected.csv", index=False)
    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"strict": "#28679D", "1hop": "#E96B4A"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    regimes = [
        ("HPRC R2", "hprc_r2", "hprc_r2"),
        ("HGSVC3", "hgsvc3", "hgsvc3"),
        ("Integrated / HPRC", "combined_hprc_r2_hgsvc3", "hprc_r2"),
        ("Integrated / HGSVC", "combined_hprc_r2_hgsvc3", "hgsvc3"),
        (
            "Integrated / combined",
            "official_hgsvc3_hprc1_integrated",
            "hgsvc3_hprc1_combined",
        ),
    ]
    for j, (label, regime, dataset) in enumerate(regimes):
        for i, (context, color) in enumerate(colors.items()):
            part = intervals.loc[
                intervals.regime.eq(regime)
                & intervals.dataset.eq(dataset)
                & intervals.closure.eq(context)
            ]
            if len(part) != 1:
                raise ValueError(f"Missing plotted regime {label} {dataset}")
            r = part.iloc[0]
            y = j + (i - 0.5) * 0.18
            axes[0].plot(
                r.auprc_mean, y, "o", color=color, label=context if j == 0 else None
            )
            axes[0].hlines(y, r.ci95_low, r.ci95_high, color=color)
    axes[0].set(
        yticks=range(len(regimes)),
        yticklabels=[r[0] for r in regimes],
        xlabel="Mean window AUPRC; 95% chromosome-block CI",
        title="a  Within-resource reconstruction",
        xlim=(0.45, 1.01),
    )
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, loc="lower left")
    transfer = pd.read_csv(evidence / "figure3_transfer.csv")
    pairs = [
        ("HPRC R2 to HGSVC3", "hprc_r2_to_hgsvc3"),
        ("HGSVC3 to HPRC R2", "hgsvc3_to_hprc_r2"),
        ("Integrated to HPRC R2", "hgsvc3_hprc1_combined_to_hprc_r2"),
        ("HPRC R1.1 to R2", "hprc_r1_1_to_hprc_r2"),
        ("HPRC R2 to R1.1", "hprc_r2_to_hprc_r1_1"),
    ]
    for j, (_, pair) in enumerate(pairs):
        for i, (context, color) in enumerate(colors.items()):
            r = transfer.loc[
                transfer.transfer_pair.eq(pair) & transfer.closure.eq(context)
            ].iloc[0]
            y = j + (i - 0.5) * 0.18
            axes[1].plot(r.auprc_mean, y, "o", color=color)
            axes[1].hlines(
                y,
                r.auprc_mean - r.auprc_sd_across_seeds,
                r.auprc_mean + r.auprc_sd_across_seeds,
                color=color,
            )
    axes[1].set(
        yticks=range(len(pairs)),
        yticklabels=[r[0] for r in pairs],
        xlabel="AUPRC; ±1 SD across seeds (not a CI)",
        title="b  Cross-resource transfer",
        xlim=(0.45, 1.01),
    )
    axes[1].invert_yaxis()
    for ax in axes:
        ax.axvline(0.5, color=".6", ls="--", lw=0.8)
    fig.suptitle("Frozen graph transfer; dashed line is approximately balanced chance")
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(
            ROOT / "figures" / f"figure5_generalization_transfer.{suffix}", dpi=300
        )
    plt.close(fig)
    rows = []
    names = {
        "P0 AS-prone cCRE": "AS-prone cCRE",
        "P1 enhancer tissue macro": "Enhancer tissue macro",
        "P2 CTCF SNV": "CTCF SNV",
        "P2 H3K27ac SNV": "H3K27ac SNV",
    }
    for r in pd.read_csv(evidence / "main_comparisons.csv").itertuples(index=False):
        rows.append(
            f"{names[r.task]} & {r.context.replace('1hop', '1-hop')} & {r.auprc_C_S:.4f} & {r.auprc_C_S_T:.4f} & {r.mean:+.5f} & [{r.ci95_low:+.5f}, {r.ci95_high:+.5f}] "
            + r"\\"
        )
    (ROOT / "entex_rows.tex").write_text("\n".join(rows) + "\n")
    (evidence / "revision_analysis_audit.json").write_text(
        json.dumps(
            dict(
                inputs={
                    n: hashlib.sha256((evidence / n).read_bytes()).hexdigest()
                    for n in sources
                },
                reconstruction_correction="Old intervals resampled equally weighted chromosome means while points averaged windows. New intervals resample chromosome blocks with window counts retained; point estimates unchanged.",
                bootstrap_seed=20260924,
                bootstrap_replicates=5000,
                scaling_or_hg008_full_results_included=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    build()
