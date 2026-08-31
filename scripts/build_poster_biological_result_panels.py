#!/usr/bin/env python3
"""Build six equally sized standalone biological-result panels for the poster."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


STRICT = "#075A9C"
ONE_HOP = "#E64B35"
GRAY = "#70777C"
UNDERPOWERED = "#A8ADB1"
GRID = "#E3E7EA"
TEXT = "#171717"

CANVAS = (10.5, 5.8)
DEFAULT_DPI = 400

CONTRAST_ORDER = [
    "frozen_sequence_fm_vs_kmer_composition",
    "topology_given_coordinate_and_frozen_sequence_fm",
    "frozen_sequence_fm_given_coordinate_and_topology",
]
CONTRAST_LABELS = {
    "frozen_sequence_fm_vs_kmer_composition": (
        "Sequence representation\nover k-mer composition"
    ),
    "topology_given_coordinate_and_frozen_sequence_fm": (
        "Graph representation beyond\ncoordinates + sequence"
    ),
    "frozen_sequence_fm_given_coordinate_and_topology": (
        "Sequence representation beyond\ncoordinates + graph"
    ),
}

AF_ORDER = [
    "af[0.001,0.01)",
    "af[0.01,0.05)",
    "af[0.05,0.5)",
    "af[0.5,1)",
]
AF_LABELS = {
    "af[0.001,0.01)": "0.001–0.01",
    "af[0.01,0.05)": "0.01–0.05",
    "af[0.05,0.5)": "0.05–0.5",
    "af[0.5,1)": "0.5–1.0",
}

LENGTH_ORDER = [
    "bp[50,100)",
    "bp[100,500)",
    "bp[500,1000)",
    "bp[1000,10000)",
    "bp[10000,100000)",
    "bp[100000,1e+06)",
]
LENGTH_LABELS = {
    "bp[50,100)": "50–100 bp",
    "bp[100,500)": "100–500 bp",
    "bp[500,1000)": "500 bp–1 kb",
    "bp[1000,10000)": "1–10 kb",
    "bp[10000,100000)": "10–100 kb",
    "bp[100000,1e+06)": "100 kb–1 Mb",
}


def _parser() -> argparse.ArgumentParser:
    root = Path(
        "server_imports/sequence_fm_v2_20260817/extracted"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--absolute",
        type=Path,
        default=root
        / "manuscript_sequence_fm_outputs_20260817"
        / "manuscript_sequence_fm_absolute_performance.csv",
    )
    parser.add_argument(
        "--contrasts",
        type=Path,
        default=root
        / "manuscript_sequence_fm_outputs_20260817"
        / "manuscript_sequence_fm_key_contrasts_v2.csv",
    )
    parser.add_argument(
        "--sv-strata",
        type=Path,
        default=root
        / "hgsvc3_sv_sequence_fm_factorial_20260815"
        / "paper_source_data_v2_20260817"
        / "sv_primary_stratified_contributions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/poster_t2t_20260831/biological_result_panels"),
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _figure(
    letter: str,
    title: str,
    *,
    left: float = 0.29,
    bottom: float = 0.20,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=CANVAS, constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.text(
        0.025,
        0.953,
        letter,
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.080,
        0.953,
        title,
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=TEXT,
    )
    fig.subplots_adjust(left=left, right=0.955, top=0.83, bottom=bottom)
    return fig, ax


def _legend(ax: plt.Axes, *, location: str = "best") -> None:
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color=STRICT,
                markerfacecolor=STRICT,
                lw=0,
                ms=8,
                label="Strict context",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color=ONE_HOP,
                markerfacecolor=ONE_HOP,
                lw=0,
                ms=7,
                label="One-hop context",
            ),
        ],
        loc=location,
        frameon=False,
        ncol=2,
        handletextpad=0.5,
        columnspacing=1.4,
    )


def _interval(
    ax: plt.Axes,
    *,
    mean: float,
    low: float,
    high: float,
    y: float,
    color: str,
    marker: str,
    size: float = 7.5,
) -> None:
    ax.errorbar(
        mean,
        y,
        xerr=[[mean - low], [high - mean]],
        fmt=marker,
        color=color,
        ecolor=color,
        markerfacecolor=color,
        markeredgecolor=color,
        ms=size,
        elinewidth=1.5,
        capsize=3.2,
        zorder=4,
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return {"png": str(png.resolve()), "pdf": str(pdf.resolve())}


def _single_row(frame: pd.DataFrame, message: str) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"{message}: expected one row, found {len(frame)}")
    return frame.iloc[0]


def absolute_panel(
    absolute: pd.DataFrame,
    *,
    task: str,
    letter: str,
    title: str,
    stem: str,
    output_dir: Path,
    dpi: int,
) -> dict[str, str]:
    subset = absolute[(absolute["task"] == task) & (absolute["metric"] == "auprc")]
    suffix = "_pair" if task == "SV" else ""
    baseline_name = f"coordinate_plus_frozen_sequence_fm{suffix}"
    combined_name = (
        f"coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm{suffix}"
    )
    baseline = _single_row(
        subset[
            (subset["feature_set"] == baseline_name)
            & (subset["closure"] == "strict")
        ],
        f"{task} baseline",
    )
    strict = _single_row(
        subset[
            (subset["feature_set"] == combined_name)
            & (subset["closure"] == "strict")
        ],
        f"{task} strict",
    )
    one_hop = _single_row(
        subset[
            (subset["feature_set"] == combined_name)
            & (subset["closure"] == "1hop")
        ],
        f"{task} one-hop",
    )
    rows = [
        ("Coordinates + sequence", baseline, GRAY, "o"),
        ("+ graph representation\n(strict context)", strict, STRICT, "o"),
        ("+ graph representation\n(one-hop context)", one_hop, ONE_HOP, "D"),
    ]

    fig, ax = _figure(letter, title, left=0.32)
    ys = np.arange(len(rows))[::-1]
    lower = min(float(row[1]["ci95_low"]) for row in rows)
    upper = max(float(row[1]["ci95_high"]) for row in rows)
    span = upper - lower
    pad = max(0.007, span * 0.18)
    ax.set_xlim(lower - pad, upper + 2.0 * pad)
    for y, (_, row, color, marker) in zip(ys, rows):
        mean = float(row["mean"])
        _interval(
            ax,
            mean=mean,
            low=float(row["ci95_low"]),
            high=float(row["ci95_high"]),
            y=float(y),
            color=color,
            marker=marker,
            size=8.5,
        )
        ax.text(
            float(row["ci95_high"]) + 0.10 * pad,
            y,
            f"{mean:.4f}",
            va="center",
            ha="left",
            color=color,
            fontsize=12,
            fontweight="bold",
        )
    ax.set_yticks(ys, [row[0] for row in rows])
    ax.tick_params(axis="y", pad=10)
    ax.grid(axis="x", color=GRID, lw=1.0)
    ax.set_xlabel("Mean chromosome-held-out AUPRC")
    ax.text(
        0.5,
        -0.25,
        "95% hierarchical bootstrap intervals; five chromosome folds × three seeds",
        transform=ax.transAxes,
        ha="center",
        color=GRAY,
        fontsize=10.5,
    )
    return _save(fig, output_dir, stem, dpi)


def contribution_panel(
    contrasts: pd.DataFrame,
    *,
    task: str,
    letter: str,
    title: str,
    stem: str,
    output_dir: Path,
    dpi: int,
) -> dict[str, str]:
    subset = contrasts[
        (contrasts["task"] == task)
        & (contrasts["metric"] == "auprc")
        & contrasts["contrast"].isin(CONTRAST_ORDER)
    ].copy()
    if len(subset) != 6:
        raise ValueError(f"{task} contribution panel: expected six rows, found {len(subset)}")

    fig, ax = _figure(letter, title, left=0.33)
    ys = np.arange(len(CONTRAST_ORDER))[::-1]
    min_low = float(subset["ci95_low"].min())
    max_high = float(subset["ci95_high"].max())
    span = max_high - min_low
    left = min(-0.01, min_low - 0.05 * span)
    right = max_high + 0.16 * span
    ax.set_xlim(left, right)
    for y, contrast in zip(ys, CONTRAST_ORDER):
        rows = subset[subset["contrast"] == contrast]
        for closure, color, marker, offset in [
            ("strict", STRICT, "o", 0.14),
            ("1hop", ONE_HOP, "D", -0.14),
        ]:
            row = _single_row(
                rows[rows["closure"] == closure],
                f"{task}/{contrast}/{closure}",
            )
            mean = float(row["mean_gain"])
            high = float(row["ci95_high"])
            _interval(
                ax,
                mean=mean,
                low=float(row["ci95_low"]),
                high=high,
                y=float(y + offset),
                color=color,
                marker=marker,
            )
            label_x = high + 0.018 * span
            ax.text(
                label_x,
                y + offset,
                f"{mean:+.4f}",
                va="center",
                ha="left",
                color=color,
                fontsize=11,
                fontweight="bold",
            )
    ax.axvline(0, color=GRAY, lw=1.2, ls="--")
    ax.set_yticks(ys, [CONTRAST_LABELS[key] for key in CONTRAST_ORDER])
    ax.tick_params(axis="y", pad=10)
    ax.grid(axis="x", color=GRID, lw=1.0)
    ax.set_xlabel("Paired ΔAUPRC (positive favors added information)")
    _legend(ax, location="lower right")
    ax.text(
        0.5,
        -0.25,
        "95% hierarchical bootstrap intervals; comparisons use identical examples",
        transform=ax.transAxes,
        ha="center",
        color=GRAY,
        fontsize=10.5,
    )
    return _save(fig, output_dir, stem, dpi)


def strata_panel(
    strata: pd.DataFrame,
    *,
    stratum: str,
    order: list[str],
    labels: dict[str, str],
    letter: str,
    title: str,
    stem: str,
    output_dir: Path,
    dpi: int,
) -> dict[str, str]:
    subset = strata[
        (strata["metric"] == "auprc")
        & (strata["stratum"] == stratum)
        & strata["stratum_value"].isin(order)
    ].copy()
    if len(subset) != 2 * len(order):
        raise ValueError(
            f"{stratum}: expected {2 * len(order)} rows, found {len(subset)}"
        )

    fig, ax = _figure(letter, title, left=0.25, bottom=0.29)
    ys = np.arange(len(order))[::-1]
    min_low = float(subset["ci95_low"].min())
    max_high = float(subset["ci95_high"].max())
    span = max_high - min_low
    has_underpowered = subset["interpretation_status"].astype(str).str.contains(
        "underpowered"
    ).any()
    left_fraction = 0.15 if has_underpowered else 0.05
    left = min(-0.012, min_low - left_fraction * span)
    right = max_high + 0.25 * span
    ax.set_xlim(left, right)
    for y, value in zip(ys, order):
        rows = subset[subset["stratum_value"] == value]
        underpowered = rows["interpretation_status"].astype(str).str.contains(
            "underpowered"
        ).any()
        for closure, color, marker, offset in [
            ("strict", STRICT, "o", 0.14),
            ("1hop", ONE_HOP, "D", -0.14),
        ]:
            row = _single_row(
                rows[rows["closure"] == closure],
                f"{stratum}/{value}/{closure}",
            )
            plot_color = UNDERPOWERED if underpowered else color
            mean = float(row["mean_gain"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            _interval(
                ax,
                mean=mean,
                low=low,
                high=high,
                y=float(y + offset),
                color=plot_color,
                marker=marker,
                size=7.2,
            )
            label_x = low - 0.012 * span if underpowered else high + 0.012 * span
            ax.text(
                label_x,
                y + offset,
                f"{mean:+.4f}",
                va="center",
                ha="right" if underpowered else "left",
                color=plot_color,
                fontsize=10.5,
                fontweight="bold",
            )
        if underpowered:
            n_examples = float(rows["mean_examples_per_run"].iloc[0])
            ax.text(
                left + 0.04 * (right - left),
                y + 0.42,
                f"Underpowered: mean n≈{n_examples:.0f} per run",
                ha="left",
                va="bottom",
                color=GRAY,
                fontsize=10.5,
                fontstyle="italic",
            )
    ax.axvline(0, color=GRAY, lw=1.2, ls="--")
    ax.set_yticks(ys, [labels[value] for value in order])
    ax.tick_params(axis="y", pad=10)
    ax.grid(axis="x", color=GRID, lw=1.0)
    ax.set_xlabel("Graph contribution, paired ΔAUPRC")
    ax.set_ylabel(
        "Allele-frequency group" if stratum == "allele_frequency_bin" else "Variant-length group"
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color=STRICT,
                markerfacecolor=STRICT,
                lw=0,
                ms=8,
                label="Strict context",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color=ONE_HOP,
                markerfacecolor=ONE_HOP,
                lw=0,
                ms=7,
                label="One-hop context",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        ncol=2,
        handletextpad=0.5,
        columnspacing=1.5,
    )
    ax.text(
        0.5,
        -0.34,
        "95% hierarchical bootstrap intervals; five chromosome folds × three seeds",
        transform=ax.transAxes,
        ha="center",
        color=GRAY,
        fontsize=10.5,
    )
    return _save(fig, output_dir, stem, dpi)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = _parser().parse_args()
    _style()
    absolute_path = args.absolute.resolve()
    contrasts_path = args.contrasts.resolve()
    strata_path = args.sv_strata.resolve()
    output_dir = args.output_dir.resolve()
    absolute = pd.read_csv(absolute_path)
    contrasts = pd.read_csv(contrasts_path)
    strata = pd.read_csv(strata_path)

    outputs = {}
    outputs["sv_b"] = absolute_panel(
        absolute,
        task="SV",
        letter="b",
        title="Structural-variant breakpoint classification",
        stem="figure_sv_panel_b_absolute_performance",
        output_dir=output_dir,
        dpi=args.dpi,
    )
    outputs["sv_c"] = contribution_panel(
        contrasts,
        task="SV",
        letter="c",
        title="Independent information-source contributions for structural variants",
        stem="figure_sv_panel_c_information_contributions",
        output_dir=output_dir,
        dpi=args.dpi,
    )
    outputs["sv_d"] = strata_panel(
        strata,
        stratum="allele_frequency_bin",
        order=AF_ORDER,
        labels=AF_LABELS,
        letter="d",
        title="Graph contribution across allele-frequency groups",
        stem="figure_sv_panel_d_allele_frequency",
        output_dir=output_dir,
        dpi=args.dpi,
    )
    outputs["sv_e"] = strata_panel(
        strata,
        stratum="length_bin",
        order=LENGTH_ORDER,
        labels=LENGTH_LABELS,
        letter="e",
        title="Graph contribution across variant-length groups",
        stem="figure_sv_panel_e_variant_length",
        output_dir=output_dir,
        dpi=args.dpi,
    )
    outputs["ccre_b"] = absolute_panel(
        absolute,
        task="cCRE",
        letter="b",
        title="Candidate regulatory-region classification",
        stem="figure_ccre_panel_b_absolute_performance",
        output_dir=output_dir,
        dpi=args.dpi,
    )
    outputs["ccre_c"] = contribution_panel(
        contrasts,
        task="cCRE",
        letter="c",
        title="Independent information-source contributions for cCRE",
        stem="figure_ccre_panel_c_information_contributions",
        output_dir=output_dir,
        dpi=args.dpi,
    )

    audit = {
        "schema_version": 1,
        "status": "complete",
        "canvas_inches": list(CANVAS),
        "dpi": args.dpi,
        "pixel_dimensions": [int(CANVAS[0] * args.dpi), int(CANVAS[1] * args.dpi)],
        "inputs": {
            "absolute": {"path": str(absolute_path), "sha256": _sha256(absolute_path)},
            "contrasts": {"path": str(contrasts_path), "sha256": _sha256(contrasts_path)},
            "sv_strata": {"path": str(strata_path), "sha256": _sha256(strata_path)},
        },
        "outputs": outputs,
        "excluded_by_request": [
            "SV panel a",
            "cCRE panel a",
            "cCRE panel d (pending stratified result)",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
