#!/usr/bin/env python3
"""Generate draft figures for the PSB 2026 GraphGenome-FM revision."""
from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/private/tmp/graphgenomefm-mpl"))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
XDG_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME", "/private/tmp/graphgenomefm-cache"))
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = ROOT / "results" / "figures"


COLORS = {
    "teal": "#1b9e77",
    "gold": "#d95f02",
    "blue": "#386cb0",
    "rose": "#b2182b",
    "gray": "#666666",
}


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def hprc_link_prediction() -> None:
    labels = [
        "strict\nrandom",
        "1-hop\nrandom",
        "strict\ndistance",
        "1-hop\ndistance",
    ]
    auroc = [0.9889, 0.9914, 0.9846, 0.9957]
    colors = [COLORS["teal"], COLORS["teal"], COLORS["gold"], COLORS["gold"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bars = ax.bar(labels, auroc, color=colors, edgecolor="#222222", linewidth=0.6)
    ax.set_ylabel("Held-out AUROC")
    ax.set_ylim(0.96, 1.0)
    ax.set_title("HPRC Chromosome-Held-Out Link Prediction")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, auroc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.0008,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _save(fig, "hprc_link_prediction_auroc.png")


def hgsvc_transfer() -> None:
    strict_path = (
        ROOT
        / "results/hgsvc3/external_eval_hardneg_strict/run_004/per_target_metrics.csv"
    )
    onehop_path = (
        ROOT
        / "results/hgsvc3/external_eval_hardneg_1hop/run_003/per_target_metrics.csv"
    )
    values: dict[str, dict[str, float]] = {}
    for closure, path in [("strict", strict_path), ("1-hop", onehop_path)]:
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                chrom = row["target_sn"].split("|")[-1]
                values.setdefault(chrom, {})[closure] = float(row["mean_auroc"])

    chroms = ["chr19", "chr21", "chr22", "chrY"]
    xs = range(len(chroms))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    strict = [values[c]["strict"] for c in chroms]
    onehop = [values[c]["1-hop"] for c in chroms]
    ax.bar(
        [x - width / 2 for x in xs],
        strict,
        width,
        label="strict",
        color=COLORS["blue"],
        edgecolor="#222222",
        linewidth=0.5,
    )
    ax.bar(
        [x + width / 2 for x in xs],
        onehop,
        width,
        label="1-hop",
        color=COLORS["gold"],
        edgecolor="#222222",
        linewidth=0.5,
    )
    ax.set_xticks(list(xs), chroms)
    ax.set_ylim(0.96, 1.0)
    ax.set_ylabel("Mean AUROC")
    ax.set_title("External HGSVC Transfer By Chromosome")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    _save(fig, "hgsvc_external_transfer_per_chromosome.png")


def ccre_class_distribution() -> None:
    path = ROOT / "data/hprc/ccre/run_003/node_labels.csv.gz"
    counts: dict[str, int] = {}
    with gzip.open(path, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            cls = row["ccre_class"]
            counts[cls] = counts.get(cls, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in ordered]
    values = [v for _, v in ordered]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.barh(labels, values, color=COLORS["teal"], edgecolor="#222222", linewidth=0.4)
    ax.set_xlabel("Mapped GRCh38 nodes")
    ax.set_title("Mapped cCRE Class Distribution")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for y, value in enumerate(values):
        ax.text(value, y, f" {value:,}", va="center", fontsize=8)
    _save(fig, "ccre_class_distribution.png")


def ccre_binary_comparison() -> None:
    runs = [
        ("Logistic", "results/hprc/ccre_binary_baseline/run_003/summary.json"),
        ("GAT scratch", "results/hprc/ccre_binary_gat_scratch/run_001/summary.json"),
        (
            "GAT frozen",
            "results/hprc/ccre_binary_gat_pretrained_frozen/run_001/summary.json",
        ),
        (
            "GAT fine-tuned",
            "results/hprc/ccre_binary_gat_pretrained_finetune/run_001/summary.json",
        ),
    ]
    auroc = []
    auprc = []
    labels = []
    for label, rel_path in runs:
        data = json.loads((ROOT / rel_path).read_text())
        metrics = data["test_metrics"]
        auroc.append(float(metrics["auroc"]))
        auprc.append(float(metrics["auprc"]))
        labels.append(label)

    xs = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(
        [x - width / 2 for x in xs],
        auroc,
        width,
        label="AUROC",
        color=COLORS["blue"],
        edgecolor="#222222",
        linewidth=0.5,
    )
    ax.bar(
        [x + width / 2 for x in xs],
        auprc,
        width,
        label="AUPRC",
        color=COLORS["rose"],
        edgecolor="#222222",
        linewidth=0.5,
    )
    ax.set_xticks(list(xs), labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Current Binary cCRE Results")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="upper right")
    _save(fig, "ccre_binary_current_comparison.png")


def main() -> None:
    hprc_link_prediction()
    hgsvc_transfer()
    ccre_class_distribution()
    ccre_binary_comparison()


if __name__ == "__main__":
    main()
