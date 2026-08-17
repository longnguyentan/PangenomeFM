#!/usr/bin/env python3
"""Build manuscript-ready sequence/topology evidence tables, figures, and notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "pangenomefm-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.server.aggregate_ccre_frozen_probes import (
    PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST,
)


SELECTED_CONTRASTS = {
    "frozen_sequence_fm_vs_kmer_composition": "Sequence-FM over k-mer",
    "topology_given_coordinate_and_sequence": "Topology beyond C + k-mer",
    PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST: "Topology beyond C + sequence-FM",
    "frozen_sequence_fm_given_coordinate_and_topology": (
        "Sequence-FM beyond C + topology"
    ),
}
PRIMARY_INTERACTION = (
    f"one_hop_minus_strict__{PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST}"
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_audit(directory: Path) -> dict[str, object]:
    path = directory / "audit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"Aggregate is not complete: {path}")
    return payload


def contribution_path(directory: Path) -> Path:
    paths = sorted(directory.glob("*_modality_contributions.csv"))
    if len(paths) != 1:
        raise ValueError(f"Expected one modality contribution table in {directory}")
    return paths[0]


def context_path(directory: Path) -> Path:
    paths = sorted(directory.glob("*_context_contributions.csv"))
    if len(paths) != 1:
        raise ValueError(f"Expected one context contribution table in {directory}")
    return paths[0]


def summary_path(directory: Path) -> Path:
    paths = sorted(directory.glob("*_summary.csv"))
    paths = [
        path
        for path in paths
        if "stratified" not in path.name and "context" not in path.name
    ]
    if len(paths) != 1:
        raise ValueError(f"Expected one task summary table in {directory}")
    return paths[0]


def key_contrasts(task: str, directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(contribution_path(directory))
    result = frame.loc[
        frame["contrast"].isin(SELECTED_CONTRASTS)
        & frame["metric"].isin(["auprc", "auroc", "nll", "brier"])
    ].copy()
    result.insert(0, "task", task)
    result.insert(
        3,
        "contrast_label",
        result["contrast"].map(SELECTED_CONTRASTS),
    )
    result["ci_excludes_zero"] = (
        result["ci95_low"].gt(0) | result["ci95_high"].lt(0)
    )
    result["interpretation"] = np.where(
        result["ci95_low"].gt(0),
        "first_feature_set_improves",
        np.where(
            result["ci95_high"].lt(0),
            "reference_feature_set_improves",
            "interval_includes_zero",
        ),
    )
    return result


def context_interaction(task: str, directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(context_path(directory))
    result = frame.loc[
        frame["contrast"].eq(PRIMARY_INTERACTION)
        & frame["metric"].isin(["auprc", "auroc", "nll", "brier"])
    ].copy()
    result.insert(0, "task", task)
    result["ci_excludes_zero"] = (
        result["ci95_low"].gt(0) | result["ci95_high"].lt(0)
    )
    return result


def absolute_performance(task: str, directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(summary_path(directory))
    suffix = "_pair" if task == "SV" else ""
    features = {
        f"coordinate_plus_frozen_sequence_fm{suffix}": "C + sequence-FM",
        f"coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm{suffix}": (
            "C + sequence-FM + topology"
        ),
    }
    result = frame.loc[
        frame["feature_set"].isin(features)
        & frame["metric"].isin(["auprc", "auroc", "nll", "brier"])
    ].copy()
    result.insert(0, "task", task)
    result.insert(3, "feature_label", result["feature_set"].map(features))
    return result


def cache_summary(
    cache_audit_path: Path | None,
    shard_audit_paths: list[Path],
) -> dict[str, object] | None:
    if cache_audit_path is None:
        return None
    cache = json.loads(cache_audit_path.read_text(encoding="utf-8"))
    sampled = 0
    embedded = 0
    maximum_shard_wall = 0.0
    for path in shard_audit_paths:
        audit = json.loads(path.read_text(encoding="utf-8"))
        sampled += int(audit.get("raw_sequences_balanced_sampled", 0))
        embedded += int(audit.get("embedded_nodes", 0))
        maximum_shard_wall = max(maximum_shard_wall, float(audit.get("wall_seconds", 0.0)))
    return {
        "status": cache.get("status"),
        "nodes": int(cache.get("embedded_nodes", 0)),
        "dimension": int(cache.get("embedding_dimension", 0)),
        "coverage_fraction": float(cache.get("coverage_fraction", 0.0)),
        "model": cache.get("model_name"),
        "revision": cache.get("resolved_revision"),
        "fine_tuned": cache.get("fine_tuned"),
        "downstream_label_access": cache.get("downstream_label_access"),
        "raw_sequences_balanced_sampled": sampled,
        "shard_embedded_nodes": embedded,
        "balanced_sample_fraction": sampled / embedded if embedded else None,
        "parallel_shard_wall_seconds": maximum_shard_wall or None,
        "raw_sampling_policy": (
            "balanced prefix/suffix separated by N; 6,000 raw-base cap and "
            "1,000-token tokenizer cap"
        ),
    }


def format_interval(row: pd.Series) -> str:
    return (
        f"{row['mean_gain']:+.4f} "
        f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]"
    )


def primary_table(key: pd.DataFrame) -> pd.DataFrame:
    result = key.loc[
        key["metric"].eq("auprc")
        & key["contrast"].isin(
            [
                "frozen_sequence_fm_vs_kmer_composition",
                "topology_given_coordinate_and_sequence",
                PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST,
                "frozen_sequence_fm_given_coordinate_and_topology",
            ]
        ),
        [
            "task",
            "closure",
            "contrast",
            "contrast_label",
            "mean_gain",
            "ci95_low",
            "ci95_high",
            "n_paired_runs",
            "n_folds",
            "n_seeds",
            "ci_excludes_zero",
            "interpretation",
        ],
    ].copy()
    result["_task_order"] = result["task"].map({"cCRE": 0, "SV": 1})
    result["_contrast_order"] = result["contrast"].map(
        {value: index for index, value in enumerate(SELECTED_CONTRASTS)}
    )
    result["_closure_order"] = result["closure"].map({"strict": 0, "1hop": 1})
    return result.sort_values(
        ["_task_order", "_contrast_order", "_closure_order"]
    ).drop(columns=["_task_order", "_contrast_order", "_closure_order"])


def write_latex_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        "% Generated by build_sequence_fm_manuscript_outputs.py; do not edit.",
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "Task & Context & Paired contrast & Gain & 95\\% CI & Runs \\\\",
        "\\midrule",
    ]
    for row in frame.itertuples(index=False):
        label = str(row.contrast_label).replace("+", "$+$").replace("_", "\\_")
        closure = "1-hop" if row.closure == "1hop" else "strict"
        lines.append(
            f"{row.task} & {closure} & {label} & {row.mean_gain:+.4f} & "
            f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] & "
            f"{row.n_paired_runs} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_forest_plot(frame: pd.DataFrame, output_stem: Path) -> None:
    tasks = ["cCRE", "SV"]
    contrast_order = list(SELECTED_CONTRASTS)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharey=True)
    colors = {"strict": "#2667A8", "1hop": "#D96C2F"}
    offsets = {"strict": -0.13, "1hop": 0.13}
    for axis, task in zip(axes, tasks, strict=True):
        task_frame = frame.loc[frame["task"].eq(task)]
        for closure in ("strict", "1hop"):
            closure_frame = task_frame.loc[task_frame["closure"].eq(closure)]
            values = []
            lows = []
            highs = []
            positions = []
            for index, contrast in enumerate(contrast_order):
                row = closure_frame.loc[closure_frame["contrast"].eq(contrast)]
                if row.empty:
                    continue
                record = row.iloc[0]
                values.append(record["mean_gain"])
                lows.append(record["mean_gain"] - record["ci95_low"])
                highs.append(record["ci95_high"] - record["mean_gain"])
                positions.append(index + offsets[closure])
            axis.errorbar(
                values,
                positions,
                xerr=np.array([lows, highs]),
                fmt="o",
                color=colors[closure],
                capsize=3,
                label="1-hop" if closure == "1hop" else "Strict",
            )
        axis.axvline(0.0, color="#555555", linewidth=1, linestyle="--")
        axis.set_title(task, fontweight="bold")
        axis.set_xlabel("Paired AUPRC gain (positive favors first model)")
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(
        range(len(contrast_order)),
        [SELECTED_CONTRASTS[value] for value in contrast_order],
    )
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Sequence and topology contributions on identical chromosome-held-out runs",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| Task | Context | Paired comparison | AUPRC gain [95% CI] | Runs |",
        "|---|---|---|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        closure = "1-hop" if row.closure == "1hop" else "strict"
        interval = (
            f"{row.mean_gain:+.4f} "
            f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}]"
        )
        lines.append(
            f"| {row.task} | {closure} | {row.contrast_label} | "
            f"{interval} | {row.n_paired_runs} |"
        )
    return lines


def write_report(
    primary: pd.DataFrame,
    interactions: pd.DataFrame,
    cache: dict[str, object] | None,
    path: Path,
) -> None:
    lines = [
        "# PangenomeFM sequence/topology manuscript evidence",
        "",
        "This report is generated from the versioned aggregate tables. Positive gains",
        "mean that the first feature set improves the metric; intervals use paired",
        "fold-then-seed bootstrap resampling.",
        "",
        "## Primary paired AUPRC results",
        "",
        *markdown_table(primary),
        "",
        "## Context interaction",
        "",
        "The context interaction is the 1-hop topology gain minus the strict topology",
        "gain, using coordinate plus frozen sequence-FM as the reference model.",
        "",
        "| Task | Interaction AUPRC gain [95% CI] | Runs |",
        "|---|---:|---:|",
    ]
    for row in interactions.loc[interactions["metric"].eq("auprc")].itertuples(
        index=False
    ):
        lines.append(
            f"| {row.task} | {row.mean_gain:+.4f} "
            f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] | "
            f"{row.n_paired_runs} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Frozen sequence-FM strongly improves over k-mer composition when used alone.",
            "- Topology remains complementary beyond coordinate and frozen sequence-FM.",
            "- The topology increment is modest for cCRE and substantially larger for SV.",
            "- The 1-hop interaction is task-specific and should not be described as a causal",
            "  comparison of the upstream closure tasks.",
            "- SV single-class strata and very small long-SV bins are not valid evidence of",
            "  within-stratum ranking performance.",
        ]
    )
    if cache is not None:
        fraction = cache.get("balanced_sample_fraction")
        fraction_text = f"{100 * float(fraction):.2f}%" if fraction is not None else "not recorded"
        lines.extend(
            [
                "",
                "## Frozen sequence-model cache",
                "",
                f"- Nodes: {int(cache['nodes']):,}",
                f"- Dimension: {int(cache['dimension']):,}",
                f"- Coverage: {float(cache['coverage_fraction']):.6f}",
                f"- Model: `{cache['model']}`",
                f"- Revision: `{cache['revision']}`",
                f"- Long-node balanced sampling: {fraction_text}",
                f"- Policy: {cache['raw_sampling_policy']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "These results support complementary topology representation learning under",
            "chromosome-held-out downstream evaluation. They do not establish donor-held-out",
            "pretraining, causality, colocalization, fine-mapping, or a general-purpose",
            "biological foundation model.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_notebook(
    *,
    ccre_aggregate: Path,
    sv_aggregate: Path,
    report_markdown: str,
    path: Path,
) -> None:
    """Write a portable review notebook with code that reloads source evidence."""

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in report_markdown.splitlines()],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Reproduce the paired evidence table\n",
                "Run from the PangenomeFM repository root. The source directories can be "
                "changed below after copying the evidence package.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import pandas as pd\n",
                f"CCRE = Path({str(ccre_aggregate)!r})\n",
                f"SV = Path({str(sv_aggregate)!r})\n",
                "ccre = pd.read_csv(CCRE / 'ccre_modality_contributions.csv')\n",
                "sv = pd.read_csv(SV / 'sv_modality_contributions.csv')\n",
                "primary = 'topology_given_coordinate_and_frozen_sequence_fm'\n",
                "pd.concat([\n",
                "    ccre.assign(task='cCRE'),\n",
                "    sv.assign(task='SV'),\n",
                "]).query(\"contrast == @primary and metric == 'auprc'\")[\n",
                "    ['task', 'closure', 'mean_gain', 'ci95_low', 'ci95_high', 'n_paired_runs']\n",
                "]\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## SV interpretation safeguards\n",
                "Do not interpret single-class or underpowered strata as stable ranking tests.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "strata = pd.read_csv(SV / 'sv_stratified_summary.csv')\n",
                "strata[['stratum', 'stratum_value', 'mean_n_per_run', "
                "'interpretation_status']].drop_duplicates()\n",
            ],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccre-aggregate", type=Path, required=True)
    parser.add_argument("--sv-aggregate", type=Path, required=True)
    parser.add_argument("--sequence-cache-audit", type=Path)
    parser.add_argument("--sequence-shard-audit", type=Path, nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    ccre_audit = read_audit(args.ccre_aggregate)
    sv_audit = read_audit(args.sv_aggregate)
    for name, audit in (("cCRE", ccre_audit), ("SV", sv_audit)):
        if audit.get("n_bootstrap") != 10_000:
            raise ValueError(f"{name} aggregate does not use 10,000 bootstrap replicates")
        if PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST not in audit.get(
            "modality_contrasts", []
        ):
            raise ValueError(f"{name} aggregate misses the primary contrast")

    key = pd.concat(
        [
            key_contrasts("cCRE", args.ccre_aggregate),
            key_contrasts("SV", args.sv_aggregate),
        ],
        ignore_index=True,
    )
    interactions = pd.concat(
        [
            context_interaction("cCRE", args.ccre_aggregate),
            context_interaction("SV", args.sv_aggregate),
        ],
        ignore_index=True,
    )
    absolute = pd.concat(
        [
            absolute_performance("cCRE", args.ccre_aggregate),
            absolute_performance("SV", args.sv_aggregate),
        ],
        ignore_index=True,
    )
    primary = primary_table(key)
    cache = cache_summary(args.sequence_cache_audit, args.sequence_shard_audit)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "key_contrasts": args.out_dir / "manuscript_sequence_fm_key_contrasts_v2.csv",
        "primary_table_csv": args.out_dir / "manuscript_sequence_fm_primary_auprc.csv",
        "context_interactions": args.out_dir / "manuscript_sequence_fm_context_interactions.csv",
        "absolute_performance": args.out_dir / "manuscript_sequence_fm_absolute_performance.csv",
        "latex_table": args.out_dir / "manuscript_sequence_fm_primary_table.tex",
        "report": args.out_dir / "README.md",
        "notebook": args.out_dir / "manuscript_sequence_fm_analysis.ipynb",
        "figure_png": args.out_dir / "figure_sequence_topology_survival.png",
        "figure_pdf": args.out_dir / "figure_sequence_topology_survival.pdf",
    }
    key.to_csv(outputs["key_contrasts"], index=False)
    primary.to_csv(outputs["primary_table_csv"], index=False)
    interactions.to_csv(outputs["context_interactions"], index=False)
    absolute.to_csv(outputs["absolute_performance"], index=False)
    write_latex_table(primary, outputs["latex_table"])
    make_forest_plot(
        primary,
        args.out_dir / "figure_sequence_topology_survival",
    )
    write_report(primary, interactions, cache, outputs["report"])
    write_notebook(
        ccre_aggregate=args.ccre_aggregate,
        sv_aggregate=args.sv_aggregate,
        report_markdown=outputs["report"].read_text(encoding="utf-8"),
        path=outputs["notebook"],
    )

    input_paths = [
        args.ccre_aggregate / "audit.json",
        contribution_path(args.ccre_aggregate),
        context_path(args.ccre_aggregate),
        summary_path(args.ccre_aggregate),
        args.sv_aggregate / "audit.json",
        contribution_path(args.sv_aggregate),
        context_path(args.sv_aggregate),
        summary_path(args.sv_aggregate),
    ]
    if args.sequence_cache_audit is not None:
        input_paths.append(args.sequence_cache_audit)
    input_paths.extend(args.sequence_shard_audit)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "primary_contrast": PRIMARY_SEQUENCE_TOPOLOGY_CONTRAST,
        "primary_metric": "auprc",
        "bootstrap_replicates": 10_000,
        "tasks": ["cCRE", "SV"],
        "contexts": ["strict", "1hop"],
        "primary_rows": int(len(primary)),
        "context_interaction_rows": int(len(interactions)),
        "cache": cache,
        "inputs": {
            str(path): sha256_file(path)
            for path in input_paths
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in outputs.items()
        },
        "claim_boundary": (
            "chromosome-held-out complementary topology evidence; not donor-held-out "
            "pretraining, causality, colocalization, fine-mapping, or a general-purpose "
            "biological foundation-model claim"
        ),
    }
    audit_path = args.out_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
