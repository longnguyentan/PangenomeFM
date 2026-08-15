#!/usr/bin/env python3
"""Analyze exact PangenomeFM-minus-baseline performance by complexity/context.

The unit of inference is a chromosome block, not an individual candidate edge.
Complexity labels must be frozen by ``extract_graph_complexity.py`` before this
program reads any model or baseline score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CATEGORY_ORDER = ["low", "medium", "high"]
CONTEXT_ORDER = ["strict", "1hop"]
METRICS = ["auprc_advantage", "model_score", "auroc_advantage"]


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def parse_score_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("score must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("score must be LABEL=PATH")
    return label.strip(), Path(raw_path)


def chromosome_block_summary(
    frame: pd.DataFrame,
    value_column: str,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Return an equal-chromosome mean and chromosome-bootstrap interval."""

    values = (
        frame.groupby("chromosome", sort=True)[value_column]
        .mean()
        .dropna()
        .to_numpy(float)
    )
    if values.size == 0:
        return {
            "n_chromosomes": 0,
            "chromosome_block_mean": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    draws = values[
        rng.integers(0, len(values), size=(int(n_bootstrap), len(values)))
    ].mean(axis=1)
    return {
        "n_chromosomes": int(len(values)),
        "chromosome_block_mean": float(values.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def sign_flip_pvalue(
    frame: pd.DataFrame,
    value_column: str,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided sign-flip test on chromosome-level mean differences."""

    values = (
        frame.groupby("chromosome", sort=True)[value_column]
        .mean()
        .dropna()
        .to_numpy(float)
    )
    if values.size == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    signs = rng.choice(
        np.asarray([-1.0, 1.0]), size=(int(n_permutations), len(values))
    )
    null = np.abs((signs * values).mean(axis=1))
    return float((1 + np.sum(null >= observed)) / (n_permutations + 1))


def load_complexity(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    required = {
        "slice_id", "chromosome", "start", "end", "context",
        "locus_complexity_score", "locus_complexity_category",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"complexity table misses columns: {sorted(missing)}")
    audit_path = path.parent / "complexity_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"missing complexity audit: {audit_path}")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS":
        raise ValueError(f"complexity audit is not PASS: {audit}")
    if audit.get("performance_columns_read") != []:
        raise ValueError("complexity was not frozen independently of performance")
    frame["context"] = frame["context"].astype(str)
    if frame.duplicated(["slice_id", "context"]).any():
        raise ValueError("complexity table has duplicate slice/context identities")
    return frame, audit


def load_scores(specs: list[tuple[str, Path]]) -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    sources: list[dict] = []
    for label, path in specs:
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {
            "region_id", "chromosome", "start", "end", "fold", "baseline",
            "context", "is_eligible", "model_score", "auprc_advantage",
            "auroc_advantage",
        }
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{path} misses columns: {sorted(missing)}")
        observed = set(frame["baseline"].dropna().astype(str))
        if observed != {label}:
            raise ValueError(
                f"score label {label!r} differs from file baseline(s) {sorted(observed)}"
            )
        audit_path = path.parent / "audit.json"
        if not audit_path.is_file():
            raise FileNotFoundError(f"missing score audit: {audit_path}")
        audit = json.loads(audit_path.read_text())
        if audit.get("status") != "complete":
            raise ValueError(f"score audit is not complete: {audit_path}")
        if str(audit.get("downstream_signal_access", "")).split(";")[0] != "none":
            raise ValueError(f"score run accessed downstream signals: {audit_path}")
        eligible = frame.loc[frame["is_eligible"].astype(bool)].copy()
        if eligible.empty:
            raise ValueError(f"score file has no eligible regions: {path}")
        frames.append(eligible)
        sources.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "sha256": sha256sum(path),
                "audit": str(audit_path.resolve()),
                "audit_sha256": sha256sum(audit_path),
            }
        )
    combined = pd.concat(frames, ignore_index=True)
    identity = ["region_id", "baseline", "context"]
    if combined.duplicated(identity).any():
        raise ValueError("score inputs duplicate a region/baseline/context identity")
    return combined, sources


def join_scores_and_complexity(
    scores: pd.DataFrame, complexity: pd.DataFrame
) -> pd.DataFrame:
    locus = complexity.loc[
        complexity["context"].astype(str).eq("strict"),
        [
            "chromosome", "start", "end", "locus_complexity_score",
            "locus_complexity_category",
        ],
    ].copy()
    if locus.duplicated(["chromosome", "start", "end"]).any():
        raise ValueError("strict complexity contains duplicate genomic loci")
    joined = scores.merge(
        locus,
        on=["chromosome", "start", "end"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[~joined["_merge"].eq("both"), "region_id"].tolist()
        raise ValueError(f"eligible score regions lack frozen complexity: {missing[:10]}")
    joined = joined.drop(columns="_merge")
    observed_columns = [
        column
        for column in [
            "slice_id", "context", "observed_context_complexity_score",
            "node_exposure_ratio_to_reference_context",
            "edge_exposure_ratio_to_reference_context",
        ]
        if column in complexity
    ]
    if {"slice_id", "context"}.issubset(observed_columns):
        observed = complexity[observed_columns].rename(
            columns={"slice_id": "region_id"}
        )
        joined = joined.merge(
            observed,
            on=["region_id", "context"],
            how="left",
            validate="many_to_one",
        )
    joined["locus_complexity_category"] = pd.Categorical(
        joined["locus_complexity_category"],
        categories=CATEGORY_ORDER,
        ordered=True,
    )
    if joined["locus_complexity_category"].isna().any():
        raise ValueError("complexity categories must be low/medium/high")
    return joined


def summarize_groups(
    frame: pd.DataFrame,
    group_columns: list[str],
    *,
    n_bootstrap: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, observed=True, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, keys))
        for metric in METRICS:
            summary = chromosome_block_summary(
                group, metric, n_bootstrap=n_bootstrap, rng=rng
            )
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n_regions": int(len(group)),
                    "region_mean": float(group[metric].mean()),
                    "region_median": float(group[metric].median()),
                    **summary,
                    "chromosome_sign_flip_p_two_sided": sign_flip_pvalue(
                        group,
                        metric,
                        n_permutations=n_permutations,
                        rng=rng,
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_context_rows(frame: pd.DataFrame) -> pd.DataFrame:
    index = [
        "baseline", "chromosome", "start", "end", "fold",
        "locus_complexity_category", "locus_complexity_score",
    ]
    value_columns = METRICS + [
        column
        for column in [
            "node_exposure_ratio_to_reference_context",
            "edge_exposure_ratio_to_reference_context",
        ]
        if column in frame
    ]
    pivot = frame.pivot_table(
        index=index,
        columns="context",
        values=value_columns,
        aggfunc="first",
        observed=True,
    )
    required = {(metric, context) for metric in METRICS for context in CONTEXT_ORDER}
    if not required.issubset(set(pivot.columns)):
        return pd.DataFrame()
    paired = pivot.reset_index()
    paired.columns = [
        "__".join(map(str, column)).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in paired.columns
    ]
    for metric in METRICS:
        paired[f"{metric}__expanded_minus_strict"] = (
            paired[f"{metric}__1hop"] - paired[f"{metric}__strict"]
        )
    return paired


def make_figure(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    baselines = list(dict.fromkeys(summary["baseline"].astype(str)))
    colors = {"strict": "#355C7D", "1hop": "#C06C84"}
    metrics = [
        ("auprc_advantage", "PangenomeFM − baseline AUPRC"),
        ("model_score", "Baseline NLL − PangenomeFM NLL"),
    ]
    fig, axes = plt.subplots(
        len(metrics),
        len(baselines),
        figsize=(3.25 * len(baselines), 6.2),
        sharex=True,
        squeeze=False,
    )
    x = np.arange(len(CATEGORY_ORDER), dtype=float)
    for column, baseline in enumerate(baselines):
        for row, (metric, ylabel) in enumerate(metrics):
            axis = axes[row, column]
            subset = summary.loc[
                summary["baseline"].astype(str).eq(baseline)
                & summary["metric"].eq(metric)
            ]
            for offset, context in zip([-0.06, 0.06], CONTEXT_ORDER):
                context_rows = (
                    subset.loc[subset["context"].astype(str).eq(context)]
                    .set_index("locus_complexity_category")
                    .reindex(CATEGORY_ORDER)
                )
                values = context_rows["chromosome_block_mean"].to_numpy(float)
                lower = values - context_rows["ci95_low"].to_numpy(float)
                upper = context_rows["ci95_high"].to_numpy(float) - values
                axis.errorbar(
                    x + offset,
                    values,
                    yerr=np.vstack([lower, upper]),
                    marker="o",
                    linewidth=1.8,
                    capsize=3,
                    color=colors[context],
                    label=context,
                )
            axis.axhline(0.0, color="#444444", linewidth=0.8, linestyle="--")
            axis.grid(axis="y", color="#dddddd", linewidth=0.6)
            axis.set_xticks(x, CATEGORY_ORDER)
            if row == 0:
                axis.set_title(baseline.replace("_", " "), fontsize=9)
            if column == 0:
                axis.set_ylabel(ylabel)
            if row == len(metrics) - 1:
                axis.set_xlabel("Frozen native-window complexity")
    axes[0, -1].legend(frameon=False, loc="best")
    fig.suptitle(
        "Exact same-example performance by graph complexity and context",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    paths = [
        out_dir / "figure4_complexity_context.png",
        out_dir / "figure4_complexity_context.pdf",
        out_dir / "figure4_complexity_context.svg",
    ]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complexity", type=Path, required=True)
    parser.add_argument(
        "--score", action="append", type=parse_score_spec, required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--require-baselines", nargs="*")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.out_dir}")
    if args.n_bootstrap < 100 or args.n_permutations < 100:
        parser.error("bootstrap and permutation counts must each be at least 100")

    started = time.monotonic()
    complexity, complexity_audit = load_complexity(args.complexity)
    scores, sources = load_scores(args.score)
    observed_baselines = set(scores["baseline"].astype(str))
    required_baselines = set(args.require_baselines or [])
    if not required_baselines.issubset(observed_baselines):
        raise ValueError(
            f"missing required baselines: {sorted(required_baselines - observed_baselines)}"
        )
    combined = join_scores_and_complexity(scores, complexity)
    rng = np.random.default_rng(args.seed)
    stratum_summary = summarize_groups(
        combined,
        ["baseline", "context", "locus_complexity_category"],
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        rng=rng,
    )
    fold_summary = (
        combined.groupby(
            ["baseline", "context", "locus_complexity_category", "fold"],
            observed=True,
            sort=True,
        )[METRICS]
        .agg(["count", "mean", "median"])
    )
    fold_summary.columns = ["__".join(column) for column in fold_summary.columns]
    fold_summary = fold_summary.reset_index()
    chromosome_summary = (
        combined.groupby(
            ["baseline", "context", "locus_complexity_category", "chromosome"],
            observed=True,
            sort=True,
        )[METRICS]
        .agg(["count", "mean", "median"])
    )
    chromosome_summary.columns = [
        "__".join(column) for column in chromosome_summary.columns
    ]
    chromosome_summary = chromosome_summary.reset_index()
    paired = paired_context_rows(combined)
    paired_metric_columns = [
        f"{metric}__expanded_minus_strict" for metric in METRICS
    ]
    paired_for_summary = paired.rename(
        columns={
            "baseline__": "baseline",
            "chromosome__": "chromosome",
            "locus_complexity_category__": "locus_complexity_category",
            **{
                column: metric
                for column, metric in zip(paired_metric_columns, METRICS)
            },
        }
    )
    paired_summary = (
        summarize_groups(
            paired_for_summary,
            ["baseline", "locus_complexity_category"],
            n_bootstrap=args.n_bootstrap,
            n_permutations=args.n_permutations,
            rng=rng,
        )
        if not paired.empty
        else pd.DataFrame()
    )

    args.out_dir.mkdir(parents=True)
    combined.to_csv(
        args.out_dir / "region_level_exact_comparisons.csv.gz",
        index=False,
        compression="gzip",
    )
    stratum_summary.to_csv(args.out_dir / "complexity_stratum_summary.csv", index=False)
    fold_summary.to_csv(args.out_dir / "fold_summary.csv", index=False)
    chromosome_summary.to_csv(args.out_dir / "chromosome_summary.csv", index=False)
    paired.to_csv(args.out_dir / "paired_context_region_deltas.csv", index=False)
    paired_summary.to_csv(args.out_dir / "paired_context_summary.csv", index=False)
    figure_paths = make_figure(stratum_summary, args.out_dir)
    output_paths = [
        args.out_dir / "region_level_exact_comparisons.csv.gz",
        args.out_dir / "complexity_stratum_summary.csv",
        args.out_dir / "fold_summary.csv",
        args.out_dir / "chromosome_summary.csv",
        args.out_dir / "paired_context_region_deltas.csv",
        args.out_dir / "paired_context_summary.csv",
        *figure_paths,
    ]
    audit = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(),
        "complexity": str(args.complexity.resolve()),
        "complexity_sha256": sha256sum(args.complexity),
        "complexity_audit": complexity_audit,
        "score_sources": sources,
        "baselines": sorted(observed_baselines),
        "contexts": sorted(combined["context"].astype(str).unique()),
        "eligible_region_rows": int(len(combined)),
        "paired_context_rows": int(len(paired)),
        "complexity_categories": CATEGORY_ORDER,
        "inference_unit": "chromosome",
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "primary_metrics": {
            "auprc_advantage": "PangenomeFM AUPRC minus exact baseline AUPRC",
            "model_score": "baseline NLL minus calibrated PangenomeFM NLL",
        },
        "multiplicity_note": (
            "Intervals and sign-flip p-values are descriptive across prespecified "
            "baseline/context/complexity strata; no edge-level independence is assumed."
        ),
        "downstream_signal_access": "none",
        "wall_seconds": time.monotonic() - started,
    }
    audit_path = args.out_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    with (args.out_dir / "SHA256SUMS").open("w") as handle:
        for path in [*output_paths, audit_path]:
            handle.write(f"{sha256sum(path)}  {path.name}\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
