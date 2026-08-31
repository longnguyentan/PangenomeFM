#!/usr/bin/env python3
"""Create manuscript-ready CSV/LaTeX tables for resolved analysis gaps."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def latex_escape(value: object) -> str:
    text = str(value)
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ]:
        text = text.replace(old, new)
    return text


def write_latex_table(
    frame: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
    columns: list[str] | None = None,
) -> None:
    table = frame[columns] if columns else frame
    align = "l" + "r" * (len(table.columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{\textbf{{{caption}}}}}",
        rf"\label{{{label}}}",
        r"\small",
        rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}",
        r"\toprule",
        " & ".join(latex_escape(column) for column in table.columns) + r" \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines))


def split_summary(path: Path, task: str) -> dict[str, object]:
    frame = pd.read_csv(path)
    runs = (
        frame.loc[frame["scope"].eq("all_test_chromosomes")]
        .sort_values(["fold", "seed", "closure", "feature_set"])
        .drop_duplicates(["fold", "seed", "closure"])
    )
    return {
        "Task": task,
        "Runs": int(len(runs)),
        "Training mean": f'{runs["n_train"].mean():,.0f}',
        "Training range": f'{runs["n_train"].min():,}–{runs["n_train"].max():,}',
        "Validation mean": f'{runs["n_validation"].mean():,.0f}',
        "Validation range": f'{runs["n_validation"].min():,}–{runs["n_validation"].max():,}',
        "Test mean": f'{runs["n_test"].mean():,.0f}',
        "Test range": f'{runs["n_test"].min():,}–{runs["n_test"].max():,}',
    }


def capacity_table(old_path: Path, principal_path: Path | None) -> pd.DataFrame:
    old = pd.read_csv(old_path)
    old = old.loc[(old["metric_scope"] == "split") & (old["split"] == "heldout_chr_test")]
    definitions = [
        ("Diagnostic small", "hprc_r2_capacity_tiny_h24_l1", 24, 1),
        ("Principal", "hprc_r2_capacity_principal_h48_l2", 48, 2),
        ("Diagnostic medium", "hprc_r2_capacity_medium_h96_l4", 96, 4),
        ("Diagnostic large", "hprc_r2_capacity_large_h192_l6", 192, 6),
    ]
    principal = pd.read_csv(principal_path) if principal_path and principal_path.is_file() else pd.DataFrame()
    rows = []
    for name, regime, hidden, layers in definitions:
        source = principal if regime.endswith("principal_h48_l2") else old
        values = source.loc[
            source.get("regime", pd.Series(dtype=str)).eq(regime)
            & source.get("metric_scope", pd.Series(dtype=str)).eq("split")
            & source.get("split", pd.Series(dtype=str)).eq("heldout_chr_test"),
            "auprc",
        ].dropna().to_numpy(float) if not source.empty else np.array([])
        rows.append(
            {
                "Configuration": name,
                "Hidden units": hidden,
                "Graph layers": layers,
                "Runs": int(len(values)),
                "Mean AUPRC": f"{values.mean():.4f}" if len(values) else "pending",
                "SD": f"{values.std(ddof=1):.4f}" if len(values) > 1 else ("0.0000" if len(values) else "pending"),
            }
        )
    return pd.DataFrame(rows)


def ccre_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame(
            [
                {
                    "Stratification": "server analysis pending",
                    "Stratum": "—",
                    "Context": "—",
                    "Graph ΔAUPRC": "pending",
                    "95% CI": "pending",
                    "Mean positive examples/run": "pending",
                }
            ]
        )
    frame = pd.read_csv(path)
    labels = {
        "ENCODE_cCRE_subtype_vs_background": "ENCODE class vs background",
        "native_graph_complexity_tertile": "Native-graph complexity",
    }
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "Stratification": labels.get(row.stratification, row.stratification),
                "Stratum": row.stratum,
                "Context": "Strict" if row.closure == "strict" else "One-hop",
                "Graph ΔAUPRC": f"{row.mean_gain:.4f}",
                "95% CI": f"({row.ci95_low:.4f}, {row.ci95_high:.4f})",
                "Mean positive examples/run": f"{row.mean_positives_per_run:,.1f}",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccre-strata", type=Path)
    parser.add_argument("--prevalence-summary", type=Path, required=True)
    parser.add_argument("--complexity-thresholds", type=Path, required=True)
    parser.add_argument("--complexity-features", type=Path, required=True)
    parser.add_argument("--ccre-fold-metrics", type=Path, required=True)
    parser.add_argument("--sv-fold-metrics", type=Path, required=True)
    parser.add_argument("--old-capacity-runs", type=Path, required=True)
    parser.add_argument("--principal-capacity-metrics", type=Path)
    parser.add_argument("--visible-graph-audit", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prevalence = pd.read_csv(args.prevalence_summary)
    chance = prevalence[
        [
            "analysis_scope",
            "analysis_name",
            "closure",
            "runs",
            "total_targets",
            "pooled_positive_fraction",
            "chance_auprc",
        ]
    ].copy()
    chance.columns = [
        "Analysis",
        "Resource or transfer",
        "Context",
        "Runs",
        "Targets",
        "Positive fraction",
        "Chance AUPRC",
    ]
    chance["Context"] = chance["Context"].replace({"strict": "Strict", "1hop": "One-hop"})
    chance["Positive fraction"] = chance["Positive fraction"].map(lambda x: f"{x:.4f}")
    chance["Chance AUPRC"] = chance["Chance AUPRC"].map(lambda x: f"{x:.4f}")
    chance.to_csv(args.out_dir / "table_link_prediction_chance.csv", index=False)
    write_latex_table(
        chance,
        args.out_dir / "table_link_prediction_chance.tex",
        caption="Candidate prevalence and chance-level AUPRC for reconstruction and transfer.",
        label="tab:s-chance-auprc",
    )

    ccre = ccre_table(args.ccre_strata)
    ccre.to_csv(args.out_dir / "table_ccre_stratification.csv", index=False)
    write_latex_table(
        ccre,
        args.out_dir / "table_ccre_stratification.tex",
        caption="Graph contribution to cCRE classification across ENCODE classes and native-graph complexity.",
        label="tab:s-ccre-strata",
    )
    if args.ccre_strata and args.ccre_strata.is_file():
        interaction_path = args.ccre_strata.with_name(
            "ccre_stratum_gain_interactions.csv"
        )
        if interaction_path.is_file():
            interaction_source = pd.read_csv(interaction_path)
            interactions = pd.DataFrame(
                {
                    "Stratification": interaction_source["stratification"].replace(
                        {
                            "ENCODE_cCRE_subtype_vs_background": "ENCODE class vs background",
                            "native_graph_complexity_tertile": "Native-graph complexity",
                        }
                    ),
                    "Context": interaction_source["closure"].replace(
                        {"strict": "Strict", "1hop": "One-hop"}
                    ),
                    "Comparison": interaction_source["first_stratum"].astype(str)
                    + " minus "
                    + interaction_source["second_stratum"].astype(str),
                    "Difference in graph ΔAUPRC": interaction_source[
                        "mean_gain_difference"
                    ].map(lambda value: f"{value:.4f}"),
                    "95% CI": interaction_source.apply(
                        lambda row: f'({row["ci95_low"]:.4f}, {row["ci95_high"]:.4f})',
                        axis=1,
                    ),
                }
            )
            interactions.to_csv(
                args.out_dir / "table_ccre_gain_interactions.csv", index=False
            )
            write_latex_table(
                interactions,
                args.out_dir / "table_ccre_gain_interactions.tex",
                caption="Pairwise differences in cCRE graph contribution across prespecified strata.",
                label="tab:s-ccre-interactions",
            )

    thresholds = json.loads(args.complexity_thresholds.read_text())
    features = pd.read_csv(args.complexity_features, sep="\t")
    strict = features.loc[features["context"].astype(str).eq("strict")]
    counts = strict["locus_complexity_category"].value_counts()
    complexity = pd.DataFrame(
        [
            {
                "Category": "Low",
                "Score definition": f'score < {thresholds["thresholds"]["low_to_medium"]:.6f}',
                "Native 5-Mb regions": int(counts.get("low", 0)),
            },
            {
                "Category": "Medium",
                "Score definition": f'{thresholds["thresholds"]["low_to_medium"]:.6f} ≤ score < {thresholds["thresholds"]["medium_to_high"]:.6f}',
                "Native 5-Mb regions": int(counts.get("medium", 0)),
            },
            {
                "Category": "High",
                "Score definition": f'score ≥ {thresholds["thresholds"]["medium_to_high"]:.6f}',
                "Native 5-Mb regions": int(counts.get("high", 0)),
            },
        ]
    )
    complexity.to_csv(args.out_dir / "table_complexity_definition.csv", index=False)
    write_latex_table(
        complexity,
        args.out_dir / "table_complexity_definition.tex",
        caption="Native-graph complexity tertiles fixed independently of model performance.",
        label="tab:s-complexity-definition",
    )

    splits = pd.DataFrame(
        [
            split_summary(args.ccre_fold_metrics, "cCRE classification"),
            split_summary(args.sv_fold_metrics, "SV breakpoint classification"),
        ]
    )
    splits.to_csv(args.out_dir / "table_downstream_split_sizes.csv", index=False)
    write_latex_table(
        splits,
        args.out_dir / "table_downstream_split_sizes.tex",
        caption="Per-run downstream split sizes; validation and test counts are reported separately.",
        label="tab:s-downstream-splits",
    )

    capacity = capacity_table(args.old_capacity_runs, args.principal_capacity_metrics)
    capacity.to_csv(args.out_dir / "table_capacity_principal_comparison.csv", index=False)
    write_latex_table(
        capacity,
        args.out_dir / "table_capacity_principal_comparison.tex",
        caption="Diagnostic model-capacity comparison under strict graph context.",
        label="tab:s-capacity-complete",
    )

    methods = pd.DataFrame(
        [
            {
                "Item": "Fusion gate",
                "Archived implementation": "g = sigmoid(W[h_coord || h_graph] + b); h = g ⊙ h_coord + (1 − g) ⊙ h_graph",
                "Source": "src/models/dual_stream_gat.py: FusionGate",
            },
            {
                "Item": "Pair scorer",
                "Archived implementation": "MLP on concatenated endpoint embeddings [h_u || h_v], with 2d→2d→d→1 layers, layer normalization, ELU and dropout",
                "Source": "src/models/dual_stream_gat.py: edge_predictor",
            },
            {
                "Item": "Pretraining loss",
                "Archived implementation": "Binary focal cross-entropy; gamma=2, alpha=0.25, positive weight n_negative/n_positive within each candidate batch",
                "Source": "src/models/losses.py and src/training/pretrain.py",
            },
            {
                "Item": "Downstream classifier",
                "Archived implementation": "StandardScaler + balanced logistic regression (lbfgs, maximum 800 iterations); temperature and F1 threshold fitted on validation chromosomes only",
                "Source": "src/tasks/ccre/aligned_baselines.py and run_ccre_frozen_probe_fold.py",
            },
            {
                "Item": "Graph context terms",
                "Archived implementation": "Strict context = native 5-Mb region; one-hop context = region plus directly connected segments",
                "Source": "benchmark manifests and Methods definition",
            },
        ]
    )
    methods.to_csv(args.out_dir / "table_implementation_details.csv", index=False)
    write_latex_table(
        methods,
        args.out_dir / "table_implementation_details.tex",
        caption="Implementation details recovered directly from the archived code and configuration.",
        label="tab:s-implementation-details",
    )

    graph_status = "not provided"
    graph_details: dict[str, object] = {}
    if args.visible_graph_audit and args.visible_graph_audit.is_file():
        graph_details = json.loads(args.visible_graph_audit.read_text())
        graph_status = str(graph_details.get("status", "unknown"))
    resolution = pd.DataFrame(
        [
            {"Manuscript gap": "cCRE subtype / complexity", "Resolution": "server-stratified paired analysis", "Status": "complete" if args.ccre_strata and args.ccre_strata.is_file() else "pending server run"},
            {"Manuscript gap": "Chance AUPRC", "Resolution": "exact candidate prevalence", "Status": "complete"},
            {"Manuscript gap": "One-hop degree definition", "Resolution": "archived-score replay on closure-specific graphs", "Status": graph_status},
            {"Manuscript gap": "Principal 48/2 capacity point", "Resolution": "exact diagnostic sweep protocol", "Status": "complete" if args.principal_capacity_metrics and args.principal_capacity_metrics.is_file() else "pending server run"},
            {"Manuscript gap": "Complexity cutoffs / counts", "Resolution": "frozen native-graph tertiles", "Status": "complete"},
            {"Manuscript gap": "Classifier / split ambiguity", "Resolution": "code-derived specification and separate counts", "Status": "complete"},
        ]
    )
    resolution.to_csv(args.out_dir / "manuscript_gap_resolution.csv", index=False)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tables": sorted(path.name for path in args.out_dir.glob("table_*.csv")),
        "visible_graph_audit_status": graph_status,
        "visible_graph_audit_details": graph_details,
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
