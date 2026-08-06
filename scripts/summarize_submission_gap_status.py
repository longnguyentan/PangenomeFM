from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "results" / "submission_gap_status.csv"
OUT_MD = ROOT / "results" / "submission_gap_status.md"


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open() as fh:
        return json.load(fh)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def _latest_summary(base: Path) -> Path | None:
    summaries = sorted(base.glob("run_*/summary.json"))
    return summaries[-1] if summaries else None


def _latest_pretrain_run(base: Path) -> Path | None:
    if not base.exists():
        return None
    for run_dir in sorted((p for p in base.glob("run_*") if p.is_dir()), reverse=True):
        if any(run_dir.glob("strict_results__*.csv")) and any(run_dir.glob("1hop_results__*.csv")):
            return run_dir
    return None


def _metric_from_test_metrics(metrics: dict[str, Any]) -> tuple[str, float | None, str]:
    if "auroc" in metrics:
        return (
            "AUROC",
            metrics.get("auroc"),
            (
                f"AUPRC={_fmt(metrics.get('auprc'))}; "
                f"macro_F1={_fmt(metrics.get('macro_f1'))}; "
                f"balanced_accuracy={_fmt(metrics.get('balanced_accuracy'))}"
            ),
        )
    return (
        "macro_F1",
        metrics.get("macro_f1"),
        f"weighted_F1={_fmt(metrics.get('weighted_f1'))}",
    )


def _add_ccre(rows: list[dict[str, Any]]) -> None:
    for base in sorted((ROOT / "results" / "hprc").glob("ccre_safe_*")):
        summary_path = _latest_summary(base)
        if summary_path is None:
            continue
        summary = _load_json(summary_path)
        if not isinstance(summary, dict):
            continue
        metrics = summary.get("test_metrics", {})
        metric, value, secondary = _metric_from_test_metrics(metrics)
        label = summary.get("label", {})
        if isinstance(label, dict):
            scheme = label.get("scheme", "")
        else:
            scheme = str(label)
        args = summary.get("args", {}) if isinstance(summary.get("args"), dict) else {}
        feature_policy = summary.get("feature_policy") or args.get("feature_policy", "")
        model = summary.get("method") or base.name.replace("ccre_safe_", "")
        if base.name.startswith("ccre_safe_gat_"):
            model = base.name.replace("ccre_safe_", "")
        notes = []
        if feature_policy == "leakage_safe":
            notes.append("SR/is_grch38 excluded")
        elif feature_policy:
            notes.append(f"feature_policy={feature_policy}")
        if summary.get("deepgene_style"):
            notes.append("DeepGene-style serialized baseline; no graph edges consumed")
        rows.append(
            {
                "area": "ccre",
                "requirement": "fair comparison and leakage control",
                "experiment": base.name,
                "model": model,
                "eval_universe": summary.get("evaluation_universe", "benchmark_windows"),
                "label_scheme": scheme,
                "primary_metric": metric,
                "primary_value": value,
                "secondary_metrics": secondary,
                "n_train": summary.get("n_train") or summary.get("n_train_slices", ""),
                "n_val": summary.get("n_val") or summary.get("n_val_slices", ""),
                "n_test": summary.get("n_test") or summary.get("n_test_slices", ""),
                "run": _rel(summary_path.parent),
                "notes": "; ".join(notes),
            }
        )


def _pretrain_rows_for_run(run_dir: Path, experiment: str, notes: str) -> list[dict[str, Any]]:
    out = []
    for closure in ["strict", "1hop"]:
        matches = sorted(run_dir.glob(f"{closure}_results__*.csv"))
        if not matches:
            continue
        df = pd.read_csv(matches[-1])
        heldout = df[df["split"].astype(str) == "heldout_chr"]
        if heldout.empty:
            continue
        out.append(
            {
                "area": "link_prediction",
                "requirement": "hard-negative robustness",
                "experiment": experiment,
                "model": "GraphGenome-FM",
                "eval_universe": closure,
                "label_scheme": "",
                "primary_metric": "heldout_AUROC",
                "primary_value": float(heldout["test_auc"].mean()),
                "secondary_metrics": f"n_slices={len(heldout)}; n_pos={int(heldout['n_pos'].sum())}",
                "n_train": "",
                "n_val": "",
                "n_test": len(heldout),
                "run": _rel(run_dir),
                "notes": notes,
            }
        )
    return out


def _add_link_robustness(rows: list[dict[str, Any]]) -> None:
    specs = [
        (
            ROOT / "results" / "hprc" / "pretrain_paper_degree_matched",
            "Degree-matched distance-negative HPRC pretraining",
            "degree-matched negatives",
        ),
        (
            ROOT / "results" / "hprc" / "pretrain_paper_hardneg_nonoverlap",
            "Non-overlapping hard-negative HPRC pretraining",
            "non-overlapping sampled windows",
        ),
    ]
    for base, experiment, notes in specs:
        run_dir = _latest_pretrain_run(base)
        if run_dir:
            rows.extend(_pretrain_rows_for_run(run_dir, experiment, notes))

    seed_values: dict[str, list[tuple[int, float, Path]]] = {"strict": [], "1hop": []}
    for base in sorted((ROOT / "results" / "hprc").glob("pretrain_paper_hardneg_seed_*")):
        match = re.search(r"seed_(\d+)$", base.name)
        seed = int(match.group(1)) if match else -1
        run_dir = _latest_pretrain_run(base)
        if not run_dir:
            continue
        for closure in ["strict", "1hop"]:
            matches = sorted(run_dir.glob(f"{closure}_results__*.csv"))
            if not matches:
                continue
            df = pd.read_csv(matches[-1])
            heldout = df[df["split"].astype(str) == "heldout_chr"]
            if not heldout.empty:
                seed_values[closure].append((seed, float(heldout["test_auc"].mean()), run_dir))
    for closure, vals in seed_values.items():
        if not vals:
            continue
        scores = [v for _, v, _ in vals]
        rows.append(
            {
                "area": "link_prediction",
                "requirement": "seed variance",
                "experiment": "HPRC hard-negative seed variance",
                "model": "GraphGenome-FM",
                "eval_universe": closure,
                "label_scheme": "",
                "primary_metric": "heldout_AUROC_mean",
                "primary_value": float(sum(scores) / len(scores)),
                "secondary_metrics": (
                    f"sd={_fmt(float(pd.Series(scores).std(ddof=1)) if len(scores) > 1 else 0.0)}; "
                    f"seeds={','.join(str(s) for s, _, _ in vals)}"
                ),
                "n_train": "",
                "n_val": "",
                "n_test": len(vals),
                "run": "; ".join(_rel(p) for _, _, p in vals),
                "notes": "mean and sd across completed seed reruns",
            }
        )


def _add_masked_link_eval(rows: list[dict[str, Any]]) -> None:
    specs = [
        (
            ROOT / "results" / "hprc" / "pretrain_paper_hardneg_masked_eval_strict",
            "HPRC held-out masked-query inference",
            "strict",
        ),
        (
            ROOT / "results" / "hprc" / "pretrain_paper_hardneg_masked_eval_1hop",
            "HPRC held-out masked-query inference",
            "1hop",
        ),
        (
            ROOT / "results" / "hgsvc3" / "external_eval_hardneg_masked_strict",
            "HGSVC masked-query transfer",
            "strict",
        ),
        (
            ROOT / "results" / "hgsvc3" / "external_eval_hardneg_masked_1hop",
            "HGSVC masked-query transfer",
            "1hop",
        ),
    ]
    for base, experiment, closure in specs:
        summary_path = _latest_summary(base)
        if summary_path is None:
            continue
        summary = _load_json(summary_path)
        if not isinstance(summary, dict):
            continue
        metrics = summary.get("pooled_metrics", {})
        rows.append(
            {
                "area": "link_prediction",
                "requirement": "query-edge masking leakage audit",
                "experiment": experiment,
                "model": "GraphGenome-FM checkpoint",
                "eval_universe": closure,
                "label_scheme": "",
                "primary_metric": "AUROC",
                "primary_value": metrics.get("auroc"),
                "secondary_metrics": (
                    f"AUPRC={_fmt(metrics.get('auprc'))}; "
                    f"Brier={_fmt(metrics.get('brier'))}; "
                    f"n_edges={metrics.get('n_edges', '')}"
                ),
                "n_train": "",
                "n_val": "",
                "n_test": summary.get("n_slices", ""),
                "run": _rel(summary_path.parent),
                "notes": "positive query edges removed during message passing at inference",
            }
        )


def _add_link_heuristics(rows: list[dict[str, Any]]) -> None:
    summary_path = _latest_summary(ROOT / "results" / "hprc" / "link_heuristics_hardneg")
    if summary_path is None:
        return
    summary = _load_json(summary_path)
    if not isinstance(summary, dict):
        return
    for item in summary.get("summary", []):
        rows.append(
            {
                "area": "link_prediction",
                "requirement": "standard topology-only baselines",
                "experiment": "HPRC hard-negative heuristic baseline",
                "model": item.get("heuristic", ""),
                "eval_universe": item.get("closure", ""),
                "label_scheme": "",
                "primary_metric": "pooled_AUROC",
                "primary_value": item.get("pooled_auroc"),
                "secondary_metrics": (
                    f"pooled_AUPRC={_fmt(item.get('pooled_auprc'))}; "
                    f"mean_AUROC={_fmt(item.get('mean_auroc'))}; "
                    f"n_edges={item.get('n_edges', '')}"
                ),
                "n_train": "",
                "n_val": "",
                "n_test": item.get("n_slices", ""),
                "run": _rel(summary_path.parent),
                "notes": "positive query edges masked before heuristic scoring",
            }
        )


def _add_imputation(rows: list[dict[str, Any]]) -> None:
    base = ROOT / "results" / "hgsvc3" / "graph_imputation_1hop"
    summary_path = _latest_summary(base)
    if summary_path is None:
        return
    summary = _load_json(summary_path)
    if not isinstance(summary, dict):
        return
    cal = summary.get("calibration_metrics_on_edge_pred", {})
    validated = "comparison_precision_at_k_unique" in summary
    rows.append(
        {
            "area": "hgsvc_imputation",
            "requirement": "newer-build validation",
            "experiment": "HGSVC candidate-edge scoring",
            "model": "Frozen HPRC one-hop checkpoint",
            "eval_universe": summary.get("closure", ""),
            "label_scheme": "",
            "primary_metric": "calibration_AUROC" if not validated else "precision@100_unique",
            "primary_value": cal.get("auroc") if not validated else summary["comparison_precision_at_k_unique"].get("100"),
            "secondary_metrics": (
                f"n_candidates={summary.get('n_candidates')}; "
                f"n_unique_pairs={summary.get('n_unique_pairs', '')}; "
                f"AUPRC={_fmt(cal.get('auprc'))}; Brier={_fmt(cal.get('brier'))}"
            ),
            "n_train": "",
            "n_val": "",
            "n_test": cal.get("n", ""),
            "run": _rel(summary_path.parent),
            "notes": "validated against comparison graph" if validated else "candidate scoring only; set HGSVC_NEW_SEGMENTS/HGSVC_NEW_LINKS to validate",
        }
    )


def _add_reliability(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "results" / "reliability_curves" / "summary.json"
    if not path.exists():
        return
    summaries = _load_json(path)
    if not isinstance(summaries, list):
        return
    for item in summaries:
        if not isinstance(item, dict) or item.get("status") != "ok":
            continue
        rows.append(
            {
                "area": "calibration",
                "requirement": "reliability curves",
                "experiment": item.get("label", ""),
                "model": "edge-probability calibration",
                "eval_universe": "",
                "label_scheme": "",
                "primary_metric": "Brier",
                "primary_value": item.get("brier"),
                "secondary_metrics": f"n={item.get('n')}; positive_fraction={_fmt(item.get('positive_fraction'))}",
                "n_train": "",
                "n_val": "",
                "n_test": item.get("n", ""),
                "run": "results/reliability_curves",
                "notes": "bins in reliability_bins.csv; plot in reliability_curve.pdf/png",
            }
        )


def build_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    _add_ccre(rows)
    _add_imputation(rows)
    _add_link_robustness(rows)
    _add_masked_link_eval(rows)
    _add_link_heuristics(rows)
    _add_reliability(rows)
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    def clean(value: Any) -> str:
        return _fmt(value).replace("|", "\\|").replace("\n", " ")

    headers = list(df.columns)
    data = [[clean(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [
        max(len(clean(header)), *(len(row[i]) for row in data)) if data else len(clean(header))
        for i, header in enumerate(headers)
    ]
    header = "| " + " | ".join(clean(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in data]
    return "\n".join([header, sep, *body])


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    with path.open("w") as fh:
        fh.write("# Submission Gap Status\n\n")
        fh.write("Generated from local artifacts. Missing requirements simply do not appear here.\n\n")
        if df.empty:
            fh.write("No submission-gap artifacts found yet.\n")
            return
        display = df.copy()
        display["primary_value"] = display["primary_value"].map(_fmt)
        cols = [
            "area",
            "requirement",
            "experiment",
            "model",
            "eval_universe",
            "primary_metric",
            "primary_value",
            "secondary_metrics",
            "run",
            "notes",
        ]
        fh.write(_markdown_table(display[cols]))
        fh.write("\n")


def main() -> int:
    df = build_rows()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    write_markdown(df, OUT_MD)
    print(f"wrote {_rel(OUT_CSV)}")
    print(f"wrote {_rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
