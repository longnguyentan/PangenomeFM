from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "results" / "master_experiment_status.csv"
OUT_MD = ROOT / "results" / "master_experiment_status.md"


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        return json.load(fh)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _latest_run_dir(base: Path) -> Path | None:
    if not base.exists():
        return None
    runs = sorted(p for p in base.glob("run_*") if p.is_dir())
    return runs[-1] if runs else None


def _latest_summary(base: Path) -> Path | None:
    if not base.exists():
        return None
    summaries = sorted(base.glob("run_*/summary.json"))
    return summaries[-1] if summaries else None


def _latest_pretrain_run(base: Path) -> Path | None:
    if not base.exists():
        return None
    for run_dir in sorted((p for p in base.glob("run_*") if p.is_dir()), reverse=True):
        has_strict = any(run_dir.glob("strict_results__*.csv"))
        has_1hop = any(run_dir.glob("1hop_results__*.csv"))
        if has_strict and has_1hop:
            return run_dir
    return None


def _add_pretrain_run(
    rows: list[dict[str, Any]],
    run_dir: Path,
    experiment: str,
    notes: str,
) -> None:
    if not run_dir.exists():
        return
    for closure in ["strict", "1hop"]:
        matches = sorted(run_dir.glob(f"{closure}_results__*.csv"))
        if not matches:
            continue
        path = matches[-1]
        df = pd.read_csv(path)
        grouped = df.groupby("split")["test_auc"].mean()
        n_by_split = df.groupby("split")["name"].count()
        n_pos_by_split = df.groupby("split")["n_pos"].sum()
        rows.append(
            {
                "area": "link_prediction",
                "experiment": experiment,
                "model": "SharedDualStreamGAT",
                "closure": closure,
                "eval_set": "heldout chr1/chr8/chr19/chrY",
                "primary_metric": "AUROC",
                "primary_value": float(grouped.get("heldout_chr", float("nan"))),
                "secondary_metrics": f"val_chr16_AUROC={_fmt(float(grouped.get('val_chr', float('nan'))))}",
                "n_train": "",
                "n_val": int(n_by_split.get("val_chr", 0)),
                "n_test": int(n_by_split.get("heldout_chr", 0)),
                "n_edges_or_nodes": int(n_pos_by_split.get("heldout_chr", 0) * 2),
                "run": _rel(run_dir),
                "notes": notes,
            }
        )


def _add_pretrain(rows: list[dict[str, Any]]) -> None:
    _add_pretrain_run(
        rows,
        ROOT / "results" / "hprc" / "pretrain_paper" / "run_003",
        "HPRC shared pretraining",
        "Original benchmark edge_pred negatives were random.",
    )
    hardneg_run = _latest_pretrain_run(ROOT / "results" / "hprc" / "pretrain_paper_hardneg")
    if hardneg_run:
        _add_pretrain_run(
            rows,
            hardneg_run,
            "HPRC shared pretraining with distance-matched negatives",
            "Paper-safety rerun using distance-matched edge_pred negatives.",
        )


def _add_external_summary(
    rows: list[dict[str, Any]],
    path: Path,
    experiment: str,
    notes: str,
) -> None:
    if not path.exists():
        return
    summary = _load_json(path)
    metrics = summary["pooled_metrics"]
    targets = [str(r["target_sn"]).split("|")[-1] for r in summary.get("per_target_metrics", [])]
    target_label = ",".join(targets) if targets else "chr22"
    rows.append(
        {
            "area": "external_transfer",
            "experiment": experiment,
            "model": "SharedDualStreamGAT frozen checkpoint",
            "closure": summary.get("closure", ""),
            "eval_set": f"HGSVC3 CHM13 {target_label}",
            "primary_metric": "AUROC",
            "primary_value": metrics["auroc"],
            "secondary_metrics": f"AUPRC={_fmt(metrics['auprc'])}; Brier={_fmt(metrics['brier'])}",
            "n_train": "",
            "n_val": "",
            "n_test": int(summary["n_slices"]),
            "n_edges_or_nodes": int(metrics["n_edges"]),
            "run": _rel(path.parent),
            "notes": notes,
        }
    )


def _add_external(rows: list[dict[str, Any]]) -> None:
    _add_external_summary(
        rows,
        ROOT / "results" / "hgsvc3" / "external_eval_chr22_strict" / "run_001" / "summary.json",
        "HPRC checkpoint to HGSVC3 CHM13 chr22",
        "Original external graph transfer; chr22 only.",
    )
    _add_external_summary(
        rows,
        ROOT / "results" / "hgsvc3" / "external_eval_chr22_1hop" / "run_001" / "summary.json",
        "HPRC checkpoint to HGSVC3 CHM13 chr22",
        "Original external graph transfer; chr22 only.",
    )
    _add_external_summary(
        rows,
        _latest_summary(ROOT / "results" / "hgsvc3" / "external_eval_hardneg_strict")
        or ROOT / "results" / "hgsvc3" / "external_eval_hardneg_strict" / "run_001" / "summary.json",
        "Hard-negative HPRC checkpoint to HGSVC3",
        "Distance-matched external eval on expanded parsed HGSVC3 targets.",
    )
    _add_external_summary(
        rows,
        _latest_summary(ROOT / "results" / "hgsvc3" / "external_eval_hardneg_1hop")
        or ROOT / "results" / "hgsvc3" / "external_eval_hardneg_1hop" / "run_001" / "summary.json",
        "Hard-negative HPRC checkpoint to HGSVC3",
        "Distance-matched external eval on expanded parsed HGSVC3 targets.",
    )



def _add_ccre_mapping(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "data" / "hprc" / "ccre" / "run_003" / "summary.json"
    summary = _load_json(path)
    totals = summary["total_by_class"]
    class_note = "; ".join(f"{k}={v}" for k, v in totals.items())
    rows.append(
        {
            "area": "ccre_labels",
            "experiment": "ENCODE cCRE labels mapped to HPRC GRCh38 nodes",
            "model": "interval overlap labeling",
            "closure": "",
            "eval_set": "all GRCh38 walk nodes",
            "primary_metric": "non_background_fraction",
            "primary_value": summary["frac_labeled_nonbg"],
            "secondary_metrics": f"labeled_nodes={summary['n_grch38_walk_segments']}",
            "n_train": "",
            "n_val": "",
            "n_test": "",
            "n_edges_or_nodes": int(summary["n_grch38_walk_segments"]),
            "run": "data/hprc/ccre/run_003",
            "notes": class_note,
        }
    )


def _add_ccre_window_coverage(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "results" / "hprc" / "ccre_window_coverage" / "coverage_by_split_closure.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    test = df[(df["split"] == "test") & (df["closure"] == "strict_or_1hop")]
    if test.empty:
        return
    row = test.iloc[0]
    binary_path = ROOT / "results" / "hprc" / "ccre_binary_baseline" / "run_003" / "summary.json"
    n_logistic_test = ""
    if binary_path.exists():
        n_logistic_test = int(_load_json(binary_path)["n_test"])
    coverage = (
        float(row["unique_labeled_nodes"]) / float(n_logistic_test)
        if n_logistic_test
        else float("nan")
    )
    rows.append(
        {
            "area": "ccre_audit",
            "experiment": "cCRE benchmark-window coverage audit",
            "model": "coverage accounting",
            "closure": "strict_or_1hop",
            "eval_set": "heldout chr8/chr19/chr22 labeled nodes",
            "primary_metric": "window_coverage_of_logistic_test_nodes",
            "primary_value": coverage,
            "secondary_metrics": (
                f"window_unique_nodes={int(row['unique_labeled_nodes'])}; "
                f"logistic_test_nodes={n_logistic_test}; "
                f"labeled_occurrences={int(row['labeled_node_occurrences'])}"
            ),
            "n_train": "",
            "n_val": "",
            "n_test": int(row["n_slices"]),
            "n_edges_or_nodes": int(row["unique_labeled_nodes"]),
            "run": "results/hprc/ccre_window_coverage",
            "notes": "Quantifies current GAT/logistic cCRE evaluation mismatch.",
        }
    )


def _add_ccre_baselines(rows: list[dict[str, Any]]) -> None:
    binary_path = ROOT / "results" / "hprc" / "ccre_binary_baseline" / "run_003" / "summary.json"
    binary = _load_json(binary_path)
    bm = binary["test_metrics"]
    rows.append(
        {
            "area": "ccre_binary",
            "experiment": "cCRE vs non-cCRE baseline",
            "model": "logistic regression",
            "closure": "",
            "eval_set": "heldout chr8/chr19/chr22 nodes",
            "primary_metric": "AUROC",
            "primary_value": bm["auroc"],
            "secondary_metrics": (
                f"AUPRC={_fmt(bm['auprc'])}; macro_F1={_fmt(bm['macro_f1'])}; "
                f"balanced_accuracy={_fmt(bm['balanced_accuracy'])}"
            ),
            "n_train": int(binary["n_train"]),
            "n_val": int(binary["n_val"]),
            "n_test": int(binary["n_test"]),
            "n_edges_or_nodes": int(binary["n_test"]),
            "run": "results/hprc/ccre_binary_baseline/run_003",
            "notes": "All labeled GRCh38 nodes; strong practical baseline.",
        }
    )

    multi_path = ROOT / "results" / "hprc" / "ccre_baseline" / "run_003" / "summary.json"
    multi = _load_json(multi_path)
    rows.append(
        {
            "area": "ccre_multiclass",
            "experiment": "9-class cCRE baseline",
            "model": "logistic regression",
            "closure": "",
            "eval_set": "heldout chr8/chr19/chr22 nodes",
            "primary_metric": "macro_F1",
            "primary_value": multi["macro_f1_test"],
            "secondary_metrics": f"val_macro_F1={_fmt(multi['macro_f1_val'])}",
            "n_train": int(multi["n_train"]),
            "n_val": int(multi["n_val"]),
            "n_test": int(multi["n_test"]),
            "n_edges_or_nodes": int(multi["n_test"]),
            "run": "results/hprc/ccre_baseline/run_003",
            "notes": "All labeled GRCh38 nodes; comparison to current GAT is imperfect.",
        }
    )


def _add_ccre_gats(rows: list[dict[str, Any]]) -> None:
    specs = [
        ("scratch", ROOT / "results" / "hprc" / "ccre_gat_scratch" / "run_001" / "summary.json"),
        ("pretrained frozen", ROOT / "results" / "hprc" / "ccre_gat_pretrained_frozen" / "run_001" / "summary.json"),
        ("pretrained fine-tuned", ROOT / "results" / "hprc" / "ccre_gat_pretrained_finetune" / "run_001" / "summary.json"),
    ]
    for model_name, path in specs:
        if not path.exists():
            continue
        summary = _load_json(path)
        rows.append(
            {
                "area": "ccre_multiclass",
                "experiment": "9-class cCRE GAT",
                "model": model_name,
                "closure": "benchmark windows",
                "eval_set": "heldout chr8/chr19/chr22 slices",
                "primary_metric": "macro_F1",
                "primary_value": summary["test_macro_f1"],
                "secondary_metrics": f"best_val_macro_F1={_fmt(summary['best_val_macro_f1'])}",
                "n_train": int(summary["n_train_slices"]),
                "n_val": int(summary["n_val_slices"]),
                "n_test": int(summary["n_test_slices"]),
                "n_edges_or_nodes": "",
                "run": _rel(path.parent),
                "notes": "Window-node evaluation only; not yet aligned with all-node logistic baseline.",
            }
        )


def _add_ccre_binary_gats(rows: list[dict[str, Any]]) -> None:
    specs = [
        ("scratch", ROOT / "results" / "hprc" / "ccre_binary_gat_scratch" / "run_001" / "summary.json"),
        ("pretrained frozen", ROOT / "results" / "hprc" / "ccre_binary_gat_pretrained_frozen" / "run_001" / "summary.json"),
        ("pretrained fine-tuned", ROOT / "results" / "hprc" / "ccre_binary_gat_pretrained_finetune" / "run_001" / "summary.json"),
    ]
    for model_name, path in specs:
        if not path.exists():
            continue
        summary = _load_json(path)
        metrics = summary.get("test_metrics", {})
        rows.append(
            {
                "area": "ccre_binary",
                "experiment": "Binary cCRE GAT",
                "model": model_name,
                "closure": "benchmark windows",
                "eval_set": "heldout chr8/chr19/chr22 slices",
                "primary_metric": "AUROC",
                "primary_value": metrics.get("auroc", float("nan")),
                "secondary_metrics": (
                    f"AUPRC={_fmt(metrics.get('auprc'))}; "
                    f"macro_F1={_fmt(metrics.get('macro_f1'))}; "
                    f"balanced_accuracy={_fmt(metrics.get('balanced_accuracy'))}"
                ),
                "n_train": int(summary["n_train_slices"]),
                "n_val": int(summary["n_val_slices"]),
                "n_test": int(summary["n_test_slices"]),
                "n_edges_or_nodes": "",
                "run": _rel(path.parent),
                "notes": "Window-node evaluation; compare cautiously against all-node logistic baseline.",
            }
        )


def build_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    _add_pretrain(rows)
    _add_external(rows)
    _add_ccre_mapping(rows)
    _add_ccre_window_coverage(rows)
    _add_ccre_baselines(rows)
    _add_ccre_gats(rows)
    _add_ccre_binary_gats(rows)
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    def clean(value: Any) -> str:
        text = _fmt(value)
        return text.replace("|", "\\|").replace("\n", " ")

    headers = list(df.columns)
    rows = [[clean(v) for v in record] for record in df.itertuples(index=False, name=None)]
    widths = [
        max(len(clean(header)), *(len(row[i]) for row in rows)) if rows else len(clean(header))
        for i, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(clean(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    display = df.copy()
    display["primary_value"] = display["primary_value"].map(_fmt)
    cols = [
        "area",
        "experiment",
        "model",
        "closure",
        "eval_set",
        "primary_metric",
        "primary_value",
        "secondary_metrics",
        "run",
        "notes",
    ]
    with path.open("w") as fh:
        fh.write("# Master Experiment Status\n\n")
        fh.write("Generated from current local artifacts.\n\n")
        fh.write(_markdown_table(display[cols]))
        fh.write("\n")


def main() -> int:
    df = build_rows()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    write_markdown(df, OUT_MD)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
