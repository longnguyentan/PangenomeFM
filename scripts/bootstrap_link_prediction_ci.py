#!/usr/bin/env python
"""Bootstrap confidence intervals for link-prediction prediction files.

Inputs are pooled prediction CSV/CSV.GZ files with at least ``y_true`` and
``p_edge`` columns. If a ``slice`` column is available, the default bootstrap
unit is the slice; otherwise rows are resampled directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _parse_input(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.parent.name, path


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {
        "n": int(len(y)),
        "positive_fraction": float(np.mean(y)) if len(y) else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
    }
    if len(y) and len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def _bootstrap(
    df: pd.DataFrame,
    *,
    unit: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    if unit == "slice" and "slice" in df.columns:
        groups = {name: sub.index.to_numpy() for name, sub in df.groupby("slice", sort=False)}
        keys = np.array(list(groups.keys()), dtype=object)
        for i in range(n_boot):
            sampled = rng.choice(keys, size=len(keys), replace=True)
            idx = np.concatenate([groups[k] for k in sampled])
            m = _metrics(
                df.loc[idx, "y_true"].to_numpy(np.float32),
                df.loc[idx, "p_edge"].to_numpy(np.float32),
            )
            m["bootstrap"] = i
            rows.append(m)
    else:
        n = len(df)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            m = _metrics(
                df.iloc[idx]["y_true"].to_numpy(np.float32),
                df.iloc[idx]["p_edge"].to_numpy(np.float32),
            )
            m["bootstrap"] = i
            rows.append(m)
    return pd.DataFrame(rows)


def _summarize_bootstrap(label: str, point: dict[str, float], boot: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {"label": label, "point": point, "ci": {}}
    for metric in ["auroc", "auprc", "brier", "positive_fraction"]:
        vals = boot[metric].dropna().to_numpy(float)
        if len(vals) == 0:
            continue
        summary["ci"][metric] = {
            "mean": float(np.mean(vals)),
            "lower": float(np.quantile(vals, 0.025)),
            "upper": float(np.quantile(vals, 0.975)),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Prediction files, optionally labeled as label=path.",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unit", choices=["auto", "slice", "row"], default="auto")
    ap.add_argument("--filter-column", default=None)
    ap.add_argument("--filter-values", nargs="+", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for raw in args.inputs:
        label, path = _parse_input(raw)
        df = pd.read_csv(path, compression="infer")
        if "y_true" not in df.columns or "p_edge" not in df.columns:
            raise ValueError(f"{path} must contain y_true and p_edge columns")
        if args.filter_column:
            if args.filter_column not in df.columns:
                raise ValueError(f"{path} does not contain filter column {args.filter_column!r}")
            allowed = set(args.filter_values or [])
            df = df[df[args.filter_column].astype(str).isin(allowed)].reset_index(drop=True)
            if df.empty:
                raise ValueError(
                    f"{path} has no rows where {args.filter_column} is in {sorted(allowed)}"
                )
        boot_unit = "slice" if args.unit == "auto" and "slice" in df.columns else args.unit
        if boot_unit == "auto":
            boot_unit = "row"
        point = _metrics(df["y_true"].to_numpy(np.float32), df["p_edge"].to_numpy(np.float32))
        boot = _bootstrap(df, unit=boot_unit, n_boot=args.n_boot, seed=args.seed)
        boot.insert(0, "label", label)
        safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
        boot.to_csv(out_dir / f"{safe_label}_bootstrap.csv", index=False)
        summary = _summarize_bootstrap(label, point, boot)
        summary["input"] = str(path)
        summary["bootstrap_unit"] = boot_unit
        summary["n_boot"] = int(args.n_boot)
        summaries.append(summary)
        row = {
            "label": label,
            "input": str(path),
            "bootstrap_unit": boot_unit,
            "n": point["n"],
            "positive_fraction": point["positive_fraction"],
        }
        for metric in ["auroc", "auprc", "brier"]:
            row[metric] = point.get(metric, float("nan"))
            ci = summary["ci"].get(metric, {})
            row[f"{metric}_ci_lower"] = ci.get("lower", float("nan"))
            row[f"{metric}_ci_upper"] = ci.get("upper", float("nan"))
        summary_rows.append(row)

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
