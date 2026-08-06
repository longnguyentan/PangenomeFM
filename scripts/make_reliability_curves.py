from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


ROOT = Path(__file__).resolve().parents[1]


def _parse_input(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Input must be LABEL=PATH, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    return label, Path(raw_path)


def reliability_table(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    if "y_true" not in df.columns or "p_edge" not in df.columns:
        raise ValueError("Prediction file must contain y_true and p_edge columns.")
    y = df["y_true"].astype(float).to_numpy()
    p = df["p_edge"].astype(float).to_numpy()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, bins, right=False) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = which == b
        if not m.any():
            rows.append(
                {
                    "bin": b,
                    "bin_left": bins[b],
                    "bin_right": bins[b + 1],
                    "n": 0,
                    "mean_predicted": np.nan,
                    "observed_fraction": np.nan,
                }
            )
            continue
        rows.append(
            {
                "bin": b,
                "bin_left": bins[b],
                "bin_right": bins[b + 1],
                "n": int(m.sum()),
                "mean_predicted": float(p[m].mean()),
                "observed_fraction": float(y[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build link-prediction reliability curves from pooled predictions.")
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more LABEL=pooled_predictions.csv.gz specs.")
    ap.add_argument("--out-dir", default="results/figures/reliability")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument(
        "--split-filter",
        default=None,
        help="Optional value in a split column, e.g. heldout_chr_test for HPRC pretraining outputs.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_tables = []

    for spec in args.inputs:
        label, path = _parse_input(spec)
        path = path if path.exists() else ROOT / path
        if not path.exists():
            summaries.append({"label": label, "path": str(path), "status": "missing"})
            continue
        df = pd.read_csv(path, compression="infer")
        if args.split_filter and "split" in df.columns:
            df = df[df["split"].astype(str) == args.split_filter]
        if df.empty:
            summaries.append({"label": label, "path": str(path), "status": "empty"})
            continue

        table = reliability_table(df, args.n_bins)
        table.insert(0, "label", label)
        all_tables.append(table)
        y = df["y_true"].astype(float).to_numpy()
        p = df["p_edge"].astype(float).to_numpy()
        summaries.append(
            {
                "label": label,
                "path": str(path),
                "status": "ok",
                "n": int(len(df)),
                "positive_fraction": float(y.mean()),
                "brier": float(brier_score_loss(y, p)),
            }
        )

    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True)
        combined.to_csv(out_dir / "reliability_bins.csv", index=False)
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            ax.plot([0, 1], [0, 1], color="#444444", lw=1, ls="--", label="ideal")
            for label, group in combined.groupby("label"):
                g = group.dropna(subset=["mean_predicted", "observed_fraction"])
                ax.plot(g["mean_predicted"], g["observed_fraction"], marker="o", lw=1.8, label=label)
            ax.set_xlabel("Mean predicted edge probability")
            ax.set_ylabel("Observed edge frequency")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / "reliability_curve.png", dpi=220)
            fig.savefig(out_dir / "reliability_curve.pdf")
            plt.close(fig)
        except Exception as exc:
            summaries.append({"plot_status": "failed", "error": str(exc)})

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

