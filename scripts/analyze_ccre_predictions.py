"""Build paper-facing cCRE PR/ROC/calibration and error-analysis artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/graphgenomefm-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    return label, Path(path)


def _probability_column(frame: pd.DataFrame) -> str:
    for column in ("p_positive", "p_ccre", "p_edge", "probability"):
        if column in frame:
            return column
    raise ValueError(f"No probability column found in {list(frame.columns)}")


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    value = 0.0
    for bin_idx in range(n_bins):
        mask = bins == bin_idx
        if mask.any():
            value += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--node-labels", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-errors", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = (
        pd.read_csv(args.node_labels, compression="infer")
        if args.node_labels
        else None
    )
    metrics = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}

    for raw_spec in args.inputs:
        label, path = _spec(raw_spec)
        frame = pd.read_csv(path, compression="infer")
        p_column = _probability_column(frame)
        frame = frame.copy()
        frame["probability"] = frame[p_column].astype(float)
        frame["y_true"] = frame["y_true"].astype(int)
        y = frame["y_true"].to_numpy()
        p = frame["probability"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        calibration_observed, calibration_predicted = calibration_curve(
            y, p, n_bins=10, strategy="quantile"
        )
        metrics.append(
            {
                "model": label,
                "path": str(path),
                "n": int(len(frame)),
                "positive_fraction": float(y.mean()),
                "auroc": float(roc_auc_score(y, p)),
                "auprc": float(average_precision_score(y, p)),
                "brier": float(brier_score_loss(y, p)),
                "ece_10": _ece(y, p, 10),
            }
        )
        curves[label] = {
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
            "calibration_observed": calibration_observed,
            "calibration_predicted": calibration_predicted,
        }
        prediction_frames[label] = frame

        frame["error_type"] = np.where(
            (frame["y_true"] == 0) & (frame["probability"] >= 0.5),
            "false_positive",
            np.where(
                (frame["y_true"] == 1) & (frame["probability"] < 0.5),
                "false_negative",
                "correct",
            ),
        )
        frame["error_confidence"] = np.where(
            frame["y_true"] == 1,
            1.0 - frame["probability"],
            frame["probability"],
        )
        errors = (
            frame[frame["error_type"] != "correct"]
            .sort_values("error_confidence", ascending=False)
            .head(args.top_errors)
        )
        if labels is not None and "segid" in errors:
            errors = errors.merge(labels, on="segid", how="left", suffixes=("", "_label"))
        errors.to_csv(out_dir / f"errors_{label}.csv", index=False)

    metrics_df = pd.DataFrame(metrics).sort_values("auroc", ascending=False)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(metrics_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for row in metrics_df.itertuples(index=False):
        curve = curves[row.model]
        axes[0].plot(curve["fpr"], curve["tpr"], linewidth=1.6, label=f"{row.model} ({row.auroc:.3f})")
        axes[1].plot(
            curve["recall"],
            curve["precision"],
            linewidth=1.6,
            label=f"{row.model} ({row.auprc:.3f})",
        )
        axes[2].plot(
            curve["calibration_predicted"],
            curve["calibration_observed"],
            marker="o",
            linewidth=1.3,
            label=f"{row.model} ({row.brier:.3f})",
        )
    axes[0].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="cCRE ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="cCRE precision–recall")
    axes[2].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axes[2].set(
        xlabel="Mean predicted probability",
        ylabel="Observed cCRE fraction",
        title="cCRE calibration",
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "ccre_roc_pr_calibration.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "ccre_roc_pr_calibration.pdf", bbox_inches="tight")
    plt.close(fig)

    keyed = {
        label: frame[["segid", "probability", "y_true", "chrom"]].rename(
            columns={"probability": label}
        )
        for label, frame in prediction_frames.items()
        if "segid" in frame
    }
    if len(keyed) >= 2:
        merged = None
        for label, frame in keyed.items():
            merged = frame if merged is None else merged.merge(
                frame.drop(columns=["y_true", "chrom"]),
                on="segid",
                how="inner",
            )
        assert merged is not None
        model_columns = list(keyed)
        merged["model_probability_range"] = (
            merged[model_columns].max(axis=1) - merged[model_columns].min(axis=1)
        )
        merged.sort_values("model_probability_range", ascending=False).head(200).to_csv(
            out_dir / "model_disagreement_case_candidates.csv",
            index=False,
        )
    print(metrics_df.to_string(index=False))
    print(f"[ccre-analysis] outputs: {out_dir}")


if __name__ == "__main__":
    main()
