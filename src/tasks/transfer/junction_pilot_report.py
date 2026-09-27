"""Recompute the fixed junction pilot's validation metrics and random controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from tasks.entex.prepare import fingerprint
from tasks.transfer.report import save_figure

ARMS = ["incoming_default", "incoming_linear", "bidirectional_default", "bidirectional_linear"]
IDENTITY = ["slice", "target_sn", "closure", "u_local", "v_local", "y_true"]


def validation_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if not set(frame.split).issubset({"train_chr_val", "val_chr_val"}):
        raise ValueError("Pilot contains unexpected held-out or training predictions")
    val = frame.loc[frame.split.eq("val_chr_val")].sort_values(IDENTITY).reset_index(drop=True)
    if (val.empty or val.duplicated(IDENTITY[:-1]).any() or set(val.y_true) != {0, 1}
            or not val.p_edge.between(0, 1).all()):
        raise ValueError("Invalid or duplicate validation candidates")
    digest = hashlib.sha256(val[IDENTITY].to_csv(index=False).encode()).hexdigest()
    return val, digest


def summarize_roots(trained: Path, random: Path) -> tuple[pd.DataFrame, list[dict]]:
    rows, sources = [], []
    for root, mode in [(trained, "trained"), (random, "frozen_random")]:
        status = json.loads((root / "status.json").read_text())
        if (status.get("status") != "complete" or status.get("heldout_predictions_requested", True)
                or status.get("biological_labels_used", True)
                or set(status["commands"]) != set(ARMS)
                or any(status["returncodes"].get(arm) != 0 for arm in ARMS)):
            raise ValueError("Incomplete or incorrectly scoped pilot")
        sources.append(fingerprint(root / "status.json"))
        for arm in ARMS:
            directory = root / arm / "run_001"
            paths = list(directory.glob("*pooled_predictions*.csv.gz"))
            checkpoints = list(directory.glob("ckpt_*.pt"))
            if len(paths) != 1 or len(checkpoints) != 1:
                raise ValueError("Expected exactly one prediction file and checkpoint per arm")
            val, digest = validation_predictions(pd.read_csv(paths[0], float_precision="round_trip"))
            checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
            cfg = checkpoint["args"]
            if (not cfg["validation_only"] or cfg["seed"] != 42
                    or cfg["junction_geometry_match"] != "signed_gap_bins"):
                raise ValueError("Checkpoint configuration differs from declared pilot")
            is_random = mode == "frozen_random"
            if bool(cfg.get("freeze_encoder", False)) != is_random:
                raise ValueError("Checkpoint random/trained attribution mismatch")
            if is_random and (not checkpoint.get("initial_encoder_sha256")
                              or checkpoint["initial_encoder_sha256"] != checkpoint["final_encoder_sha256"]):
                raise ValueError("Frozen random encoder weights changed")
            macro = np.mean([roc_auc_score(g.y_true, g.p_edge) for _, g in val.groupby("slice")
                             if g.y_true.nunique() == 2])
            if not np.isclose(macro, checkpoint["best_val_auc"], atol=1e-10, rtol=0):
                raise ValueError("Prediction replay differs from selected checkpoint score")
            rows.append(dict(arm=arm, encoder=mode, n=len(val), n_windows=val.slice.nunique(),
                             positive_prevalence=val.y_true.mean(), auprc=average_precision_score(val.y_true, val.p_edge),
                             auroc=roc_auc_score(val.y_true, val.p_edge), macro_window_auroc=macro,
                             epochs_run=checkpoint["epochs_run"], candidates_sha256=digest,
                             checkpoint_sha256=fingerprint(checkpoints[0])["sha256"],
                             prediction_sha256=fingerprint(paths[0])["sha256"],
                             evaluation_partition="development_validation"))
    frame = pd.DataFrame(rows)
    if frame.candidates_sha256.nunique() != 1:
        raise ValueError("Candidate identity or labels differ across trained/random arms")
    return frame, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-root", type=Path, required=True)
    parser.add_argument("--random-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    frame, sources = summarize_roots(args.trained_root, args.random_root)
    contrasts = []
    for arm, group in frame.groupby("arm", sort=False):
        indexed = group.set_index("encoder")
        for metric in ["auprc", "auroc", "macro_window_auroc"]:
            contrasts.append(dict(arm=arm, metric=metric,
                                  trained_minus_random=indexed.loc["trained", metric] - indexed.loc["frozen_random", metric]))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.out_dir / "validation_metrics.csv", index=False)
    pd.DataFrame(contrasts).to_csv(args.out_dir / "paired_differences.csv", index=False)
    selected = frame.loc[frame.encoder.eq("trained")].sort_values("macro_window_auroc", ascending=False).iloc[0]
    (args.out_dir / "audit.json").write_text(json.dumps(dict(
        status="complete", source_receipts=sources, candidate_identity="passed",
        frozen_random_parameter_identity="passed", checkpoint_metric_replay="passed",
        selected_arm=selected.arm, selection_criterion="native window-macro validation AUROC",
        selected_checkpoint_sha256=selected.checkpoint_sha256,
        scope="single-fold development comparison; not independent biological evidence",
        confidence_intervals="not estimated after selection on this development partition",
    ), indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for mode, color, offset in [("trained", "#245a81", -0.15), ("frozen_random", "#be6831", 0.15)]:
        data = frame.loc[frame.encoder.eq(mode)].set_index("arm").loc[ARMS]
        ax.bar(np.arange(4) + offset, data.auprc, width=0.3, label=mode.replace("_", " "), color=color)
    ax.axhline(0.5, color="0.4", ls="--", lw=1, label="Prevalence")
    ax.set(xticks=range(4), xticklabels=["Incoming\nMLP", "Incoming\nLinear", "Bidirectional\nMLP", "Bidirectional\nLinear"],
           ylim=(0, 1), ylabel="Validation AUPRC", title="Junction reconstruction: identical validation candidates")
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, args.out_dir, "junction_validation")
    plt.close(fig)


if __name__ == "__main__":
    main()
